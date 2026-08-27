// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gollm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
)

func TestNewClient(t *testing.T) {
	_, err := NewClient(context.Background(), "gemini")
	if err == nil || err.Error() != "GEMINI_API_KEY environment variable not set" {
		t.Fatalf("Unexpected error: %v", err)
	}

	_, err = NewClient(context.Background(), "invalid")
	if err == nil || !strings.Contains(err.Error(), "provider \"invalid\" not registered") {
		t.Fatalf("Unexpected error: %v", err)
	}
}

type stubRetryChat struct {
	calls int
	errs  []error
}

func (s *stubRetryChat) Send(context.Context, ...any) (ChatResponse, error) {
	return nil, nil
}
func (s *stubRetryChat) SendStreaming(context.Context, ...any) (ChatResponseIterator, error) {
	s.calls++
	if s.calls <= len(s.errs) {
		return nil, s.errs[s.calls-1]
	}
	return ChatResponseIterator(func(func(ChatResponse, error) bool) {}), nil
}
func (s *stubRetryChat) SetFunctionDefinitions([]*FunctionDefinition) error { return nil }
func (s *stubRetryChat) IsRetryableError(err error) bool                    { return errors.Is(err, errStubRetryable) }
func (s *stubRetryChat) Initialize([]*api.Message) error                    { return nil }

var errStubRetryable = errors.New("429 rate limited")

func TestRetryChatRetriesSendStreaming(t *testing.T) {
	// The agentic loop only ever calls SendStreaming; previously it bypassed
	// the retry policy entirely, so a transient 429 killed the turn.
	underlying := &stubRetryChat{errs: []error{errStubRetryable, errStubRetryable}}
	chat := NewRetryChat(underlying, RetryConfig{
		MaxAttempts:    3,
		InitialBackoff: time.Millisecond,
		MaxBackoff:     5 * time.Millisecond,
		BackoffFactor:  1,
	})

	if _, err := chat.SendStreaming(context.Background(), "hi"); err != nil {
		t.Fatalf("SendStreaming: %v", err)
	}
	if underlying.calls != 3 {
		t.Errorf("calls = %d, want 3 (2 failures + 1 success)", underlying.calls)
	}

	// Exhausting the budget surfaces the last error.
	underlying2 := &stubRetryChat{errs: []error{errStubRetryable, errStubRetryable, errStubRetryable}}
	chat2 := NewRetryChat(underlying2, RetryConfig{
		MaxAttempts:    2,
		InitialBackoff: time.Millisecond,
		MaxBackoff:     5 * time.Millisecond,
		BackoffFactor:  1,
	})
	if _, err := chat2.SendStreaming(context.Background(), "hi"); !errors.Is(err, errStubRetryable) {
		t.Fatalf("SendStreaming after exhaustion = %v, want %v", err, errStubRetryable)
	}
}

type historyRecordingChat struct {
	history []string
	errs    []error
	calls   int
}

func (s *historyRecordingChat) Send(context.Context, ...any) (ChatResponse, error) { return nil, nil }
func (s *historyRecordingChat) SendStreaming(_ context.Context, contents ...any) (ChatResponseIterator, error) {
	// Eager-history provider shape: contents land in history BEFORE the call.
	for _, c := range contents {
		if text, ok := c.(string); ok {
			s.history = append(s.history, text)
		}
	}
	s.calls++
	if s.calls <= len(s.errs) {
		// Correct behavior: roll back the staged contents on failure.
		s.history = s.history[:len(s.history)-len(contents)]
		return nil, s.errs[s.calls-1]
	}
	return ChatResponseIterator(func(func(ChatResponse, error) bool) {}), nil
}
func (s *historyRecordingChat) SetFunctionDefinitions([]*FunctionDefinition) error { return nil }
func (s *historyRecordingChat) IsRetryableError(err error) bool {
	return errors.Is(err, errStubRetryable)
}
func (s *historyRecordingChat) Initialize([]*api.Message) error { return nil }

func TestRetryChatDoesNotDuplicateHistory(t *testing.T) {
	// Eager providers stage the user contents before the API call; a retry
	// must not leave N copies in the provider-side history.
	underlying := &historyRecordingChat{errs: []error{errStubRetryable, errStubRetryable}}
	chat := NewRetryChat(underlying, RetryConfig{
		MaxAttempts:    3,
		InitialBackoff: time.Millisecond,
		MaxBackoff:     5 * time.Millisecond,
		BackoffFactor:  1,
	})
	if _, err := chat.SendStreaming(context.Background(), "the query"); err != nil {
		t.Fatalf("SendStreaming: %v", err)
	}
	if len(underlying.history) != 1 {
		t.Errorf("history has %d copies of the query after retries, want 1", len(underlying.history))
	}
}

func TestRetryInvokesOnRetryCallback(t *testing.T) {
	underlying := &stubRetryChat{errs: []error{errStubRetryable}}
	var retries []time.Duration
	chat := NewRetryChat(underlying, RetryConfig{
		MaxAttempts:    2,
		InitialBackoff: time.Millisecond,
		MaxBackoff:     5 * time.Millisecond,
		BackoffFactor:  1,
		OnRetry: func(attempt int, err error, wait time.Duration) {
			retries = append(retries, wait)
		},
	})
	if _, err := chat.SendStreaming(context.Background(), "hi"); err != nil {
		t.Fatalf("SendStreaming: %v", err)
	}
	if len(retries) != 1 {
		t.Fatalf("OnRetry called %d times, want 1", len(retries))
	}
}
