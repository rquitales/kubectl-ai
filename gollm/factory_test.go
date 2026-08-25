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
