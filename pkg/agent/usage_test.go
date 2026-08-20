// Copyright 2025 Google LLC
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

package agent

import (
	"encoding/json"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
)

// Provider-shaped usage structs, mirroring what the SDKs return from
// UsageMetadata().
type openAILikeUsage struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
}

type geminiLikeUsage struct {
	PromptTokenCount     int32
	CandidatesTokenCount int32
	TotalTokenCount      int32
}

type anthropicLikeUsage struct {
	InputTokens  int64
	OutputTokens int64
}

func TestUsageTotalTokens(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want int
	}{
		{name: "nil", in: nil, want: 0},
		{name: "empty map", in: map[string]any{}, want: 0},
		{name: "unusable shape", in: "junk", want: 0},
		{name: "map total_tokens", in: map[string]any{"total_tokens": 42}, want: 42},
		{name: "map totalTokens", in: map[string]any{"totalTokens": 42}, want: 42},
		{name: "map totalTokenCount", in: map[string]any{"totalTokenCount": 42}, want: 42},
		{name: "map float64 from JSON", in: map[string]any{"total_tokens": float64(42)}, want: 42},
		{name: "map json.Number", in: map[string]any{"total_tokens": json.Number("42")}, want: 42},
		{name: "map prompt+completion fallback", in: map[string]any{"prompt_tokens": 30, "completion_tokens": 12}, want: 42},
		{name: "map total wins over parts", in: map[string]any{"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 12}, want: 50},
		{name: "map non-numeric", in: map[string]any{"total_tokens": "lots"}, want: 0},
		{name: "struct TotalTokens", in: openAILikeUsage{TotalTokens: 42}, want: 42},
		{name: "struct pointer", in: &openAILikeUsage{TotalTokens: 42}, want: 42},
		{name: "struct TotalTokenCount", in: geminiLikeUsage{TotalTokenCount: 42}, want: 42},
		{name: "struct prompt+completion fallback", in: openAILikeUsage{PromptTokens: 30, CompletionTokens: 12}, want: 42},
		{name: "struct input+output fallback", in: anthropicLikeUsage{InputTokens: 30, OutputTokens: 12}, want: 42},
		{name: "struct zero", in: openAILikeUsage{}, want: 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := usageTotalTokens(c.in); got != c.want {
				t.Errorf("usageTotalTokens(%v) = %d, want %d", c.in, got, c.want)
			}
		})
	}
}

func TestAddModelTextMessageRecordsTokens(t *testing.T) {
	store := sessions.NewInMemoryChatStore()
	a := &Agent{
		Output:  make(chan any, 1),
		Session: &api.Session{ID: "test-session", ChatMessageStore: store},
	}

	msg := a.addModelTextMessage("hello", 123)
	if msg.Tokens != 123 {
		t.Errorf("message Tokens = %d, want 123", msg.Tokens)
	}
	if msg.Source != api.MessageSourceModel || msg.Type != api.MessageTypeText {
		t.Errorf("unexpected message source/type: %v/%v", msg.Source, msg.Type)
	}

	stored := store.ChatMessages()
	if len(stored) != 1 || stored[0].Tokens != 123 {
		t.Fatalf("stored messages = %+v, want one message with 123 tokens", stored)
	}

	select {
	case out := <-a.Output:
		if out.(*api.Message).Tokens != 123 {
			t.Errorf("output message Tokens = %d, want 123", out.(*api.Message).Tokens)
		}
	default:
		t.Error("expected the message on the output channel")
	}
}
