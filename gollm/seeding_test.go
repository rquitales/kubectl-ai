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
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
)

func TestSeedableMessagesFilters(t *testing.T) {
	messages := []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "real question"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "real answer"},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: "agent note"},
		// Display-only: must never enter model context.
		{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: "/help output", Ephemeral: true},
		{Source: api.MessageSourceModel, Type: api.MessageTypeThinking, Payload: "pondering", Ephemeral: true},
		// Tool records lack pairing IDs.
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods"},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeToolCallResponse, Payload: map[string]any{"stdout": "x"}},
		// Empty.
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: ""},
	}

	got := SeedableMessages(messages)
	if len(got) != 3 {
		t.Fatalf("SeedableMessages = %d, want 3; got %+v", len(got), got)
	}
	if got[0].Role != "user" || got[1].Role != "assistant" || got[2].Role != "user" {
		t.Errorf("roles = %q/%q/%q, want user/assistant/user (agent text is user-side)", got[0].Role, got[1].Role, got[2].Role)
	}
}

func TestAnthropicInitializeSkipsEphemeral(t *testing.T) {
	c := &anthropicChatSession{}
	err := c.Initialize([]*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeThinking, Payload: "thinking", Ephemeral: true},
	})
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if len(c.messages) != 1 {
		t.Errorf("anthropic history = %d, want 1 (ephemeral skipped)", len(c.messages))
	}
}

func TestGeminiInitializeSkipsEphemeralAndToolRecords(t *testing.T) {
	c := &GeminiChat{}
	err := c.Initialize([]*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods"},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: "/tools output", Ephemeral: true},
	})
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if len(c.history) != 2 {
		t.Errorf("gemini history = %d, want 2 (tool record + ephemeral skipped)", len(c.history))
	}
}

func TestOllamaInitializeSeedsHistory(t *testing.T) {
	c := &OllamaChat{}
	err := c.Initialize([]*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a"},
	})
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if len(c.history) != 2 {
		t.Errorf("ollama history = %d, want 2 (resume amnesia fixed)", len(c.history))
	}
}
