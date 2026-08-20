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
	"context"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/internal/mocks"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"go.uber.org/mock/gomock"
)

// TestAgentStreamsTextDeltas verifies that a streamed model turn emits
// ephemeral text-delta messages (all sharing one ID, carrying the text
// accumulated so far) followed by a single final text message with the same
// ID, and that only the final message is persisted to the session store.
func TestAgentStreamsTextDeltas(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	store := sessions.NewInMemoryChatStore()

	client := mocks.NewMockClient(ctrl)
	chat := mocks.NewMockChat(ctrl)

	client.EXPECT().StartChat(gomock.Any(), "test-model").Return(chat)
	chat.EXPECT().Initialize(gomock.Any()).Return(nil)
	chat.EXPECT().SetFunctionDefinitions(gomock.Any()).Return(nil)
	// The first model text also triggers async session-title generation.
	client.EXPECT().GenerateCompletion(gomock.Any(), gomock.Any()).Return(&fakeCompletionResponse{text: "a title"}, nil).AnyTimes()

	iter := gollm.ChatResponseIterator(func(yield func(gollm.ChatResponse, error) bool) {
		yield(chatWith(fText("Hello, ")), nil)
		yield(chatWith(fText("world")), nil)
		yield(chatWith(fText("!")), nil)
	})
	chat.EXPECT().SendStreaming(gomock.Any(), gomock.Any()).Return(iter, nil)

	var toolset tools.Tools
	toolset.Init()

	a := &Agent{
		ChatMessageStore: store,
		LLM:              client,
		Model:            "test-model",
		Tools:            toolset,
		MaxIterations:    4,
		Session: &api.Session{
			ID:               "test-session",
			ChatMessageStore: store,
			AgentState:       api.AgentStateIdle,
		},
	}

	if err := a.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}
	if err := a.Run(ctx, ""); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Startup prompt, then send a query (UI -> Agent).
	if m := recvMsg(t, ctx, a.Output); m.Type != api.MessageTypeUserInputRequest {
		t.Fatalf("expected user-input-request, got %v", m.Type)
	}
	a.Input <- &api.UserInputResponse{Query: "hi"}

	// The user's query is echoed back first.
	if m := recvMsg(t, ctx, a.Output); m.Type != api.MessageTypeText || m.Source != api.MessageSourceUser {
		t.Fatalf("expected user text message, got %v (%v)", m.Type, m.Source)
	}

	// Collect deltas until the final model text message arrives.
	var deltas []*api.Message
	var final *api.Message
	for final == nil {
		m := recvMsg(t, ctx, a.Output)
		switch {
		case m.Type == api.MessageTypeTextDelta:
			deltas = append(deltas, m)
		case m.Type == api.MessageTypeText && m.Source == api.MessageSourceModel:
			final = m
		}
	}

	// Deltas carry the accumulated text, share one ID, and precede the final.
	wantPayloads := []string{"Hello, ", "Hello, world", "Hello, world!"}
	if len(deltas) != len(wantPayloads) {
		t.Fatalf("expected %d delta messages, got %d", len(wantPayloads), len(deltas))
	}
	for i, d := range deltas {
		if d.Source != api.MessageSourceModel {
			t.Errorf("delta %d source = %v, want model", i, d.Source)
		}
		if d.Payload != wantPayloads[i] {
			t.Errorf("delta %d payload = %q, want %q", i, d.Payload, wantPayloads[i])
		}
		if d.ID == "" || d.ID != deltas[0].ID {
			t.Errorf("delta %d ID = %q, want all deltas to share one ID", i, d.ID)
		}
	}

	// The final message reuses the stream ID and holds the complete text.
	if final.ID != deltas[0].ID {
		t.Errorf("final message ID = %q, want stream ID %q", final.ID, deltas[0].ID)
	}
	if final.Payload != "Hello, world!" {
		t.Errorf("final payload = %q, want %q", final.Payload, "Hello, world!")
	}

	// The store holds ONLY the user message and the final message: deltas
	// are never persisted.
	stored := store.ChatMessages()
	if len(stored) != 2 {
		t.Fatalf("expected 2 stored messages (user + final), got %d", len(stored))
	}
	if stored[0].Source != api.MessageSourceUser {
		t.Errorf("stored[0] source = %v, want user", stored[0].Source)
	}
	if stored[1].ID != final.ID || stored[1].Type != api.MessageTypeText || stored[1].Payload != "Hello, world!" {
		t.Errorf("stored[1] = %+v, want the final text message", stored[1])
	}
	for _, m := range stored {
		if m.Type == api.MessageTypeTextDelta {
			t.Errorf("text-delta must never be stored, got %+v", m)
		}
	}
}
