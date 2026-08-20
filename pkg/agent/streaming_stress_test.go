package agent

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/internal/mocks"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"go.uber.org/mock/gomock"
)

// A fast provider emitting hundreds of chunks must not stall the run:
// throttled deltas keep the output channel draining.
func TestAgentStreamingBackpressure(t *testing.T) {
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
	client.EXPECT().GenerateCompletion(gomock.Any(), gomock.Any()).Return(&fakeCompletionResponse{text: "t"}, nil).AnyTimes()

	const chunks = 500
	iter := gollm.ChatResponseIterator(func(yield func(gollm.ChatResponse, error) bool) {
		for i := 0; i < chunks; i++ {
			yield(chatWith(fText(fmt.Sprintf("w%d ", i))), nil)
		}
		yield(chatWith(fText("done.")), nil)
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

	if m := recvMsg(t, ctx, a.Output); m.Type != api.MessageTypeUserInputRequest {
		t.Fatalf("expected user-input-request, got %v", m.Type)
	}
	a.Input <- &api.UserInputResponse{Query: "hi"}

	// Drain like the TUI forwarder does, with per-message render cost.
	deltaCount := 0
	start := time.Now()
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("stalled: run did not finish with %d chunks (%d deltas in %v)", chunks, deltaCount, time.Since(start))
		case m := <-a.Output:
			if msg, ok := m.(*api.Message); ok && msg.Type == api.MessageTypeTextDelta {
				deltaCount++
			}
			if a.AgentState() == api.AgentStateDone {
				if deltaCount > chunks/2 {
					t.Errorf("throttle broken: %d deltas for %d instant chunks", deltaCount, chunks)
				}
				t.Logf("finished: %d chunks -> %d deltas in %v", chunks, deltaCount, time.Since(start))
				return
			}
		}
	}
}
