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
	"github.com/google/uuid"
	"go.uber.org/mock/gomock"
)

func TestAgentParallelToolCallsNotDuplicated(t *testing.T) {
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

	// The first model text reply triggers an async session-title request.
	client.EXPECT().GenerateCompletion(gomock.Any(), gomock.Any()).Return(nil, nil).AnyTimes()

	// Iteration 1: text + two parallel calls. Iteration 2: text only.
	firstResp := chatWith(
		fText("Let me check the available contexts and switch if needed."),
		fCalls("mocktool", map[string]any{"command": "kubectl config get-contexts"}),
		fCalls("mocktool", map[string]any{"command": "kubectl config current-context"}),
	)
	secondResp := chatWith(fText("done"))

	firstIter := gollm.ChatResponseIterator(func(yield func(gollm.ChatResponse, error) bool) {
		yield(firstResp, nil)
	})
	secondIter := gollm.ChatResponseIterator(func(yield func(gollm.ChatResponse, error) bool) {
		yield(secondResp, nil)
	})

	gomock.InOrder(
		chat.EXPECT().SendStreaming(gomock.Any(), gomock.Any()).Return(firstIter, nil),
		chat.EXPECT().SendStreaming(gomock.Any(), gomock.Any()).Return(secondIter, nil),
	)

	tool := mocks.NewMockTool(ctrl)
	tool.EXPECT().Name().Return("mocktool").AnyTimes()
	tool.EXPECT().Description().Return("mock tool").AnyTimes()
	tool.EXPECT().FunctionDefinition().Return(&gollm.FunctionDefinition{Name: "mocktool"}).AnyTimes()
	tool.EXPECT().IsInteractive(gomock.Any()).Return(false, nil).AnyTimes()
	tool.EXPECT().CheckModifiesResource(gomock.Any()).Return("no").AnyTimes()
	tool.EXPECT().Run(gomock.Any(), gomock.Any()).Return(map[string]any{"result": "ok"}, nil).Times(2)

	var toolset tools.Tools
	toolset.Init()
	toolset.RegisterTool(tool)

	a := &Agent{
		ChatMessageStore: store,
		LLM:              client,
		Model:            "test-model",
		Tools:            toolset,
		MaxIterations:    4,
		Session:          &api.Session{ID: uuid.New().String(), ChatMessageStore: store, AgentState: api.AgentStateIdle},
	}

	if err := a.Init(ctx); err != nil {
		t.Fatalf("init: %v", err)
	}
	if err := a.Run(ctx, ""); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Wait for prompt, then send the query.
	m1 := recvMsg(t, ctx, a.Output)
	if m1.Type != api.MessageTypeUserInputRequest {
		t.Fatalf("expected user-input-request, got %v", m1.Type)
	}
	a.Input <- &api.UserInputResponse{Query: "wrong kubectx. check again"}

	// Drain output until the agent goes idle/done.
	deadline := time.After(4 * time.Second)
	counts := map[string]int{}
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for agent to finish")
		case v := <-a.Output:
			msg, ok := v.(*api.Message)
			if !ok {
				continue
			}
			if msg.Type == api.MessageTypeToolCallRequest {
				if p, ok := msg.Payload.(string); ok {
					counts[p]++
				}
			}
			if msg.Type == api.MessageTypeText && msg.Source == api.MessageSourceModel {
				if p, ok := msg.Payload.(string); ok {
					counts[p]++
				}
			}
			if a.AgentState() == api.AgentStateDone || a.AgentState() == api.AgentStateIdle {
				goto drained
			}
		}
	}
drained:
	for k, n := range counts {
		if n > 1 {
			t.Errorf("DUPLICATED message %q appears %d times", k, n)
		}
	}
	t.Logf("message counts: %v", counts)
}
