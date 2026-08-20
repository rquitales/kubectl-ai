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
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/internal/mocks"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"go.uber.org/mock/gomock"
)

func TestSessionTitleFromResponse(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "trims whitespace", in: "  Debug Crashlooping Pods  ", want: "Debug Crashlooping Pods"},
		{name: "strips quotes", in: `"Nginx Crashloop Debug"`, want: "Nginx Crashloop Debug"},
		{name: "strips control characters", in: "pods\ncrash", want: "podscrash"},
		{name: "empty", in: "   ", want: ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := sessionTitleFromResponse(c.in); got != c.want {
				t.Errorf("sessionTitleFromResponse(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}

	long := strings.Repeat("a very long title word ", 10)
	if got := sessionTitleFromResponse(long); len([]rune(got)) > 60 {
		t.Errorf("expected title capped at 60 runes, got %d (%q)", len([]rune(got)), got)
	}
}

func TestTitleSnippets(t *testing.T) {
	messages := []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "why is\nnginx\t crashlooping"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "Let me check the logs."},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "second reply ignored"},
	}
	user, model := titleSnippets(messages)
	if user != "why is nginx crashlooping" {
		t.Errorf("user snippet = %q", user)
	}
	if model != "Let me check the logs." {
		t.Errorf("model snippet = %q", model)
	}

	long := strings.Repeat("x", 400)
	user, _ = titleSnippets([]*api.Message{{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: long}})
	if len([]rune(user)) != titleSnippetLen {
		t.Errorf("expected snippet capped at %d runes, got %d", titleSnippetLen, len([]rune(user)))
	}

	if user, model := titleSnippets(nil); user != "" || model != "" {
		t.Errorf("expected empty snippets for no messages, got %q/%q", user, model)
	}
}

func TestGenerateSessionTitle(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	llm := mocks.NewMockClient(ctrl)
	var gotPrompt string
	llm.EXPECT().GenerateCompletion(gomock.Any(), gomock.Any()).
		DoAndReturn(func(_ context.Context, req *gollm.CompletionRequest) (gollm.CompletionResponse, error) {
			gotPrompt = req.Prompt
			return &fakeCompletionResponse{text: `"Nginx Crashloop Debug"`}, nil
		})

	a := &Agent{LLM: llm, Model: "test-model"}
	title, err := a.generateSessionTitle(context.Background(), "why is nginx crashlooping", "Let me check the logs.")
	if err != nil {
		t.Fatalf("generateSessionTitle failed: %v", err)
	}
	if title != "Nginx Crashloop Debug" {
		t.Errorf("title = %q, want %q", title, "Nginx Crashloop Debug")
	}
	for _, want := range []string{"User: why is nginx crashlooping", "Assistant: Let me check the logs.", "Title:"} {
		if !strings.Contains(gotPrompt, want) {
			t.Errorf("prompt missing %q:\n%s", want, gotPrompt)
		}
	}
}

// waitForTitle polls until the agent's session gets a name or the deadline
// passes; returns the final name.
func waitForTitle(a *Agent, timeout time.Duration) string {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if name := a.GetSession().Name; name != "" {
			return name
		}
		time.Sleep(10 * time.Millisecond)
	}
	return a.GetSession().Name
}

func TestMaybeGenerateSessionTitleFlow(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mgr, err := sessions.NewSessionManager("memory")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	s, err := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	t.Cleanup(func() { _ = mgr.DeleteSession(s.ID) })

	if err := s.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "why is nginx crashlooping"}); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	if err := s.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "Let me check the logs."}); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}

	llm := mocks.NewMockClient(ctrl)
	// Exactly one completion, even though we trigger the check twice.
	llm.EXPECT().GenerateCompletion(gomock.Any(), gomock.Any()).
		Return(&fakeCompletionResponse{text: "Nginx Crashloop Debug"}, nil).
		Times(1)

	a := &Agent{LLM: llm, Model: "m", Session: s, SessionBackend: "memory"}
	a.maybeGenerateSessionTitle()
	a.maybeGenerateSessionTitle() // second call must be a no-op

	if got := waitForTitle(a, 2*time.Second); got != "Nginx Crashloop Debug" {
		t.Fatalf("session name = %q, want %q", got, "Nginx Crashloop Debug")
	}

	reloaded, err := mgr.FindSessionByID(s.ID)
	if err != nil {
		t.Fatalf("FindSessionByID: %v", err)
	}
	if reloaded.Name != "Nginx Crashloop Debug" {
		t.Errorf("persisted name = %q, want %q", reloaded.Name, "Nginx Crashloop Debug")
	}
	if reloaded.ManuallyNamed {
		t.Error("LLM-generated title must not be marked as manual")
	}
	if !a.llmTitleGenerated() {
		t.Error("expected titleGenerated to be set after a successful title")
	}
}

func TestMaybeGenerateSessionTitleSkipsNamedSessions(t *testing.T) {
	cases := []struct {
		name          string
		sessionName   string
		manuallyNamed bool
	}{
		{name: "manual name", sessionName: "mine", manuallyNamed: true},
		{name: "existing auto name", sessionName: "previous title", manuallyNamed: false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mgr, err := sessions.NewSessionManager("memory")
			if err != nil {
				t.Fatalf("NewSessionManager: %v", err)
			}
			s, err := mgr.NewSession(sessions.Metadata{ModelID: "m"})
			if err != nil {
				t.Fatalf("NewSession: %v", err)
			}
			t.Cleanup(func() { _ = mgr.DeleteSession(s.ID) })
			if err := mgr.SetSessionName(s.ID, c.sessionName, c.manuallyNamed); err != nil {
				t.Fatalf("SetSessionName: %v", err)
			}

			// No GenerateCompletion expectation: any call fails the test.
			llm := mocks.NewMockClient(ctrl)
			a := &Agent{LLM: llm, Model: "m", Session: s, SessionBackend: "memory"}
			a.maybeGenerateSessionTitle()

			if a.titleAttempted {
				t.Error("titleAttempted must stay false when the check is skipped")
			}
			time.Sleep(100 * time.Millisecond)
			if got := a.GetSession().Name; got != c.sessionName {
				t.Errorf("session name = %q, want unchanged %q", got, c.sessionName)
			}
		})
	}
}

func TestAgentCloseKeepsLLMGeneratedTitle(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)

	mgr, err := sessions.NewSessionManager("filesystem")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	s, err := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	if err := s.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "why is nginx crashlooping"}); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	if err := s.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "Let me investigate."}); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}

	a := &Agent{Session: s, SessionBackend: "filesystem", titleGenerated: true}
	if err := mgr.SetSessionName(s.ID, "Nginx Crashloop Debug", false); err != nil {
		t.Fatalf("SetSessionName: %v", err)
	}
	s.Name = "Nginx Crashloop Debug"

	if err := a.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	reloaded, err := mgr.FindSessionByID(s.ID)
	if err != nil {
		t.Fatalf("session missing after close: %v", err)
	}
	if reloaded.Name != "Nginx Crashloop Debug" {
		t.Errorf("LLM title was overridden on exit: %q", reloaded.Name)
	}
}
