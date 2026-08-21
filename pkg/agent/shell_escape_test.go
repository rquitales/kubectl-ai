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
	"errors"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
)

// fakeExecutor records executed commands and returns a canned result.
type fakeExecutor struct {
	commands []string
	result   *sandbox.ExecResult
	err      error
}

func (f *fakeExecutor) Execute(ctx context.Context, command string, env []string, workDir string) (*sandbox.ExecResult, error) {
	f.commands = append(f.commands, command)
	return f.result, f.err
}

func (f *fakeExecutor) Close(ctx context.Context) error { return nil }

func TestShellEscapeCommand(t *testing.T) {
	cases := []struct {
		in      string
		wantCmd string
		wantOK  bool
	}{
		{"!ls -la", "ls -la", true},
		{"  ! kubectl get pods  ", "kubectl get pods", true},
		{"!", "", true},
		{"ls -la", "", false},
		{"/exit", "", false},
		{"", "", false},
		{"email me!", "", false},
	}
	for _, c := range cases {
		gotCmd, gotOK := shellEscapeCommand(c.in)
		if gotOK != c.wantOK || gotCmd != c.wantCmd {
			t.Errorf("shellEscapeCommand(%q) = %q, %v; want %q, %v", c.in, gotCmd, gotOK, c.wantCmd, c.wantOK)
		}
	}
}

func newShellEscapeTestAgent(executor sandbox.Executor) *Agent {
	a := &Agent{
		Output:   make(chan any, 10),
		executor: executor,
	}
	a.Session = &api.Session{ChatMessageStore: sessions.NewInMemoryChatStore()}
	return a
}

func TestRunShellEscape(t *testing.T) {
	fake := &fakeExecutor{result: &sandbox.ExecResult{Stdout: "hello\n", Stderr: "warn\n"}}
	a := newShellEscapeTestAgent(fake)

	a.runShellEscape(context.Background(), "echo hello")

	if len(fake.commands) != 1 || fake.commands[0] != "echo hello" {
		t.Fatalf("executor commands = %v, want [echo hello]", fake.commands)
	}

	// Transcript gets an agent message formatted like a tool result.
	messages := a.Session.ChatMessageStore.ChatMessages()
	if len(messages) != 1 {
		t.Fatalf("transcript messages = %d, want 1", len(messages))
	}
	msg := messages[0]
	if msg.Source != api.MessageSourceAgent || msg.Type != api.MessageTypeText {
		t.Errorf("message = %v/%v, want agent text", msg.Source, msg.Type)
	}
	payload, _ := msg.Payload.(string)
	if !strings.Contains(payload, "$ echo hello") || !strings.Contains(payload, "hello") || !strings.Contains(payload, "warn") {
		t.Errorf("transcript payload = %q, want command and stdout+stderr", payload)
	}

	// The observation is appended to the chat context for the next turn.
	if len(a.currChatContent) != 1 {
		t.Fatalf("currChatContent = %v, want one observation", a.currChatContent)
	}
	obs, _ := a.currChatContent[0].(string)
	if !strings.Contains(obs, "User ran: echo hello") || !strings.Contains(obs, "Output:\nhello\nwarn") {
		t.Errorf("observation = %q, want command and output", obs)
	}
}

func TestRunShellEscapeExecutorError(t *testing.T) {
	fake := &fakeExecutor{err: errors.New("boom")}
	a := newShellEscapeTestAgent(fake)

	a.runShellEscape(context.Background(), "explode")

	messages := a.Session.ChatMessageStore.ChatMessages()
	if len(messages) != 1 {
		t.Fatalf("transcript messages = %d, want 1", len(messages))
	}
	payload, _ := messages[0].Payload.(string)
	if !strings.Contains(payload, "boom") {
		t.Errorf("transcript payload = %q, want the error", payload)
	}
	if len(a.currChatContent) != 1 {
		t.Fatalf("currChatContent = %v, want one observation", a.currChatContent)
	}
	obs, _ := a.currChatContent[0].(string)
	if !strings.Contains(obs, "User ran: explode") || !strings.Contains(obs, "boom") {
		t.Errorf("observation = %q, want command and error", obs)
	}
}

func TestRunShellEscapeNoExecutor(t *testing.T) {
	a := newShellEscapeTestAgent(nil)

	// Must not panic with a nil executor.
	a.runShellEscape(context.Background(), "ls")

	messages := a.Session.ChatMessageStore.ChatMessages()
	if len(messages) != 1 {
		t.Fatalf("transcript messages = %d, want 1", len(messages))
	}
	if messages[0].Type != api.MessageTypeError {
		t.Errorf("message type = %v, want error", messages[0].Type)
	}
	if len(a.currChatContent) != 0 {
		t.Errorf("currChatContent = %v, want no observation without an executor", a.currChatContent)
	}
}
