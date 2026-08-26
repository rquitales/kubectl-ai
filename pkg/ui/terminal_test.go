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

package ui

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
)

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = orig
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func newTestTerminalUI(t *testing.T) *TerminalUI {
	t.Helper()
	a := &agent.Agent{Session: &api.Session{ID: "test"}}
	u, err := NewTerminalUI(a, false, false, &journal.LogRecorder{})
	if err != nil {
		t.Fatalf("NewTerminalUI: %v", err)
	}
	return u
}

func TestTerminalUIStreamsDeltasLive(t *testing.T) {
	u := newTestTerminalUI(t)

	out := captureStdout(t, func() {
		u.handleMessage(&api.Message{ID: "s1", Source: api.MessageSourceModel, Type: api.MessageTypeTextDelta, Payload: "Hello"})
		u.handleMessage(&api.Message{ID: "s1", Source: api.MessageSourceModel, Type: api.MessageTypeTextDelta, Payload: "Hello, world"})
		// The final message (same ID) only closes the line — no duplicate.
		u.handleMessage(&api.Message{ID: "s1", Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "Hello, world"})
	})

	if got := strings.Count(out, "Hello"); got != 1 {
		t.Errorf("final message duplicated streamed content: %q", out)
	}
	if !strings.Contains(out, ", world") {
		t.Errorf("delta suffix missing: %q", out)
	}
	if !strings.HasSuffix(out, "\n") {
		t.Errorf("final message should close the line: %q", out)
	}
}

func TestTerminalUIThinkingDeltasSilent(t *testing.T) {
	u := newTestTerminalUI(t)
	out := captureStdout(t, func() {
		u.handleMessage(&api.Message{ID: "t1", Source: api.MessageSourceModel, Type: api.MessageTypeThinkingDelta, Payload: "hmm"})
		u.handleMessage(&api.Message{ID: "t1", Source: api.MessageSourceModel, Type: api.MessageTypeThinking, Payload: "hmm\nmore", Ephemeral: true})
	})
	if strings.Contains(out, "hmm") || strings.Contains(out, "more") {
		t.Errorf("thinking content should not print on the plain CLI: %q", out)
	}
	if !strings.Contains(out, "thought for 2 lines") {
		t.Errorf("expected a one-line thinking note, got: %q", out)
	}
}

func TestTerminalUISafeOnNonStringPayloads(t *testing.T) {
	u := newTestTerminalUI(t)
	// Must not panic on unexpected payload types.
	captureStdout(t, func() {
		u.handleMessage(&api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: 42})
		u.handleMessage(&api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeError, Payload: map[string]any{"x": 1}})
		u.handleMessage(&api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeToolCallRequest, Payload: nil})
	})
}
