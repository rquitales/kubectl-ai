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

package ui

import (
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	tea "github.com/charmbracelet/bubbletea"
)

func pasteMsg(s string) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s), Paste: true}
}

func TestHandlePasteCollapsesLargePastes(t *testing.T) {
	m := newModel(nil)

	pasted := "line one\nline two\nline three\nline four"
	if _, cmd := m.handleKey(pasteMsg(pasted)); cmd != nil {
		t.Fatalf("expected no command from paste, got %v", cmd)
	}

	if len(m.pastes) != 1 {
		t.Fatalf("expected 1 pasted block, got %d", len(m.pastes))
	}
	if m.pastes[0].content != pasted {
		t.Errorf("stored paste content = %q, want %q", m.pastes[0].content, pasted)
	}
	if m.pastes[0].lines != 4 {
		t.Errorf("stored paste lines = %d, want 4", m.pastes[0].lines)
	}
	// The paste must NOT flood the input; it is attached as a chip.
	if got := m.input.Value(); got != "" {
		t.Errorf("input value = %q, want empty (paste attached as chip)", got)
	}
}

func TestHandlePasteInsertsShortPastesVerbatim(t *testing.T) {
	m := newModel(nil)

	_, _ = m.handleKey(pasteMsg("just one line"))
	if len(m.pastes) != 0 {
		t.Fatalf("expected no pasted blocks for short paste, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "just one line" {
		t.Errorf("input value = %q, want %q", got, "just one line")
	}

	// Two lines (below collapse threshold) are inserted verbatim.
	m.input.Reset()
	_, _ = m.handleKey(pasteMsg("one\ntwo"))
	if len(m.pastes) != 0 {
		t.Fatalf("expected no pasted blocks for two-line paste, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "one\ntwo" {
		t.Errorf("input value = %q, want %q", got, "one\ntwo")
	}
}

func TestHandlePasteTrimsTrailingNewlineAndNormalizesCRLF(t *testing.T) {
	m := newModel(nil)

	_, _ = m.handleKey(pasteMsg("hello\r\n"))
	if len(m.pastes) != 0 {
		t.Fatalf("expected no pasted blocks, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "hello" {
		t.Errorf("input value = %q, want %q", got, "hello")
	}
}

func TestCtrlJInsertsNewline(t *testing.T) {
	m := newModel(nil)

	m.input.SetValue("first")
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlJ})
	if !strings.Contains(m.input.Value(), "\n") {
		t.Errorf("expected ctrl+j to insert a newline, input = %q", m.input.Value())
	}
}

func TestSubmitAttachesPasteContent(t *testing.T) {
	m := newModel(nil)

	big := "alpha\nbeta\ngamma\ndelta"
	_, _ = m.handleKey(pasteMsg(big))
	if len(m.pastes) != 1 {
		t.Fatalf("expected 1 pasted block, got %d", len(m.pastes))
	}

	// Submit; note we deliberately do not run the returned tea.Cmd, which is
	// what would deliver the message to the agent.
	_, _ = m.handleEnter()
	if len(m.messages) != 1 {
		t.Fatalf("expected 1 message after submit, got %d", len(m.messages))
	}
	if got := m.messages[0].Payload; got != big {
		t.Errorf("submitted payload = %q, want attached paste %q", got, big)
	}
	if len(m.pastes) != 0 {
		t.Errorf("expected pastes to be cleared after submit, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("expected input to be cleared after submit, got %q", got)
	}
}

func TestSubmitJoinsDraftAndPasteContent(t *testing.T) {
	m := newModel(nil)

	m.input.SetValue("what does this do?")
	_, _ = m.handleKey(pasteMsg("a\nb\nc\nd"))

	_, _ = m.handleEnter()
	if len(m.messages) != 1 {
		t.Fatalf("expected 1 message after submit, got %d", len(m.messages))
	}
	want := "what does this do?\n\na\nb\nc\nd"
	if got := m.messages[0].Payload; got != want {
		t.Errorf("submitted payload = %q, want %q", got, want)
	}
}

func TestBackspaceDetachesLastPaste(t *testing.T) {
	m := newModel(nil)

	_, _ = m.handleKey(pasteMsg("a\nb\nc\nd"))
	_, _ = m.handleKey(pasteMsg("e\nf\ng\nh"))
	if len(m.pastes) != 2 {
		t.Fatalf("expected 2 pasted blocks, got %d", len(m.pastes))
	}

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if len(m.pastes) != 1 {
		t.Fatalf("expected backspace to detach one paste, got %d", len(m.pastes))
	}
	if m.pastes[0].content != "a\nb\nc\nd" {
		t.Errorf("wrong paste detached, remaining = %q", m.pastes[0].content)
	}

	// Detaching the last one; submitting with no input sends nothing.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if len(m.pastes) != 0 {
		t.Fatalf("expected all pastes detached, got %d", len(m.pastes))
	}
	_, _ = m.handleEnter()
	if len(m.messages) != 0 {
		t.Fatalf("expected no messages after submitting empty input, got %d", len(m.messages))
	}
}

func TestEscClearsInputAndPastes(t *testing.T) {
	m := newModel(nil)

	_, _ = m.handleKey(pasteMsg("a\nb\nc\nd"))
	if len(m.pastes) != 1 {
		t.Fatalf("expected 1 pasted block, got %d", len(m.pastes))
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if len(m.pastes) != 0 {
		t.Errorf("expected esc to clear pastes, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("expected esc to clear input, got %q", got)
	}
}

func TestViewRendersWithMultiLineInput(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	m.input.SetValue("line one\nline two\nline three")
	m.syncInputHeight()
	if m.inputHeight != 3 {
		t.Errorf("inputHeight = %d, want 3", m.inputHeight)
	}

	// Must not panic and must produce output for both small and tall inputs.
	if got := m.View(); got == "" {
		t.Error("expected non-empty view")
	}
}

// Regression test: after typing on line 1 and inserting a newline with
// Ctrl+J, all lines must remain visible (previously the textarea's internal
// viewport scrolled and hid the first line).
func TestCtrlJKeepsAllLinesVisible(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	m.input.SetValue("first line")
	m.syncInputHeight()
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlJ})
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("second")})

	view := m.View()
	if !strings.Contains(view, "first line") {
		t.Errorf("first line not visible after ctrl+j; view:\n%s", view)
	}
	if !strings.Contains(view, "second") {
		t.Errorf("second line not visible after ctrl+j; view:\n%s", view)
	}
}

// Long lines that soft-wrap must also grow the input box.
func TestSoftWrapGrowsInput(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	wrapWidth := m.input.Width()
	long := strings.Repeat("x", wrapWidth*2)
	m.input.SetValue(long)
	m.syncInputHeight()
	if m.inputHeight < 2 {
		t.Errorf("inputHeight = %d, want at least 2 for a soft-wrapped line", m.inputHeight)
	}
}

func TestVisualLines(t *testing.T) {
	cases := []struct {
		s     string
		width int
		want  int
	}{
		{"", 10, 1},
		{"hello", 10, 1},
		{"0123456789", 10, 1},
		{"0123456789a", 10, 2},
		{"a\nb", 10, 2},
		{"a\n\nb", 10, 3},
		{strings.Repeat("x", 25), 10, 3},
	}
	for _, c := range cases {
		if got := visualLines(c.s, c.width); got != c.want {
			t.Errorf("visualLines(%q, %d) = %d, want %d", c.s, c.width, got, c.want)
		}
	}
}

func TestAltEnterInsertsNewline(t *testing.T) {
	m := newModel(nil)
	m.input.SetValue("first")

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyEnter, Alt: true})
	if !strings.Contains(m.input.Value(), "\n") {
		t.Errorf("expected alt+enter to insert a newline, input = %q", m.input.Value())
	}
	if len(m.messages) != 0 {
		t.Errorf("alt+enter must not submit, got %d messages", len(m.messages))
	}
}

func historyTestMessages() []*api.Message {
	return []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "first query"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "answer one"},
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "second query"},
		{Source: api.MessageSourceUser, Type: api.MessageTypeToolCallResponse, Payload: "not text"},
	}
}

func TestHistoryNavigation(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	press := func(k tea.KeyType) {
		_, _ = m.handleKey(tea.KeyMsg{Type: k})
	}

	press(tea.KeyUp)
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after 1st up: input = %q, want %q", got, "second query")
	}
	press(tea.KeyUp)
	if got := m.input.Value(); got != "first query" {
		t.Errorf("after 2nd up: input = %q, want %q", got, "first query")
	}
	press(tea.KeyUp) // at oldest: stays
	if got := m.input.Value(); got != "first query" {
		t.Errorf("at oldest: input = %q, want %q", got, "first query")
	}
	press(tea.KeyDown)
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after down: input = %q, want %q", got, "second query")
	}
	press(tea.KeyDown) // past newest: restores (empty) draft
	if got := m.input.Value(); got != "" {
		t.Errorf("past newest: input = %q, want %q", got, "")
	}
	if m.historyIdx != -1 {
		t.Errorf("expected history navigation to end, historyIdx = %d", m.historyIdx)
	}
}

func TestHistoryRestoresDraft(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	m.input.SetValue("my draft")
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Value(); got != "second query" {
		t.Fatalf("after up: input = %q, want %q", got, "second query")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if got := m.input.Value(); got != "my draft" {
		t.Errorf("after down: input = %q, want draft %q", got, "my draft")
	}
}

func TestUpMovesCursorWithinMultiLineDraft(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	m.input.SetValue("line one\nline two")
	// Cursor is at the end (line 2). Up must move the cursor, not recall history.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Line(); got != 0 {
		t.Errorf("cursor line = %d, want 0", got)
	}
	if got := m.input.Value(); got != "line one\nline two" {
		t.Errorf("draft changed during cursor movement: %q", got)
	}
	// On the first line now: Up recalls history.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after up on first line: input = %q, want %q", got, "second query")
	}
}

func TestHistorySkipsConsecutiveDuplicates(t *testing.T) {
	m := newModel(nil)
	m.messages = []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "same"},
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "same"},
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "other"},
	}
	m.rebuildInputHistory()
	if len(m.inputHistory) != 2 {
		t.Errorf("expected 2 deduped history entries, got %d: %v", len(m.inputHistory), m.inputHistory)
	}
}

// Regression test for terminals that send CRLF for Return: the CR submits,
// and the following LF must be swallowed (not turned into a phantom newline
// in the next draft).
func TestPhantomLFAfterSubmitIsSwallowed(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)
	m.input.SetValue("abc")

	// CR: submits.
	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("expected enter to submit")
	}
	go cmd()
	if got := <-a.Input; got.(*api.UserInputResponse).Query != "abc" {
		t.Fatalf("expected query %q", "abc")
	}

	// Phantom LF: must be swallowed.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlJ})
	if got := m.input.Value(); got != "" {
		t.Errorf("phantom LF was not swallowed: value = %q", got)
	}

	// Typing afterwards starts on the first line of a fresh draft.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if got := m.input.Value(); got != "d" {
		t.Errorf("value = %q, want %q", got, "d")
	}
	if m.inputHeight != 1 {
		t.Errorf("inputHeight = %d, want 1", m.inputHeight)
	}
}
