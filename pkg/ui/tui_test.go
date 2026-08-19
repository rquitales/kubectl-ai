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
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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
	if m.pastes[0].token != "[+4 lines]" {
		t.Errorf("stored paste token = %q, want %q", m.pastes[0].token, "[+4 lines]")
	}
	// The paste must NOT flood the input; only a placeholder is inserted.
	if got := m.input.Value(); got != "[+4 lines]" {
		t.Errorf("input value = %q, want placeholder token", got)
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

	altKey := func(r rune) {
		_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}, Alt: true})
	}

	altKey('p')
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after 1st alt+p: input = %q, want %q", got, "second query")
	}
	altKey('p')
	if got := m.input.Value(); got != "first query" {
		t.Errorf("after 2nd alt+p: input = %q, want %q", got, "first query")
	}
	altKey('p') // at oldest: stays
	if got := m.input.Value(); got != "first query" {
		t.Errorf("at oldest: input = %q, want %q", got, "first query")
	}
	altKey('n')
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after alt+n: input = %q, want %q", got, "second query")
	}
	altKey('n') // past newest: restores (empty) draft
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
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("p"), Alt: true})
	if got := m.input.Value(); got != "second query" {
		t.Fatalf("after ctrl+p: input = %q, want %q", got, "second query")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n"), Alt: true})
	if got := m.input.Value(); got != "my draft" {
		t.Errorf("after ctrl+n: input = %q, want draft %q", got, "my draft")
	}
}

func TestUpMovesCursorWithinMultiLineDraft(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	m.input.SetValue("line one\nline two")
	// Cursor is at the end (line 2). Up must move the cursor.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Line(); got != 0 {
		t.Errorf("cursor line = %d, want 0", got)
	}
	if got := m.input.Value(); got != "line one\nline two" {
		t.Errorf("draft changed during cursor movement: %q", got)
	}
}

func TestUpDownScrollsViewportOnSingleLineDraft(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 24
	m.resize()
	// Fill the transcript so it scrolls.
	for i := 0; i < 50; i++ {
		m.messages = append(m.messages, &api.Message{
			Source: api.MessageSourceModel, Type: api.MessageTypeText,
			Payload: fmt.Sprintf("line %d", i), Timestamp: time.Now(),
		})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()
	atBottom := m.viewport.YOffset
	if atBottom == 0 {
		t.Fatal("precondition: viewport should be scrolled to the bottom with tall content")
	}

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if m.viewport.YOffset >= atBottom {
		t.Error("expected KeyUp to scroll the viewport up (away from bottom)")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if m.viewport.YOffset != atBottom {
		t.Errorf("expected KeyDown to scroll back to bottom: YOffset = %d, want %d", m.viewport.YOffset, atBottom)
	}
	// History must NOT be triggered by plain Up/Down.
	if got := m.input.Value(); got != "" {
		t.Errorf("expected input to stay empty on Up/Down scroll, got %q", got)
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

func testSessions() []api.SessionInfo {
	return []api.SessionInfo{
		{ID: "s1", Name: "first", ModelID: "model-a", LastModified: time.Now(), MessageCount: 3},
		{ID: "s2", Name: "second", ModelID: "model-b", LastModified: time.Now().Add(-time.Hour), MessageCount: 5},
		{ID: "s3", ModelID: "model-c", LastModified: time.Now().Add(-24 * time.Hour), MessageCount: 1},
	}
}

func newBrowserModel() model {
	a := &agent.Agent{
		Session: &api.Session{ID: "s2", AgentState: api.AgentStateIdle},
		Input:   make(chan any, 1),
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	return m
}

func TestBrowserOpensWithCurrentSessionSelected(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	if !m.browserOpen {
		t.Fatal("expected browser to be open")
	}
	if m.browserIndex != 1 {
		t.Errorf("browserIndex = %d, want 1 (current session s2)", m.browserIndex)
	}
}

func TestBrowserNavigationWraps(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	m.moveBrowserSelection(-1)
	if m.browserIndex != 0 {
		t.Errorf("browserIndex = %d, want 0", m.browserIndex)
	}
	m.moveBrowserSelection(-1)
	if m.browserIndex != 2 {
		t.Errorf("browserIndex = %d, want 2 (wrapped)", m.browserIndex)
	}
	m.moveBrowserSelection(1)
	if m.browserIndex != 0 {
		t.Errorf("browserIndex = %d, want 0 (wrapped)", m.browserIndex)
	}
}

func TestBrowserEnterSwitchesSession(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	_, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyEnter})
	if m.browserOpen {
		t.Error("expected browser to close after enter")
	}
	if cmd == nil {
		t.Fatal("expected a command sending the picker response")
	}
	go cmd()
	got := <-m.agent.Input
	resp, ok := got.(*api.SessionPickerResponse)
	if !ok {
		t.Fatalf("expected *api.SessionPickerResponse, got %T", got)
	}
	if resp.SessionID != "s2" {
		t.Errorf("SessionID = %q, want %q", resp.SessionID, "s2")
	}
}

func TestBrowserRenameFlow(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	// 'r' starts rename with the current name prefilled.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
	if !m.renaming {
		t.Fatal("expected rename mode to start")
	}
	if got := m.renameInput.Value(); got != "second" {
		t.Errorf("rename input = %q, want %q", got, "second")
	}

	// Esc cancels without renaming.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.renaming {
		t.Error("expected rename mode to end on esc")
	}
}

func TestBrowserNewSession(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	// Single 'n' must NOT create a session (too easy to hit accidentally).
	if _, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")}); cmd != nil {
		t.Error("expected 'n' alone to be a no-op")
	}
	if !m.browserOpen {
		t.Error("expected browser to stay open on 'n'")
	}

	_, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyCtrlN})
	if m.browserOpen {
		t.Error("expected browser to close on ctrl+n")
	}
	if cmd == nil {
		t.Fatal("expected a command sending the new-session request")
	}
	go cmd()
	got := <-m.agent.Input
	if _, ok := got.(*api.NewSessionRequest); !ok {
		t.Fatalf("expected *api.NewSessionRequest, got %T", got)
	}
}

func TestBrowserPasteGoesToRenameField(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	// Start renaming, then paste: it must land in the rename field, not the
	// hidden main input.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
	if !m.renaming {
		t.Fatal("expected rename mode to start")
	}
	_, _ = m.handleBrowserKey(pasteMsg("pasted-name"))
	if got := m.renameInput.Value(); got != "secondpasted-name" {
		t.Errorf("rename input = %q, want %q", got, "secondpasted-name")
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("main input must stay empty, got %q", got)
	}
}

func TestBrowserPasteOutsideRenameIsSwallowedWithNote(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	_, _ = m.handleBrowserKey(pasteMsg("a\nb\nc\nd"))
	if len(m.pastes) != 0 {
		t.Errorf("expected no pastes attached while browsing, got %d", len(m.pastes))
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("main input must stay empty, got %q", got)
	}
	if m.browserStatus.text == "" {
		t.Error("expected a footer note explaining the paste was swallowed")
	}
}

func TestBrowserEnterQueuesSwitchWhenAgentBusy(t *testing.T) {
	m := newBrowserModel()
	m.agent.Session.AgentState = api.AgentStateRunning
	m.openBrowser(testSessions())

	_, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyEnter})
	if !m.browserOpen {
		t.Error("expected browser to stay open when the agent is busy")
	}
	if m.browserStatus.text == "" {
		t.Error("expected a status note about the queued switch")
	}
	if cmd == nil {
		t.Fatal("expected a command sending the picker response")
	}
	go cmd()
	got := <-m.agent.Input
	if resp, ok := got.(*api.SessionPickerResponse); !ok || resp.SessionID != "s2" {
		t.Errorf("expected picker response for s2, got %v", got)
	}
}

func TestBrowserAdaptsRowsToSmallTerminal(t *testing.T) {
	m := newBrowserModel()
	m.height = 22
	sessions := testSessions()
	for i := 0; i < 10; i++ {
		sessions = append(sessions, api.SessionInfo{ID: fmt.Sprintf("sx%d", i), ModelID: "m", LastModified: time.Now()})
	}
	m.openBrowser(sessions)
	m.updateViewportHeight()

	frameHeight := lipgloss.Height(m.View())
	if frameHeight > m.height {
		t.Errorf("frame height %d exceeds terminal height %d", frameHeight, m.height)
	}
}

func TestBrowserRenameErrorShownInFooter(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	updated, _ := m.Update(sessionRenamedMsg{err: errTest})
	got := updated.(model)
	if got.browserStatus.text == "" || !got.browserStatus.isErr {
		t.Errorf("expected rename error in browser footer, got %+v", got.browserStatus)
	}
	if !got.browserOpen {
		t.Error("expected browser to stay open after rename error")
	}
}

var errTest = &testError{}

type testError struct{}

func (e *testError) Error() string { return "boom" }

func TestBrowserShrinksViewport(t *testing.T) {
	m := newBrowserModel()
	before := m.viewport.Height
	m.openBrowser(testSessions())
	m.updateViewportHeight()
	if m.viewport.Height >= before {
		t.Errorf("expected viewport to shrink with browser open: before=%d after=%d", before, m.viewport.Height)
	}

	// Rendering the whole view with the browser open must not panic.
	if got := m.View(); got == "" {
		t.Error("expected non-empty view with browser open")
	}
}

func TestSlashSessionsCommandOpensBrowser(t *testing.T) {
	for _, value := range []string{"sessions", "/sessions", "/session"} {
		m := newBrowserModel()
		m.input.SetValue(value)

		_, cmd := m.handleEnter()
		if cmd == nil {
			t.Fatalf("%q: expected a command (fetch sessions)", value)
		}
		select {
		case got := <-m.agent.Input:
			t.Fatalf("intercepted %q must not reach the agent, got %v", value, got)
		case <-time.After(50 * time.Millisecond):
		}
		if m.input.Value() != "" {
			t.Errorf("%q: expected input to be cleared", value)
		}
	}
}

func TestSlashCommandPassesThroughToAgent(t *testing.T) {
	m := newBrowserModel()
	m.input.SetValue("/rename my session")

	_, cmd := m.handleEnter()
	if cmd == nil {
		t.Fatal("expected a command sending the query")
	}
	go cmd()
	got := <-m.agent.Input
	resp, ok := got.(*api.UserInputResponse)
	if !ok {
		t.Fatalf("expected *api.UserInputResponse, got %T", got)
	}
	// The agent resolves slash commands centrally; the TUI forwards them
	// verbatim.
	if resp.Query != "/rename my session" {
		t.Errorf("Query = %q, want %q", resp.Query, "/rename my session")
	}
}

func TestShiftTabTogglesAutoMode(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	if a.SkipPermissionsEnabled() {
		t.Fatal("precondition: auto mode off")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyShiftTab})
	if !a.SkipPermissionsEnabled() {
		t.Error("expected auto mode on after shift+tab")
	}
	if len(m.messages) == 0 {
		t.Error("expected a transcript confirmation message")
	}
	if got := m.View(); !strings.Contains(got, "AUTO") {
		t.Error("expected AUTO indicator in status bar")
	}

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyShiftTab})
	if a.SkipPermissionsEnabled() {
		t.Error("expected auto mode off after second shift+tab")
	}
}

func TestLastCopyableText(t *testing.T) {
	m := newModel(nil)
	if _, ok := m.lastCopyableText(); ok {
		t.Error("expected nothing copyable with no messages")
	}

	m.messages = []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "question"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "first answer"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods"},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: "final answer"},
	}
	got, ok := m.lastCopyableText()
	if !ok {
		t.Fatal("expected something copyable")
	}
	if got != "final answer" {
		t.Errorf("lastCopyableText = %q, want %q", got, "final answer")
	}
}

func TestOsc52Write(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdout
	os.Stdout = w
	osc52Write("copy me")
	w.Close()
	os.Stdout = orig

	buf := make([]byte, 4096)
	n, _ := r.Read(buf)
	out := string(buf[:n])
	want := "\x1b]52;c;" + base64.StdEncoding.EncodeToString([]byte("copy me")) + "\a"
	if out != want {
		t.Errorf("OSC52 output = %q, want %q", out, want)
	}
}

func TestCopyToClipboard(t *testing.T) {
	if _, err := exec.LookPath("pbcopy"); err != nil {
		t.Skip("pbcopy not available on this platform")
	}
	if _, err := exec.LookPath("pbpaste"); err != nil {
		t.Skip("pbpaste not available on this platform")
	}
	if err := copyToClipboard("kubectl-ai clipboard test"); err != nil {
		t.Fatalf("copyToClipboard failed: %v", err)
	}
	out, err := exec.Command("pbpaste").Output()
	if err != nil {
		t.Fatalf("pbpaste failed: %v", err)
	}
	if string(out) != "kubectl-ai clipboard test" {
		t.Errorf("clipboard = %q, want %q", string(out), "kubectl-ai clipboard test")
	}
}

func TestCtrlYConfirmsAndCopies(t *testing.T) {
	m := newModel(nil)
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "copy me"},
	}

	_, cmd := m.copyLastResponse()
	if cmd == nil {
		t.Fatal("expected a copy command")
	}
	cmd()
	if len(m.messages) == 0 || m.messages[len(m.messages)-1].Payload != "📋 Copied last response to clipboard." {
		t.Error("expected a transcript confirmation message")
	}
}

func TestTypeAfterPastePreservesOrder(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)

	// Paste first, then type a message after it.
	_, _ = m.handleKey(pasteMsg("line one\nline two\nline three"))
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("what is wrong with this pod?")})

	_, cmd := m.handleEnter()
	if cmd == nil {
		t.Fatal("expected a submit command")
	}
	go cmd()
	got := <-a.Input
	query := got.(*api.UserInputResponse).Query
	want := "line one\nline two\nline three\n\nwhat is wrong with this pod?"
	if query != want {
		t.Errorf("query = %q, want %q", query, want)
	}
}

func TestExpansionSeparatesGluedTextAndPaste(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)

	// Type first, paste immediately after: the submitted message must
	// separate them with a blank line, not glue them.
	m.input.SetValue("explain this")
	_, _ = m.handleKey(pasteMsg("a\nb\nc\nd"))

	_, cmd := m.handleEnter()
	if cmd == nil {
		t.Fatal("expected a submit command")
	}
	go cmd()
	got := <-a.Input
	query := got.(*api.UserInputResponse).Query
	want := "explain this\n\na\nb\nc\nd"
	if query != want {
		t.Errorf("query = %q, want %q", query, want)
	}
}

func TestMultipleSameSizePastesExpandInInsertionOrder(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)

	_, _ = m.handleKey(pasteMsg("first\npaste\nhere"))
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(" and ")})
	_, _ = m.handleKey(pasteMsg("second\npaste\nhere"))

	_, cmd := m.handleEnter()
	if cmd == nil {
		t.Fatal("expected a submit command")
	}
	go cmd()
	got := <-a.Input
	query := got.(*api.UserInputResponse).Query
	want := "first\npaste\nhere\n\nand\n\nsecond\npaste\nhere"
	if query != want {
		t.Errorf("query = %q, want %q", query, want)
	}
}

func TestEscInterruptsRunningAgent(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateRunning}}
	m := newModel(a)

	runCtx := a.StartRun(context.Background())
	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if cmd == nil {
		t.Fatal("expected an interrupt command")
	}
	go cmd()
	select {
	case <-runCtx.Done():
	case <-time.After(time.Second):
		t.Error("expected esc to cancel the running agent")
	}
	// The input must NOT be cleared by an interrupt.
	if m.input.Value() != "" {
		t.Error("expected input untouched by interrupt")
	}
}

func TestEscClearsInputWhenIdle(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.input.SetValue("draft")

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if got := m.input.Value(); got != "" {
		t.Errorf("expected esc to clear input when idle, got %q", got)
	}
}

func TestEscDeclinesPermissionPrompt(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}, Input: make(chan any, 1)}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "confirm"

	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.inChoiceMode {
		t.Error("expected choice mode to close on esc")
	}
	if cmd == nil {
		t.Fatal("expected a decline command")
	}
	go cmd()
	got := <-a.Input
	resp, ok := got.(*api.UserChoiceResponse)
	if !ok {
		t.Fatalf("expected *api.UserChoiceResponse, got %T", got)
	}
	if resp.Choice != 3 {
		t.Errorf("expected decline (choice 3), got %d", resp.Choice)
	}
}

func TestPaletteOpenNavigateClose(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlP})
	if !m.paletteOpen {
		t.Fatal("expected palette to open on ctrl+p")
	}
	if got := m.View(); !strings.Contains(got, "Commands") {
		t.Error("expected palette to render in the view")
	}

	n := len(m.paletteItems())
	if n == 0 {
		t.Fatal("expected palette items")
	}
	_, _ = m.handlePaletteKey(tea.KeyMsg{Type: tea.KeyDown})
	if m.paletteIndex != 1 {
		t.Errorf("paletteIndex = %d, want 1", m.paletteIndex)
	}
	_, _ = m.handlePaletteKey(tea.KeyMsg{Type: tea.KeyUp})
	if m.paletteIndex != 0 {
		t.Errorf("paletteIndex = %d, want 0", m.paletteIndex)
	}
	// Wrap around.
	_, _ = m.handlePaletteKey(tea.KeyMsg{Type: tea.KeyUp})
	if m.paletteIndex != n-1 {
		t.Errorf("paletteIndex = %d, want %d (wrapped)", m.paletteIndex, n-1)
	}

	_, _ = m.handlePaletteKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.paletteOpen {
		t.Error("expected palette to close on esc")
	}
}

func TestPaletteModelSendsSlashQuery(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)

	// Find the Switch model action and run it.
	var item *paletteItem
	for i, it := range m.paletteItems() {
		if it.label == "Switch model" {
			item = &m.paletteItems()[i]
			break
		}
	}
	if item == nil {
		t.Fatal("expected a Switch model action")
	}
	_, cmd := item.run(&m)
	if cmd == nil {
		t.Fatal("expected a command")
	}
	go cmd()
	got := <-a.Input
	resp, ok := got.(*api.UserInputResponse)
	if !ok || resp.Query != "/model" {
		t.Errorf("expected /model query, got %v", got)
	}
}

func TestPaletteAutoModeToggleInstant(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	_, _ = m.toggleAutoMode()
	if !a.SkipPermissionsEnabled() {
		t.Error("expected auto mode on after palette toggle")
	}
}
