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
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
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

func TestPgUpPgDownScrollsViewport(t *testing.T) {
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

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyPgUp})
	if m.viewport.YOffset >= atBottom {
		t.Error("expected PgUp to scroll the viewport up (away from bottom)")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyPgDown})
	if m.viewport.YOffset != atBottom {
		t.Errorf("expected PgDown to scroll back to bottom: YOffset = %d, want %d", m.viewport.YOffset, atBottom)
	}
}

func TestUpDownRecallsHistory(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after up: input = %q, want %q", got, "second query")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Value(); got != "first query" {
		t.Errorf("after 2nd up: input = %q, want %q", got, "first query")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if got := m.input.Value(); got != "second query" {
		t.Errorf("after down: input = %q, want %q", got, "second query")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	if got := m.input.Value(); got != "" {
		t.Errorf("past newest: input = %q, want empty", got)
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
	m.input.SetValue("/clear")

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
	if resp.Query != "/clear" {
		t.Errorf("Query = %q, want %q", resp.Query, "/clear")
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

func newRenameModel(t *testing.T) (model, *agent.Agent, string) {
	t.Helper()
	mgr, err := sessions.NewSessionManager("memory")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	s, err := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	t.Cleanup(func() { _ = mgr.DeleteSession(s.ID) })

	a := &agent.Agent{
		Session:        s,
		SessionBackend: "memory",
		Input:          make(chan any, 1),
	}
	s.AgentState = api.AgentStateIdle
	return newModel(a), a, s.ID
}

func TestRenameBareCommandEntersRenameMode(t *testing.T) {
	m, a, _ := newRenameModel(t)
	a.Session.Name = "old name"

	m.input.SetValue("/rename")
	_, _ = m.handleEnter()

	if !m.sessionRename {
		t.Fatal("expected rename mode to start on bare /rename")
	}
	if got := m.input.Value(); got != "old name" {
		t.Errorf("rename input = %q, want prefilled %q", got, "old name")
	}
	if got := m.input.Placeholder; got != "New session name..." {
		t.Errorf("placeholder = %q, want %q", got, "New session name...")
	}
}

func TestRenameSubmitAppliesAndPersists(t *testing.T) {
	m, a, _ := newRenameModel(t)

	m.enterSessionRename()
	m.input.SetValue("my debug session")
	_, _ = m.handleEnter()

	if m.sessionRename {
		t.Error("expected rename mode to end after submit")
	}
	if a.Session.Name != "my debug session" {
		t.Errorf("session name = %q, want %q", a.Session.Name, "my debug session")
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("expected input cleared after rename, got %q", got)
	}
	if len(m.messages) == 0 || !strings.Contains(m.messages[len(m.messages)-1].Payload.(string), "my debug session") {
		t.Error("expected a rename confirmation message")
	}
}

func TestRenameEscCancels(t *testing.T) {
	m, _, _ := newRenameModel(t)

	m.input.SetValue("/rename")
	_, _ = m.handleEnter()
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})

	if m.sessionRename {
		t.Error("expected rename mode to end on esc")
	}
	if got := m.input.Value(); got != "" {
		t.Errorf("expected input cleared after cancel, got %q", got)
	}
	if got := m.input.Placeholder; got != "Ask kubectl-ai anything..." {
		t.Errorf("placeholder = %q, want default restored", got)
	}
}

func TestRenameWithArgsAppliesImmediately(t *testing.T) {
	m, a, _ := newRenameModel(t)

	m.input.SetValue("/rename quick name")
	_, _ = m.handleEnter()
	if m.sessionRename {
		t.Error("must not enter rename mode when a name is given")
	}
	if a.Session.Name != "quick name" {
		t.Errorf("session name = %q, want %q", a.Session.Name, "quick name")
	}
}

func TestBrowserDeleteFlow(t *testing.T) {
	mgr, err := sessions.NewSessionManager("memory")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	cur, _ := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	other, _ := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	_ = mgr.RenameSession(other.ID, "delete me")
	t.Cleanup(func() { _ = mgr.DeleteSession(cur.ID); _ = mgr.DeleteSession(other.ID) })

	a := &agent.Agent{
		Session:        cur,
		SessionBackend: "memory",
		Input:          make(chan any, 1),
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	sessionsList := []api.SessionInfo{
		{ID: cur.ID, Name: "current", LastModified: time.Now()},
		{ID: other.ID, Name: "delete me", LastModified: time.Now().Add(-time.Hour)},
	}
	m.openBrowser(sessionsList)
	m.browserIndex = 1

	// 'd' stages the delete with a confirmation prompt.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if m.pendingDeleteID != other.ID {
		t.Fatalf("expected pendingDeleteID %q, got %q", other.ID, m.pendingDeleteID)
	}

	// 'y' confirms and deletes.
	_, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("y")})
	if m.pendingDeleteID != "" {
		t.Error("expected pending delete to clear after confirm")
	}
	if cmd == nil {
		t.Fatal("expected a delete command")
	}
	msg := cmd()
	if _, ok := msg.(sessionDeletedMsg); !ok {
		t.Fatalf("expected sessionDeletedMsg, got %T", msg)
	}
	if _, err := mgr.FindSessionByID(other.ID); err == nil {
		t.Error("expected session to be deleted from the store")
	}
}

func TestBrowserDeleteCancelOnOtherKey(t *testing.T) {
	mgr, err := sessions.NewSessionManager("memory")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	cur, _ := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	other, _ := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	t.Cleanup(func() { _ = mgr.DeleteSession(cur.ID); _ = mgr.DeleteSession(other.ID) })

	a := &agent.Agent{Session: cur, SessionBackend: "memory", Input: make(chan any, 1)}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.openBrowser([]api.SessionInfo{{ID: cur.ID, LastModified: time.Now()}, {ID: other.ID, LastModified: time.Now()}})
	m.browserIndex = 1

	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if m.pendingDeleteID != other.ID {
		t.Fatal("expected staged delete")
	}
	_, cmd := m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("x")})
	if m.pendingDeleteID != "" {
		t.Error("expected delete to be cancelled on any other key")
	}
	if cmd != nil {
		t.Error("expected no command after cancel")
	}
	if _, err := mgr.FindSessionByID(other.ID); err != nil {
		t.Error("expected session NOT to be deleted after cancel")
	}
}

func TestBrowserCannotDeleteCurrentSession(t *testing.T) {
	mgr, err := sessions.NewSessionManager("memory")
	if err != nil {
		t.Fatalf("NewSessionManager: %v", err)
	}
	cur, _ := mgr.NewSession(sessions.Metadata{ModelID: "m"})
	t.Cleanup(func() { _ = mgr.DeleteSession(cur.ID) })

	a := &agent.Agent{Session: cur, SessionBackend: "memory", Input: make(chan any, 1)}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.openBrowser([]api.SessionInfo{{ID: cur.ID, LastModified: time.Now()}})
	m.browserIndex = 0

	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	if m.pendingDeleteID != "" {
		t.Error("expected no staged delete for the current session")
	}
	if m.browserStatus.text == "" {
		t.Error("expected an error status about not deleting the current session")
	}
}

func TestCtrlLClearsViewButScrollUpReveals(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.messages = []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "secret question", Timestamp: time.Now()},
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "secret answer", Timestamp: time.Now()},
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	// Ctrl+L clears the current view: only the marker remains visible.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlL})
	got := m.renderMessages()
	if !strings.Contains(got, "transcript cleared") {
		t.Error("expected a 'transcript cleared' marker")
	}
	if strings.Contains(got, "secret") {
		t.Error("expected the current view cleared of old messages after ctrl+l")
	}
	if len(m.messages) != 2 {
		t.Errorf("messages must stay in state, got %d", len(m.messages))
	}

	// Scrolling up reveals the hidden transcript again.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyPgUp})
	got = m.renderMessages()
	if !strings.Contains(got, "secret") {
		t.Error("expected scrolling up to reveal the hidden transcript")
	}
	if !strings.Contains(got, "transcript cleared") {
		t.Error("expected the marker to stay visible while revealed")
	}

	// Scrolling back to the bottom re-hides it.
	m.viewport.GotoBottom()
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyPgDown})
	if got := m.renderMessages(); strings.Contains(got, "secret") {
		t.Error("expected returning to the bottom to re-hide the transcript")
	}

	// Ctrl+L again removes the cleared state entirely.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlL})
	if got := m.renderMessages(); !strings.Contains(got, "secret") || strings.Contains(got, "transcript cleared") {
		t.Error("expected second ctrl+l to fully restore the transcript")
	}
}

func TestRenderToolResultCollapsed(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	msg := &api.Message{
		Type:    api.MessageTypeToolCallResponse,
		Payload: map[string]any{"stdout": "NAME   READY\ncoredns   1/1\nmetrics-server   1/1\nkube-proxy   1/1\n"},
	}
	got := m.renderMessage(msg, nil, 90)
	if !strings.Contains(got, "⎿") {
		t.Error("expected a collapsed result block")
	}
	if !strings.Contains(got, "coredns") {
		t.Error("expected first output line shown")
	}
	if !strings.Contains(got, "+1 more lines") {
		t.Errorf("expected '+1 more lines', got:\n%s", got)
	}
}

func TestRenderToolResultShimStringAndEmpty(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	if got := m.renderMessage(&api.Message{Type: api.MessageTypeToolCallResponse, Payload: "Result of running \"x\":\nok"}, nil, 90); !strings.Contains(got, "⎿") {
		t.Error("expected shim string payloads to render")
	}
	if got := m.renderMessage(&api.Message{Type: api.MessageTypeToolCallResponse, Payload: map[string]any{}}, nil, 90); got != "" {
		t.Errorf("expected empty result to render nothing, got %q", got)
	}
}

func TestRenderToolGroupPairedRequestAndResult(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	req := &api.Message{
		Type:    api.MessageTypeToolCallRequest,
		Payload: "kubectl get pods",
	}
	resp := &api.Message{
		Type:    api.MessageTypeToolCallResponse,
		Payload: map[string]any{"stdout": "NAME   READY\ncoredns   1/1\nmetrics-server   1/1\n"},
	}

	got := m.renderToolGroup(req, resp, 90)
	// The command appears once as the header line.
	if !strings.Contains(got, "kubectl get pods") {
		t.Errorf("expected the command header, got:\n%s", got)
	}
	if strings.Count(got, "kubectl get pods") != 1 {
		t.Errorf("expected the command once, got:\n%s", got)
	}
	// The result is nested under the command, not a separate "Running" box.
	if !strings.Contains(got, "⎿") {
		t.Errorf("expected a nested result marker, got:\n%s", got)
	}
	if strings.Contains(got, "Running") {
		t.Errorf("a completed call must not show 'Running', got:\n%s", got)
	}
	if !strings.Contains(got, "coredns") {
		t.Errorf("expected the result content, got:\n%s", got)
	}
}

func TestRenderToolGroupEmptyResultKeepsHeader(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	req := &api.Message{Type: api.MessageTypeToolCallRequest, Payload: "kubectl delete pod foo"}
	resp := &api.Message{Type: api.MessageTypeToolCallResponse, Payload: map[string]any{}}

	got := m.renderToolGroup(req, resp, 90)
	if !strings.Contains(got, "delete pod foo") {
		t.Errorf("expected the header for an empty result, got:\n%s", got)
	}
	if strings.Contains(got, "⎿") {
		t.Errorf("expected no nested marker for an empty result, got:\n%s", got)
	}
}

func TestRenderMessagesGroupsAdjacentToolCallAndResult(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	m.messages = []*api.Message{
		{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "show pods", Timestamp: time.Now()},
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods", Timestamp: time.Now()},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeToolCallResponse, Payload: map[string]any{"stdout": "coredns 1/1\n"}, Timestamp: time.Now()},
	}
	m.dirty = true

	got := m.renderMessages()
	if !strings.Contains(got, "kubectl get pods") {
		t.Errorf("expected the command header, got:\n%s", got)
	}
	if !strings.Contains(got, "⎿") {
		t.Errorf("expected the paired result nested under the command, got:\n%s", got)
	}
	if !strings.Contains(got, "coredns") {
		t.Errorf("expected the result content, got:\n%s", got)
	}
	if strings.Contains(got, "Running") {
		t.Errorf("a completed tool call must not show 'Running', got:\n%s", got)
	}
}

func TestRenderMessagesLoneToolRequestShowsRunning(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateRunning}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// A request whose result has not arrived yet renders standalone.
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get nodes", Timestamp: time.Now()},
	}
	m.dirty = true

	got := m.renderMessages()
	if !strings.Contains(got, "kubectl get nodes") {
		t.Errorf("expected the command, got:\n%s", got)
	}
	if !strings.Contains(got, "Running") {
		t.Errorf("expected a 'Running' indicator for an in-flight call, got:\n%s", got)
	}
}

func TestToolResultText(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{"plain string", "plain string"},
		{map[string]any{"stdout": "out", "stderr": "err"}, "out"},
		{map[string]any{"stderr": "err only"}, "err only"},
		{map[string]any{"content": "mcp content"}, "mcp content"},
		{42, "42"},
	}
	for _, c := range cases {
		if got := toolResultText(c.in); got != c.want {
			t.Errorf("toolResultText(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestToolResultFailed(t *testing.T) {
	cases := []struct {
		in   any
		want bool
	}{
		{map[string]any{"stdout": "ok", "exit_code": float64(0)}, false},
		{map[string]any{"stdout": "ok"}, false},
		{map[string]any{"content": "mcp result"}, false},
		{map[string]any{}, false},
		{map[string]any{"stderr": "boom", "exit_code": float64(1)}, true},
		{map[string]any{"error": "connection refused"}, true},
		{"Result of running \"x\":\nok", false},
		{"connection refused: get pods", true},
		{"", false},
	}
	for _, c := range cases {
		if got := toolResultFailed(c.in); got != c.want {
			t.Errorf("toolResultFailed(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestToolResultErrorText(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{map[string]any{"error": "boom", "stderr": "lines"}, "boom"},
		{map[string]any{"stderr": "lines"}, "lines"},
		{map[string]any{"stdout": "ok"}, ""},
		{"plain string", ""},
	}
	for _, c := range cases {
		if got := toolResultErrorText(c.in); got != c.want {
			t.Errorf("toolResultErrorText(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestRenderToolResultFailedStyle(t *testing.T) {
	m := newModel(nil)

	// A non-zero exit code renders in the error color with a ✗ marker.
	msg := &api.Message{
		Type: api.MessageTypeToolCallResponse,
		Payload: map[string]any{
			"stdout":   "partial output",
			"stderr":   "Error from server: not found",
			"exit_code": float64(1),
		},
	}
	got := m.renderToolResult(msg)
	if !strings.Contains(got, "✗") {
		t.Errorf("expected a ✗ marker on a failed result, got:\n%s", got)
	}
	// The cause (stderr) is shown, not the normal stdout.
	if !strings.Contains(got, "not found") {
		t.Errorf("expected stderr cause on a failed result, got:\n%s", got)
	}
	if strings.Contains(got, "partial output") {
		t.Errorf("stdout must not be shown when stderr is present on failure, got:\n%s", got)
	}
}

func TestRenderToolResultSuccessStaysDim(t *testing.T) {
	m := newModel(nil)

	msg := &api.Message{
		Type:    api.MessageTypeToolCallResponse,
		Payload: map[string]any{"stdout": "all good", "exit_code": float64(0)},
	}
	got := m.renderToolResult(msg)
	if strings.Contains(got, "✗") {
		t.Errorf("a successful result must not show a ✗ marker, got:\n%s", got)
	}
	if !strings.Contains(got, "all good") {
		t.Errorf("expected the success output, got:\n%s", got)
	}
}

func TestRenderToolGroupFailedHeader(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	req := &api.Message{Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pod missing"}
	resp := &api.Message{
		Type: api.MessageTypeToolCallResponse,
		Payload: map[string]any{"stderr": "not found", "exit_code": float64(1)},
	}
	got := m.renderToolGroup(req, resp, 90)
	if !strings.Contains(got, "✗") {
		t.Errorf("expected a ✗ header marker for a failed grouped call, got:\n%s", got)
	}
	if !strings.Contains(got, "not found") {
		t.Errorf("expected the error body nested under the header, got:\n%s", got)
	}
}

func writeKubeConfig(t *testing.T, currentContext string, withNamespace bool) string {
	t.Helper()
	ns := ""
	if withNamespace {
		ns = "\n    namespace: dev-ns"
	}
	content := fmt.Sprintf(`apiVersion: v1
kind: Config
current-context: %s
clusters:
- cluster:
    server: https://dev.example.com
  name: dev-cluster
contexts:
- context:
    cluster: dev-cluster
    user: dev-user%s
  name: dev-context
users:
- name: dev-user
  user: {}
`, currentContext, ns)
	path := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadKubeContext(t *testing.T) {
	info, ok := loadKubeContext(writeKubeConfig(t, "dev-context", true))
	if !ok {
		t.Fatal("expected kubeconfig to load")
	}
	if info.context != "dev-context" {
		t.Errorf("context = %q, want %q", info.context, "dev-context")
	}
	if info.namespace != "dev-ns" {
		t.Errorf("namespace = %q, want %q", info.namespace, "dev-ns")
	}

	// A context without a namespace falls back to "default".
	info, ok = loadKubeContext(writeKubeConfig(t, "dev-context", false))
	if !ok {
		t.Fatal("expected kubeconfig to load")
	}
	if info.namespace != "default" {
		t.Errorf("namespace = %q, want %q", info.namespace, "default")
	}

	// A missing file fails silently.
	if _, ok := loadKubeContext(filepath.Join(t.TempDir(), "missing")); ok {
		t.Error("expected ok=false for a missing kubeconfig")
	}
}

func TestKubeContextInfoStringAndProd(t *testing.T) {
	cases := []struct {
		info kubeContextInfo
		want string
		prod bool
	}{
		{kubeContextInfo{context: "minikube", namespace: "default"}, "minikube", false},
		{kubeContextInfo{context: "minikube", namespace: ""}, "minikube", false},
		{kubeContextInfo{context: "staging", namespace: "qa"}, "staging/qa", false},
		{kubeContextInfo{context: "gke-acme-eu-prod", namespace: "default"}, "gke-acme-eu-prod", true},
		{kubeContextInfo{context: "prod-cluster", namespace: "payments"}, "prod-cluster/payments", true},
	}
	for _, c := range cases {
		if got := c.info.String(); got != c.want {
			t.Errorf("%+v: String() = %q, want %q", c.info, got, c.want)
		}
		if got := c.info.isProd(); got != c.prod {
			t.Errorf("%+v: isProd() = %v, want %v", c.info, got, c.prod)
		}
	}
}

func TestFormatTokens(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "0"},
		{42, "42"},
		{999, "999"},
		{1000, "1.0k"},
		{1234, "1.2k"},
		{45200, "45.2k"},
	}
	for _, c := range cases {
		if got := formatTokens(c.in); got != c.want {
			t.Errorf("formatTokens(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestResolveKubeContextPrefersAgentPath(t *testing.T) {
	path := writeKubeConfig(t, "agent-context", true)
	a := &agent.Agent{
		Session:    &api.Session{ID: "test", AgentState: api.AgentStateIdle},
		Kubeconfig: path,
	}
	m := newModel(a)
	if !m.kubeContextOK {
		t.Fatal("expected the kube context to resolve at startup")
	}
	if m.kubeContext.context != "agent-context" {
		t.Errorf("context = %q, want %q (from the agent's kubeconfig)", m.kubeContext.context, "agent-context")
	}
}

func TestViewStatusShowsKubeContext(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width = 100
	m.kubeContext = kubeContextInfo{context: "dev-context", namespace: "dev-ns"}
	m.kubeContextOK = true

	if got := m.viewStatus(a.Session); !strings.Contains(got, "⎈ dev-context/dev-ns") {
		t.Errorf("expected kube context in the status bar, got %q", got)
	}

	// The default namespace renders as the bare context.
	m.kubeContext.namespace = "default"
	if got := m.viewStatus(a.Session); !strings.Contains(got, "⎈ dev-context") || strings.Contains(got, "dev-context/") {
		t.Errorf("expected bare context for the default namespace, got %q", got)
	}
}

func TestCtrlOTogglesToolResultExpansion(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	msg := &api.Message{
		Type:    api.MessageTypeToolCallResponse,
		Payload: map[string]any{"stdout": "one\ntwo\nthree\nfour\nfive\n"},
	}

	collapsed := m.renderToolResult(msg)
	if !strings.Contains(collapsed, "+2 more lines (ctrl+o to expand)") {
		t.Errorf("expected the collapsed expand hint, got:\n%s", collapsed)
	}
	if strings.Contains(collapsed, "four") {
		t.Error("expected the collapsed result to hide lines past the first 3")
	}

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	if !m.expandToolResults {
		t.Fatal("expected ctrl+o to expand tool results")
	}
	expanded := m.renderToolResult(msg)
	if !strings.Contains(expanded, "five") {
		t.Error("expected the expanded result to show all lines")
	}
	if !strings.Contains(expanded, "(ctrl+o to collapse)") {
		t.Errorf("expected the collapse hint on the last line, got:\n%s", expanded)
	}
	if strings.Contains(expanded, "more lines") {
		t.Error("expected no '+N more lines' tail when everything fits expanded")
	}

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	if m.expandToolResults {
		t.Error("expected a second ctrl+o to collapse again")
	}
}

func TestRenderToolResultExpandedCapsAt200Lines(t *testing.T) {
	m := newModel(nil)
	m.expandToolResults = true

	var sb strings.Builder
	for i := 0; i < 210; i++ {
		fmt.Fprintf(&sb, "line-%d\n", i)
	}
	msg := &api.Message{Type: api.MessageTypeToolCallResponse, Payload: sb.String()}
	got := m.renderToolResult(msg)
	if !strings.Contains(got, "line-199") {
		t.Error("expected the first 200 lines to be shown")
	}
	if strings.Contains(got, "line-209") {
		t.Error("expected lines beyond 200 to stay hidden")
	}
	if !strings.Contains(got, "+10 more lines (ctrl+o to collapse)") {
		t.Errorf("expected the capped tail with the collapse hint, got:\n%s", got)
	}
}

func TestSlashCompletions(t *testing.T) {
	cases := []struct {
		input string
		want  []string
	}{
		{"/mo", []string{"/model", "/models"}},
		{"/models", []string{"/models"}},
		{"/s", []string{"/sessions", "/session", "/save"}},
		{"hello", nil},
		{"", nil},
		{"/nope", nil},
	}
	for _, c := range cases {
		got := slashCompletions(c.input)
		if len(got) != len(c.want) {
			t.Errorf("slashCompletions(%q) = %v, want %v", c.input, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("slashCompletions(%q) = %v, want %v", c.input, got, c.want)
				break
			}
		}
	}
	if got := slashCompletions("/"); len(got) != len(slashCommands) {
		t.Errorf("slashCompletions(\"/\") returned %d matches, want all %d commands", len(got), len(slashCommands))
	}
}

func TestTabCyclesSlashCompletions(t *testing.T) {
	m := newModel(nil)
	m.input.SetValue("/mo")

	tab := tea.KeyMsg{Type: tea.KeyTab}
	_, _ = m.handleKey(tab)
	if got := m.input.Value(); got != "/model" {
		t.Errorf("after 1st tab: input = %q, want %q", got, "/model")
	}
	_, _ = m.handleKey(tab)
	if got := m.input.Value(); got != "/models" {
		t.Errorf("after 2nd tab: input = %q, want %q", got, "/models")
	}
	_, _ = m.handleKey(tab)
	if got := m.input.Value(); got != "/model" {
		t.Errorf("after 3rd tab: input = %q, want %q (wrapped)", got, "/model")
	}
}

func TestTabWithoutSlashLeavesInputAlone(t *testing.T) {
	m := newModel(nil)
	m.input.SetValue("hello")

	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	if got := m.input.Value(); got != "hello" {
		t.Errorf("tab without a slash prefix changed the input to %q", got)
	}
}

func TestCompletionHintGrowsInputBlock(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	base := m.inputBlockHeight()
	m.input.SetValue("/mo")
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the hint visible", got, base+1)
	}
	if hint := m.completionHint(); !strings.Contains(hint, "/model") || !strings.Contains(hint, "/models") {
		t.Errorf("completion hint = %q, want the matching commands", hint)
	}
	if view := m.viewInput(api.AgentStateIdle); !strings.Contains(view, "/models") {
		t.Error("expected the completion hint rendered inside the input box")
	}

	m.input.SetValue("hello")
	if got := m.inputBlockHeight(); got != base {
		t.Errorf("inputBlockHeight = %d, want %d without a slash prefix", got, base)
	}
}

func TestRenderTextMsgShowsTokenCount(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	r, err := m.cache.getRenderer(80)
	if err != nil {
		t.Fatalf("getRenderer: %v", err)
	}

	msg := &api.Message{
		Source: api.MessageSourceModel, Type: api.MessageTypeText,
		Payload: "answer", Timestamp: time.Now(), Tokens: 1234,
	}
	if got := m.renderTextMsg(msg, r, 80); !strings.Contains(got, "· 1.2k") {
		t.Errorf("expected a dim token count in the label, got:\n%s", got)
	}

	// No token count when the provider reported none.
	msg.Tokens = 0
	if got := m.renderTextMsg(msg, r, 80); strings.Contains(got, "·") {
		t.Errorf("expected no token count for Tokens=0, got:\n%s", got)
	}
}

func TestViewStatusShowsTokenTotal(t *testing.T) {
	store := sessions.NewInMemoryChatStore()
	_ = store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a", Tokens: 40000})
	_ = store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "b", Tokens: 5200})

	a := &agent.Agent{Session: &api.Session{
		ID: "test", AgentState: api.AgentStateIdle, ModelID: "m", ChatMessageStore: store,
	}}
	m := newModel(a)
	m.width = 100

	if got := m.viewStatus(a.GetSession()); !strings.Contains(got, "Σ 45.2k") {
		t.Errorf("expected session token total in status bar, got:\n%s", got)
	}

	// Hidden when no usage was reported.
	empty := &api.Session{ID: "test", AgentState: api.AgentStateIdle}
	if got := m.viewStatus(empty); strings.Contains(got, "Σ") {
		t.Errorf("expected no token total for an empty session, got:\n%s", got)
	}
}

func newStreamModel() (model, *sessions.InMemoryChatStore) {
	store := sessions.NewInMemoryChatStore()
	a := &agent.Agent{Session: &api.Session{
		ID:               "test",
		AgentState:       api.AgentStateRunning,
		ChatMessageStore: store,
	}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	return m, store
}

func textDelta(id, payload string) *api.Message {
	return &api.Message{
		ID:        id,
		Source:    api.MessageSourceModel,
		Type:      api.MessageTypeTextDelta,
		Payload:   payload,
		Timestamp: time.Now(),
	}
}

func TestTextDeltaUpdateInPlaceAndFinalReplacement(t *testing.T) {
	m, store := newStreamModel()
	const streamID = "stream-1"

	// First delta appends a new transcript entry.
	m.handleAgentMsg(textDelta(streamID, "Hello"))
	if len(m.messages) != 1 {
		t.Fatalf("expected 1 message after first delta, got %d", len(m.messages))
	}

	// Subsequent deltas with the same ID update the entry in place.
	m.handleAgentMsg(textDelta(streamID, "Hello, world"))
	if len(m.messages) != 1 {
		t.Fatalf("expected delta to update in place, got %d messages", len(m.messages))
	}
	if got := m.messages[0].Payload; got != "Hello, world" {
		t.Errorf("delta payload = %q, want %q", got, "Hello, world")
	}
	if got := m.messages[0].Type; got != api.MessageTypeTextDelta {
		t.Errorf("streaming entry type = %q, want text-delta", got)
	}

	// The final text message (same ID; the agent stores it before sending)
	// replaces the delta entry at the same index.
	final := &api.Message{
		ID:        streamID,
		Source:    api.MessageSourceModel,
		Type:      api.MessageTypeText,
		Payload:   "Hello, world",
		Timestamp: time.Now(),
	}
	if err := store.AddChatMessage(final); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	m.handleAgentMsg(final)
	if len(m.messages) != 1 {
		t.Fatalf("expected final message to replace the delta entry, got %d messages", len(m.messages))
	}
	if got := m.messages[0]; got.ID != streamID || got.Type != api.MessageTypeText || got.Payload != "Hello, world" {
		t.Errorf("final entry = %+v, want the stored final text message", got)
	}

	// Rendering the final message must not serve a stale cached delta render
	// (glamour interleaves ANSI codes, so match on a word unique to the final).
	if got := m.renderMessages(); !strings.Contains(got, "world") {
		t.Errorf("expected rendered transcript to contain the final text, got:\n%s", got)
	}
}

func TestTextDeltaThrottlesViewportRefresh(t *testing.T) {
	m, _ := newStreamModel()
	const streamID = "stream-1"

	// The first delta refreshes the viewport immediately and renders like a
	// normal model text message.
	m.handleAgentMsg(textDelta(streamID, "first"))
	if got := m.viewport.View(); !strings.Contains(got, "first") || !strings.Contains(got, "kubectl-ai") {
		t.Errorf("expected viewport to render the first delta as a model message, got:\n%s", got)
	}

	// A delta arriving within deltaRefreshInterval updates the payload but
	// skips the (expensive) transcript re-render.
	m.handleAgentMsg(textDelta(streamID, "first second"))
	if got := m.messages[0].Payload; got != "first second" {
		t.Fatalf("payload = %q, want %q", got, "first second")
	}
	if got := m.viewport.View(); strings.Contains(got, "first second") {
		t.Errorf("expected throttled viewport to keep the previous render, got:\n%s", got)
	}
	if !m.dirty {
		t.Error("expected model to stay dirty so a later refresh picks up the delta")
	}
}

func TestShellModeStyling(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	if m.shellMode() {
		t.Error("expected shellMode=false for normal input")
	}
	if m.completionHintVisible() {
		t.Error("expected no hint for normal input")
	}

	m.input.SetValue("!kubectl get pods")
	if !m.shellMode() {
		t.Error("expected shellMode=true for '!' prefix")
	}
	if !m.completionHintVisible() {
		t.Error("expected the shell hint line to show")
	}
	if got := m.completionHint(); !strings.Contains(got, "shell command") {
		t.Errorf("completionHint = %q, want shell marker", got)
	}
	base := m.inputHeight + 2
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the shell hint", got, base+1)
	}
	// The box border switches to the warning color for shell commands.
	if got := m.inputBox().GetBorderTopForeground(); got != colorWarning {
		t.Errorf("shell mode border = %v, want colorWarning %v", got, colorWarning)
	}
	m.input.SetValue("normal prompt")
	if got := m.inputBox().GetBorderTopForeground(); got != colorPrimary {
		t.Errorf("normal mode border = %v, want colorPrimary %v", got, colorPrimary)
	}
}

func writeFiles(t *testing.T, dir string, names ...string) {
	t.Helper()
	for _, n := range names {
		p := filepath.Join(dir, n)
		if strings.HasSuffix(n, "/") {
			if err := os.MkdirAll(p, 0o755); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFileMentionCompletion(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	dir := t.TempDir()
	writeFiles(t, dir, "deploy.yaml", "deployment.yaml", "service.yaml", "scripts/")

	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(oldwd) })

	// Prefix match with directory trailing slash.
	m.input.SetValue("fix @depl")
	matches := m.fileMatches()
	if len(matches) != 2 {
		t.Fatalf("fileMatches(@depl) = %v, want 2 matches", matches)
	}

	// Hint shows matches and grows the block.
	if !m.completionHintVisible() {
		t.Error("expected hint visible for @ mention")
	}
	if got := m.completionHint(); !strings.Contains(got, "deploy") {
		t.Errorf("completionHint = %q, want file matches", got)
	}
	base := m.inputHeight + 2
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the mention hint", got, base+1)
	}

	// Tab cycles through matches, keeping preceding text.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	got := m.input.Value()
	if !strings.HasPrefix(got, "fix @") || !(strings.Contains(got, "deploy") || strings.Contains(got, "deployment")) {
		t.Errorf("after tab: input = %q, want completion of @depl", got)
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	got2 := m.input.Value()
	if got2 == got {
		t.Errorf("expected tab to cycle to the next match, got %q", got2)
	}

	// Directory completion gets a trailing slash.
	m.input.SetValue("@scr")
	if matches := m.fileMatches(); len(matches) != 1 || matches[0] != "scripts/" {
		t.Errorf("fileMatches(@scr) = %v, want [scripts/]", matches)
	}
}

func TestFileMentionNonToken(t *testing.T) {
	m := newModel(nil)
	m.input.SetValue("plain text")
	if got := m.fileMatches(); got != nil {
		t.Errorf("expected no file matches for non-mention token, got %v", got)
	}
	m.input.SetValue("user@example.com")
	if got := m.fileMatches(); got != nil {
		t.Errorf("expected no file matches inside an email-like token, got %v", got)
	}
}

func writeAgentKubeConfig(t *testing.T, currentContext string, names ...string) string {
	t.Helper()
	content := "apiVersion: v1\nkind: Config\ncurrent-context: " + currentContext + "\nclusters: []\ncontexts:\n"
	for _, n := range names {
		content += "- context:\n    cluster: c\n    user: u\n  name: " + n + "\n"
	}
	content += "users: []\n"
	path := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestContextAutocomplete(t *testing.T) {
	path := writeAgentKubeConfig(t, "prod", "dev", "prod", "staging")
	a := &agent.Agent{
		Session:    &api.Session{ID: "test", AgentState: api.AgentStateIdle},
		Kubeconfig: path,
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// No matches outside the /context command.
	m.input.SetValue("hello")
	if got := m.contextMatches(); got != nil {
		t.Errorf("expected no matches outside /context, got %v", got)
	}

	// Prefix matches and hint.
	m.input.SetValue("/context s")
	if got := m.contextMatches(); len(got) != 1 || got[0] != "staging" {
		t.Errorf("contextMatches(/context s) = %v, want [staging]", got)
	}
	if !m.completionHintVisible() {
		t.Error("expected hint visible for /context partial")
	}
	if got := m.completionHint(); !strings.Contains(got, "staging") {
		t.Errorf("completionHint = %q, want staging", got)
	}
	base := m.inputHeight + 2
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the context hint", got, base+1)
	}

	// Tab cycles through all contexts, keeping the command prefix.
	m.input.SetValue("/context ")
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	first := m.input.Value()
	if !strings.HasPrefix(first, "/context ") || first == "/context " {
		t.Errorf("after tab: input = %q, want a context name", first)
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	second := m.input.Value()
	if second == first {
		t.Errorf("expected tab to cycle to the next context, got %q", second)
	}
}

// fakeExecutor returns a canned result for executor calls.
type fakeExecutor struct {
	result *sandbox.ExecResult
	err    error
}

func (f *fakeExecutor) Execute(ctx context.Context, command string, env []string, workDir string) (*sandbox.ExecResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.result, nil
}

func (f *fakeExecutor) Close(ctx context.Context) error { return nil }

func TestNamespaceAutocomplete(t *testing.T) {
	path := writeAgentKubeConfig(t, "prod", "prod", "staging")
	a := &agent.Agent{
		Session:    &api.Session{ID: "test", AgentState: api.AgentStateIdle},
		Kubeconfig: path,
		Executor:   &fakeExecutor{result: &sandbox.ExecResult{Stdout: "default\nkube-system\npayments\n"}},
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// No matches outside the namespace command.
	m.input.SetValue("hello")
	if got := m.namespaceMatches(); got != nil {
		t.Errorf("expected no matches outside /namespace, got %v", got)
	}

	// Prefix matches and hint.
	m.input.SetValue("/namespace pay")
	if got := m.namespaceMatches(); len(got) != 1 || got[0] != "payments" {
		t.Errorf("namespaceMatches(/namespace pay) = %v, want [payments]", got)
	}
	if !m.completionHintVisible() {
		t.Error("expected hint visible for /namespace partial")
	}
	if got := m.completionHint(); !strings.Contains(got, "payments") {
		t.Errorf("completionHint = %q, want payments", got)
	}
	base := m.inputHeight + 2
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the namespace hint", got, base+1)
	}

	// /ns alias matches too.
	m.input.SetValue("/ns kube")
	if got := m.namespaceMatches(); len(got) != 1 || got[0] != "kube-system" {
		t.Errorf("namespaceMatches(/ns kube) = %v, want [kube-system]", got)
	}

	// Tab cycles, keeping the command prefix.
	m.input.SetValue("/ns ")
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	first := m.input.Value()
	if !strings.HasPrefix(first, "/ns ") || first == "/ns " {
		t.Errorf("after tab: input = %q, want a namespace name", first)
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyTab})
	second := m.input.Value()
	if second == first {
		t.Errorf("expected tab to cycle to the next namespace, got %q", second)
	}
}

func TestFitHintsDropsLeastImportant(t *testing.T) {
	hints := []string{"Enter: send", "Ctrl+P: commands", "Ctrl+C: quit"}
	sep := " • "

	// Everything fits.
	got := fitHints(hints, sep, 100)
	if len(got) != 3 {
		t.Errorf("wide: got %d hints, want 3: %v", len(got), got)
	}

	// Only the first two fit; the least-important (quit) drops off, but the
	// first two are preserved in priority order.
	got = fitHints(hints, sep, lipgloss.Width("Enter: send")+lipgloss.Width(sep)+lipgloss.Width("Ctrl+P: commands"))
	if len(got) != 2 || got[0] != "Enter: send" || got[1] != "Ctrl+P: commands" {
		t.Errorf("narrow: got %v, want first two in order", got)
	}

	// Only the first fits.
	got = fitHints(hints, sep, lipgloss.Width("Enter: send"))
	if len(got) != 1 || got[0] != "Enter: send" {
		t.Errorf("very narrow: got %v, want only the first", got)
	}

	// Nothing fits at all.
	got = fitHints(hints, sep, 0)
	if got != nil {
		t.Errorf("zero width: got %v, want nil", got)
	}
}

func TestViewHelpNeverWraps(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	for _, w := range []int{20, 30, 40, 60, 80, 120, 200} {
		m.width = w
		help := m.viewHelp(api.AgentStateIdle)
		// The help bar is styled with bottom padding (one blank row under it),
		// so the invariant is that the *content* never wraps: its rendered
		// width must not exceed the terminal width. The leading/trailing
		// padding (2 + 2) is included in the rendered width.
		if got := lipgloss.Width(help); got > w {
			t.Errorf("width=%d: rendered help width %d exceeds terminal (wrapped):\n%s", w, got, help)
		}
		// The primary action is always present, even on the narrowest terminal.
		if !strings.Contains(help, "Enter: send") {
			t.Errorf("width=%d: expected 'Enter: send' to survive, got:\n%s", w, help)
		}
	}
}

func TestViewHelpWideShowsAllHints(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 200, 40
	m.resize()
	// Fill the transcript so the scroll hint is added.
	for i := 0; i < 50; i++ {
		m.messages = append(m.messages, &api.Message{
			Source: api.MessageSourceModel, Type: api.MessageTypeText,
			Payload: fmt.Sprintf("line %d", i), Timestamp: time.Now(),
		})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	help := m.viewHelp(api.AgentStateIdle)
	for _, want := range []string{"Enter: send", "Ctrl+P: commands", "Ctrl+C: quit", "Shift+Tab: auto", "PgUp/PgDn: scroll"} {
		if !strings.Contains(help, want) {
			t.Errorf("wide terminal: expected %q in help, got:\n%s", want, help)
		}
	}
}

func TestViewHelpRunningShowsCancel(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width = 60
	help := m.viewHelp(api.AgentStateRunning)
	if !strings.Contains(help, "Ctrl+C: cancel") {
		t.Errorf("running: expected cancel hint, got:\n%s", help)
	}
	if strings.Contains(help, "Enter: send") {
		t.Errorf("running: must not show the idle send hint, got:\n%s", help)
	}
}
