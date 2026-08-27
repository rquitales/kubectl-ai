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

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// stubTool implements tools.Tool for /tools-command tests without spinning up
// the full agent tool registry.
type stubTool struct{ name, desc string }

func (t *stubTool) Name() string                                     { return t.name }
func (t *stubTool) Description() string                              { return t.desc }
func (t *stubTool) FunctionDefinition() *gollm.FunctionDefinition    { return nil }
func (t *stubTool) Run(context.Context, map[string]any) (any, error) { return nil, nil }
func (t *stubTool) IsInteractive(map[string]any) (bool, error)       { return false, nil }
func (t *stubTool) CheckModifiesResource(map[string]any) string      { return "no" }

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

func TestDraftCounter(t *testing.T) {
	got := draftCounter("hello world foo", 80)
	// "hello world foo" is 15 runes / 3 words.
	if !strings.Contains(got, "15 chars") {
		t.Errorf("expected 15 chars (runes), got %q", got)
	}
	if !strings.Contains(got, "3 words") {
		t.Errorf("expected 3 words, got %q", got)
	}
	// A small draft has no large-draft hint.
	if strings.Contains(got, "large") {
		t.Errorf("expected no large hint for a small draft, got %q", got)
	}
}

func TestDraftCounterLargeHint(t *testing.T) {
	big := strings.Repeat("x", largeDraftThreshold)
	got := draftCounter(big, 120)
	// A large draft appends the @file/compact hint.
	if !strings.Contains(got, "large") || !strings.Contains(got, "@file") || !strings.Contains(got, "/compact") {
		t.Errorf("expected a large-draft hint, got %q", got)
	}
}

func TestDraftCounterVisible(t *testing.T) {
	m := newModel(nil)
	// Empty draft: no counter.
	if m.draftCounterVisible() {
		t.Error("expected counter hidden for an empty draft")
	}
	// Slash completion: counter hidden (completion hint takes the line).
	m.input.SetValue("/mo")
	if m.draftCounterVisible() {
		t.Error("expected counter hidden when a completion hint is visible")
	}
	// Non-empty plain draft: counter visible.
	m.input.SetValue("hello")
	if !m.draftCounterVisible() {
		t.Error("expected counter visible for a non-empty plain draft")
	}
	// Rename mode: counter hidden (input is a name, not a prompt).
	m.sessionRename = true
	if m.draftCounterVisible() {
		t.Error("expected counter hidden in rename mode")
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

func TestBottomDividerPlainAtBottom(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 24
	m.resize()
	for i := 0; i < 50; i++ {
		m.messages = append(m.messages, &api.Message{
			Source: api.MessageSourceModel, Type: api.MessageTypeText,
			Payload: fmt.Sprintf("line %d", i), Timestamp: time.Now(),
		})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	got := m.viewBottomDivider()
	if strings.Contains(got, "PgDn") {
		t.Errorf("at the bottom, the divider must be plain (no scroll cue), got:\n%s", got)
	}
}

func TestBottomDividerShowsScrollCueWhenScrolledUp(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 24
	m.resize()
	for i := 0; i < 50; i++ {
		m.messages = append(m.messages, &api.Message{
			Source: api.MessageSourceModel, Type: api.MessageTypeText,
			Payload: fmt.Sprintf("line %d", i), Timestamp: time.Now(),
		})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	// Scroll up: the divider now carries the position cue.
	m.viewport.ScrollUp(m.viewport.Height / 2)
	got := m.viewBottomDivider()
	if !strings.Contains(got, "PgDn for latest") {
		t.Errorf("expected a scroll cue when scrolled up, got:\n%s", got)
	}
	if !strings.Contains(got, "%") {
		t.Errorf("expected a percentage in the scroll cue, got:\n%s", got)
	}
	// The cue and divider together must stay on one line of the terminal width.
	if n := lipgloss.Height(got); n != 1 {
		t.Errorf("expected the bottom divider to be one line, got %d:\n%s", n, got)
	}
}

func TestBottomDividerPlainWhenShortTranscript(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 24
	m.resize()
	// A transcript shorter than the viewport cannot scroll: no cue.
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "hi", Timestamp: time.Now()},
	}
	m.dirty = true
	m.refresh()

	got := m.viewBottomDivider()
	if strings.Contains(got, "PgDn") {
		t.Errorf("a short transcript must not show a scroll cue, got:\n%s", got)
	}
}

func TestUpDownRecallsHistory(t *testing.T) {
	m := newModel(nil)
	m.messages = historyTestMessages()

	// Up/Down recall input history (the scroll wheel handles transcript
	// scrolling via mouse capture, not the arrow keys).
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

func TestCtrlGTogglesMouseCapture(t *testing.T) {
	m := newModel(nil)
	if !m.mouseEnabled {
		t.Fatal("mouse capture should start enabled")
	}

	// Off: the terminal gets the mouse back so text can be selected.
	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlG})
	if m.mouseEnabled {
		t.Error("Ctrl+G did not disable mouse capture")
	}
	if cmd == nil {
		t.Error("expected a command to disable mouse capture")
	}
	// The toggle is a transient UI state, shown in the status bar — it must
	// NOT append a message to the transcript.
	if got := lastMessageText(m); got != "" {
		t.Errorf("Ctrl+G appended a transcript message %q; want none (status-bar indicator only)", got)
	}

	// On again: the wheel scrolls once more.
	_, cmd = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlG})
	if !m.mouseEnabled {
		t.Error("Ctrl+G did not re-enable mouse capture")
	}
	if cmd == nil {
		t.Error("expected a command to re-enable mouse capture")
	}
	if got := lastMessageText(m); got != "" {
		t.Errorf("re-enabling appended a transcript message %q; want none", got)
	}
}

func TestStatusBarShowsMouseSelectIndicator(t *testing.T) {
	// A nil agent means viewStatus can't run, so drive it via the agent.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()

	// Default (capture on): no SELECT indicator in the status bar.
	gotOn := m.viewStatus(a.Session)
	if strings.Contains(gotOn, "SELECT") {
		t.Errorf("status bar should not show SELECT while capture is on: %q", gotOn)
	}

	// After toggling off: the SELECT indicator appears.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlG})
	gotOff := m.viewStatus(a.Session)
	if !strings.Contains(gotOff, "SELECT") {
		t.Errorf("status bar should show SELECT while capture is off: %q", gotOff)
	}
}

func TestResizeDoesNotReEnableDisabledMouse(t *testing.T) {
	// A resize re-enables capture so split reports don't leak — but it must
	// not override a user who turned capture off to select text.
	m := newModel(nil)
	m.mouseEnabled = false
	updated, cmd := m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	if cmd != nil {
		t.Error("resize must not re-enable mouse capture while it is toggled off")
	}
	if got := updated.(model); got.mouseEnabled {
		t.Error("resize flipped mouseEnabled back on")
	}

	// With capture on, a resize does re-establish it.
	m2 := newModel(nil)
	if _, cmd := m2.Update(tea.WindowSizeMsg{Width: 100, Height: 30}); cmd == nil {
		t.Error("resize should re-enable mouse capture when it is on")
	}
}

func TestStatusClickSpansMatchRenderedSegments(t *testing.T) {
	// The status bar is one line at y==0. The right-hand segments
	// (model, kube context, kube namespace) must produce click spans
	// that are half-open and end before m.width-1. Drive via an agent
	// so statusLayout can run.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()

	got, spans := m.statusLayout(a.Session)
	// The rendered bar must be exactly m.width (the invariant the
	// right-edge anchoring depends on).
	if w := lipgloss.Width(got); w != m.width {
		t.Errorf("status bar width = %d, want %d", w, m.width)
	}
	// A non-zero span must end at or before m.width-1.
	for _, s := range []statusSpan{spans.model, spans.kubeContext, spans.kubeNamespace} {
		if s.contains(0) && s.end > m.width-1 {
			t.Errorf("span %v ends at %d, past the bar edge %d", s, s.end, m.width-1)
		}
	}
	// Without a kube context the kube spans are absent.
	if spans.kubeContext.contains(0) {
		t.Error("kubeContext span should be zero with no kube context")
	}
}

func TestStatusClickSpansDefaultNamespace(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()
	// A default/empty namespace: no namespace target, context target present.
	m.kubeContext = kubeContextInfo{context: "dev-cluster", namespace: "default"}
	m.kubeContextOK = true
	_, spans := m.statusLayout(a.Session)
	if spans.kubeNamespace.contains(0) {
		t.Errorf("default namespace should produce no namespace span, got %v", spans.kubeNamespace)
	}
	if !spans.kubeContext.contains(spans.kubeContext.start) {
		t.Errorf("default namespace should produce a context span, got %v", spans.kubeContext)
	}
}

func TestStatusClickSpansARNContext(t *testing.T) {
	// An EKS ARN context name contains a literal "/"; splitting the
	// rendered string on "/" would misplace the boundary. Deriving from
	// the struct fields keeps the context span correct.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)
	m.width, m.height = 200, 24
	m.resize()
	m.kubeContext = kubeContextInfo{context: "arn:aws:eks:us-west-2:123:cluster/prod-a", namespace: "payments"}
	m.kubeContextOK = true
	_, spans := m.statusLayout(a.Session)
	// The context span must be non-zero and the namespace span must start
	// after the context (the "/" separator belongs to neither).
	ctx := spans.kubeContext
	if !ctx.contains(ctx.start) {
		t.Fatalf("expected a context span, got %v", ctx)
	}
	ns := spans.kubeNamespace
	if !ns.contains(ns.start) {
		t.Fatalf("expected a namespace span, got %v", ns)
	}
	if ns.start <= ctx.end {
		t.Errorf("namespace span %v must start after context span %v", ns, ctx)
	}
}

func TestStatusClickOpensPicker(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev-cluster", namespace: "payments"}
	m.kubeContextOK = true

	// Click the context span: opens the context picker.
	_, spans := m.statusLayout(a.Session)
	x := spans.kubeContext.start
	if !spans.kubeContext.contains(x) {
		t.Fatal("test precondition: no context span to click")
	}
	_, cmd := m.handleStatusClick(x, 0)
	if !m.pickerOpen || m.pickerKind != pickerContext {
		t.Errorf("click should open the context picker, got open=%v kind=%v", m.pickerOpen, m.pickerKind)
	}
	if cmd == nil {
		t.Error("expected a fetch command from openPicker")
	}
}

func TestStatusClickModelSpanSendsModelQuery(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev-cluster"}
	m.kubeContextOK = true

	_, spans := m.statusLayout(a.Session)
	x := spans.model.start
	if !spans.model.contains(x) {
		t.Fatal("test precondition: no model span to click")
	}
	_, cmd := m.handleStatusClick(x, 0)
	if cmd == nil {
		t.Fatal("expected a command from the model click")
	}
	// The model click sends /model (agent-driven picker), not openPicker.
	// Verify the returned cmd is a non-nil tea.Cmd that would deliver the
	// query when executed by the bubbletea runtime.
	if _, ok := cmd().(*api.UserInputResponse); ok {
		t.Errorf("model click returned %T, want a tea.Cmd", cmd())
	}
}

func TestStatusClickNoOpWhenChoicePromptActive(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev-cluster", namespace: "payments"}
	m.kubeContextOK = true
	m.inChoiceMode = true // an agent prompt is capturing input

	_, spans := m.statusLayout(a.Session)
	// Clicking any span while a choice prompt is active is a no-op.
	for _, s := range []statusSpan{spans.model, spans.kubeContext, spans.kubeNamespace} {
		if !s.contains(0) {
			continue
		}
		if _, cmd := m.handleStatusClick(s.start, 0); cmd != nil {
			t.Errorf("click at span %v should be a no-op while choice prompt active, got cmd=%v", s, cmd)
		}
	}
}

func TestStatusClickNoOpWhenFlashActive(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev-cluster", namespace: "payments"}
	m.kubeContextOK = true
	m.setFlash("flash test") // a flash replaces the right segment

	_, spans := m.statusLayout(a.Session)
	// While the flash is active all spans are zero (no targets).
	for _, s := range []statusSpan{spans.model, spans.kubeContext, spans.kubeNamespace} {
		if s.contains(0) {
			t.Errorf("flash active should zero all spans, got %v", s)
		}
	}
}

func TestStatusClickNoOpWhenNarrow(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.kubeContext = kubeContextInfo{context: "dev-cluster", namespace: "payments"}
	m.kubeContextOK = true
	// Very narrow: the gap<0 path joins+truncates, spans are zero.
	m.width, m.height = 20, 4
	m.resize()
	_, spans := m.statusLayout(a.Session)
	for _, s := range []statusSpan{spans.model, spans.kubeContext, spans.kubeNamespace} {
		if s.contains(0) {
			t.Errorf("narrow width should zero all spans, got %v", s)
		}
	}
}

func TestPickerNavigationAndSelect(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}, Input: make(chan any, 1)}
	m := newModel(a)
	m.width, m.height = 80, 24
	m.resize()
	m.pickerItems = []pickerItem{
		{value: "ctx-a", current: true},
		{value: "ctx-b"},
		{value: "ctx-c"},
	}
	m.pickerOpen = true
	m.pickerKind = pickerContext
	m.pickerIndex = 0

	// Down/Up wrap around.
	m.handlePickerKey(tea.KeyMsg{Type: tea.KeyDown})
	if m.pickerIndex != 1 {
		t.Errorf("after down: index = %d, want 1", m.pickerIndex)
	}
	m.handlePickerKey(tea.KeyMsg{Type: tea.KeyDown})
	m.handlePickerKey(tea.KeyMsg{Type: tea.KeyDown})
	if m.pickerIndex != 0 {
		t.Errorf("after 3x down (wrap): index = %d, want 0", m.pickerIndex)
	}
	m.handlePickerKey(tea.KeyMsg{Type: tea.KeyUp})
	if m.pickerIndex != 2 {
		t.Errorf("after up (wrap): index = %d, want 2", m.pickerIndex)
	}

	// Enter selects: sends /context <value> for the highlighted row.
	// Capture the selection before handlePickerKey, which closes the
	// picker and nils pickerItems — reading it afterwards panics.
	sel := m.pickerItems[m.pickerIndex].value
	_, cmd := m.handlePickerKey(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("expected a command from enter")
	}
	cmd()
	select {
	case v := <-m.agent.Input:
		if q, ok := v.(*api.UserInputResponse); !ok || q.Query != "/context "+sel {
			t.Errorf("enter sent %q, want %q", q, "/context "+sel)
		}
	case <-time.After(time.Second):
		t.Error("enter did not send a query")
	}
	if m.pickerOpen {
		t.Error("picker should close after enter")
	}

	// Esc closes without sending.
	m.pickerItems = []pickerItem{{value: "x"}, {value: "y"}}
	m.pickerOpen = true
	m.handlePickerKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.pickerOpen {
		t.Error("esc should close the picker")
	}
}

func TestPickerIsBounded(t *testing.T) {
	m := newModel(nil)
	m.height = 10
	m.pickerItems = make([]pickerItem, 50)
	for i := range m.pickerItems {
		m.pickerItems[i] = pickerItem{value: fmt.Sprintf("item-%d", i)}
	}
	m.pickerOpen = true
	m.pickerKind = pickerNamespace
	m.pickerIndex = 40
	// pickerRows windows; a 50-row list must not overflow a 10-row viewport.
	if rows := m.pickerRows(); rows > m.height {
		t.Errorf("pickerRows = %d, must fit within height %d", rows, m.height)
	}
}

func TestMouseReportBurstDoesNotLeakIntoInput(t *testing.T) {
	// A fast scroll wheel packs many SGR mouse reports into one read; a
	// chunk boundary that splits a sequence drops the ESC and/or the
	// trailing M, and the leftovers arrive as literal runes.
	bursts := []string{
		"\x1b[<64;140;44M",                     // one full report
		"[<64;140;44M",                         // ESC eaten by the parser
		"[<64;140;44M[<64;140;44M[<65;140;44M", // fast-scroll burst
		"[<64;140;44M[<64;140",                 // split mid-sequence
		"[<",                                   // bare fragment
	}
	for _, burst := range bursts {
		m := newModel(nil)
		_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(burst)})
		if got := m.input.Value(); got != "" {
			t.Errorf("burst %q leaked into the input: %q", burst, got)
		}
	}
}

func TestSplitMouseReportAcrossMessagesDoesNotLeak(t *testing.T) {
	// When a fast scroll splits an SGR report at the read boundary, the
	// parser's ESC branch stops after one rune, so the report arrives as a
	// lone Alt+'[' followed by the remainder in a separate message.
	m := newModel(nil)
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("["), Alt: true})
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("<64;140")})
	if got := m.input.Value(); got != "" {
		t.Errorf("split report leaked into the input: %q", got)
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(";44M")})
	if got := m.input.Value(); got != "" {
		t.Errorf("report tail leaked into the input: %q", got)
	}
	// Typing resumes normally once the report has been consumed.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("k")})
	if got := m.input.Value(); got != "k" {
		t.Errorf("typing after a split report: value = %q, want %q", got, "k")
	}
}

func TestSwallowModeIsBounded(t *testing.T) {
	// A false positive must not eat input indefinitely.
	m := newModel(nil)
	m.swallowMouseSeq = true
	m.swallowedRunes = 1
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(strings.Repeat("a", maxSwallowedRunes+8))})
	if m.swallowMouseSeq {
		t.Error("swallow mode should have given up after maxSwallowedRunes")
	}
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("ok")})
	if got := m.input.Value(); !strings.HasSuffix(got, "ok") {
		t.Errorf("input after bounded swallow = %q, want it to end with %q", got, "ok")
	}
}

// lastMessageText returns the payload of the most recent transcript message
// as a string, or "" when there are none.
func lastMessageText(m model) string {
	if len(m.messages) == 0 {
		return ""
	}
	s, _ := m.messages[len(m.messages)-1].Payload.(string)
	return s
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

// wireEphemeral gives a test agent the store and output channel that
// AddEphemeralMessage needs.
func wireEphemeral(m *model) {
	m.agent.Output = make(chan any, 10)
	m.agent.Session.ChatMessageStore = sessions.NewInMemoryChatStore()
}

// drainEphemeral feeds n emitted agent messages back through the model's
// message handler, mirroring the live reader goroutine.
func drainEphemeral(t *testing.T, m *model, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		select {
		case got := <-m.agent.Output:
			m.handleAgentMsg(got.(*api.Message))
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for ephemeral message %d", i)
		}
	}
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

func TestBrowserMarksCurrentSessionWithIcon(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())

	got := m.viewSessionBrowser()
	// The current session (s2) is marked with a "●" icon so it's obvious at a
	// glance, not just buried in the meta text.
	if !strings.Contains(got, "●") {
		t.Errorf("expected a ● current-session marker, got:\n%s", got)
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

func TestSlashHelpPrintsReferenceLocally(t *testing.T) {
	for _, value := range []string{"/help", "/?"} {
		m := newBrowserModel()
		wireEphemeral(&m)
		m.input.SetValue(value)

		_, cmd := m.handleEnter()
		// /help is handled locally: no command is sent to the agent's input.
		if cmd != nil {
			select {
			case got := <-m.agent.Input:
				t.Fatalf("%q must not reach the agent, got %v", value, got)
			case <-time.After(50 * time.Millisecond):
			}
		}
		// The echo + reference arrive as ephemeral agent messages.
		drainEphemeral(t, &m, 2)
		if len(m.messages) != 2 {
			t.Fatalf("%q: expected 2 transcript messages (query + help), got %d", value, len(m.messages))
		}
		if !m.messages[1].Ephemeral {
			t.Errorf("%q: help message must be ephemeral (excluded from LLM history)", value)
		}
		help, ok := m.messages[1].Payload.(string)
		if !ok {
			t.Fatalf("%q: expected a string help payload, got %T", value, m.messages[1].Payload)
		}
		if !strings.Contains(help, "Keyboard shortcuts") || !strings.Contains(help, "Slash commands") {
			t.Errorf("%q: help text missing sections, got:\n%s", value, help)
		}
		// The reference lists the /mcp and /tools commands.
		if !strings.Contains(help, "/mcp") || !strings.Contains(help, "/tools") {
			t.Errorf("%q: help text missing /mcp or /tools, got:\n%s", value, help)
		}
		// /help is offered for autocomplete.
		if !strings.Contains(strings.Join(slashCompletions("/he"), ","), "/help") {
			t.Errorf("%q: expected /help in slash autocomplete", value)
		}
	}
}

func TestSlashMCPPrintsDetailsLocally(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{
		ID:         "test",
		AgentState: api.AgentStateIdle,
		MCPStatus: &api.MCPStatus{
			TotalServers:   2,
			ConnectedCount: 1,
			FailedCount:    1,
			ServerInfoList: []api.ServerConnectionInfo{
				{Name: "k8s-tools", IsConnected: true, AvailableTools: []api.MCPTool{{Name: "get-pods"}, {Name: "get-nodes"}}},
				{Name: "broken-server", IsConnected: false},
			},
		},
	}}
	m := newModel(a)
	wireEphemeral(&m)
	m.input.SetValue("/mcp")

	_, cmd := m.handleEnter()
	drainEphemeral(t, &m, 2)
	// /mcp is handled locally: no command is sent to the agent.
	if cmd != nil {
		select {
		case got := <-a.Input:
			t.Fatalf("/mcp must not reach the agent, got %v", got)
		case <-time.After(50 * time.Millisecond):
		}
	}
	// An MCP details message is appended to the transcript (query + details).
	if len(m.messages) != 2 {
		t.Fatalf("expected 2 transcript messages (query + mcp), got %d", len(m.messages))
	}
	detail, ok := m.messages[1].Payload.(string)
	if !ok {
		t.Fatalf("expected a string mcp payload, got %T", m.messages[1].Payload)
	}
	if !strings.Contains(detail, "MCP servers") || !strings.Contains(detail, "1/2 connected") {
		t.Errorf("mcp text missing summary, got:\n%s", detail)
	}
	if !strings.Contains(detail, "k8s-tools") || !strings.Contains(detail, "broken-server") {
		t.Errorf("mcp text missing server rows, got:\n%s", detail)
	}
	// /mcp is offered for autocomplete.
	if !strings.Contains(strings.Join(slashCompletions("/m"), ","), "/mcp") {
		t.Error("expected /mcp in slash autocomplete")
	}
}

func TestSlashMCPUnconfigured(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	wireEphemeral(&m)
	m.input.SetValue("/mcp")
	_, _ = m.handleEnter()
	drainEphemeral(t, &m, 2)
	if len(m.messages) != 2 {
		t.Fatalf("expected 2 transcript messages, got %d", len(m.messages))
	}
	if got := m.messages[1].Payload.(string); !strings.Contains(got, "No MCP servers") {
		t.Errorf("expected a no-MCP message when unconfigured, got:\n%s", got)
	}
}

func TestToolsTextFormatsSortedList(t *testing.T) {
	tls := []tools.Tool{
		&stubTool{name: "kubectl", desc: "Run kubectl commands against the cluster."},
		&stubTool{name: "bash", desc: "Run a bash command.\nWith more detail."},
		&stubTool{name: "mcp__srv__get-pods", desc: "List pods in a namespace."},
	}
	got := toolsText(tls)
	if !strings.Contains(got, "Available tools") {
		t.Errorf("expected an 'Available tools' header, got:\n%s", got)
	}
	if !strings.Contains(got, "kubectl") || !strings.Contains(got, "bash") {
		t.Errorf("expected kubectl and bash in the tools list, got:\n%s", got)
	}
	// Multi-line description collapses to its first line.
	if strings.Contains(got, "With more detail") {
		t.Errorf("expected the description collapsed to its first line, got:\n%s", got)
	}
}

func TestToolsTextEmpty(t *testing.T) {
	if got := toolsText(nil); !strings.Contains(got, "No tools are available") {
		t.Errorf("expected a no-tools message, got %q", got)
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
	// Auto-mode confirmation is a status-bar flash, not a transcript message.
	if !strings.Contains(m.flash, "Auto mode") {
		t.Errorf("expected an auto-mode flash, got %q", m.flash)
	}
	if len(m.messages) != 0 {
		t.Errorf("auto-mode toggle leaked a transcript message: %v", m.messages)
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
	before := len(m.messages)

	_, cmd := m.copyLastResponse()
	if cmd == nil {
		t.Fatal("expected a copy command")
	}
	cmd()
	// The confirmation is a status-bar flash, not a transcript message.
	if len(m.messages) != before {
		t.Errorf("copy leaked a transcript message: %v", m.messages[before:])
	}
	if !strings.Contains(m.flash, "Copied") {
		t.Errorf("expected a copy flash, got %q", m.flash)
	}
}

func TestCopyToolCommandAndOutput(t *testing.T) {
	m := newModel(nil)
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "here is the plan"},
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods -n kube-system"},
		{Source: api.MessageSourceAgent, Type: api.MessageTypeToolCallResponse, Payload: map[string]any{"stdout": "coredns 1/1\nkube-proxy 1/1\n"}},
	}
	before := len(m.messages)

	// lastToolCommand finds the most recent tool-call request.
	cmd, ok := m.lastToolCommand()
	if !ok || cmd != "kubectl get pods -n kube-system" {
		t.Fatalf("lastToolCommand = %q ok=%v, want the kubectl command", cmd, ok)
	}
	// lastToolOutput finds the most recent tool-call response's text.
	out, ok := m.lastToolOutput()
	if !ok || !strings.Contains(out, "coredns") {
		t.Fatalf("lastToolOutput = %q ok=%v, want the pod list", out, ok)
	}

	// copyToolCommand confirms via a status-bar flash, not the transcript.
	_, cmdFn := m.copyToolCommand()
	if cmdFn == nil {
		t.Fatal("expected a copy command")
	}
	cmdFn()
	if len(m.messages) != before {
		t.Errorf("copy command leaked a transcript message: %v", m.messages[before:])
	}
	if !strings.Contains(m.flash, "Copied") {
		t.Errorf("expected a copy flash, got %q", m.flash)
	}

	// copyToolOutput confirms via a status-bar flash, not the transcript.
	_, outFn := m.copyToolOutput()
	if outFn == nil {
		t.Fatal("expected a copy command")
	}
	outFn()
	if len(m.messages) != before {
		t.Errorf("copy output leaked a transcript message: %v", m.messages[before:])
	}
	if !strings.Contains(m.flash, "Copied") {
		t.Errorf("expected a copy flash, got %q", m.flash)
	}
}

func TestCopyToolNothingToCopy(t *testing.T) {
	m := newModel(nil)
	// No tool calls: both report nothing to copy via a status-bar flash
	// (the returned cmd is the flash auto-clear timer, not a copy action).
	m.copyToolCommand()
	if !strings.Contains(m.flash, "Nothing to copy") {
		t.Errorf("expected a 'Nothing to copy' flash, got %q", m.flash)
	}
	if len(m.messages) != 0 {
		t.Errorf("nothing-to-copy leaked a transcript message: %v", m.messages)
	}
}

func TestPaletteHasCopyCommandOutputActions(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	labels := make(map[string]bool)
	for _, it := range m.paletteItems() {
		labels[it.label] = true
	}
	for _, want := range []string{"Copy last response", "Copy last command", "Copy last output"} {
		if !labels[want] {
			t.Errorf("expected %q in the palette, missing", want)
		}
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
	// The interrupt is confirmed via a status-bar flash, not the transcript.
	if !strings.Contains(m.flash, "Interrupted") {
		t.Errorf("expected an 'Interrupted' flash, got %q", m.flash)
	}
	if len(m.messages) != 0 {
		t.Errorf("interrupt leaked a transcript message: %v", m.messages)
	}
}

func TestInterruptRunConfirmsAndNothingRunning(t *testing.T) {
	// Nothing running: a status-bar flash, no transcript message.
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	_, _ = m.interruptRun()
	if !strings.Contains(m.flash, "Nothing running") {
		t.Errorf("expected a 'Nothing running' flash, got %q", m.flash)
	}
	if len(m.messages) != 0 {
		t.Errorf("interrupt leaked a transcript message: %v", m.messages)
	}

	// A running agent: the interrupt is confirmed via flash.
	a2 := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateRunning}}
	m2 := newModel(a2)
	runCtx := a2.StartRun(context.Background())
	_, _ = m2.interruptRun()
	select {
	case <-runCtx.Done():
	case <-time.After(time.Second):
		t.Error("expected interruptRun to cancel the running agent")
	}
	if !strings.Contains(m2.flash, "Interrupted") {
		t.Errorf("expected an 'Interrupted' flash, got %q", m2.flash)
	}
	if len(m2.messages) != 0 {
		t.Errorf("interrupt leaked a transcript message: %v", m2.messages)
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

func TestRenderChoicePromptHighlightsCommands(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "confirm"
	m.choicePrompt = "The following commands require your approval to run:\n* kubectl delete deployment nginx\n* kubectl scale deployment web --replicas=0\n\nDo you want to proceed ?"

	got := m.renderChoicePrompt()
	// The commands are pulled out onto their own marked lines.
	if !strings.Contains(got, "› kubectl delete deployment nginx") {
		t.Errorf("expected the first command on its own line, got:\n%s", got)
	}
	if !strings.Contains(got, "› kubectl scale deployment web --replicas=0") {
		t.Errorf("expected the second command on its own line, got:\n%s", got)
	}
	// The question is still present.
	if !strings.Contains(got, "Do you want to proceed") {
		t.Errorf("expected the proceed question, got:\n%s", got)
	}
	// The commands must no longer be inline bullets glued to the prose.
	if strings.Contains(got, "run:\n* kubectl") {
		t.Errorf("expected the bullet prose form to be replaced, got:\n%s", got)
	}
}

func TestRenderChoicePromptShowsDryRunPreview(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "confirm"
	m.choicePrompt = "The following commands require your approval to run:\n* kubectl apply -f pod.yaml\n\nDry-run preview (safe, not applied):\n* kubectl apply -f pod.yaml --dry-run=server\n\nDo you want to proceed ?"

	got := m.renderChoicePrompt()
	// The command is on its own marked line.
	if !strings.Contains(got, "› kubectl apply -f pod.yaml") {
		t.Errorf("expected the command on its own line, got:\n%s", got)
	}
	// The dry-run preview is rendered distinctly (dimmed, nested).
	if !strings.Contains(got, "dry-run preview (safe, not applied)") {
		t.Errorf("expected a dry-run preview header, got:\n%s", got)
	}
	if !strings.Contains(got, "⎿ kubectl apply -f pod.yaml --dry-run=server") {
		t.Errorf("expected the dry-run preview nested under the command, got:\n%s", got)
	}
	// The question is still present.
	if !strings.Contains(got, "Do you want to proceed") {
		t.Errorf("expected the proceed question, got:\n%s", got)
	}
}

func TestRenderChoicePromptSessionPickerSimple(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "session"
	m.choicePrompt = "Select a session to resume"

	got := m.renderChoicePrompt()
	if !strings.HasPrefix(got, "? Select a session to resume") {
		t.Errorf("expected the simple '? prompt' form for a session picker, got:\n%s", got)
	}
	if strings.Contains(got, "›") {
		t.Errorf("a session picker must not render command lines, got:\n%s", got)
	}
}

func TestRenderChoicePromptFallsBackForUnknownShape(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "confirm"
	// A confirm prompt with no bullet commands: fall back to the raw prompt.
	m.choicePrompt = "Something unusual. Continue?"

	got := m.renderChoicePrompt()
	if !strings.Contains(got, "Something unusual. Continue?") {
		t.Errorf("expected the raw prompt as a fallback, got:\n%s", got)
	}
	if strings.Contains(got, "›") {
		t.Errorf("the fallback must not fabricate command lines, got:\n%s", got)
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

func TestChoicePickerTruncatesLongLabels(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}}
	m := newModel(a)
	m.width, m.height = 50, 24
	m.resize()
	req := &api.UserChoiceRequest{
		Prompt: "Select a model:",
		Options: []api.UserChoiceOption{
			{Label: "accounts/fireworks/models/deepseek-v3 (current)", Value: "a"},
			{Label: "accounts/fireworks/models/llama4-scout", Value: "b"},
		},
	}
	msg := &api.Message{Type: api.MessageTypeUserChoiceRequest, Payload: req}
	m.handleAgentMsg(msg)

	got := m.renderMessages()
	// The long label is truncated to the list width so the picker doesn't
	// wrap or push its border. The ellipsis confirms truncation.
	if !strings.Contains(got, "…") {
		t.Errorf("expected a long choice label to be truncated, got:\n%s", got)
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

func TestPaletteHasExpandToggles(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	labels := make(map[string]bool)
	for _, it := range m.paletteItems() {
		labels[it.label] = true
	}
	// Both expand toggles are present and reflect the (collapsed) state.
	if !labels["Expand tool results"] {
		t.Error("expected 'Expand tool results' in the palette")
	}
	if !labels["Expand thinking"] {
		t.Error("expected 'Expand thinking' in the palette")
	}
	// Running the tool-results toggle flips the state and the label.
	_, _ = m.toggleExpandToolResults()
	if !m.expandToolResults {
		t.Error("expected toggleExpandToolResults to expand")
	}
	for _, it := range m.paletteItems() {
		if it.label == "Collapse tool results" {
			goto found
		}
	}
	t.Error("expected 'Collapse tool results' label after expanding")
found:
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
	// The rename confirmation is a status-bar flash, not a transcript
	// message, so the transcript must not have gained a rename message.
	if len(m.messages) > 0 {
		if last, ok := m.messages[len(m.messages)-1].Payload.(string); ok && strings.Contains(last, "Renamed session") {
			t.Errorf("rename leaked a transcript confirmation: %q", last)
		}
	}
	if !strings.Contains(m.flash, "my debug session") {
		t.Errorf("expected rename flash to mention the new name, got %q", m.flash)
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
	if got := m.input.Placeholder; got != "Ask anything, or / for commands, !shell, @file…" {
		t.Errorf("placeholder = %q, want default restored", got)
	}
}

func TestRenameModeGreenBorder(t *testing.T) {
	m, _, _ := newRenameModel(t)
	m.enterSessionRename()
	// Rename mode gets a green (secondary) border as a subtle cue.
	if got := m.inputBox().GetBorderTopForeground(); got != colorSecondary {
		t.Errorf("rename border = %v, want colorSecondary %v", got, colorSecondary)
	}
	// Exiting rename mode restores the primary border.
	m.exitSessionRename()
	if got := m.inputBox().GetBorderTopForeground(); got != colorPrimary {
		t.Errorf("normal border after rename = %v, want colorPrimary %v", got, colorPrimary)
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

func TestBrowserFilterTypingNarrows(t *testing.T) {
	m := newBrowserModel()
	sessions := testSessions()
	sessions = append(sessions, api.SessionInfo{ID: "sx9", Name: "deploy-debug", ModelID: "m", LastModified: time.Now()})
	m.openBrowser(sessions)
	full := len(m.browserSessions)

	// Typing a non-reserved character ('e') filters the list.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("e")})
	if m.browserFilter != "e" {
		t.Fatalf("filter = %q, want %q", m.browserFilter, "e")
	}
	if len(m.browserSessions) >= full {
		t.Errorf("expected the list to narrow after filtering, got %d (full %d)", len(m.browserSessions), full)
	}
	// Every surviving session matches the filter.
	for _, s := range m.browserSessions {
		if !sessionMatchesFilter(s, "e") {
			t.Errorf("session %q does not match filter 'e' but survived", s.ID)
		}
	}
}

func TestBrowserFilterBackspaceAndEsc(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("e")})
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("c")})
	if m.browserFilter != "ec" {
		t.Fatalf("filter = %q, want %q", m.browserFilter, "ec")
	}

	// Backspace edits the filter.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if m.browserFilter != "e" {
		t.Errorf("after backspace: filter = %q, want %q", m.browserFilter, "e")
	}

	// Esc clears an active filter (it doesn't close the browser).
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.browserFilter != "" {
		t.Errorf("after esc: filter = %q, want cleared", m.browserFilter)
	}
	if !m.browserOpen {
		t.Error("expected esc to clear the filter, not close the browser")
	}
}

func TestBrowserFilterNoMatchMessage(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())
	// A filter that matches nothing.
	_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("z")})
	got := m.viewSessionBrowser()
	if !strings.Contains(got, "No sessions match") {
		t.Errorf("expected a 'No sessions match' message, got:\n%s", got)
	}
}

func TestBrowserReservedKeysDoNotFilter(t *testing.T) {
	m := newBrowserModel()
	m.openBrowser(testSessions())
	before := len(m.browserSessions)
	// 'd', 'r', 'j', 'k' are reserved commands and must not start a filter.
	for _, r := range "drjk" {
		_, _ = m.handleBrowserKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
	}
	if m.browserFilter != "" {
		t.Errorf("reserved keys started a filter: %q (want empty)", m.browserFilter)
	}
	// The list wasn't narrowed by reserved keys.
	if len(m.browserSessions) != before {
		t.Errorf("reserved keys narrowed the list: %d (want %d)", len(m.browserSessions), before)
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

func TestWelcomeShowsKubeContext(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 90, 30
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev-context", namespace: "payments"}
	m.kubeContextOK = true

	got := m.renderWelcome()
	if !strings.Contains(got, "dev-context/payments") {
		t.Errorf("expected the kube context in the welcome panel, got:\n%s", got)
	}
	if !strings.Contains(got, "Connected to") {
		t.Errorf("expected a 'Connected to' label, got:\n%s", got)
	}
}

func TestWelcomeFlagsProdContext(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 90, 30
	m.resize()
	m.kubeContext = kubeContextInfo{context: "gke-prod-eu", namespace: "default"}
	m.kubeContextOK = true

	got := m.renderWelcome()
	if !strings.Contains(got, "prod") {
		t.Errorf("expected a prod warning label for a prod context, got:\n%s", got)
	}
}

func TestWelcomeShowsNoKubeconfig(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 90, 30
	m.resize()
	m.kubeContextOK = false

	got := m.renderWelcome()
	if !strings.Contains(got, "No kubeconfig") {
		t.Errorf("expected a no-kubeconfig hint when the context is missing, got:\n%s", got)
	}
}

func TestWelcomeListsCommands(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 110, 30
	m.resize()
	m.kubeContext = kubeContextInfo{context: "dev"}
	m.kubeContextOK = true

	got := m.renderWelcome()
	for _, cmd := range []string{"/sessions", "/context", "/namespace", "/model", "/rename", "/compact", "/clear", "/exit"} {
		if !strings.Contains(got, cmd) {
			t.Errorf("expected %q in the welcome command reference, got:\n%s", cmd, got)
		}
	}
}

func TestWelcomeNeverExceedsTerminalWidth(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.kubeContext = kubeContextInfo{context: "dev"}
	m.kubeContextOK = true

	// The ASCII logo is a fixed 48-column art block (pre-existing) that can't
	// shrink, so the sweep starts at 50 — wide enough for the logo and tight
	// enough to exercise the single-column command layout. The dynamic content
	// (context panel, command rows, tagline, footer) must always fit.
	for _, w := range []int{50, 60, 80, 110} {
		m.width, m.height = w, 30
		m.resize()
		got := m.renderWelcome()
		for _, line := range strings.Split(got, "\n") {
			if lipgloss.Width(line) > w {
				t.Errorf("width=%d: line exceeds terminal (%d): %q", w, lipgloss.Width(line), line)
			}
		}
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

func TestRenderToolGroupWrapsLongCommand(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	long := "kubectl apply -f https://raw.githubusercontent.com/example-org/manifests/main/big-deployment-with-long-name.yaml"
	req := &api.Message{Type: api.MessageTypeToolCallRequest, Payload: long}
	resp := &api.Message{Type: api.MessageTypeToolCallResponse, Payload: map[string]any{"stdout": "pod created\n"}}

	got := m.renderToolGroup(req, resp, 90)
	// The tail of the command (the part truncation would have eaten) is visible
	// because the command wraps rather than being cut to "…".
	if !strings.Contains(got, "deployment-with-long-name.yaml") {
		t.Errorf("expected the long command to wrap and keep its tail visible, got:\n%s", got)
	}
	if strings.Contains(got, "…") {
		t.Errorf("expected no truncation ellipsis on a wrapped long command, got:\n%s", got)
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

	// A request whose result has not arrived yet renders standalone with a
	// live spinner frame (not a static "Running" label).
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get nodes", Timestamp: time.Now()},
	}
	m.dirty = true

	got := m.renderMessages()
	if !strings.Contains(got, "kubectl get nodes") {
		t.Errorf("expected the command, got:\n%s", got)
	}
	// The MiniDot spinner's first frame is ⠋; the in-flight call shows it
	// inside the tool box (a rounded border) instead of a literal word.
	if !strings.Contains(got, "⠋") {
		t.Errorf("expected a live spinner frame for an in-flight call, got:\n%s", got)
	}
	if strings.Contains(got, "Running") {
		t.Errorf("an in-flight call must not show the static 'Running' label, got:\n%s", got)
	}
}

func TestRenderMessagesInFlightToolShowsElapsed(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateRunning}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	// thinkStart spans the turn; an in-flight tool call shows the turn's
	// elapsed time so a long-running command doesn't look frozen.
	m.thinkStart = time.Now().Add(-5 * time.Second)
	m.messages = []*api.Message{
		{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl wait pod/x", Timestamp: time.Now()},
	}
	m.dirty = true
	got := m.renderMessages()
	if !strings.Contains(got, "kubectl wait pod/x") {
		t.Errorf("expected the command, got:\n%s", got)
	}
	if !strings.Contains(got, "5s") {
		t.Errorf("expected the elapsed timer (5s) on an in-flight call, got:\n%s", got)
	}
}

func TestHasInFlightToolCall(t *testing.T) {
	m := newModel(nil)
	if m.hasInFlightToolCall() {
		t.Error("expected no in-flight call with no messages")
	}
	// A trailing tool-call request with no response is in-flight.
	m.messages = []*api.Message{
		{Type: api.MessageTypeText, Payload: "hi"},
		{Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods"},
	}
	if !m.hasInFlightToolCall() {
		t.Error("expected an in-flight call when the last message is a tool-call request")
	}
	// A tool-call request followed by its response is NOT in-flight.
	m.messages = append(m.messages, &api.Message{Type: api.MessageTypeToolCallResponse, Payload: "ok"})
	if m.hasInFlightToolCall() {
		t.Error("expected no in-flight call once the response has arrived")
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
			"stdout":    "partial output",
			"stderr":    "Error from server: not found",
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
	// The exit code is surfaced on the failed marker line.
	if !strings.Contains(got, "exit 1") {
		t.Errorf("expected 'exit 1' on the failed result, got:\n%s", got)
	}
}

func TestRenderToolResultExitCodeDistinguishes(t *testing.T) {
	m := newModel(nil)

	// exit 137 (OOM-killed) reads distinctly from exit 1.
	msg := &api.Message{
		Type: api.MessageTypeToolCallResponse,
		Payload: map[string]any{
			"stderr":    "OOMKilled",
			"exit_code": float64(137),
		},
	}
	got := m.renderToolResult(msg)
	if !strings.Contains(got, "exit 137") {
		t.Errorf("expected 'exit 137' surfaced, got:\n%s", got)
	}

	// A failed result with no exit code (err.Error() string) doesn't fabricate one.
	got = m.renderToolResult(&api.Message{Type: api.MessageTypeToolCallResponse, Payload: "connection refused"})
	if strings.Contains(got, "exit ") {
		t.Errorf("a string error must not show a fabricated exit code, got:\n%s", got)
	}
	if !strings.Contains(got, "✗") {
		t.Errorf("expected a ✗ marker on a string error, got:\n%s", got)
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

func TestRenderToolResultColorizesDiff(t *testing.T) {
	m := newModel(nil)
	m.expandToolResults = true // show the full diff, not the collapsed 3 lines
	msg := &api.Message{
		Type: api.MessageTypeToolCallResponse,
		Payload: map[string]any{
			"stdout":    "--- a\n+++ b\n@@ -1 +1 @@\n-foo\n+bar\n",
			"exit_code": float64(0),
		},
	}
	got := m.renderToolResult(msg)
	// The diff's addition (+bar) and deletion (-foo) lines render with
	// diff coloring (green/red); assert the content is present.
	if !strings.Contains(got, "+bar") {
		t.Errorf("expected the diff addition line present, got:\n%s", got)
	}
	if !strings.Contains(got, "-foo") {
		t.Errorf("expected the diff deletion line present, got:\n%s", got)
	}
}

func TestRenderToolGroupFailedHeader(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)

	req := &api.Message{Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pod missing"}
	resp := &api.Message{
		Type:    api.MessageTypeToolCallResponse,
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

func TestKubeContextRefreshesAfterTurn(t *testing.T) {
	// Start with a kubeconfig pointing at "agent-context".
	path := writeKubeConfig(t, "agent-context", true)
	store := sessions.NewInMemoryChatStore()
	a := &agent.Agent{
		Session:    &api.Session{ID: "test", AgentState: api.AgentStateRunning, ChatMessageStore: store},
		Kubeconfig: path,
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	if m.kubeContext.context != "agent-context" {
		t.Fatalf("precondition: expected agent-context at startup, got %q", m.kubeContext.context)
	}

	// Simulate a /context switch: rewrite the kubeconfig to a new context,
	// then deliver the agent's "Switched to context" text message with the
	// session going done — exactly what handleAgentMsg sees.
	newPath := writeKubeConfig(t, "new-context", true)
	a.Kubeconfig = newPath
	switched := &api.Message{
		Source:    api.MessageSourceAgent,
		Type:      api.MessageTypeText,
		Payload:   "Switched to context `new-context` (session only — global kubeconfig unchanged).",
		Timestamp: time.Now(),
	}
	a.Session.AgentState = api.AgentStateDone
	m.handleAgentMsg(switched)

	// The status bar reflects the new context immediately, without waiting
	// for the kubeContextTTL tick.
	if m.kubeContext.context != "new-context" {
		t.Errorf("expected the kube context to refresh after the turn, got %q", m.kubeContext.context)
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

func TestViewStatusOneLineAtNarrowWidth(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "a-very-long-model-name", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.kubeContext = kubeContextInfo{context: "gke-prod-eu-cluster", namespace: "payments"}
	m.kubeContextOK = true
	// At a very narrow width the shrink loop can't shrink everything to fit;
	// the status bar must still stay on one line (truncate) rather than wrap.
	for _, w := range []int{20, 30, 40} {
		m.width = w
		got := m.viewStatus(a.GetSession())
		if n := lipgloss.Height(got); n != 1 {
			t.Errorf("width=%d: status bar is %d lines, want 1:\n%s", w, n, got)
		}
	}
}

func TestViewStateShowsTurnDurationOnDone(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateDone}}
	m := newModel(a)
	m.lastTurnDuration = 12 * time.Second

	got := m.viewState(api.AgentStateDone)
	if !strings.Contains(got, "Done") {
		t.Errorf("expected the Done label, got %q", got)
	}
	if !strings.Contains(got, "12s") {
		t.Errorf("expected the persisted turn duration '12s' on Done, got %q", got)
	}
}

func TestViewStateHidesDurationOnIdle(t *testing.T) {
	m := newModel(nil)
	m.lastTurnDuration = 12 * time.Second
	// Idle state must not carry the done turn's duration.
	if got := m.viewState(api.AgentStateIdle); strings.Contains(got, "12s") {
		t.Errorf("expected no duration on Idle, got %q", got)
	}
}

func TestDoneTransitionStashesTurnDuration(t *testing.T) {
	m, store := newStreamModel() // AgentStateRunning
	// Simulate a turn that started, then a final text message that flips the
	// session to Done.
	m.thinkStart = time.Now().Add(-5 * time.Second)
	final := &api.Message{
		ID: "s1", Source: api.MessageSourceModel, Type: api.MessageTypeText,
		Payload: "done", Timestamp: time.Now(),
	}
	if err := store.AddChatMessage(final); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	m.agent.Session.AgentState = api.AgentStateDone
	m.handleAgentMsg(final)

	// The turn duration is stashed and thinkStart cleared.
	if m.lastTurnDuration <= 0 {
		t.Errorf("expected lastTurnDuration to be stashed, got %v", m.lastTurnDuration)
	}
	if !m.thinkStart.IsZero() {
		t.Error("expected thinkStart to be cleared after the turn completes")
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
	// A non-empty draft with no completion hint shows the draft-size counter
	// line, so the block grows by one.
	if got := m.inputBlockHeight(); got != base+1 {
		t.Errorf("inputBlockHeight = %d, want %d with the draft counter visible", got, base+1)
	}
	if view := m.viewInput(api.AgentStateIdle); !strings.Contains(view, "chars") || !strings.Contains(view, "words") {
		t.Errorf("expected the draft counter rendered in the input box, got:\n%s", view)
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

func TestRenderErrorWrapsLongMessage(t *testing.T) {
	m := newModel(nil)
	// A long unbreakable error line must wrap, not overflow the box.
	long := strings.Repeat("x", 80)
	msg := &api.Message{Type: api.MessageTypeError, Payload: long, Timestamp: time.Now()}
	got := m.renderError(msg, 60)
	// The error body wraps: the 80-char line is split across multiple lines,
	// each within the box's content width. Assert no content line exceeds
	// the box frame width (62 = 60 inner + 2 border).
	for _, line := range strings.Split(got, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "╭") || strings.HasPrefix(trimmed, "╰") ||
			strings.HasPrefix(trimmed, "│") || trimmed == "" {
			continue
		}
		if w := lipgloss.Width(line); w > 62 {
			t.Errorf("error content line exceeds box width (62): %q", line)
		}
	}
}

func TestRenderTextMsgMarksShellEscape(t *testing.T) {
	m := newModel(nil)
	r, err := m.cache.getRenderer(80)
	if err != nil {
		t.Fatalf("getRenderer: %v", err)
	}

	// A shell-escape user message ("!command") is marked distinctly.
	shellMsg := &api.Message{
		Source: api.MessageSourceUser, Type: api.MessageTypeText,
		Payload: "!kubectl get nodes", Timestamp: time.Now(),
	}
	got := m.renderTextMsg(shellMsg, r, 80)
	if !strings.Contains(got, "shell") {
		t.Errorf("expected a shell-escape marker on a '!command' user message, got:\n%s", got)
	}

	// A normal user message is not marked as a shell escape.
	normalMsg := &api.Message{
		Source: api.MessageSourceUser, Type: api.MessageTypeText,
		Payload: "what pods are running?", Timestamp: time.Now(),
	}
	if got := m.renderTextMsg(normalMsg, r, 80); strings.Contains(got, "shell") {
		t.Errorf("a normal user message must not show a shell marker, got:\n%s", got)
	}
}

func TestViewStatusShowsContextBudget(t *testing.T) {
	store := sessions.NewInMemoryChatStore()
	_ = store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a", Tokens: 44000})
	_ = store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "b", Tokens: 45200})

	a := &agent.Agent{Session: &api.Session{
		ID: "test", AgentState: api.AgentStateIdle, ModelID: "m", ChatMessageStore: store,
	}}
	m := newModel(a)
	m.width = 100

	// The cache updates on message arrival (mirroring the live flow).
	m.handleAgentMsg(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "b", Tokens: 45200})

	// The LATEST model turn's total (45200) is the context-window fill:
	// 45200 / 128000 ≈ 35%.
	got := m.viewStatus(a.GetSession())
	if !strings.Contains(got, "ctx ") || !strings.Contains(got, "35%") {
		t.Errorf("expected a context budget indicator at 35%%, got:\n%s", got)
	}

	// Hidden when no usage was reported (fresh model, empty session — the
	// token cache is zero).
	emptyAgent := &agent.Agent{Session: &api.Session{ID: "test2", AgentState: api.AgentStateIdle, ChatMessageStore: sessions.NewInMemoryChatStore()}}
	m2 := newModel(emptyAgent)
	m2.width = 100
	if g := m2.viewStatus(emptyAgent.GetSession()); strings.Contains(g, "ctx ") {
		t.Errorf("expected no context indicator for an empty session, got:\n%s", g)
	}
}

func TestViewContextBudgetTiers(t *testing.T) {
	m := newModel(nil)

	// No usage: nothing rendered.
	if got := m.viewContextBudget(0); got != "" {
		t.Errorf("expected no indicator for 0 tokens, got %q", got)
	}

	// Low usage (<50%): contains a bar and percentage.
	got := m.viewContextBudget(10_000) // ~7%
	if !strings.Contains(got, "ctx ") || !strings.Contains(got, "%") {
		t.Errorf("expected a low-usage indicator, got %q", got)
	}
	if !strings.Contains(got, "7%") {
		t.Errorf("expected 7%% for 10000/128000, got %q", got)
	}

	// High usage (>=80%): still capped at 100%.
	got = m.viewContextBudget(200_000)
	if !strings.Contains(got, "100%") {
		t.Errorf("expected usage capped at 100%%, got %q", got)
	}
	// The /compact hint appears at high usage.
	if !strings.Contains(got, "/compact") {
		t.Errorf("expected a /compact hint at high usage, got %q", got)
	}
}

func TestViewContextBudgetEnvOverride(t *testing.T) {
	// A smaller budget makes a given token count read as a higher percentage.
	prev := contextBudgetTokens
	contextBudgetTokens = 50_000
	defer func() { contextBudgetTokens = prev }()

	m := newModel(nil)
	got := m.viewContextBudget(25_000) // 50% with the override
	if !strings.Contains(got, "50%") {
		t.Errorf("expected 50%% with a 50000 budget override, got %q", got)
	}
}

func TestViewStatusShowsMCPStatus(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{
		ID: "test", AgentState: api.AgentStateIdle, ModelID: "m",
		MCPStatus: &api.MCPStatus{TotalServers: 2, ConnectedCount: 2, FailedCount: 0, ClientEnabled: true},
	}}
	m := newModel(a)
	m.width = 100

	got := m.viewStatus(a.Session)
	if !strings.Contains(got, "🔌 2/2") {
		t.Errorf("expected the MCP connected indicator, got:\n%s", got)
	}
}

func TestViewStatusMCPFailsYellow(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{
		ID: "test", AgentState: api.AgentStateIdle, ModelID: "m",
		MCPStatus: &api.MCPStatus{TotalServers: 2, ConnectedCount: 1, FailedCount: 1, ClientEnabled: true},
	}}
	m := newModel(a)
	m.width = 100

	got := m.viewStatus(a.Session)
	if !strings.Contains(got, "🔌 1/2") {
		t.Errorf("expected the MCP partial indicator, got:\n%s", got)
	}
}

func TestViewStatusHidesMCPWhenUnconfigured(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{
		ID: "test", AgentState: api.AgentStateIdle, ModelID: "m",
		MCPStatus: &api.MCPStatus{TotalServers: 0, ClientEnabled: false},
	}}
	m := newModel(a)
	m.width = 100

	got := m.viewStatus(a.Session)
	if strings.Contains(got, "🔌") {
		t.Errorf("expected no MCP indicator when unconfigured, got:\n%s", got)
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

func thinkingDelta(id, payload string) *api.Message {
	return &api.Message{
		ID:        id,
		Source:    api.MessageSourceModel,
		Type:      api.MessageTypeThinkingDelta,
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

func TestTextDeltaShowsLiveCursorWhileRunning(t *testing.T) {
	m, _ := newStreamModel() // AgentStateRunning
	const streamID = "stream-1"
	m.handleAgentMsg(textDelta(streamID, "partial reply"))

	got := m.renderMessages()
	// Glamour interleaves ANSI codes, so check the words individually rather
	// than the full phrase.
	if !strings.Contains(got, "partial") || !strings.Contains(got, "reply") {
		t.Fatalf("expected the streamed text, got:\n%s", got)
	}
	// A live-streaming delta shows a cursor at the tail (rendered outside
	// glamour, so the glyph is contiguous).
	if !strings.Contains(got, "▋") {
		t.Errorf("expected a live cursor on the streaming delta, got:\n%s", got)
	}
}

func TestFinalTextHasNoLiveCursor(t *testing.T) {
	m, store := newStreamModel()
	const streamID = "stream-1"
	m.handleAgentMsg(textDelta(streamID, "the answer"))

	// The agent finishes: the final Text message (same ID) replaces the delta
	// and the agent goes idle. The cursor must not appear on the final reply.
	final := &api.Message{
		ID: streamID, Source: api.MessageSourceModel, Type: api.MessageTypeText,
		Payload: "the answer", Timestamp: time.Now(),
	}
	if err := store.AddChatMessage(final); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	m.agent.Session.AgentState = api.AgentStateIdle
	m.handleAgentMsg(final)

	got := m.renderMessages()
	if !strings.Contains(got, "answer") {
		t.Fatalf("expected the final text, got:\n%s", got)
	}
	if strings.Contains(got, "▋") {
		t.Errorf("the final reply must not show a live cursor, got:\n%s", got)
	}
}

func TestThinkingDeltaShowsStreamingHeader(t *testing.T) {
	m, _ := newStreamModel() // AgentStateRunning
	const thinkID = "think-1"
	m.handleAgentMsg(thinkingDelta(thinkID, "Let me consider the pods"))

	got := m.renderMessages()
	if !strings.Contains(got, "Thinking") {
		t.Errorf("expected a streaming thinking header, got:\n%s", got)
	}
	// The partial reasoning is shown dimmed while streaming.
	if !strings.Contains(got, "consider") {
		t.Errorf("expected the partial reasoning visible while streaming, got:\n%s", got)
	}
}

func TestFinalThinkingCollapsesToSummary(t *testing.T) {
	m, _ := newStreamModel()
	const thinkID = "think-1"
	m.handleAgentMsg(thinkingDelta(thinkID, "step one\nstep two\nstep three"))

	// The final thinking message replaces the delta and collapses to a summary.
	final := &api.Message{
		ID: thinkID, Source: api.MessageSourceModel, Type: api.MessageTypeThinking,
		Payload: "step one\nstep two\nstep three", Timestamp: time.Now(),
	}
	m.agent.Session.AgentState = api.AgentStateIdle
	m.handleAgentMsg(final)

	got := m.renderMessages()
	if !strings.Contains(got, "Thought") {
		t.Errorf("expected a collapsed 'Thought' summary, got:\n%s", got)
	}
	if !strings.Contains(got, "3 lines") {
		t.Errorf("expected the line count in the summary, got:\n%s", got)
	}
	// Collapsed by default: the full reasoning isn't shown inline.
	if strings.Contains(got, "step three") {
		t.Errorf("collapsed thinking must not show the full reasoning, got:\n%s", got)
	}
}

func TestCtrlTTogglesThinkingExpansion(t *testing.T) {
	m, _ := newStreamModel()
	const thinkID = "think-1"
	final := &api.Message{
		ID: thinkID, Source: api.MessageSourceModel, Type: api.MessageTypeThinking,
		Payload: "the reasoning here", Timestamp: time.Now(),
	}
	m.agent.Session.AgentState = api.AgentStateIdle
	m.handleAgentMsg(final)

	collapsed := m.renderMessages()
	if strings.Contains(collapsed, "the reasoning here") {
		t.Errorf("expected thinking collapsed by default, got:\n%s", collapsed)
	}

	// Ctrl+T expands.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlT})
	if !m.expandThinking {
		t.Fatal("expected ctrl+t to expand thinking")
	}
	expanded := m.renderMessages()
	if !strings.Contains(expanded, "the reasoning here") {
		t.Errorf("expected expanded thinking to show the reasoning, got:\n%s", expanded)
	}

	// A second Ctrl+T collapses again.
	_, _ = m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlT})
	if m.expandThinking {
		t.Error("expected a second ctrl+t to collapse thinking")
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

func TestAutoModeWarningBorderIdle(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// Auto mode off: the idle input border is the primary color.
	if got := m.inputBox().GetBorderTopForeground(); got != colorPrimary {
		t.Errorf("auto-off border = %v, want colorPrimary %v", got, colorPrimary)
	}

	// Turn auto mode on: the border switches to warning so it's unmistakable.
	if !a.ToggleSkipPermissions() {
		t.Fatal("expected ToggleSkipPermissions to enable auto mode")
	}
	if got := m.inputBox().GetBorderTopForeground(); got != colorWarning {
		t.Errorf("auto-on border = %v, want colorWarning %v", got, colorWarning)
	}

	// The rendered idle input box carries the warning border (asserted via
	// the style directly — lipgloss strips ANSI codes in non-terminal tests,
	// so the rendered string can't be compared for color).
}

func TestAutoModeWarningBorderRunning(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateRunning}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// While running with auto mode off, the (dim) running box uses the dim
	// border.
	if got := m.runningBox().GetBorderTopForeground(); got != colorDim {
		t.Errorf("auto-off running border = %v, want colorDim %v", got, colorDim)
	}

	// While running with auto mode on, the running box keeps the warning
	// border so auto mode is still visible during execution.
	if !a.ToggleSkipPermissions() {
		t.Fatal("expected ToggleSkipPermissions to enable auto mode")
	}
	if got := m.runningBox().GetBorderTopForeground(); got != colorWarning {
		t.Errorf("auto-on running border = %v, want colorWarning %v", got, colorWarning)
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

	// The cache starts cold; warm it via the async fetch path (Update has a
	// value receiver, so apply the returned model).
	m.input.SetValue("/namespace ")
	fetchCmd := m.maybeFetchNamespaces()
	if fetchCmd == nil {
		t.Fatal("expected a namespace fetch to be triggered for a /namespace draft")
	}
	updated, _ := m.Update(fetchCmd())
	m = updated.(model)

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
	m.width, m.height = 220, 40
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
	// The bar must tell the truth: Ctrl+C QUITS (there is no cancel path);
	// Esc interrupts the run.
	if !strings.Contains(help, "Ctrl+C: quit") {
		t.Errorf("running: expected the quit hint (Ctrl+C quits the app), got:\n%s", help)
	}
	if !strings.Contains(help, "Esc: interrupt") {
		t.Errorf("running: expected the interrupt hint, got:\n%s", help)
	}
	if !strings.Contains(help, "scroll") {
		t.Errorf("running: expected the arrow scroll hint, got:\n%s", help)
	}
	if strings.Contains(help, "Enter: send") {
		t.Errorf("running: must not show the idle send hint, got:\n%s", help)
	}
}

func TestViewInputInitializingLabel(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "t", AgentState: api.AgentStateInitializing}}
	m := newModel(a)
	m.width, m.height = 80, 24
	m.resize()
	got := m.viewInput(api.AgentStateInitializing)
	if !strings.Contains(got, "Initializing...") {
		t.Errorf("expected 'Initializing...' label during init, got:\n%s", got)
	}
	if strings.Contains(got, "Thinking...") {
		t.Errorf("init phase must not show 'Thinking...', got:\n%s", got)
	}
	// Running still shows 'Thinking...'.
	a2 := &agent.Agent{Session: &api.Session{ID: "t", AgentState: api.AgentStateRunning}}
	m2 := newModel(a2)
	m2.width, m2.height = 80, 24
	m2.resize()
	got2 := m2.viewInput(api.AgentStateRunning)
	if !strings.Contains(got2, "Thinking...") {
		t.Errorf("expected 'Thinking...' label while running, got:\n%s", got2)
	}
}

func TestFrameStaysExactlyOneScreenTallWhileTyping(t *testing.T) {
	// Regression: typing the first character shows the draft-counter line,
	// growing the input block without shrinking the viewport, so the frame
	// became one line taller than the terminal and the status bar scrolled
	// out of view until a window resize recalculated the layout.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 120, 24
	m.resize()

	if got := lipgloss.Height(m.View()); got != m.height {
		t.Fatalf("initial frame height = %d, want exactly %d (one screen)", got, m.height)
	}

	// Type a character: the draft counter appears under the input box.
	m.input.SetValue("a")
	m.syncInputHeight()

	if got := lipgloss.Height(m.View()); got != m.height {
		t.Errorf("frame height after typing = %d, want exactly %d (input block grew but viewport did not shrink)", got, m.height)
	}

	// Typing enough to soft-wrap grows the content lines too; the frame
	// must still fit exactly.
	m.input.SetValue(strings.Repeat("x", m.input.Width()+10))
	m.syncInputHeight()
	if got := lipgloss.Height(m.View()); got != m.height {
		t.Errorf("frame height after wrap = %d, want exactly %d", got, m.height)
	}
}

func TestEscCancelsModelPicker(t *testing.T) {
	// Regression: Esc on the /model picker used to send the generic decline
	// (choice 3), which handleModelChoice interpreted as "select the 3rd
	// model" — silently switching and persisting a model on cancel.
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}, Input: make(chan any, 1)}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "model"

	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.inChoiceMode {
		t.Error("expected choice mode to close on esc")
	}
	if cmd == nil {
		t.Fatal("expected a cancel command")
	}
	go cmd()
	got := <-a.Input
	resp, ok := got.(*api.UserChoiceResponse)
	if !ok {
		t.Fatalf("expected *api.UserChoiceResponse, got %T", got)
	}
	if resp.Choice != 0 {
		t.Errorf("expected cancel (choice 0), got %d", resp.Choice)
	}
}

func TestChoiceKindStoredFromRequest(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}}
	m := newModel(a)
	m.handleAgentMsg(&api.Message{
		Type:    api.MessageTypeUserChoiceRequest,
		Payload: &api.UserChoiceRequest{Prompt: "Select a model:", Options: []api.UserChoiceOption{{Label: "m1"}}, Kind: "model"},
	})
	if !m.inChoiceMode {
		t.Fatal("expected choice mode to open")
	}
	if m.choiceType != "model" {
		t.Errorf("choiceType = %q, want %q", m.choiceType, "model")
	}
}

type slowExecutor struct{ delay time.Duration }

func (e *slowExecutor) Execute(ctx context.Context, _ string, _ []string, _ string) (*sandbox.ExecResult, error) {
	select {
	case <-time.After(e.delay):
		return &sandbox.ExecResult{Stdout: "default\n"}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (e *slowExecutor) Close(ctx context.Context) error { return nil }

func TestNamespaceCompletionNeverBlocksOnCluster(t *testing.T) {
	// Regression: namespaceMatches used to shell out to kubectl synchronously
	// on the UI goroutine (10s timeout, no negative caching), freezing all
	// input/rendering on slow or unreachable clusters.
	a := &agent.Agent{
		Session:  &api.Session{ID: "test", AgentState: api.AgentStateIdle},
		Executor: &slowExecutor{delay: 10 * time.Second},
	}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.input.SetValue("/namespace pay")

	start := time.Now()
	for i := 0; i < 100; i++ {
		_ = m.namespaceMatches()
		_ = m.completionHintVisible()
		_ = m.completionHint()
	}
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("300 completion-path calls took %v; the UI goroutine must never block on the cluster", elapsed)
	}

	// The fetch is triggered asynchronously instead.
	if cmd := m.maybeFetchNamespaces(); cmd == nil {
		t.Fatal("expected an async fetch to be triggered")
	}

	// An error result is negative-cached: no immediate refetch.
	m2, _ := m.Update(nsCompletionsMsg{err: fmt.Errorf("cluster unreachable")})
	m = m2.(model)
	if m.namespacesFetching {
		t.Error("fetching flag should clear on result")
	}
	if cmd := m.maybeFetchNamespaces(); cmd != nil {
		t.Error("expected negative caching to suppress an immediate refetch")
	}
}

func TestExitStateQuitsProgram(t *testing.T) {
	// Regression: /exit set AgentStateExited and closed Output, but nothing
	// ever returned tea.Quit — the TUI kept running as a zombie.
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateExited}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	_, cmd := m.handleAgentMsg(&api.Message{
		Source:  api.MessageSourceAgent,
		Type:    api.MessageTypeText,
		Payload: "It has been a pleasure assisting you. Have a great day!",
	})
	if cmd == nil {
		t.Fatal("expected a quit command when the agent state is Exited")
	}
	if msg := cmd(); msg == nil {
		t.Fatal("expected the command to produce quit-related messages")
	}

	// The channel-close path quits too.
	m2 := newModel(a)
	_, cmd2 := m2.Update(agentExitedMsg{})
	if cmd2 == nil {
		t.Fatal("expected a quit command on agentExitedMsg")
	}
}

func TestFrameStaysExactWithOpenPanels(t *testing.T) {
	// Regression: the picker/browser chrome was under-budgeted by one line
	// and the palette had no height windowing at all, so opening any panel
	// on a standard 80x24 terminal made the frame taller than the screen
	// and pushed the status bar out of view.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	newTestModel := func() model {
		m := newModel(a)
		m.width, m.height = 80, 24
		m.resize()
		return m
	}
	check := func(m model, what string) {
		t.Helper()
		if got := lipgloss.Height(m.View()); got != m.height {
			t.Errorf("%s: frame height = %d, want exactly %d", what, got, m.height)
		}
	}

	// Picker with a height-capped long list.
	m := newTestModel()
	m.pickerOpen = true
	m.pickerKind = pickerNamespace
	m.pickerTitle = "Namespaces"
	for i := 0; i < 30; i++ {
		m.pickerItems = append(m.pickerItems, pickerItem{value: fmt.Sprintf("namespace-%d", i)})
	}
	m.updateViewportHeight()
	check(m, "picker with 30 items")

	// Picker showing an error line instead of rows.
	m.pickerItems = nil
	m.pickerStatus = "cluster unreachable"
	m.updateViewportHeight()
	check(m, "picker error")

	// Palette (13 items) on a 24-row terminal.
	m = newTestModel()
	m.paletteOpen = true
	m.updateViewportHeight()
	check(m, "palette")

	// Session browser with many sessions.
	m = newTestModel()
	m.browserOpen = true
	for i := 0; i < 20; i++ {
		m.browserSessions = append(m.browserSessions, api.SessionInfo{ID: fmt.Sprintf("s%d", i), ModelID: "m"})
	}
	m.updateViewportHeight()
	check(m, "session browser")

	// Browser with a long error status in the footer.
	m.browserStatus = browserStatusMsg{text: strings.Repeat("x", 200), isErr: true}
	check(m, "browser long error footer")
}

func TestFrameStaysExactAcrossAgentStateTransitions(t *testing.T) {
	// Regression: while the agent runs, the input block renders a 3-line
	// spinner box; when the turn finished with a draft in the box, the block
	// grew (draft lines + counter) but nothing re-budgeted the viewport, so
	// the frame overflowed and the status bar scrolled out of view.
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 80, 24
	m.resize()

	// User drafts a follow-up, then the agent starts running.
	m.input.SetValue("follow-up question while it works")
	m.syncInputHeight()
	a.Session.AgentState = api.AgentStateRunning
	m.handleAgentMsg(&api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: "working…"})
	if got := lipgloss.Height(m.View()); got != m.height {
		t.Fatalf("frame height while running = %d, want exactly %d", got, m.height)
	}

	// Turn finishes: the draft box (with counter line) returns.
	a.Session.AgentState = api.AgentStateDone
	m.handleAgentMsg(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "done"})
	if got := lipgloss.Height(m.View()); got != m.height {
		t.Errorf("frame height after turn end with a draft = %d, want exactly %d", got, m.height)
	}
}

func TestTruncateCellsHandlesWideChars(t *testing.T) {
	// 10 CJK chars = 20 cells; rune-truncation to 10 would render 20 cells
	// and wrap a budgeted single line.
	s := strings.Repeat("界", 10)
	if got := truncateCells(s, 10); lipgloss.Width(got) > 10 {
		t.Errorf("truncateCells(10 CJK, 10) renders %d cells, want <= 10", lipgloss.Width(got))
	}
	if got := truncateCells("ascii", 10); got != "ascii" {
		t.Errorf("truncateCells short string = %q, want unchanged", got)
	}
}

func TestFinalThinkingSurvivesStoreSnapshot(t *testing.T) {
	// Regression: the final thinking message was never stored, so the next
	// store snapshot (any subsequent message) wiped it — the collapsed
	// "Thought for N lines" block and Ctrl+T expansion never appeared in
	// real usage.
	store := sessions.NewInMemoryChatStore()
	a := &agent.Agent{Session: &api.Session{ID: "t", ModelID: "m", AgentState: api.AgentStateRunning, ChatMessageStore: store}, Output: make(chan any, 10)}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()

	// Live deltas accumulate an ephemeral entry...
	m.handleAgentMsg(&api.Message{ID: "think-1", Source: api.MessageSourceModel, Type: api.MessageTypeThinkingDelta, Payload: "pondering"})
	// ...then the final thinking message is STORED (ephemeral)...
	store.AddChatMessage(&api.Message{ID: "think-1", Source: api.MessageSourceModel, Type: api.MessageTypeThinking, Payload: "pondering deeply", Ephemeral: true})
	// ...and the next ordinary message triggers a store snapshot, which
	// previously dropped the thinking entry.
	m.handleAgentMsg(&api.Message{ID: "txt-1", Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "answer"})

	found := false
	for _, msg := range m.messages {
		if msg.Type == api.MessageTypeThinking && msg.Payload == "pondering deeply" {
			found = true
		}
	}
	if !found {
		t.Error("final thinking message did not survive the store snapshot")
	}
}

func TestContextBudgetUsesLatestTurnTokens(t *testing.T) {
	// Regression: the ctx% indicator summed per-message token totals — each
	// of which already includes the whole conversation — so it grew
	// quadratically and hit 100% red long before the window filled.
	a := &agent.Agent{Session: &api.Session{ID: "t", ModelID: "m", AgentState: api.AgentStateIdle}}
	store := sessions.NewInMemoryChatStore()
	a.Session.ChatMessageStore = store
	store.AddChatMessage(&api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q1"})
	store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a1", Tokens: 10_000})
	store.AddChatMessage(&api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q2"})
	store.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "a2", Tokens: 20_000})

	m := newModel(a)
	// Latest turn (20k = real context size) against the 128k budget: ~15%,
	// not (10k+20k)/128k = 23%, and definitely not runaway growth.
	if got := currentContextTokens(a.Session); got != 20_000 {
		t.Errorf("currentContextTokens = %d, want 20000 (latest model turn, not the sum)", got)
	}
	bar := m.viewContextBudget(20_000)
	if !strings.Contains(bar, "15%") {
		t.Errorf("budget for 20k/128k = %q, want 15%%", bar)
	}
	if got := m.viewContextBudget(0); got != "" {
		t.Errorf("expected empty budget with no usage, got %q", got)
	}
}

func TestToolOutputSanitizesANSI(t *testing.T) {
	// Forced-color output must not bleed escape sequences into the frame.
	got := toolResultText(map[string]any{"stdout": "\x1b[31mred text\x1b[0m\r\nnext\x1b[1m"})
	if strings.Contains(got, "\x1b") || strings.Contains(got, "\r") {
		t.Errorf("toolResultText kept escape/control chars: %q", got)
	}
	if !strings.Contains(got, "red text") {
		t.Errorf("sanitized output lost content: %q", got)
	}
}

func TestDiffColoringOnlyForRealDiffs(t *testing.T) {
	if looksLikeUnifiedDiff([]string{"- name: foo", "+ something"}) {
		t.Error("YAML list items misclassified as a diff")
	}
	if !looksLikeUnifiedDiff([]string{"diff -u -N /tmp/a /tmp/b", "-old", "+new"}) {
		t.Error("kubectl diff output not recognized as a diff")
	}
	if !looksLikeUnifiedDiff([]string{"@@ -1,2 +1,2 @@", "-old", "+new"}) {
		t.Error("hunk-only diff not recognized")
	}
}

func TestLastToolOutputPrefersStderrOnFailure(t *testing.T) {
	m := newModel(nil)
	m.messages = append(m.messages, &api.Message{
		Source: api.MessageSourceAgent,
		Type:   api.MessageTypeToolCallResponse,
		Payload: map[string]any{
			"stdout":    "normal output",
			"stderr":    "the real error",
			"exit_code": float64(1),
			"error":     "command failed",
		},
	})
	got, ok := m.lastToolOutput()
	if !ok {
		t.Fatal("expected a copyable output")
	}
	if !strings.Contains(got, "the real error") && !strings.Contains(got, "command failed") {
		t.Errorf("copy got %q, want the failure channel (what the transcript shows)", got)
	}
}

func TestArrowKeysScrollWhileRunning(t *testing.T) {
	// While the agent runs, the input box is hidden behind the spinner —
	// Up/Down must scroll the transcript, not edit the invisible draft.
	a := &agent.Agent{Session: &api.Session{ID: "t", AgentState: api.AgentStateRunning}}
	m := newModel(a)
	m.width, m.height = 80, 24
	m.resize()
	for i := 0; i < 50; i++ {
		m.messages = append(m.messages, &api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: fmt.Sprintf("line %d", i)})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	m.input.SetValue("draft in progress")
	m.handleKey(tea.KeyMsg{Type: tea.KeyUp})
	if got := m.input.Value(); got != "draft in progress" {
		t.Errorf("Up while running edited the hidden draft: %q", got)
	}
	if m.viewport.YOffset == m.viewport.TotalLineCount()-m.viewport.Height {
		t.Error("Up while running did not scroll the transcript")
	}
}

func TestEscDuringRunningFlashesOnlyWhenCancellable(t *testing.T) {
	// A running state without a run context (old /compact) must not flash
	// a lying "Interrupted" message.
	a := &agent.Agent{Session: &api.Session{ID: "t", AgentState: api.AgentStateRunning}}
	m := newModel(a)
	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyEsc})
	if m.flash != "" {
		t.Errorf("flash = %q, want none when nothing was cancellable", m.flash)
	}
	_ = cmd
}

func TestRenamePreservesDraft(t *testing.T) {
	// Regression: entering rename mode overwrote the in-progress draft and
	// Esc/submit cleared it; now the draft is stashed and restored.
	m := newBrowserModel()
	m.input.SetValue("my half-written question")

	m.enterSessionRename()
	if m.input.Value() == "my half-written question" {
		t.Fatal("rename should prefill the session name, not keep the draft")
	}
	m.exitSessionRename()
	if got := m.input.Value(); got != "my half-written question" {
		t.Errorf("draft after rename exit = %q, want restored", got)
	}
}

func TestRenamePasteFlattensToSingleLine(t *testing.T) {
	m := newBrowserModel()
	m.enterSessionRename()
	m.handlePaste(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("multi\nline\nname"), Paste: true})
	if got := m.input.Value(); strings.Contains(got, "\n") || strings.Contains(got, "[+") {
		t.Errorf("rename paste = %q, want a flat single line (no tokens)", got)
	}
	m.exitSessionRename()
}

func TestWholeTokenBackspaceRequiresCursorAtEnd(t *testing.T) {
	// Regression: backspace with the cursor mid-draft ate the paste token at
	// the tail of the input.
	m := newBrowserModel()
	m.input.SetValue("fix this")
	m.handlePaste(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(strings.Repeat("log line\n", 12)), Paste: true})
	if len(m.pastes) != 1 {
		t.Fatalf("expected one attached paste, got %d", len(m.pastes))
	}
	// Cursor to the start (not at the token).
	m.input.CursorStart()
	m.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if len(m.pastes) != 1 {
		t.Error("mid-draft backspace must not remove the tail paste token")
	}
	// At the end, it removes the whole token.
	m.input.CursorEnd()
	m.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if len(m.pastes) != 0 {
		t.Error("end-of-draft backspace should remove the paste token")
	}
}

func TestHistoryRecallClearsPasteAttachments(t *testing.T) {
	m := newBrowserModel()
	// History is rebuilt from the transcript: give it a real user message.
	m.messages = append(m.messages, &api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "first query ever"})

	m.handlePaste(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(strings.Repeat("blob\n", 12)), Paste: true})
	if len(m.pastes) != 1 {
		t.Fatalf("expected one attached paste, got %d", len(m.pastes))
	}
	m.historyPrev()
	if len(m.pastes) != 0 {
		t.Error("history recall must clear orphaned paste attachments")
	}
	if got := m.input.Value(); got != "first query ever" {
		t.Errorf("recalled %q, want the history entry", got)
	}
}

func TestShellModeTrimsLeadingWhitespace(t *testing.T) {
	m := newBrowserModel()
	m.input.SetValue("  !rm -rf /")
	if !m.shellMode() {
		t.Error("shellMode must trim leading whitespace (the agent does)")
	}
}

func TestWheelScrollsOpenPanelNotTranscript(t *testing.T) {
	m := newBrowserModel()
	for i := 0; i < 10; i++ {
		m.browserSessions = append(m.browserSessions, api.SessionInfo{ID: fmt.Sprintf("s%d", i)})
	}
	m.browserOpen = true
	m.browserIndex = 5

	m.handleMouse(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	if m.browserIndex != 4 {
		t.Errorf("wheel-up with browser open: index = %d, want 4", m.browserIndex)
	}
	m.handleMouse(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	m.handleMouse(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	if m.browserIndex != 6 {
		t.Errorf("wheel-down with browser open: index = %d, want 6", m.browserIndex)
	}
}

func TestWheelRevealsAndRehidesClearedTranscript(t *testing.T) {
	m := newBrowserModel()
	for i := 0; i < 40; i++ {
		m.messages = append(m.messages, &api.Message{Source: api.MessageSourceAgent, Type: api.MessageTypeText, Payload: fmt.Sprintf("line %d", i)})
	}
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	// Clear, then wheel up: history reveals (previously only PgUp did this).
	m.clearedAt = 30
	m.refresh()
	m.handleMouse(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	if !m.revealAll {
		t.Error("wheel-up after Ctrl+L should reveal the hidden transcript")
	}
}

func TestCtrlLExcludesTrailingStreamDelta(t *testing.T) {
	// The in-flight reply's delta entry must not count toward the clear
	// boundary — its final message (same ID) would land above the marker.
	m := newBrowserModel()
	m.messages = append(m.messages,
		&api.Message{ID: "u1", Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "q"},
		&api.Message{ID: "s1", Source: api.MessageSourceModel, Type: api.MessageTypeTextDelta, Payload: "streaming…"},
	)
	m.handleKey(tea.KeyMsg{Type: tea.KeyCtrlL})
	if m.clearedAt != 1 {
		t.Errorf("clearedAt = %d, want 1 (the ephemeral delta entry excluded)", m.clearedAt)
	}
}

func TestSpinnerTickChainStopsWhenIdle(t *testing.T) {
	// The spinner chain used to reschedule at ~12fps for the process
	// lifetime; it must only run while something visibly animates.
	m := newBrowserModel() // agent Idle
	_, cmd := m.Update(spinner.TickMsg{})
	if cmd != nil {
		t.Error("idle spinner tick must not reschedule")
	}

	a := &agent.Agent{Session: &api.Session{ID: "t", AgentState: api.AgentStateRunning}}
	m2 := newModel(a)
	_, cmd2 := m2.Update(spinner.TickMsg{})
	if cmd2 == nil {
		t.Error("running spinner tick must keep animating")
	}
}

func TestStatusBarUsesCachedContextTokens(t *testing.T) {
	store := sessions.NewInMemoryChatStore()
	a := &agent.Agent{Session: &api.Session{ID: "t", ModelID: "m", AgentState: api.AgentStateIdle, ChatMessageStore: store}}
	m := newModel(a)
	m.width = 100

	// No cache yet → no indicator.
	if got := m.viewStatus(a.GetSession()); strings.Contains(got, "ctx ") {
		t.Errorf("no indicator expected before any message, got:\n%s", got)
	}

	// A message arrival updates the cache (not per-frame store reads).
	// Production stores before broadcasting, so mirror that.
	msg := &api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "hi", Tokens: 64000}
	_ = store.AddChatMessage(msg)
	m.handleAgentMsg(msg)
	if m.contextTokens != 64000 {
		t.Fatalf("contextTokens cache = %d, want 64000", m.contextTokens)
	}
	if got := m.viewStatus(a.GetSession()); !strings.Contains(got, "ctx ") || !strings.Contains(got, "50%") {
		t.Errorf("expected a 50%% context indicator from the cache, got:\n%s", got)
	}
}

func TestWaitingForInputShowsApprovalCue(t *testing.T) {
	m := newBrowserModel()
	got := m.viewState(api.AgentStateWaitingForInput)
	if !strings.Contains(got, "Approval") {
		t.Errorf("WaitingForInput should read as an approval prompt, got %q", got)
	}
}

func TestNumberKeySelectsChoiceOption(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}, Input: make(chan any, 1)}
	m := newModel(a)
	m.inChoiceMode = true
	m.choiceType = "permission"
	m.list.SetItems([]list.Item{item("Yes"), item("Yes, don't ask again"), item("No")})
	m.list.Select(0)

	_, cmd := m.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("3")})
	if cmd == nil {
		t.Fatal("expected a choice command from number key 3")
	}
	go cmd()
	got := <-a.Input
	resp, ok := got.(*api.UserChoiceResponse)
	if !ok {
		t.Fatalf("expected *api.UserChoiceResponse, got %T", got)
	}
	if resp.Choice != 3 {
		t.Errorf("number key 3 selected choice %d, want 3", resp.Choice)
	}
	if m.inChoiceMode {
		t.Error("choice mode should close after a number-key selection")
	}
}

func TestChoiceRequestClosesPalette(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", AgentState: api.AgentStateWaitingForInput}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.paletteOpen = true

	m.handleAgentMsg(&api.Message{
		Type:    api.MessageTypeUserChoiceRequest,
		Payload: &api.UserChoiceRequest{Prompt: "p", Options: []api.UserChoiceOption{{Label: "Yes"}}, Kind: "permission"},
	})
	if m.paletteOpen {
		t.Error("a choice request must close the palette — the prompt must never hide behind it")
	}
	if !m.inChoiceMode {
		t.Error("expected choice mode to open")
	}
}

func TestStatusClickIgnoredWhilePanelOpen(t *testing.T) {
	a := &agent.Agent{Session: &api.Session{ID: "test", ModelID: "m", AgentState: api.AgentStateIdle}}
	m := newModel(a)
	m.width, m.height = 100, 40
	m.resize()
	m.paletteOpen = true
	_, cmd := m.handleStatusClick(50, 0)
	if cmd != nil {
		t.Error("status clicks must be no-ops while an overlay is open")
	}
}

func TestToolResultsCachedWithToggleAwareKey(t *testing.T) {
	// Regression: making tool responses uncacheable re-rendered every tool
	// result on every 150ms streaming refresh — visibly laggy streaming on
	// sessions with many tool calls.
	m := newBrowserModel()
	m.width, m.height = 100, 40
	m.resize()
	msg := &api.Message{ID: "r1", Source: api.MessageSourceAgent, Type: api.MessageTypeToolCallResponse, Payload: map[string]any{"stdout": "some output"}}
	r, _ := m.cache.getRenderer(80)

	first := m.renderMessage(msg, r, 80)
	if _, ok := m.cache.get("r1"); !ok {
		t.Fatal("collapsed tool response should be cached")
	}
	if got := m.renderMessage(msg, r, 80); got != first {
		t.Error("collapsed re-render did not hit the cache")
	}

	// Toggling expands — a different cache entry, and the orphan-response
	// fix still holds (the toggle re-renders).
	m.expandToolResults = true
	_ = m.renderMessage(msg, r, 80)
	if _, ok := m.cache.get("r1|expanded"); !ok {
		t.Fatal("expanded tool response should be cached under the toggle-aware key")
	}
	m.expandToolResults = false
	if got := m.renderMessage(msg, r, 80); got != first {
		t.Error("collapsing again should hit the original cache entry")
	}
}
