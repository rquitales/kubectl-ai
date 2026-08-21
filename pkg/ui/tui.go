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
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"k8s.io/klog/v2"
)

const logo = `
 _          _               _   _             _
| | ___   _| |__   ___  ___| |_| |       __ _(_)
| |/ / | | | '_ \ / _ \/ __| __| |_____ / _  | |
|   <| |_| | |_) |  __/ (__| |_| |_____| (_| | |
|_|\_\\__,_|_.__/ \___|\___|\__|_|      \__,_|_|
`

// Color palette - Google Material Design colors
var (
	colorPrimary   = lipgloss.Color("#8AB4F8") // Blue 200
	colorSecondary = lipgloss.Color("#81C995") // Green 200
	colorError     = lipgloss.Color("#F28B82") // Red 200
	colorWarning   = lipgloss.Color("#FDD663") // Yellow 200
	colorText      = lipgloss.Color("#E8EAED") // Grey 200
	colorMuted     = lipgloss.Color("#9AA0A6") // Grey 500
	colorDim       = lipgloss.Color("#5F6368") // Grey 700
	colorBgSubtle  = lipgloss.Color("#303134") // Surface variant
	colorBgCode    = lipgloss.Color("#1E1E1E") // Code background
)

// Styles - consolidated for reuse
var (
	textStyle   = lipgloss.NewStyle().Foreground(colorText)
	mutedStyle  = lipgloss.NewStyle().Foreground(colorMuted)
	dimStyle    = lipgloss.NewStyle().Foreground(colorDim)
	primaryText = lipgloss.NewStyle().Foreground(colorPrimary).Bold(true)
	successText = lipgloss.NewStyle().Foreground(colorSecondary).Bold(true)
	errorText   = lipgloss.NewStyle().Foreground(colorError).Bold(true)
	warnText    = lipgloss.NewStyle().Foreground(colorWarning).Bold(true)

	statusBar = lipgloss.NewStyle().Background(colorBgSubtle).Foreground(colorText)

	userMsg = lipgloss.NewStyle().
		BorderLeft(true).BorderStyle(lipgloss.ThickBorder()).
		BorderForeground(colorPrimary).PaddingLeft(1).MarginBottom(1)
	agentMsg = lipgloss.NewStyle().
			BorderLeft(true).BorderStyle(lipgloss.ThickBorder()).
			BorderForeground(colorSecondary).PaddingLeft(1).MarginBottom(1)

	toolBox = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).BorderForeground(colorSecondary).
		Padding(0, 1).MarginBottom(1)
	errorBox = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).BorderForeground(colorError).
			Padding(0, 1).MarginBottom(1)
	inputBox    = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(colorPrimary).Padding(0, 1)
	inputBoxDim = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(colorDim).Padding(0, 1)
	codeStyle   = lipgloss.NewStyle().Foreground(colorText).Background(colorBgCode).Padding(0, 1)
)

// List item for choice selection
type item string

func (i item) FilterValue() string { return "" }

type itemDelegate struct{}

func (d itemDelegate) Height() int                             { return 1 }
func (d itemDelegate) Spacing() int                            { return 0 }
func (d itemDelegate) Update(_ tea.Msg, _ *list.Model) tea.Cmd { return nil }
func (d itemDelegate) Render(w io.Writer, m list.Model, idx int, li list.Item) {
	s, ok := li.(item)
	if !ok {
		return
	}
	if idx == m.Index() {
		fmt.Fprint(w, primaryText.Render("> "+string(s)))
	} else {
		fmt.Fprint(w, mutedStyle.PaddingLeft(2).Render(string(s)))
	}
}

// TUI is the terminal user interface for the agent.
type TUI struct {
	program *tea.Program
	agent   *agent.Agent
}

func NewTUI(agent *agent.Agent) *TUI {
	// Mouse capture is intentionally NOT enabled: selecting text with the
	// mouse copies natively in every terminal (no modifier keys). Scrolling
	// still works because terminals translate the wheel to arrow keys in
	// alt-screen mode, which scroll the viewport (plus PgUp/PgDn).
	return &TUI{
		program: tea.NewProgram(newModel(agent), tea.WithAltScreen()),
		agent:   agent,
	}
}

func (u *TUI) Run(ctx context.Context) error {
	// Suppress stderr to prevent klog from breaking TUI
	if devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0); err == nil {
		orig := os.Stderr
		os.Stderr = devNull
		defer func() { os.Stderr = orig; devNull.Close() }()
	}
	klog.SetOutput(io.Discard)
	klog.LogToStderr(false)
	// The standard log package captured the original stderr at init time;
	// silence it as well so provider logs can't corrupt the UI.
	log.SetOutput(io.Discard)

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case msg, ok := <-u.agent.Output:
				if !ok {
					return
				}
				u.program.Send(msg)
			}
		}
	}()

	_, err := u.program.Run()
	return err
}

func (u *TUI) ClearScreen() {}

type sessionListMsg []api.SessionInfo

// sessionListErrMsg reports a failure to list sessions.
type sessionListErrMsg struct{ err error }

func (m *model) fetchSessions() tea.Msg {
	sessions, err := m.agent.ListSessions()
	if err != nil {
		return sessionListErrMsg{err: err}
	}
	return sessionListMsg(sessions)
}

type tickMsg time.Time

const (
	// maxInputHeight is the maximum number of lines the input box grows to
	// before it starts scrolling internally.
	maxInputHeight = 8
	// pasteCollapseLines is the minimum number of lines in a pasted chunk
	// for it to be collapsed into a compact placeholder token instead of
	// being inserted into the input verbatim.
	pasteCollapseLines = 3
)

// pastedBlock holds the contents of a large paste. Instead of flooding the
// input with the full text, a compact "[+N lines]" placeholder token is
// inserted at the cursor in the draft (like opencode), so users can keep
// typing before or after it. On submit the token expands back to the full
// content in place, preserving the order the user arranged. If the input is
// nearly full, the paste falls back to a bottom-attached chip (token == "")
// appended on submit.
type pastedBlock struct {
	id      int
	lines   int
	token   string // empty for chip-fallback blocks
	content string
}

const (
	// maxBrowserRows is the maximum number of sessions listed in the
	// session browser before the list starts scrolling.
	maxBrowserRows = 8
)

// sessionRenamedMsg reports the result of a rename attempt from the
// session browser.
type sessionRenamedMsg struct{ err error }

// browserStatusMsg is a transient status line shown in the session browser
// footer; isErr selects error vs info styling.
type browserStatusMsg struct {
	text  string
	isErr bool
}

// Render cache for markdown
type renderCache struct {
	mu       sync.RWMutex
	cache    map[string]string
	width    int
	renderer *glamour.TermRenderer
}

func newRenderCache() *renderCache {
	return &renderCache{cache: make(map[string]string)}
}

func (rc *renderCache) get(id string) (string, bool) {
	rc.mu.RLock()
	defer rc.mu.RUnlock()
	v, ok := rc.cache[id]
	return v, ok
}

func (rc *renderCache) set(id, content string) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.cache[id] = content
}

func (rc *renderCache) getRenderer(width int) (*glamour.TermRenderer, error) {
	rc.mu.Lock()
	defer rc.mu.Unlock()

	if rc.width != width {
		rc.cache = make(map[string]string)
		rc.width = width
		rc.renderer = nil
	}
	if rc.renderer == nil {
		r, err := glamour.NewTermRenderer(glamour.WithStylePath("dark"), glamour.WithWordWrap(width))
		if err != nil {
			return nil, err
		}
		rc.renderer = r
	}
	return rc.renderer, nil
}

// Model state
type model struct {
	agent       *agent.Agent
	viewport    viewport.Model
	input       textarea.Model
	inputHeight int
	pastes      []pastedBlock
	nextPasteID int
	// Input history navigation (previous submitted messages, oldest first).
	inputHistory []string
	historyIdx   int // -1 when not navigating
	historyDraft string
	// justSubmitted is set after a submit. Terminals that send CRLF for
	// Return deliver a phantom LF (KeyCtrlJ) right after the CR; it is
	// swallowed so it doesn't leave a stray newline in the next draft.
	justSubmitted bool
	spinner       spinner.Model
	list          list.Model
	cache         *renderCache
	messages      []*api.Message
	width         int
	height        int
	dirty         bool
	quitting      bool
	thinkStart    time.Time
	// Choice mode tracking
	inChoiceMode   bool
	choicePrompt   string
	choiceOptionID string // Track which choice request we initialized for
	choiceType     string // "confirm" or "session"
	sessionIDs     []string
	// Session browser state
	browserOpen     bool
	browserSessions []api.SessionInfo
	browserIndex    int
	browserStatus   browserStatusMsg // transient info/error shown in the browser footer
	renaming        bool
	renameInput     textinput.Model
	// Command palette state
	paletteOpen  bool
	paletteIndex int
	// Session rename mode: the input captures a new session name instead
	// of a chat message.
	sessionRename bool
}

func newModel(agent *agent.Agent) model {
	ti := textarea.New()
	ti.Placeholder = "Ask kubectl-ai anything..."
	ti.Focus()
	ti.Prompt = ""
	ti.CharLimit = 4096
	ti.ShowLineNumbers = false
	ti.SetWidth(80)
	// The textarea keeps a fixed internal height and we clip the rendered
	// view to the content's height (see viewInput). This avoids a class of
	// scroll bugs: the textarea scrolls its internal viewport to follow the
	// cursor, and shrinking/growing the height afterwards leaves the scroll
	// offset stale (e.g. the first line vanishing after Ctrl+J).
	ti.SetHeight(maxInputHeight)
	ti.FocusedStyle.Base = textStyle
	ti.FocusedStyle.Text = textStyle
	ti.FocusedStyle.Placeholder = dimStyle
	ti.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ti.Cursor.Style = primaryText
	// Enter submits the message (handled by us); Ctrl+J or Alt+Enter inserts
	// a newline.
	ti.KeyMap.InsertNewline = key.NewBinding(key.WithKeys("ctrl+j", "alt+enter"))

	sp := spinner.New()
	sp.Spinner = spinner.MiniDot
	sp.Style = primaryText

	l := list.New(nil, itemDelegate{}, 40, 5)
	l.SetShowStatusBar(false)
	l.SetFilteringEnabled(false)
	l.SetShowHelp(false)
	l.SetShowPagination(false)
	l.SetShowTitle(false)

	vp := viewport.New(80, 20)

	ri := textinput.New()
	ri.Prompt = ""
	ri.CharLimit = 128
	ri.TextStyle = textStyle
	ri.PlaceholderStyle = dimStyle
	ri.Cursor.Style = primaryText

	return model{
		agent:       agent,
		input:       ti,
		inputHeight: 1,
		historyIdx:  -1,
		viewport:    vp,
		spinner:     sp,
		list:        l,
		cache:       newRenderCache(),
		renameInput: ri,
		dirty:       true,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.spinner.Tick, m.tick())
}

func (m model) tick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.dirty = true
		m.resize()
		return m, nil

	case tea.KeyMsg:
		return m.handleKey(msg)

	case *api.Message:
		return m.handleAgentMsg(msg)

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case tickMsg:
		return m, m.tick()

	case sessionListMsg:
		// A refresh landing after the user closed the browser must not
		// reopen it.
		if m.browserOpen {
			m.openBrowser([]api.SessionInfo(msg))
			m.updateViewportHeight()
			return m, nil
		}
		if len(msg) == 0 {
			m.messages = append(m.messages, &api.Message{
				Source:    api.MessageSourceAgent,
				Type:      api.MessageTypeText,
				Payload:   "No sessions found.",
				Timestamp: time.Now(),
			})
			m.dirty = true
			m.refresh()
			m.viewport.GotoBottom()
			return m, nil
		}
		if m.inChoiceMode {
			// Don't stack the browser on top of an active choice picker.
			m.messages = append(m.messages, &api.Message{
				Source:    api.MessageSourceAgent,
				Type:      api.MessageTypeText,
				Payload:   "Finish the current prompt, then try 'sessions' again.",
				Timestamp: time.Now(),
			})
			m.dirty = true
			m.refresh()
			m.viewport.GotoBottom()
			return m, nil
		}
		m.openBrowser([]api.SessionInfo(msg))
		m.updateViewportHeight()
		return m, nil

	case sessionRenamedMsg:
		if msg.err != nil {
			if m.browserOpen {
				m.setBrowserStatus("Rename failed: "+msg.err.Error(), true)
				return m, nil
			}
			m.messages = append(m.messages, &api.Message{
				Source:    api.MessageSourceAgent,
				Type:      api.MessageTypeError,
				Payload:   "Rename failed: " + msg.err.Error(),
				Timestamp: time.Now(),
			})
			m.dirty = true
			m.refresh()
			m.viewport.GotoBottom()
			return m, nil
		}
		if m.browserOpen {
			m.setBrowserStatus("Renamed ✓", false)
			// Refresh the browser contents, keeping the browser open.
			return m, m.fetchSessions
		}
		return m, nil

	case sessionListErrMsg:
		if m.browserOpen {
			m.setBrowserStatus("Failed to list sessions: "+msg.err.Error(), true)
			return m, nil
		}
		m.messages = append(m.messages, &api.Message{
			Source:    api.MessageSourceAgent,
			Type:      api.MessageTypeError,
			Payload:   "Failed to list sessions: " + msg.err.Error(),
			Timestamp: time.Now(),
		})
		m.dirty = true
		m.refresh()
		m.viewport.GotoBottom()
		return m, nil
	}
	return m, nil
}

// setBrowserStatus shows a transient status line in the browser footer.
func (m *model) setBrowserStatus(text string, isErr bool) {
	m.browserStatus = browserStatusMsg{text: text, isErr: isErr}
}

// appendLocalMessage adds a transient agent-styled message to the
// transcript (for UI-originated events; it is not persisted).
func (m *model) appendLocalMessage(text string) {
	m.messages = append(m.messages, &api.Message{
		Source:    api.MessageSourceAgent,
		Type:      api.MessageTypeText,
		Payload:   text,
		Timestamp: time.Now(),
	})
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()
}

// openBrowser opens the session browser with the given sessions, selecting
// the current session when possible and preserving the selection across
// refreshes by session ID. When the browser is already open (a refresh),
// the footer status and rename state are preserved.
func (m *model) openBrowser(sessions []api.SessionInfo) {
	selectedID := ""
	if m.browserOpen && m.browserIndex >= 0 && m.browserIndex < len(m.browserSessions) {
		selectedID = m.browserSessions[m.browserIndex].ID
	} else if s := m.agent.GetSession(); s != nil {
		selectedID = s.ID
	}

	m.browserSessions = sessions
	m.browserIndex = 0
	for i, s := range sessions {
		if s.ID == selectedID {
			m.browserIndex = i
			break
		}
	}
	if !m.browserOpen {
		m.browserStatus = browserStatusMsg{}
		m.renaming = false
	}
	m.browserOpen = true
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()
}

// closeBrowser closes the session browser.
func (m *model) closeBrowser() {
	m.browserOpen = false
	m.renaming = false
	m.browserStatus = browserStatusMsg{}
	m.updateViewportHeight()
}

func (m *model) resize() {
	m.viewport.Width = m.width - 2
	// The textarea must fit the input box's content area exactly: box border
	// (2) + box padding (2) + outer padding (2) = 6, plus 2 cells of slack
	// so rendered lines never reach the terminal's last column and wrap.
	m.input.SetWidth(max(m.width-8, 20))
	m.list.SetWidth(m.width - 4)
	m.renameInput.Width = max(m.width-30, 10)
	m.updateViewportHeight()
	m.refresh()
	m.viewport.GotoBottom()
}

func (m *model) updateViewportHeight() {
	// Layout: status(1) + 2 dividers(2) + input block + help(1) + bottom padding(1)
	contentH := m.height - (m.inputBlockHeight() + 5)
	if m.browserOpen && m.width > 0 {
		contentH -= lipgloss.Height(m.viewSessionBrowser())
	}
	if m.paletteOpen && m.width > 0 {
		contentH -= lipgloss.Height(m.viewPalette())
	}

	contentH = max(contentH, 5)
	m.viewport.Height = contentH
}

// inputBlockHeight returns the number of terminal rows the input area
// occupies. It must match what viewInput renders.
func (m *model) inputBlockHeight() int {
	if m.inChoiceMode || m.agentState() == api.AgentStateRunning || m.agentState() == api.AgentStateInitializing {
		return 3 // one-line hint/spinner box + 2 borders
	}
	h := m.inputHeight + 2 // +2 for the box borders
	if m.sessionRename {
		h++ // the "Rename session:" label line
	}
	for _, p := range m.pastes {
		if p.token == "" {
			h++ // chip-fallback pastes line
			break
		}
	}
	return h
}

func (m *model) agentState() api.AgentState {
	if m.agent == nil {
		return api.AgentStateIdle
	}
	return m.agent.AgentState()
}

// syncInputHeight recomputes the input box height from its (soft-wrapped)
// content, capped at maxInputHeight lines, and adjusts the viewport height
// accordingly. The textarea itself keeps a fixed internal height; we only
// clip how many of its rendered lines we show (see viewInput).
func (m *model) syncInputHeight() {
	h := min(visualLines(m.input.Value(), m.input.Width()), maxInputHeight)
	if h == m.inputHeight {
		return
	}
	m.inputHeight = h
	m.updateViewportHeight()
}

// visualLines estimates how many terminal rows s occupies when soft-wrapped
// at the given width.
func visualLines(s string, width int) int {
	if width <= 0 {
		return 1
	}
	lines := 0
	for _, l := range strings.Split(s, "\n") {
		if w := lipgloss.Width(l); w == 0 {
			lines++
		} else {
			lines += (w + width - 1) / width
		}
	}
	return max(lines, 1)
}

func (m *model) navigateList(keyType tea.KeyType) tea.Cmd {
	var cmd tea.Cmd
	m.list, cmd = m.list.Update(tea.KeyMsg{Type: keyType})
	m.dirty = true
	m.refresh()
	return cmd
}

func (m *model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.justSubmitted {
		m.justSubmitted = false
		if msg.Type == tea.KeyCtrlJ {
			// Terminals that send CRLF for Return: swallow the phantom LF
			// that follows the CR, so it doesn't leave a stray newline in
			// the next draft.
			return m, nil
		}
	}

	// While the session browser is open it captures all keys except quit
	// (including pastes, which are routed by handleBrowserKey).
	if m.browserOpen && msg.Type != tea.KeyCtrlC && msg.Type != tea.KeyCtrlD {
		return m.handleBrowserKey(msg)
	}

	// While the command palette is open it captures all keys except quit.
	if m.paletteOpen && msg.Type != tea.KeyCtrlC && msg.Type != tea.KeyCtrlD {
		return m.handlePaletteKey(msg)
	}

	// Bracketed paste arrives as a single key message with Paste set.
	// Handle it before anything else so pasted text never triggers shortcuts.
	if msg.Paste {
		return m.handlePaste(msg)
	}

	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyCtrlD:
		m.quitting = true
		return m, tea.Quit
	case tea.KeyEsc:
		return m.handleEsc()
	case tea.KeyCtrlP:
		m.openPalette()
		return m, nil
	case tea.KeyShiftTab:
		// Toggle auto-accept mode (skip permission prompts), like opencode
		// and Claude Code.
		return m.toggleAutoMode()
	case tea.KeyEnter:
		if msg.Alt {
			// Alt+Enter inserts a newline (bound in the textarea keymap).
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			m.syncInputHeight()
			return m, cmd
		}
		return m.handleEnter()
	case tea.KeyUp:
		if m.inChoiceMode {
			return m, m.navigateList(tea.KeyUp)
		}
		// Within a multi-line draft, Up moves the cursor until the first
		// line; from there (and for single-line drafts) it recalls older
		// input history, like opencode and Claude Code. Transcript
		// scrolling lives on PgUp/PgDn.
		if m.input.LineCount() > 1 && m.input.Line() > 0 {
			m.input.CursorUp()
			return m, nil
		}
		m.historyPrev()
	case tea.KeyDown:
		if m.inChoiceMode {
			return m, m.navigateList(tea.KeyDown)
		}
		if m.input.LineCount() > 1 && m.input.Line() < m.input.LineCount()-1 {
			m.input.CursorDown()
			return m, nil
		}
		m.historyNext()
	case tea.KeyCtrlY:
		return m.copyLastResponse()
	case tea.KeyPgUp:
		m.viewport.ScrollUp(m.viewport.Height / 2)
	case tea.KeyPgDown:
		m.viewport.ScrollDown(m.viewport.Height / 2)
	case tea.KeyBackspace:
		if m.inChoiceMode {
			return m, nil
		}
		// Backspace right after an inline paste placeholder removes the
		// whole placeholder and its paste (like deleting an attachment in
		// opencode).
		if m.input.Line() == m.input.LineCount()-1 {
			if token, ok := m.tokenAtEndOfInput(); ok {
				m.removeToken(token)
				m.syncInputHeight()
				return m, nil
			}
		}
		// With an empty input, Backspace detaches the most recent
		// chip-fallback paste.
		if m.input.Value() == "" && len(m.pastes) > 0 {
			m.pastes = m.pastes[:len(m.pastes)-1]
			m.syncInputHeight()
			return m, nil
		}
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		m.syncInputHeight()
		return m, cmd
	default:
		if m.inChoiceMode {
			switch msg.String() {
			case "j":
				return m, m.navigateList(tea.KeyDown)
			case "k":
				return m, m.navigateList(tea.KeyUp)
			}
			// Don't let keystrokes accumulate invisibly in the input
			// while a choice picker is active.
			return m, nil
		}
		switch msg.String() {
		case "alt+p":
			// Alt+P recalls older input history (emacs M-p style; works on
			// macOS terminals where Alt+arrows aren't delivered).
			m.historyPrev()
			return m, nil
		case "alt+n":
			// Alt+N recalls newer input history.
			m.historyNext()
			return m, nil
		}
		// Default: send to text input
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		m.syncInputHeight()
		return m, cmd
	}
	return m, nil
}

// rebuildInputHistory refreshes the input history from the current session's
// user messages (oldest first, consecutive duplicates removed).
func (m *model) rebuildInputHistory() {
	m.inputHistory = m.inputHistory[:0]
	for _, msg := range m.messages {
		if msg.Source != api.MessageSourceUser || msg.Type != api.MessageTypeText {
			continue
		}
		p, ok := msg.Payload.(string)
		if !ok || strings.TrimSpace(p) == "" {
			continue
		}
		if n := len(m.inputHistory); n > 0 && m.inputHistory[n-1] == p {
			continue
		}
		m.inputHistory = append(m.inputHistory, p)
	}
}

// historyPrev recalls the previous submitted message into the input, saving
// the current draft when navigation starts.
func (m *model) historyPrev() {
	if m.historyIdx == -1 {
		m.rebuildInputHistory()
		if len(m.inputHistory) == 0 {
			return
		}
		m.historyDraft = m.input.Value()
		m.historyIdx = len(m.inputHistory) - 1
	} else if m.historyIdx > 0 {
		m.historyIdx--
	} else {
		return // already at the oldest entry
	}
	m.input.SetValue(m.inputHistory[m.historyIdx])
	m.syncInputHeight()
}

// historyNext moves towards newer history entries, finally restoring the
// draft that was being edited when navigation started.
func (m *model) historyNext() {
	if m.historyIdx == -1 {
		return
	}
	if m.historyIdx < len(m.inputHistory)-1 {
		m.historyIdx++
		m.input.SetValue(m.inputHistory[m.historyIdx])
	} else {
		m.historyIdx = -1
		m.input.SetValue(m.historyDraft)
		m.historyDraft = ""
	}
	m.syncInputHeight()
}

// copyLastResponse copies the most recent agent/model text message to the
// system clipboard, and confirms in the transcript. On macOS it uses pbcopy
// (always works, no terminal support needed); elsewhere it falls back to the
// OSC 52 escape sequence (iTerm2, kitty, WezTerm, foot, Windows Terminal).
func (m *model) copyLastResponse() (tea.Model, tea.Cmd) {
	payload, ok := m.lastCopyableText()
	if !ok {
		m.appendLocalMessage("Nothing to copy yet.")
		return m, nil
	}
	m.appendLocalMessage("📋 Copied last response to clipboard.")
	return m, func() tea.Msg {
		_ = copyToClipboard(payload)
		return nil
	}
}

// copyToClipboard puts s on the system clipboard, preferring pbcopy (macOS)
// and falling back to OSC 52.
func copyToClipboard(s string) error {
	if path, err := exec.LookPath("pbcopy"); err == nil {
		cmd := exec.Command(path)
		cmd.Stdin = strings.NewReader(s)
		if err := cmd.Run(); err == nil {
			return nil
		}
	}
	// Fallback: OSC 52 (zero visual output, so the renderer's cursor
	// accounting is unaffected; tea.Println is a no-op in alt-screen).
	osc52Write(s)
	return nil
}

// osc52Write emits the OSC 52 clipboard sequence to the terminal.
func osc52Write(s string) {
	fmt.Fprintf(os.Stdout, "\x1b]52;c;%s\a", base64.StdEncoding.EncodeToString([]byte(s)))
}

// lastCopyableText returns the payload of the most recent agent/model text
// message, suitable for copying.
func (m *model) lastCopyableText() (string, bool) {
	for i := len(m.messages) - 1; i >= 0; i-- {
		msg := m.messages[i]
		if (msg.Source == api.MessageSourceModel || msg.Source == api.MessageSourceAgent) &&
			msg.Type == api.MessageTypeText {
			if payload, ok := msg.Payload.(string); ok && strings.TrimSpace(payload) != "" {
				return payload, true
			}
		}
	}
	return "", false
}

// normalizePasteContent normalizes line endings in pasted runes and drops a
// single trailing newline (almost always a copy artifact, so pasting one
// line doesn't grow the input).
func normalizePasteContent(runes []rune) string {
	content := strings.ReplaceAll(string(runes), "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	return strings.TrimSuffix(content, "\n")
}

// handleEsc routes the Esc key by priority: close panels, decline/stop a
// pending prompt, interrupt a running agent, then fall back to clearing
// the input.
func (m *model) handleEsc() (tea.Model, tea.Cmd) {
	// Cancel an in-progress rename first, restoring the draft.
	if m.sessionRename {
		m.exitSessionRename()
		return m, nil
	}

	// Decline/stop a pending permission prompt or session picker.
	if m.inChoiceMode {
		if m.choiceType == "session" {
			m.inChoiceMode = false
			m.choicePrompt = ""
			m.choiceOptionID = ""
			return m, func() tea.Msg {
				m.agent.Input <- &api.SessionPickerResponse{Cancelled: true}
				return nil
			}
		}
		// Permission prompt: decline and interrupt the run.
		m.inChoiceMode = false
		m.choicePrompt = ""
		m.choiceOptionID = ""
		return m, func() tea.Msg {
			m.agent.Input <- &api.UserChoiceResponse{Choice: 3} // 3 == No/decline
			m.agent.CancelRun()
			return nil
		}
	}

	// Interrupt a running agent.
	if m.agent != nil {
		if s := m.agentState(); s == api.AgentStateRunning || s == api.AgentStateInitializing {
			return m, func() tea.Msg {
				m.agent.CancelRun()
				return nil
			}
		}
	}

	// Idle: clear the input and attachments as before.
	m.input.Reset()
	m.pastes = nil
	m.historyIdx = -1
	m.historyDraft = ""
	m.syncInputHeight()
	return m, nil
}

// paletteItem is a single action in the command palette.
type paletteItem struct {
	label string
	hint  string
	run   func(m *model) (tea.Model, tea.Cmd)
}

// paletteItems returns the current palette actions (some are dynamic).
func (m *model) paletteItems() []paletteItem {
	items := []paletteItem{
		{label: "Switch model", hint: "/model", run: func(m *model) (tea.Model, tea.Cmd) {
			return m, m.sendQuery("/model")
		}},
		{label: "Browse sessions", hint: "/sessions", run: func(m *model) (tea.Model, tea.Cmd) {
			return m, m.fetchSessions
		}},
		{label: "Rename session", hint: "/rename", run: func(m *model) (tea.Model, tea.Cmd) {
			m.enterSessionRename()
			return m, nil
		}},
		{label: "New session", hint: "", run: func(m *model) (tea.Model, tea.Cmd) {
			return m, m.sendMsg(&api.NewSessionRequest{})
		}},
	}
	if m.agent != nil && m.agent.SkipPermissionsEnabled() {
		items = append(items, paletteItem{label: "Auto mode: on (shift+tab to disable)", hint: "shift+tab", run: (*model).toggleAutoMode})
	} else {
		items = append(items, paletteItem{label: "Auto mode: off (shift+tab to enable)", hint: "shift+tab", run: (*model).toggleAutoMode})
	}
	items = append(items,
		paletteItem{label: "Copy last response", hint: "ctrl+y", run: (*model).copyLastResponse},
	)
	if s := m.agentState(); s == api.AgentStateRunning || s == api.AgentStateInitializing || m.inChoiceMode {
		items = append(items, paletteItem{label: "Interrupt agent", hint: "esc", run: (*model).interruptRun})
	}
	items = append(items,
		paletteItem{label: "Clear conversation", hint: "/clear", run: func(m *model) (tea.Model, tea.Cmd) {
			return m, m.sendQuery("/clear")
		}},
		paletteItem{label: "Quit", hint: "/exit", run: func(m *model) (tea.Model, tea.Cmd) {
			return m, m.sendQuery("/exit")
		}},
	)
	return items
}

func (m *model) openPalette() {
	if m.browserOpen {
		m.closeBrowser()
	}
	m.paletteOpen = true
	m.paletteIndex = 0
	m.updateViewportHeight()
}

func (m *model) closePalette() {
	m.paletteOpen = false
	m.updateViewportHeight()
}

func (m *model) movePaletteSelection(delta int) {
	n := len(m.paletteItems())
	if n == 0 {
		return
	}
	m.paletteIndex = (m.paletteIndex + delta + n) % n
}

func (m *model) handlePaletteKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEsc:
		m.closePalette()
		return m, nil
	case tea.KeyUp:
		m.movePaletteSelection(-1)
		return m, nil
	case tea.KeyDown:
		m.movePaletteSelection(1)
		return m, nil
	case tea.KeyEnter:
		items := m.paletteItems()
		if m.paletteIndex < 0 || m.paletteIndex >= len(items) {
			return m, nil
		}
		item := items[m.paletteIndex]
		m.closePalette()
		return item.run(m)
	}
	switch msg.String() {
	case "j":
		m.movePaletteSelection(1)
	case "k":
		m.movePaletteSelection(-1)
	}
	return m, nil
}

// sendQuery sends a query (e.g. a slash command) to the agent, exactly as
// if the user had typed and submitted it.
func (m *model) sendQuery(query string) tea.Cmd {
	return func() tea.Msg {
		m.agent.Input <- &api.UserInputResponse{Query: query}
		return nil
	}
}

// sendMsg sends an arbitrary input message to the agent.
func (m *model) sendMsg(v any) tea.Cmd {
	return func() tea.Msg {
		m.agent.Input <- v
		return nil
	}
}

// toggleAutoMode flips auto-accept mode with a transcript confirmation.
func (m *model) toggleAutoMode() (tea.Model, tea.Cmd) {
	if m.agent == nil {
		return m, nil
	}
	if enabled := m.agent.ToggleSkipPermissions(); enabled {
		m.appendLocalMessage("⚡ Auto mode on — the agent will run tools without asking for permission.")
	} else {
		m.appendLocalMessage("Auto mode off — you'll be asked to approve modifying commands.")
	}
	return m, nil
}

// interruptRun cancels the current agentic run.
func (m *model) interruptRun() (tea.Model, tea.Cmd) {
	if m.agent == nil || !m.agent.CancelRun() {
		m.appendLocalMessage("Nothing running to interrupt.")
		return m, nil
	}
	return m, nil
}

// tokenAtEndOfInput returns the paste placeholder token (if any) that the
// current input value ends with.
func (m *model) tokenAtEndOfInput() (string, bool) {
	value := m.input.Value()
	for i := len(m.pastes) - 1; i >= 0; i-- {
		if t := m.pastes[i].token; t != "" && strings.HasSuffix(value, t) {
			return t, true
		}
	}
	return "", false
}

// removeToken removes the last paste block with the given placeholder token
// and deletes the token text (and its separator space) from the input.
func (m *model) removeToken(token string) {
	v := strings.TrimSuffix(m.input.Value(), token)
	v = strings.TrimSuffix(v, " ") // the separator added at insert time
	m.input.SetValue(v)
	for i := len(m.pastes) - 1; i >= 0; i-- {
		if m.pastes[i].token == token {
			m.pastes = append(m.pastes[:i], m.pastes[i+1:]...)
			return
		}
	}
}

// expandPasteToken replaces the first occurrence of token in value with
// content, separating the pasted content from adjacent typed text with a
// blank line when they would otherwise be glued together.
func expandPasteToken(value, token, content string) string {
	idx := strings.Index(value, token)
	if idx < 0 {
		return value
	}
	end := idx + len(token)

	pre := ""
	if idx > 0 {
		if value[idx-1] == ' ' || value[idx-1] == '\t' {
			idx-- // absorb the space into the separator
			pre = "\n\n"
		} else if value[idx-1] != '\n' {
			pre = "\n\n"
		}
	}
	post := ""
	if end < len(value) {
		if value[end] == ' ' || value[end] == '\t' {
			end++ // absorb the space into the separator
			post = "\n\n"
		} else if value[end] != '\n' {
			post = "\n\n"
		}
	}
	return value[:idx] + pre + content + post + value[end:]
}

// handlePaste inserts pasted text into the input. Large multi-line pastes
// are collapsed into a compact "[+N lines]" placeholder token inserted at
// the cursor (like opencode), so the user can keep typing before or after
// the paste; the token expands back to the full content on submit. If the
// token doesn't fit (CharLimit), the paste attaches as a bottom chip
// instead. Short pastes are inserted verbatim.
func (m *model) handlePaste(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.inChoiceMode {
		return m, nil
	}

	content := normalizePasteContent(msg.Runes)
	if content == "" {
		return m, nil
	}

	if n := strings.Count(content, "\n") + 1; n >= pasteCollapseLines {
		m.nextPasteID++
		token := fmt.Sprintf("[+%d lines]", n)
		block := pastedBlock{id: m.nextPasteID, lines: n, content: content}
		// Insert the placeholder inline at the cursor when it fits;
		// otherwise attach it as a bottom chip.
		if m.input.CharLimit == 0 || m.input.CharLimit-m.input.Length() >= len(token) {
			block.token = token
			// Keep a space between the token and preceding text.
			if v := m.input.Value(); v != "" && !strings.HasSuffix(v, " ") && !strings.HasSuffix(v, "\n") {
				m.input.InsertString(" ")
			}
			m.input.InsertString(token)
		}
		m.pastes = append(m.pastes, block)
	} else {
		m.input.InsertString(content)
	}
	m.syncInputHeight()
	return m, nil
}

// handleBrowserKey routes keys while the session browser is open.
func (m *model) handleBrowserKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Scrolling the transcript works even with the browser open, and must
	// not dismiss a status line.
	if msg.Type == tea.KeyPgUp {
		m.viewport.ScrollUp(m.viewport.Height / 2)
		return m, nil
	}
	if msg.Type == tea.KeyPgDown {
		m.viewport.ScrollDown(m.viewport.Height / 2)
		return m, nil
	}

	m.browserStatus = browserStatusMsg{}

	// Pastes land in the rename field while renaming; otherwise they would
	// silently accumulate in the hidden main input.
	if msg.Paste {
		if m.renaming {
			// textinput has no insert-at-cursor API; appending is fine for a
			// rename field.
			m.renameInput.SetValue(m.renameInput.Value() + normalizePasteContent(msg.Runes))
			m.renameInput.CursorEnd()
			return m, nil
		}
		m.setBrowserStatus("Close the browser (esc) to paste into the input", false)
		return m, nil
	}

	// Rename mode captures all input for the rename field.
	if m.renaming {
		switch msg.Type {
		case tea.KeyEnter:
			newName := strings.TrimSpace(m.renameInput.Value())
			m.renaming = false
			m.renameInput.Blur()
			if newName == "" || m.browserIndex >= len(m.browserSessions) {
				return m, nil
			}
			sessionID := m.browserSessions[m.browserIndex].ID
			return m, func() tea.Msg {
				return sessionRenamedMsg{err: m.agent.RenameSession(sessionID, newName)}
			}
		case tea.KeyEsc:
			m.renaming = false
			m.renameInput.Blur()
			return m, nil
		default:
			var cmd tea.Cmd
			m.renameInput, cmd = m.renameInput.Update(msg)
			return m, cmd
		}
	}

	switch msg.Type {
	case tea.KeyEsc:
		m.closeBrowser()
		return m, nil
	case tea.KeyUp:
		m.moveBrowserSelection(-1)
		return m, nil
	case tea.KeyDown:
		m.moveBrowserSelection(1)
		return m, nil
	case tea.KeyEnter:
		if m.browserIndex >= len(m.browserSessions) {
			return m, nil
		}
		selectedID := m.browserSessions[m.browserIndex].ID
		// The agent only reads its input channel when idle; while it is
		// running, the switch queues up. Say so instead of pretending the
		// switch already happened.
		if s := m.agentState(); s == api.AgentStateRunning || s == api.AgentStateInitializing {
			m.setBrowserStatus("Agent is busy — it will switch when done", false)
		} else {
			m.closeBrowser()
		}
		return m, func() tea.Msg {
			m.agent.Input <- &api.SessionPickerResponse{SessionID: selectedID}
			return nil
		}
	}

	switch msg.String() {
	case "k":
		m.moveBrowserSelection(-1)
	case "j":
		m.moveBrowserSelection(1)
	case "r":
		if m.browserIndex < len(m.browserSessions) {
			m.renaming = true
			s := m.browserSessions[m.browserIndex]
			m.renameInput.SetValue(s.Name)
			m.renameInput.Placeholder = s.ID
			m.renameInput.CursorEnd()
			return m, m.renameInput.Focus()
		}
	case "ctrl+n":
		m.closeBrowser()
		return m, func() tea.Msg {
			m.agent.Input <- &api.NewSessionRequest{}
			return nil
		}
	}
	return m, nil
}

func (m *model) moveBrowserSelection(delta int) {
	if len(m.browserSessions) == 0 {
		return
	}
	m.browserIndex = (m.browserIndex + delta + len(m.browserSessions)) % len(m.browserSessions)
}

// enterSessionRename switches the input into rename mode for the current
// session, prefilling the current name.
func (m *model) enterSessionRename() {
	m.sessionRename = true
	name := ""
	if s := m.agent.GetSession(); s != nil {
		name = s.Name
	}
	m.input.SetValue(name)
	m.input.Placeholder = "New session name..."
	m.input.CursorEnd()
	m.syncInputHeight()
}

// exitSessionRename leaves rename mode, clearing the input.
func (m *model) exitSessionRename() {
	m.sessionRename = false
	m.input.Placeholder = "Ask kubectl-ai anything..."
	m.input.Reset()
	m.syncInputHeight()
}

// submitSessionRename applies the typed name to the current session.
func (m *model) submitSessionRename() (tea.Model, tea.Cmd) {
	name := strings.TrimSpace(m.input.Value())
	if name == "" {
		return m, nil
	}
	m.exitSessionRename()
	return m.renameCurrentSession(name)
}

// renameCurrentSession renames the current session immediately and confirms.
func (m *model) renameCurrentSession(name string) (tea.Model, tea.Cmd) {
	if m.agent == nil {
		return m, nil
	}
	s := m.agent.GetSession()
	if err := m.agent.RenameSession(s.ID, name); err != nil {
		m.appendLocalMessage("Rename failed: " + err.Error())
		return m, nil
	}
	m.appendLocalMessage(fmt.Sprintf("Renamed session to %q.", name))
	return m, nil
}

func (m *model) handleEnter() (tea.Model, tea.Cmd) {
	// Rename mode captures the input for the new session name.
	if m.sessionRename {
		return m.submitSessionRename()
	}

	// Handle choice selection
	if m.inChoiceMode {
		if _, ok := m.list.SelectedItem().(item); ok {
			if m.choiceType == "session" {
				idx := m.list.Index()
				if idx >= 0 && idx < len(m.sessionIDs) {
					selectedID := m.sessionIDs[idx]
					m.inChoiceMode = false
					m.choicePrompt = ""
					m.choiceOptionID = ""
					// Don't reset choiceType/sessionIDs yet or it might race, but actually we are done.
					m.dirty = true
					m.refresh()
					return m, func() tea.Msg {
						m.agent.Input <- &api.SessionPickerResponse{SessionID: selectedID}
						return nil
					}
				}
			} else {
				choice := m.list.Index() + 1
				m.inChoiceMode = false
				m.choicePrompt = ""
				m.choiceOptionID = ""
				m.dirty = true
				m.refresh()
				return m, func() tea.Msg {
					m.agent.Input <- &api.UserChoiceResponse{Choice: choice}
					return nil
				}
			}
		}
		return m, nil
	}

	value := m.input.Value()
	// Expand inline paste placeholders back to their full content, in
	// insertion order (so typed text and pastes keep the order the user
	// arranged). Placeholders the user deleted simply drop their paste.
	// Chip-fallback blocks (no token) are appended at the end.
	var tail []string
	for _, p := range m.pastes {
		if p.token != "" {
			value = expandPasteToken(value, p.token, p.content)
		} else {
			tail = append(tail, p.content)
		}
	}
	if len(tail) > 0 {
		var parts []string
		if strings.TrimSpace(value) != "" {
			parts = append(parts, value)
		}
		parts = append(parts, tail...)
		value = strings.Join(parts, "\n\n")
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return m, nil
	}

	// Add user message
	m.messages = append(m.messages, &api.Message{
		Source:    api.MessageSourceUser,
		Type:      api.MessageTypeText,
		Payload:   value,
		Timestamp: time.Now(),
	})
	m.input.Reset()
	m.pastes = nil
	m.historyIdx = -1
	m.historyDraft = ""
	m.justSubmitted = true
	m.syncInputHeight()
	m.dirty = true
	m.refresh()
	m.viewport.GotoBottom()

	// Intercept the sessions command (handled locally, not sent to the LLM)
	if v := strings.ToLower(value); v == "/sessions" || v == "/session" || v == "sessions" {
		return m, m.fetchSessions
	}

	// Intercept the rename command (handled locally and instantly).
	if v := strings.ToLower(value); v == "/rename" {
		m.enterSessionRename()
		return m, nil
	}
	if strings.HasPrefix(strings.ToLower(value), "/rename ") {
		name := strings.TrimSpace(value[len("/rename "):])
		return m.renameCurrentSession(name)
	}

	m.thinkStart = time.Now()

	return m, func() tea.Msg {
		m.agent.Input <- &api.UserInputResponse{Query: value}
		return nil
	}
}

func (m *model) handleAgentMsg(msg *api.Message) (tea.Model, tea.Cmd) {
	session := m.agent.GetSession()
	m.messages = session.AllMessages()
	m.dirty = true

	// Check if we're entering choice mode - use the incoming message directly
	// to avoid race conditions where the message isn't yet in AllMessages()
	if msg.Type == api.MessageTypeUserChoiceRequest {
		// A permission prompt supersedes the session browser.
		if m.browserOpen {
			m.closeBrowser()
		}
		if req, ok := msg.Payload.(*api.UserChoiceRequest); ok {
			items := make([]list.Item, len(req.Options))
			for i, opt := range req.Options {
				items[i] = item(opt.Label)
			}
			m.list.SetItems(items)
			m.list.Select(0)
			m.inChoiceMode = true
			m.choicePrompt = req.Prompt
			m.choiceOptionID = msg.ID
			m.choiceType = "confirm"
		}
	} else if msg.Type == api.MessageTypeSessionPickerRequest {
		if req, ok := msg.Payload.(*api.SessionPickerRequest); ok {
			items := make([]list.Item, len(req.Sessions))
			ids := make([]string, len(req.Sessions))
			for i, s := range req.Sessions {
				label := fmt.Sprintf("%s (%s) • %d msgs", s.ID, s.ModelID, s.MessageCount)
				if s.Name != "" {
					label = fmt.Sprintf("%s (%s) • %s • %d msgs", s.Name, s.ModelID, s.ID, s.MessageCount)
				}
				items[i] = item(label)
				ids[i] = s.ID
			}
			m.list.SetItems(items)
			m.list.Select(0)
			m.inChoiceMode = true
			m.choicePrompt = "Select a session to resume"
			m.choiceOptionID = msg.ID
			m.choiceType = "session"
			m.sessionIDs = ids
		}
	} else if session.AgentState == api.AgentStateDone || session.AgentState == api.AgentStateExited {
		// Clear choice mode if we're done or exited
		m.inChoiceMode = false
		m.choicePrompt = ""
		m.choiceOptionID = ""
	}

	// Only follow the transcript to the bottom if the user is already at the
	// bottom; yanking the viewport down while the user scrolled up (e.g. to
	// select text for copying) is hostile.
	atBottom := m.viewport.AtBottom()
	m.refresh()
	if atBottom {
		m.viewport.GotoBottom()
	}

	if session.AgentState == api.AgentStateRunning || session.AgentState == api.AgentStateInitializing {
		return m, m.spinner.Tick
	}
	return m, nil
}

func (m *model) refresh() {
	if !m.dirty {
		return
	}
	m.viewport.SetContent(m.renderMessages())
	m.dirty = false
}

func (m model) renderMessages() string {
	var sb strings.Builder

	if len(m.messages) == 0 {
		sb.WriteString(fmt.Sprintf("\n%s\n\n%s\n%s\n%s\n%s\n",
			primaryText.Render(logo),
			mutedStyle.PaddingLeft(1).Render("Your AI-powered Kubernetes assistant"),
			dimStyle.PaddingLeft(1).Render("Type a message to get started"),
			dimStyle.PaddingLeft(1).Render("Type /sessions to browse and resume past sessions"),
			dimStyle.PaddingLeft(1).Render("Drag-select with your mouse to copy, or press Ctrl+Y to copy the last reply")))
	} else {
		width := min(m.viewport.Width-6, 90)
		if width < 40 {
			width = 40
		}

		renderer, err := m.cache.getRenderer(width)
		if err != nil {
			return "Error rendering messages"
		}

		for _, msg := range m.messages {
			if s := m.renderMessage(msg, renderer, width); s != "" {
				sb.WriteString(s)
			}
		}
	}

	// Render choice picker inline at the end of messages
	if m.inChoiceMode {
		sb.WriteString("\n")
		sb.WriteString(warnText.Render("? " + m.choicePrompt))
		sb.WriteString("\n\n")
		sb.WriteString(m.list.View())
		sb.WriteString("\n")
	}

	return sb.String()
}

func (m model) renderMessage(msg *api.Message, r *glamour.TermRenderer, w int) string {
	// Skip certain message types
	if msg.Type == api.MessageTypeUserInputRequest {
		if p, ok := msg.Payload.(string); ok && p == ">>>" {
			return ""
		}
	}
	if msg.Type == api.MessageTypeToolCallResponse {
		return ""
	}
	// Skip choice requests - they're rendered in the input area instead
	if msg.Type == api.MessageTypeUserChoiceRequest || msg.Type == api.MessageTypeSessionPickerRequest {
		return ""
	}

	// Check cache (except tool calls which show status)
	if msg.ID != "" && msg.Type != api.MessageTypeToolCallRequest {
		if cached, ok := m.cache.get(msg.ID); ok {
			return cached
		}
	}

	var result string
	switch msg.Type {
	case api.MessageTypeToolCallRequest:
		result = m.renderToolCall(msg, w)
	case api.MessageTypeError:
		result = m.renderError(msg, w)
	default:
		result = m.renderTextMsg(msg, r, w)
	}

	// Cache result
	if msg.ID != "" && result != "" && msg.Type != api.MessageTypeToolCallRequest {
		m.cache.set(msg.ID, result)
	}
	return result
}

func (m model) renderTextMsg(msg *api.Message, r *glamour.TermRenderer, w int) string {
	payload, ok := msg.Payload.(string)
	if !ok {
		return ""
	}

	ts := ""
	if !msg.Timestamp.IsZero() {
		ts = dimStyle.Italic(true).Render(" " + msg.Timestamp.Format("15:04"))
	}

	switch msg.Source {
	case api.MessageSourceUser:
		label := primaryText.Render("You") + ts
		content := textStyle.Width(w).Render(payload)
		return userMsg.Width(w+2).Render(label+"\n"+content) + "\n"
	case api.MessageSourceModel, api.MessageSourceAgent:
		label := successText.Render("kubectl-ai") + ts
		rendered, _ := r.Render(payload)
		return agentMsg.Width(w+2).Render(label+"\n"+strings.TrimSpace(rendered)) + "\n"
	}
	return ""
}

func (m model) renderToolCall(msg *api.Message, w int) string {
	payload, ok := msg.Payload.(string)
	if !ok {
		return ""
	}
	content := successText.Render("⚡ Running") + "\n" + codeStyle.Render(payload)
	return toolBox.Width(w).Render(content) + "\n"
}

func (m model) renderError(msg *api.Message, w int) string {
	payload, ok := msg.Payload.(string)
	if !ok {
		return ""
	}
	content := errorText.Render("✗ Error") + "\n" + errorText.Render(payload)
	return errorBox.Width(w).Render(content) + "\n"
}

func (m model) View() string {
	if m.quitting {
		return mutedStyle.Padding(1).Render("Goodbye!")
	}

	session := m.agent.GetSession()
	sections := []string{
		m.viewStatus(session),
		m.viewDivider(),
		lipgloss.NewStyle().PaddingLeft(1).Render(m.viewport.View()),
	}
	if m.browserOpen {
		sections = append(sections, m.viewSessionBrowser())
	}
	if m.paletteOpen {
		sections = append(sections, m.viewPalette())
	}
	sections = append(sections,
		m.viewDivider(),
		m.viewInput(session.AgentState),
		m.viewHelp(session.AgentState),
	)
	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// browserRows is the number of session rows the browser shows, adapted to
// the terminal height so the whole frame always fits on screen.
func (m *model) browserRows() int {
	n := len(m.browserSessions)
	if n == 0 {
		n = 1 // the "no sessions" row
	}
	// Reserve: browser chrome (title+blank+footer+2 borders = 5) and at
	// least 5 transcript lines.
	avail := m.height - m.inputBlockHeight() - 5 - 5 - 5
	if avail < 2 {
		avail = 2
	}
	return min(min(n, maxBrowserRows), avail)
}

// viewPalette renders the command palette panel shown above the input.
func (m model) viewPalette() string {
	var sb strings.Builder
	sb.WriteString(primaryText.Render("Commands"))
	sb.WriteString("\n\n")

	items := m.paletteItems()
	if len(items) == 0 {
		sb.WriteString(mutedStyle.Render("  No actions available."))
		sb.WriteString("\n")
	}
	for i, item := range items {
		hint := dimStyle.Render("  " + item.hint)
		if i == m.paletteIndex {
			sb.WriteString(primaryText.Render("> "+item.label) + hint)
		} else {
			sb.WriteString("  " + textStyle.Render(item.label) + hint)
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\n")
	sb.WriteString(dimStyle.Render("↑/↓/j/k: navigate • enter: run • esc: close"))

	return lipgloss.NewStyle().Padding(0, 1).Render(
		inputBox.Width(max(m.width-4, 20)).Render(sb.String()))
}

// viewSessionBrowser renders the session browser panel shown above the input.
func (m model) viewSessionBrowser() string {
	var sb strings.Builder
	sb.WriteString(primaryText.Render("Sessions"))
	sb.WriteString("\n\n")

	if len(m.browserSessions) == 0 {
		sb.WriteString(mutedStyle.Render("  No sessions found."))
		sb.WriteString("\n")
	} else {
		currentID := ""
		if s := m.agent.GetSession(); s != nil {
			currentID = s.ID
		}

		// Scroll window around the selected row.
		rows := m.browserRows()
		start := 0
		if m.browserIndex >= rows {
			start = m.browserIndex - rows + 1
		}
		end := min(start+rows, len(m.browserSessions))

		for i := start; i < end; i++ {
			s := m.browserSessions[i]
			name := s.Name
			nameStyle := textStyle
			if name == "" {
				// Unnamed sessions get a first-message preview (or the bare
				// ID) as a fallback title, shown dimmed.
				nameStyle = mutedStyle
				if s.FirstMessage != "" {
					name = s.FirstMessage
				} else {
					name = s.ID
				}
			}
			meta := fmt.Sprintf("%s • %d msgs • %s", s.ModelID, s.MessageCount, formatRelativeTime(s.LastModified))
			if s.ID == currentID {
				meta = "current • " + meta
			}

			if i == m.browserIndex {
				sb.WriteString(primaryText.Render("> "+name) + "  " + dimStyle.Render(meta))
			} else {
				sb.WriteString("  " + nameStyle.Render(name) + "  " + dimStyle.Render(meta))
			}
			sb.WriteString("\n")
		}
	}

	sb.WriteString("\n")
	switch {
	case m.renaming:
		sb.WriteString(warnText.Render("Rename: ") + m.renameInput.View() + dimStyle.Render("  (enter: save • esc: cancel)"))
	case m.browserStatus.text != "":
		if m.browserStatus.isErr {
			sb.WriteString(errorText.Render(m.browserStatus.text))
		} else {
			sb.WriteString(successText.Render(m.browserStatus.text))
		}
	default:
		hint := "↑/↓/j/k: navigate • enter: switch • r: rename • ctrl+n: new • esc: close"
		if len(m.browserSessions) > m.browserRows() {
			hint += fmt.Sprintf(" • %d/%d", m.browserIndex+1, len(m.browserSessions))
		}
		sb.WriteString(dimStyle.Render(hint))
	}

	return lipgloss.NewStyle().Padding(0, 1).Render(
		inputBox.Width(max(m.width-4, 20)).Render(sb.String()))
}

func (m model) viewStatus(session *api.Session) string {
	sep := dimStyle.Render(" | ")

	name := session.Name
	if name == "" {
		name = session.ID
	}
	name = truncateRunes(name, 40)

	model := session.ModelID
	if model == "" {
		model = "unknown"
	}
	model = truncateRunes(model, 30)

	left := primaryText.Render("kubectl-ai") + sep + mutedStyle.Render(name) + sep + m.viewState(session.AgentState)
	if m.agent != nil && m.agent.SkipPermissionsEnabled() {
		left += sep + warnText.Render("⚡AUTO")
	}
	right := lipgloss.NewStyle().Foreground(colorSecondary).Render(model)

	// The status bar must always be exactly one line, no matter the
	// terminal width: shrink the name (then the model) until it fits.
	for lipgloss.Width(left)+lipgloss.Width(right) > m.width-2 && len([]rune(name)) > 8 {
		name = truncateRunes(name, len([]rune(name))-4)
		left = primaryText.Render("kubectl-ai") + sep + mutedStyle.Render(name) + sep + m.viewState(session.AgentState)
	}
	for lipgloss.Width(left)+lipgloss.Width(right) > m.width-2 && len([]rune(model)) > 7 {
		model = truncateRunes(model, len([]rune(model))-4)
		right = lipgloss.NewStyle().Foreground(colorSecondary).Render(model)
	}

	gap := m.width - lipgloss.Width(left) - lipgloss.Width(right) - 2
	if gap < 0 {
		gap = 0
	}
	return statusBar.Width(m.width).Render(" " + left + strings.Repeat(" ", gap) + right + " ")
}

// truncateRunes shortens s to at most n runes, ending with an ellipsis when
// truncated.
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	if n <= 1 {
		return "…"
	}
	return string(r[:n-1]) + "…"
}

func (m model) viewState(state api.AgentState) string {
	states := map[api.AgentState]struct {
		icon, text string
		style      lipgloss.Style
	}{
		api.AgentStateRunning:         {"●", "Running", successText},
		api.AgentStateInitializing:    {"", "Initializing...", mutedStyle},
		api.AgentStateWaitingForInput: {"●", "Ready", successText},
		api.AgentStateIdle:            {"○", "Idle", mutedStyle},
		api.AgentStateDone:            {"✓", "Done", successText},
		api.AgentStateExited:          {"○", "Exited", mutedStyle},
	}

	if s, ok := states[state]; ok {
		txt := s.style.Render(s.icon + " " + s.text)
		if state == api.AgentStateRunning && !m.thinkStart.IsZero() {
			txt += mutedStyle.Render(" " + formatDuration(time.Since(m.thinkStart)))
		}
		return txt
	}
	return mutedStyle.Render(string(state))
}

func (m model) viewDivider() string {
	return dimStyle.Render(strings.Repeat("─", m.width))
}

func (m model) viewInput(state api.AgentState) string {
	// Show dimmed input hint when in choice mode (picker is inline above)
	if m.inChoiceMode {
		content := mutedStyle.Render("Use ↑/↓ to navigate, Enter to select")
		return lipgloss.NewStyle().Padding(0, 1).Render(inputBoxDim.Width(m.width - 4).Render(content))
	}

	// Show spinner or input
	if state == api.AgentStateRunning || state == api.AgentStateInitializing {
		elapsed := ""
		if !m.thinkStart.IsZero() {
			elapsed = " " + formatDuration(time.Since(m.thinkStart))
		}
		content := primaryText.Render(m.spinner.View()+" Thinking...") + mutedStyle.Render(elapsed)
		return lipgloss.NewStyle().Padding(0, 1).Render(inputBoxDim.Width(m.width - 4).Render(content))
	}

	// The textarea has a fixed internal height; show only as many lines as
	// the content needs (it scrolls internally once content exceeds
	// maxInputHeight, so the cursor line is always within the window).
	view := m.input.View()
	lines := strings.Split(view, "\n")
	if len(lines) > m.inputHeight {
		lines = lines[:m.inputHeight]
	}
	content := strings.Join(lines, "\n")
	if m.sessionRename {
		content = warnText.Render("Rename session:") + "\n" + content
	}

	// Chip-fallback pastes (those that didn't fit inline) are shown as
	// chips below the input.
	if len(m.pastes) > 0 {
		var chips []string
		for _, p := range m.pastes {
			if p.token == "" {
				chips = append(chips, warnText.Render(fmt.Sprintf("[+%d lines]", p.lines)))
			}
		}
		if len(chips) > 0 {
			content += "\n" + strings.Join(chips, " ")
		}
	}

	return lipgloss.NewStyle().Padding(0, 1).Render(inputBox.Width(m.width - 4).Render(content))
}

func (m model) viewHelp(state api.AgentState) string {
	var hints []string
	switch {
	case m.browserOpen:
		return "" // the browser renders its own key hints
	case m.inChoiceMode:
		hints = []string{"↑/↓: navigate", "Enter: select", "Ctrl+C: quit"}
	case state == api.AgentStateRunning:
		hints = []string{"Ctrl+C: cancel"}
	default:
		hints = []string{"Enter: send", "Ctrl+J: newline", "Ctrl+P: commands", "↑/↓: history", "Ctrl+Y: copy", "Shift+Tab: auto", "Esc: clear/stop", "Ctrl+C: quit"}
		if m.viewport.TotalLineCount() > m.viewport.Height {
			hints = append(hints, "PgUp/PgDn: scroll")
		}
	}
	return dimStyle.Padding(0, 2, 1, 2).Render(strings.Join(hints, " • "))
}

// formatRelativeTime renders t as a short relative duration like "5m ago".
func formatRelativeTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	}
}

func formatDuration(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
	}
}
