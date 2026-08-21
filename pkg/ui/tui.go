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
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
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
	// Mouse capture is enabled for smooth wheel scrolling. Text can still be
	// selected/copied with the terminal's standard bypass for mouse-capturing
	// apps (hold Shift on most terminals, Option on iTerm2, while dragging) —
	// the same convention opencode, vim and tmux users are used to. We do NOT
	// use WithMouseAllMotion: we only care about the wheel, and cell motion
	// keeps the event volume down.
	return &TUI{
		program: tea.NewProgram(newModel(agent), tea.WithAltScreen(), tea.WithMouseCellMotion()),
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

func (m *model) fetchSessions() tea.Msg {
	sessions, err := m.agent.ListSessions()
	if err != nil {
		return api.Message{
			Type:    api.MessageTypeError,
			Payload: fmt.Sprintf("Failed to list sessions: %v", err),
		}
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
// input with the full text, the paste is attached to the draft and shown as
// a compact "[+N lines]" chip below the input (like opencode). The attached
// content is appended to the message when it is submitted.
type pastedBlock struct {
	id      int
	lines   int
	content string
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
	// Plain Up/Down are handled by us (input history navigation, and cursor
	// movement within multi-line drafts); Ctrl+P/N also moves between lines.
	ti.KeyMap.LinePrevious = key.NewBinding(key.WithKeys("ctrl+p"))
	ti.KeyMap.LineNext = key.NewBinding(key.WithKeys("ctrl+n"))

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

	return model{
		agent:       agent,
		input:       ti,
		inputHeight: 1,
		historyIdx:  -1,
		viewport:    vp,
		spinner:     sp,
		list:        l,
		cache:       newRenderCache(),
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

	case tea.MouseMsg:
		switch msg.Button {
		case tea.MouseButtonWheelUp:
			m.viewport.ScrollUp(3)
		case tea.MouseButtonWheelDown:
			m.viewport.ScrollDown(3)
		}
		return m, nil

	case *api.Message:
		return m.handleAgentMsg(msg)

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case tickMsg:
		return m, m.tick()

	case sessionListMsg:
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

		items := make([]list.Item, len(msg))
		ids := make([]string, len(msg))
		for i, s := range msg {
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
		m.choiceOptionID = "manual-session-picker"
		m.choiceType = "session"
		m.sessionIDs = ids
		m.dirty = true
		m.refresh()
		m.viewport.GotoBottom()
		return m, nil
	}
	return m, nil
}

func (m *model) resize() {
	m.viewport.Width = m.width - 2
	// The textarea must fit the input box's content area exactly: box border
	// (2) + box padding (2) + outer padding (2) = 6, plus 2 cells of slack
	// so rendered lines never reach the terminal's last column and wrap.
	m.input.SetWidth(max(m.width-8, 20))
	m.list.SetWidth(m.width - 4)
	m.updateViewportHeight()
	m.refresh()
	m.viewport.GotoBottom()
}

func (m *model) updateViewportHeight() {
	// Layout: status(1) + 2 dividers(2) + input block + help(1) + bottom padding(1)
	contentH := m.height - (m.inputBlockHeight() + 5)

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
	if len(m.pastes) > 0 {
		h++ // paste attachment chips line
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
		m.input.Reset()
		m.pastes = nil
		m.historyIdx = -1
		m.historyDraft = ""
		m.syncInputHeight()
		return m, nil
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
		// input history, like opencode and Claude Code.
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
	case tea.KeyPgUp:
		m.viewport.ScrollUp(m.viewport.Height / 2)
	case tea.KeyPgDown:
		m.viewport.ScrollDown(m.viewport.Height / 2)
	case tea.KeyBackspace:
		if m.inChoiceMode {
			return m, nil
		}
		// With an empty input, Backspace detaches the most recent paste.
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

// handlePaste inserts pasted text into the input. Large multi-line pastes are
// attached to the draft and shown as a compact "[+N lines]" chip below the
// input (like opencode) instead of flooding the input; the attached content
// is appended to the message on submit. Short pastes are inserted verbatim.
func (m *model) handlePaste(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if m.inChoiceMode {
		return m, nil
	}

	content := strings.ReplaceAll(string(msg.Runes), "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	// A trailing newline is almost always a copy artifact; drop it so that
	// pasting a single line doesn't grow the input.
	content = strings.TrimSuffix(content, "\n")
	if content == "" {
		return m, nil
	}

	if n := strings.Count(content, "\n") + 1; n >= pasteCollapseLines {
		m.nextPasteID++
		m.pastes = append(m.pastes, pastedBlock{id: m.nextPasteID, lines: n, content: content})
	} else {
		m.input.InsertString(content)
	}
	m.syncInputHeight()
	return m, nil
}

func (m *model) handleEnter() (tea.Model, tea.Cmd) {
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
	// Attach the full content of collapsed pastes to the submitted message.
	if len(m.pastes) > 0 {
		var parts []string
		if strings.TrimSpace(value) != "" {
			parts = append(parts, value)
		}
		for _, p := range m.pastes {
			parts = append(parts, p.content)
		}
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

	// Intercept "sessions" command
	if strings.EqualFold(value, "sessions") {
		return m, m.fetchSessions
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
			dimStyle.PaddingLeft(1).Render("Type 'sessions' to browse and resume past sessions"),
			dimStyle.PaddingLeft(1).Render("Hold Shift (Option on iTerm2) and drag to copy text")))
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
	return lipgloss.JoinVertical(lipgloss.Left,
		m.viewStatus(session),
		m.viewDivider(),
		lipgloss.NewStyle().PaddingLeft(1).Render(m.viewport.View()),
		m.viewDivider(),
		m.viewInput(session.AgentState),
		m.viewHelp(session.AgentState),
	)
}

func (m model) viewStatus(session *api.Session) string {
	sep := dimStyle.Render(" | ")

	name := session.Name
	if name == "" {
		name = session.ID
	}
	left := primaryText.Render("kubectl-ai") + sep + mutedStyle.Render(name) + sep + m.viewState(session.AgentState)

	model := session.ModelID
	if model == "" {
		model = "unknown"
	}
	right := lipgloss.NewStyle().Foreground(colorSecondary).Render(model)

	gap := m.width - lipgloss.Width(left) - lipgloss.Width(right) - 2
	if gap < 0 {
		gap = 0
	}
	return statusBar.Width(m.width).Render(" " + left + strings.Repeat(" ", gap) + right + " ")
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

	// Collapsed pastes are shown as chips below the input.
	if len(m.pastes) > 0 {
		chips := make([]string, len(m.pastes))
		for i, p := range m.pastes {
			chips[i] = warnText.Render(fmt.Sprintf("[+%d lines]", p.lines))
		}
		content += "\n" + strings.Join(chips, " ")
	}

	return lipgloss.NewStyle().Padding(0, 1).Render(inputBox.Width(m.width - 4).Render(content))
}

func (m model) viewHelp(state api.AgentState) string {
	var hints []string
	if m.inChoiceMode {
		hints = []string{"↑/↓: navigate", "Enter: select", "Ctrl+C: quit"}
	} else if state == api.AgentStateRunning {
		hints = []string{"Ctrl+C: cancel"}
	} else {
		hints = []string{"Enter: send", "Ctrl+J: newline", "↑/↓: history", "Esc: clear", "Ctrl+C: quit"}
		if m.viewport.TotalLineCount() > m.viewport.Height {
			hints = append(hints, "PgUp/PgDn: scroll")
		}
	}
	return dimStyle.Padding(0, 2, 1, 2).Render(strings.Join(hints, " • "))
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
