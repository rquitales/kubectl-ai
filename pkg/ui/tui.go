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
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/agent"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/kube"
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
	// kubeContextTTL is how often the status bar's kube context is
	// re-resolved from the kubeconfig file (driven by the 1s tick).
	kubeContextTTL = 10 * time.Second
	// deltaRefreshInterval is the minimum time between viewport refreshes
	// for live-streaming text deltas: chunks arrive far faster than is
	// worth re-rendering the transcript, so refreshes are gated (the final
	// text message always refreshes, so the tail is never lost).
	deltaRefreshInterval = 150 * time.Millisecond
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

// sessionDeletedMsg reports the result of a delete attempt from the
// session browser.
type sessionDeletedMsg struct {
	id  string
	err error
}

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
	// clearedAt is the transcript position of the Ctrl+L "cleared"
	// boundary marker (0 = no marker). Content before it leaves the
	// current view but is revealed again by scrolling up (revealAll).
	clearedAt int
	// revealAll temporarily shows the whole transcript (user scrolled up
	// while cleared); it resets when scrolling back to the bottom or on
	// new messages.
	revealAll bool
	// expandToolResults renders tool call results in full (ctrl+o toggle);
	// collapsed results show only the first few lines plus a count.
	expandToolResults bool
	// Slash-command autocomplete cycling state: the matches captured when
	// Tab was first pressed, so cycling doesn't re-narrow once the input
	// becomes a full command name.
	tabMatches []string
	tabIndex   int
	// Kube context names for /context autocomplete, cached briefly.
	contextNames     []string
	contextNamesLast time.Time
	// Cluster namespaces for /namespace autocomplete, cached briefly.
	namespaceNames     []string
	namespaceNamesLast time.Time
	// sessionID tracks the active session so switches are detectable.
	sessionID string
	// Kube context/namespace shown in the status bar; resolved once at
	// startup and re-resolved on a TTL by the 1s tick (a cheap file read).
	kubeContext     kubeContextInfo
	kubeContextOK   bool
	kubeContextLast time.Time
	width           int
	height          int
	dirty           bool
	quitting        bool
	thinkStart      time.Time
	// lastDeltaRefresh is when the viewport last refreshed for a live
	// text-delta; used to throttle re-renders while streaming.
	lastDeltaRefresh time.Time
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
	pendingDeleteID string // session staged for deletion, awaiting 'y'
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

	m := model{
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
	if agent != nil {
		if s := agent.GetSession(); s != nil {
			m.sessionID = s.ID
		}
	}
	m.resolveKubeContext()
	return m
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
		// Re-resolve the kube context on a TTL so an external
		// `kubectl config use-context` is reflected in the status bar.
		if time.Since(m.kubeContextLast) > kubeContextTTL {
			m.resolveKubeContext()
		}
		// Keep the spinner alive while the agent is running, even when no
		// other messages are arriving (e.g. during quiet stretches between
		// live text deltas, without spawning per-chunk tick chains).
		if m.agentState() == api.AgentStateRunning || m.agentState() == api.AgentStateInitializing {
			m.spinner, _ = m.spinner.Update(msg)
		}
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

	case sessionDeletedMsg:
		if msg.err != nil {
			m.setBrowserStatus("Delete failed: "+msg.err.Error(), true)
			return m, nil
		}
		m.setBrowserStatus("Deleted "+msg.id, false)
		// Refresh the browser contents, keeping the browser open.
		return m, m.fetchSessions

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
	if m.completionHintVisible() {
		h++ // the completion/shell hint line
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
	case tea.KeyCtrlL:
		// Clear the current view (transcript leaves the screen) while
		// keeping everything one PgUp away. Pressing again restores.
		if m.clearedAt > 0 {
			m.clearedAt = 0
			m.revealAll = false
		} else {
			m.clearedAt = len(m.messages)
			m.revealAll = false
		}
		m.dirty = true
		m.refresh()
		m.viewport.GotoBottom()
		return m, nil
	case tea.KeyCtrlP:
		m.openPalette()
		return m, nil
	case tea.KeyCtrlO:
		// Toggle expanded tool call results (collapsed by default so
		// back-to-back tool calls stay compact).
		m.expandToolResults = !m.expandToolResults
		m.dirty = true
		m.refresh()
		return m, nil
	case tea.KeyShiftTab:
		// Toggle auto-accept mode (skip permission prompts), like opencode
		// and Claude Code.
		return m.toggleAutoMode()
	case tea.KeyTab:
		if m.inChoiceMode {
			return m, nil
		}
		// Namespace autocomplete: cycle the partial after "/namespace " (or
		// "/ns ") through live cluster namespaces.
		if v := m.input.Value(); strings.HasPrefix(v, "/namespace ") || strings.HasPrefix(v, "/ns ") {
			cmd := "/namespace "
			if strings.HasPrefix(v, "/ns ") {
				cmd = "/ns "
			}
			if m.tabIndex < len(m.tabMatches) && v == cmd+m.tabMatches[m.tabIndex] {
				m.tabIndex = (m.tabIndex + 1) % len(m.tabMatches)
			} else {
				m.tabMatches = m.namespaceMatches()
				m.tabIndex = 0
			}
			if len(m.tabMatches) > 0 {
				m.input.SetValue(cmd + m.tabMatches[m.tabIndex])
				m.input.CursorEnd()
				m.syncInputHeight()
			}
			return m, nil
		}
		// Kube context autocomplete: cycle the partial after "/context "
		// through context names.
		if v := m.input.Value(); strings.HasPrefix(v, "/context ") {
			if m.tabIndex < len(m.tabMatches) && v == "/context "+m.tabMatches[m.tabIndex] {
				m.tabIndex = (m.tabIndex + 1) % len(m.tabMatches)
			} else {
				m.tabMatches = m.contextMatches()
				m.tabIndex = 0
			}
			if len(m.tabMatches) > 0 {
				m.input.SetValue("/context " + m.tabMatches[m.tabIndex])
				m.input.CursorEnd()
				m.syncInputHeight()
			}
			return m, nil
		}
		// File mention autocomplete: cycle the current @token through
		// filesystem path matches, keeping preceding text intact.
		if tok := m.lastToken(); strings.HasPrefix(tok, "@") && tok != "@" {
			if m.tabIndex < len(m.tabMatches) && strings.HasSuffix(m.input.Value(), m.tabMatches[m.tabIndex]) {
				m.tabIndex = (m.tabIndex + 1) % len(m.tabMatches)
			} else {
				m.tabMatches = m.fileMatches()
				m.tabIndex = 0
			}
			if len(m.tabMatches) > 0 {
				m.input.SetValue(strings.TrimSuffix(m.input.Value(), tok) + "@" + m.tabMatches[m.tabIndex])
				m.input.CursorEnd()
				m.syncInputHeight()
			}
			return m, nil
		}
		// Slash-command autocomplete: cycle the input through the matches.
		if m.tabIndex < len(m.tabMatches) && m.input.Value() == m.tabMatches[m.tabIndex] {
			// Mid-cycle: advance to the next captured match.
			m.tabIndex = (m.tabIndex + 1) % len(m.tabMatches)
		} else {
			// Start a new cycle from the current prefix.
			m.tabMatches = slashCompletions(m.input.Value())
			m.tabIndex = 0
		}
		if len(m.tabMatches) > 0 {
			m.input.SetValue(m.tabMatches[m.tabIndex])
			m.input.CursorEnd()
			m.syncInputHeight()
			return m, nil
		}
		// Not a slash command: keep the textarea's default behavior.
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		m.syncInputHeight()
		return m, cmd
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
		// Scrolling up while cleared reveals the hidden transcript.
		if m.clearedAt > 0 && !m.revealAll {
			m.revealAll = true
			m.dirty = true
			m.refresh()
		}
		m.viewport.ScrollUp(m.viewport.Height / 2)
	case tea.KeyPgDown:
		m.viewport.ScrollDown(m.viewport.Height / 2)
		// Back at the bottom: the cleared view applies again.
		if m.revealAll && m.viewport.AtBottom() {
			m.revealAll = false
			m.dirty = true
			m.refresh()
		}
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

	// A staged delete confirmation captures y/n before anything else.
	if m.pendingDeleteID != "" {
		if msg.String() == "y" {
			id := m.pendingDeleteID
			m.pendingDeleteID = ""
			return m, func() tea.Msg {
				return sessionDeletedMsg{err: m.agent.DeleteSession(id), id: id}
			}
		}
		// Anything else cancels.
		m.pendingDeleteID = ""
		m.browserStatus = browserStatusMsg{}
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
	case "d":
		if m.browserIndex < len(m.browserSessions) {
			s := m.browserSessions[m.browserIndex]
			if cur := m.agent.GetSession(); cur != nil && cur.ID == s.ID {
				m.setBrowserStatus("Can't delete the current session", true)
				return m, nil
			}
			m.pendingDeleteID = s.ID
			label := s.Name
			if label == "" {
				label = s.ID
			}
			m.setBrowserStatus(fmt.Sprintf("Delete %q? (y to confirm)", label), true)
			return m, nil
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

	// Intercept the help command (handled locally, prints a reference into the
	// transcript instead of round-tripping to the LLM).
	if v := strings.ToLower(value); v == "/help" || v == "/?" {
		m.appendLocalMessage(helpText())
		return m, nil
	}

	m.thinkStart = time.Now()

	return m, func() tea.Msg {
		m.agent.Input <- &api.UserInputResponse{Query: value}
		return nil
	}
}

// handleTextDelta folds a live-streaming text delta into the transcript:
// the entry matching the delta's ID is updated in place, or appended on the
// first chunk. The final text message (same ID) arrives via handleAgentMsg
// and replaces the delta entry through the store snapshot.
func (m *model) handleTextDelta(msg *api.Message) (tea.Model, tea.Cmd) {
	updated := false
	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].ID == msg.ID {
			m.messages[i].Payload = msg.Payload
			m.messages[i].Timestamp = msg.Timestamp
			updated = true
			break
		}
	}
	if !updated {
		m.messages = append(m.messages, msg)
	}
	m.dirty = true

	// Throttle viewport refreshes while streaming (see deltaRefreshInterval);
	// note we don't restart the spinner tick here to avoid spawning extra
	// tick chains per chunk.
	if time.Since(m.lastDeltaRefresh) < deltaRefreshInterval {
		return m, nil
	}
	m.lastDeltaRefresh = time.Now()

	// Mirror the follow-bottom behavior of handleAgentMsg.
	if m.clearedAt > 0 && m.revealAll {
		m.revealAll = false
		m.refresh()
		m.viewport.GotoBottom()
	} else {
		atBottom := m.viewport.AtBottom()
		m.refresh()
		if atBottom {
			m.viewport.GotoBottom()
		}
	}
	return m, nil
}

func (m *model) handleAgentMsg(msg *api.Message) (tea.Model, tea.Cmd) {
	// Live-streamed model text arrives as ephemeral deltas that are not in
	// the session store; fold them into the transcript separately.
	if msg.Type == api.MessageTypeTextDelta {
		return m.handleTextDelta(msg)
	}

	session := m.agent.GetSession()
	// A session switch resets the cleared boundary so the resumed
	// session's transcript shows in full.
	if session.ID != m.sessionID {
		m.clearedAt = 0
		m.sessionID = session.ID
	}
	// Re-snapshot from the store. When this is the final text message of a
	// streamed iteration, the snapshot contains it (with the stream ID) and
	// drops the ephemeral delta entry: the streaming entry is replaced in
	// place by the final stored message.
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

	// A new message snaps the view back to the cleared state: hide the
	// revealed history again and land on the fresh content below the marker.
	if m.clearedAt > 0 && m.revealAll {
		m.revealAll = false
		m.dirty = true
		m.refresh()
		m.viewport.GotoBottom()
	} else {
		// Only follow the transcript to the bottom if the user is already at
		// the bottom; yanking the viewport down while the user scrolled up
		// (e.g. to select text for copying) is hostile.
		atBottom := m.viewport.AtBottom()
		m.refresh()
		if atBottom {
			m.viewport.GotoBottom()
		}
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

// renderWelcome renders the empty-state panel shown when the transcript is
// empty: the logo and tagline, the current kube context (front and center,
// since "which cluster am I pointed at?" is the most important thing to know
// before you act), and a compact command reference.
func (m model) renderWelcome() string {
	var sb strings.Builder
	sb.WriteString("\n")
	sb.WriteString(primaryText.Render(logo))
	sb.WriteString("\n")
	tagline := "Your AI-powered Kubernetes assistant"
	sb.WriteString(mutedStyle.PaddingLeft(1).Render(truncateRunes(tagline, max(m.width-2, 10))))
	sb.WriteString("\n\n")

	// Kube context panel — the safety-critical "what am I pointed at?" cue.
	// A prod-looking context is rendered in the warning color.
	if m.kubeContextOK {
		ctxStyle := lipgloss.NewStyle().Foreground(colorSecondary).Bold(true)
		label := "Connected to"
		if m.kubeContext.isProd() {
			ctxStyle = lipgloss.NewStyle().Foreground(colorWarning).Bold(true)
			label = "⚠ Connected to (prod)"
		}
		sb.WriteString(inputBox.Width(max(m.width-6, 30)).
			Render("⎈ " + dimStyle.Render(label+" ") + ctxStyle.Render(m.kubeContext.String())))
		sb.WriteString("\n\n")
	} else {
		sb.WriteString(dimStyle.PaddingLeft(1).Render("⎈ No kubeconfig found — set $KUBECONFIG or ~/.kube/config"))
		sb.WriteString("\n\n")
	}

	// Command reference: two compact columns when there's room, a single
	// column when the terminal is narrow so the rows never wrap.
	commands := [][2]string{
		{"/sessions", "browse & resume sessions"},
		{"/context", "switch kube context"},
		{"/namespace", "switch namespace"},
		{"/model", "switch model"},
		{"/rename", "rename this session"},
		{"/compact", "summarize & free context"},
		{"/clear", "clear the transcript"},
		{"/exit", "quit"},
	}
	renderRow := func(c [2]string) string {
		return primaryText.Render("  "+c[0]) + " " + dimStyle.Render(c[1])
	}
	// A two-column row needs room for the widest left + widest right entry
	// plus a gap; fall back to a single column below that width.
	colWidth := 0
	for _, c := range commands {
		if w := lipgloss.Width(c[0] + " " + c[1]); w > colWidth {
			colWidth = w
		}
	}
	twoCol := m.width >= colWidth*2+6
	var rows []string
	if twoCol {
		half := (len(commands) + 1) / 2
		for i := 0; i < half; i++ {
			left := renderRow(commands[i])
			right := ""
			if i+half < len(commands) {
				right = "  " + renderRow(commands[i+half])
			}
			rows = append(rows, left+right)
		}
	} else {
		for _, c := range commands {
			rows = append(rows, renderRow(c))
		}
	}
	sb.WriteString(strings.Join(rows, "\n"))
	sb.WriteString("\n\n")
	footer := "Type a message, or drag-select to copy (Ctrl+Y copies the last reply)"
	sb.WriteString(dimStyle.PaddingLeft(1).Render(truncateRunes(footer, max(m.width-2, 10))))
	sb.WriteString("\n")
	return sb.String()
}

// renderChoicePrompt renders the choice picker's prompt. For a permission
// prompt (confirm), the commands being approved are pulled out of the prose
// and rendered on their own highlighted lines so the user immediately sees
// what they're authorizing — the safety-critical part must not blend into the
// surrounding text. Other prompts (e.g. session pickers) render as a simple
// "? <prompt>" line.
func (m model) renderChoicePrompt() string {
	if m.choiceType != "confirm" {
		return warnText.Render("? " + m.choicePrompt)
	}
	// Permission prompt: "The following commands require your approval to run:\n* cmd\n...\n\nDo you want to proceed ?"
	// Split the bullet command lines out of the prose.
	lines := strings.Split(m.choicePrompt, "\n")
	var header, commands []string
	question := ""
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "":
			continue
		case strings.HasPrefix(trimmed, "* "):
			commands = append(commands, strings.TrimPrefix(trimmed, "* "))
		case strings.HasPrefix(trimmed, "Do you want to proceed"):
			question = trimmed
		default:
			header = append(header, line)
		}
	}

	var sb strings.Builder
	if len(header) > 0 {
		sb.WriteString(warnText.Render("? " + strings.Join(header, " ")))
	} else {
		sb.WriteString(warnText.Render("? Approval required"))
	}
	// The commands are the thing the user is actually approving: render each
	// distinctly (a warning-tinted code line) so they pop out of the prose.
	if len(commands) == 0 {
		// Unknown shape — fall back to the raw prompt.
		return warnText.Render("? " + m.choicePrompt)
	}
	cmdStyle := lipgloss.NewStyle().Foreground(colorWarning)
	for _, cmd := range commands {
		sb.WriteString("\n")
		sb.WriteString("  " + cmdStyle.Render("› " + cmd))
	}
	if question != "" {
		sb.WriteString("\n\n")
		sb.WriteString(warnText.Bold(true).Render(question))
	}
	return sb.String()
}

func (m model) renderMessages() string {
	var sb strings.Builder

	if len(m.messages) == 0 {
		sb.WriteString(m.renderWelcome())
	} else {
		width := min(m.viewport.Width-6, 90)
		if width < 40 {
			width = 40
		}

		renderer, err := m.cache.getRenderer(width)
		if err != nil {
			return "Error rendering messages"
		}

		from := min(m.clearedAt, len(m.messages))
		start := 0
		if from > 0 && !m.revealAll {
			// Cleared view: only the marker and what came after it.
			start = from
		}
		for i := start; i < len(m.messages); i++ {
			if i == from && from > 0 {
				sb.WriteString(dimStyle.PaddingLeft(1).Render("── transcript cleared (ctrl+l) ──") + "\n\n")
			}
			msg := m.messages[i]
			// A tool-call request immediately followed by its response renders
			// as one grouped, nested block (the command header with the result
			// indented under it) — like Claude Code — instead of two
			// disconnected boxes. A request whose result hasn't arrived yet
			// renders standalone with a "Running" indicator.
			if msg.Type == api.MessageTypeToolCallRequest && i+1 < len(m.messages) &&
				m.messages[i+1].Type == api.MessageTypeToolCallResponse {
				if s := m.renderToolGroup(msg, m.messages[i+1], width); s != "" {
					sb.WriteString(s)
				}
				i++ // consume the paired response
				continue
			}
			if s := m.renderMessage(msg, renderer, width); s != "" {
				sb.WriteString(s)
			}
		}
		if from == len(m.messages) && from > 0 {
			sb.WriteString(dimStyle.PaddingLeft(1).Render("── transcript cleared (ctrl+l) ──") + "\n\n")
		}
	}

	// Render choice picker inline at the end of messages
	if m.inChoiceMode {
		sb.WriteString("\n")
		sb.WriteString(m.renderChoicePrompt())
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
		return m.renderToolResult(msg)
	}
	// Skip choice requests - they're rendered in the input area instead
	if msg.Type == api.MessageTypeUserChoiceRequest || msg.Type == api.MessageTypeSessionPickerRequest {
		return ""
	}

	// Check cache (except tool calls which show status, and text deltas
	// whose content changes per chunk; the final text message shares the
	// delta's ID and must render fresh).
	cacheable := msg.ID != "" && msg.Type != api.MessageTypeToolCallRequest && msg.Type != api.MessageTypeTextDelta
	if cacheable {
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
	if cacheable && result != "" {
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
		if msg.Tokens > 0 {
			label += dimStyle.Italic(true).Render(" · " + formatTokens(msg.Tokens))
		}
		rendered, _ := r.Render(payload)
		body := strings.TrimSpace(rendered)
		// A live-streaming delta is the incomplete text the model is still
		// producing (the final Text message replaces it). Append a cursor so
		// it's visually obvious the reply is still being typed — like Claude
		// Code and opencode — instead of looking like a stalled reply.
		if msg.Type == api.MessageTypeTextDelta && m.agentState() == api.AgentStateRunning {
			body += primaryText.Render(" ▋")
		}
		return agentMsg.Width(w+2).Render(label+"\n"+body) + "\n"
	}
	return ""
}

func (m model) renderToolCall(msg *api.Message, w int) string {
	payload, ok := msg.Payload.(string)
	if !ok {
		return ""
	}
	content := successText.Render("⚡ Running") + " " + dimStyle.Render("·") + " " + textStyle.Render(payload)
	return toolBox.Width(w).Render(content) + "\n"
}

// renderToolGroup renders a tool-call request and its result as a single
// grouped block: the command on the header line (in the running/secondary
// color), and the result nested under a "⎿" indent — the same shape Claude
// Code and opencode use, so back-to-back tool calls read as one action with
// its output instead of two unrelated boxes. The result is collapsed/expanded
// exactly like renderToolResult, reusing the same line caps and the
// ctrl+o toggle.
func (m model) renderToolGroup(req, resp *api.Message, w int) string {
	cmd, ok := req.Payload.(string)
	if !ok {
		return ""
	}
	failed := toolResultFailed(resp.Payload)
	marker, headStyle := "⚡ ", successText
	if failed {
		marker, headStyle = "✗ ", errorText
	}
	header := headStyle.Render(marker + truncateRunes(cmd, max(w-4, 20)))

	body := m.renderToolResult(resp)
	if strings.TrimSpace(body) == "" {
		// No displayable output yet (e.g. still running, or empty success):
		// keep the block compact with just the command header.
		return toolBox.Width(w).Render(header) + "\n"
	}
	content := header + "\n" + strings.TrimRight(body, "\n")
	return toolBox.Width(w).Render(content) + "\n"
}

const (
	// toolResultCollapsedLines is the number of output lines shown for a
	// collapsed tool result.
	toolResultCollapsedLines = 3
	// toolResultExpandedLines caps an expanded tool result; beyond it a
	// "+N more lines" tail is shown.
	toolResultExpandedLines = 200
	// toolResultCollapsedWidth and toolResultExpandedWidth cap the
	// rendered width of each output line.
	toolResultCollapsedWidth = 100
	toolResultExpandedWidth  = 200
)

// renderToolResult renders a tool call's result collapsed (opencode-style):
// the first couple of output lines plus a "+N more lines" count, so back-to-
// back tool calls are visually distinct instead of looking duplicated.
// Ctrl+O toggles expandToolResults to show the full output (still capped at
// toolResultExpandedLines lines and toolResultExpandedWidth columns).
func (m model) renderToolResult(msg *api.Message) string {
	failed := toolResultFailed(msg.Payload)
	text := toolResultText(msg.Payload)
	if failed {
		// For a failed command, prefer stderr/error over stdout so the cause
		// is front and center rather than the command's normal output.
		if s := toolResultErrorText(msg.Payload); strings.TrimSpace(s) != "" {
			text = s
		}
	}
	if strings.TrimSpace(text) == "" {
		return ""
	}

	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	maxLines, lineWidth := toolResultCollapsedLines, toolResultCollapsedWidth
	if m.expandToolResults {
		maxLines, lineWidth = toolResultExpandedLines, toolResultExpandedWidth
	}
	shown := lines
	if len(shown) > maxLines {
		shown = shown[:maxLines]
	}

	// Failed results render in the error color so a back-to-back run of tool
	// calls makes the failure obvious at a glance; successes stay dim.
	lineStyle := dimStyle
	if failed {
		lineStyle = lipgloss.NewStyle().Foreground(colorError)
	}

	var out []string
	for i, l := range shown {
		prefix := "    "
		if i == 0 {
			if failed {
				prefix = "  ⎿ ✗ "
			} else {
				prefix = "  ⎿ "
			}
		}
		out = append(out, lineStyle.Render(prefix+truncateRunes(l, lineWidth)))
	}
	if len(lines) > maxLines {
		hint := fmt.Sprintf("    +%d more lines", len(lines)-maxLines)
		if m.expandToolResults {
			hint += " (ctrl+o to collapse)"
		} else {
			hint += " (ctrl+o to expand)"
		}
		out = append(out, lineStyle.Render(hint))
	} else if m.expandToolResults {
		out[len(out)-1] += lineStyle.Render(" (ctrl+o to collapse)")
	}
	return strings.Join(out, "\n") + "\n"
}

// toolResultText extracts displayable output from a tool call result payload
// (ExecResult-shaped maps, MCP content maps, or shim observation strings).
func toolResultText(payload any) string {
	switch p := payload.(type) {
	case string:
		return p
	case map[string]any:
		if len(p) == 0 {
			return ""
		}
		for _, k := range []string{"stdout", "stderr", "content", "result", "output"} {
			if s, ok := p[k].(string); ok && s != "" {
				return s
			}
		}
		if b, err := json.Marshal(p); err == nil {
			return string(b)
		}
	}
	return fmt.Sprintf("%v", payload)
}

// toolResultFailed reports whether a tool-call result represents a failure:
// a non-zero exit code, a non-empty "error" field in the result map, or a
// plain string payload emitted on the tool-error path (err.Error()). Success-
// shaped results (empty maps, stdout/content wrappers) report false.
func toolResultFailed(payload any) bool {
	switch p := payload.(type) {
	case string:
		// The only string results are shim observations or err.Error();
		// treat a non-empty string with no "Result of running" prefix as a
		// failure. A shim observation wraps successful output.
		return p != "" && !strings.HasPrefix(p, "Result of running ")
	case map[string]any:
		if code, ok := p["exit_code"]; ok {
			if n, ok := toInt(code); ok && n != 0 {
				return true
			}
		}
		if e, ok := p["error"].(string); ok && e != "" {
			return true
		}
	}
	return false
}

// toolResultErrorText extracts the failure explanation (stderr/error) from a
// result payload, preferring the cause over normal stdout.
func toolResultErrorText(payload any) string {
	if p, ok := payload.(map[string]any); ok {
		for _, k := range []string{"error", "stderr"} {
			if s, ok := p[k].(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}

// toInt converts a JSON-decoded numeric value (float64) to an int.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	}
	return 0, false
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
		m.viewBottomDivider(),
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
		hint := "↑/↓/j/k: navigate • enter: switch • r: rename • d: delete • ctrl+n: new • esc: close"
		if len(m.browserSessions) > m.browserRows() {
			hint += fmt.Sprintf(" • %d/%d", m.browserIndex+1, len(m.browserSessions))
		}
		sb.WriteString(dimStyle.Render(hint))
	}

	return lipgloss.NewStyle().Padding(0, 1).Render(
		inputBox.Width(max(m.width-4, 20)).Render(sb.String()))
}

// kubeContextInfo is the resolved current kube context/namespace shown in
// the status bar.
type kubeContextInfo struct {
	context   string
	namespace string
}

// String renders the info as "context/namespace", or the bare context for
// the default namespace.
func (k kubeContextInfo) String() string {
	if k.namespace == "" || k.namespace == "default" {
		return k.context
	}
	return k.context + "/" + k.namespace
}

// isProd reports whether the context name looks like a production cluster
// (rendered in the warning color in the status bar).
func (k kubeContextInfo) isProd() bool {
	return strings.Contains(k.context, "prod")
}

// loadKubeContext reads the current context and namespace from a kubeconfig
// file (path, or the KUBECONFIG env / ~/.kube/config default when empty).
// A missing or unreadable config reports ok=false so the status bar simply
// omits the indicator.
func loadKubeContext(path string) (info kubeContextInfo, ok bool) {
	context, namespace, ok := kube.CurrentContext(path)
	if !ok {
		return kubeContextInfo{}, false
	}
	info.context = context
	info.namespace = namespace
	return info, true
}

// resolveKubeContext (re)loads the status bar's kube context from the
// agent's active kubeconfig (the session override when one is applied, else
// the base path). Failures are silent: the indicator simply disappears.
func (m *model) resolveKubeContext() {
	path := ""
	if m.agent != nil {
		path = m.agent.ActiveKubeconfig()
	}
	m.kubeContext, m.kubeContextOK = loadKubeContext(path)
	m.kubeContextLast = time.Now()
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

	// Running session token total, hidden until the provider reports usage.
	totalTokens := 0
	for _, msg := range session.AllMessages() {
		totalTokens += msg.Tokens
	}

	left := primaryText.Render("kubectl-ai") + sep + mutedStyle.Render(name) + sep + m.viewState(session.AgentState)
	if m.agent != nil && m.agent.SkipPermissionsEnabled() {
		left += sep + warnText.Render("⚡AUTO")
	}

	// The kube context sits on the right next to the model name; contexts
	// that look like production get the warning color.
	kubeStyle := lipgloss.NewStyle().Foreground(colorSecondary)
	kube := ""
	if m.kubeContextOK {
		if m.kubeContext.isProd() {
			kubeStyle = lipgloss.NewStyle().Foreground(colorWarning)
		}
		kube = "⎈ " + m.kubeContext.String()
	}
	renderRight := func(model string) string {
		s := lipgloss.NewStyle().Foreground(colorSecondary).Render(model)
		if kube != "" {
			s += sep + kubeStyle.Render(kube)
		}
		if totalTokens > 0 {
			s = dimStyle.Render("Σ "+formatTokens(totalTokens)) + " " + s
		}
		return s
	}
	right := renderRight(model)

	// The status bar must always be exactly one line, no matter the
	// terminal width: shrink the name (then the model, then the kube
	// context) until it fits.
	for lipgloss.Width(left)+lipgloss.Width(right) > m.width-2 && len([]rune(name)) > 8 {
		name = truncateRunes(name, len([]rune(name))-4)
		left = primaryText.Render("kubectl-ai") + sep + mutedStyle.Render(name) + sep + m.viewState(session.AgentState)
	}
	for lipgloss.Width(left)+lipgloss.Width(right) > m.width-2 && len([]rune(model)) > 7 {
		model = truncateRunes(model, len([]rune(model))-4)
		right = renderRight(model)
	}
	for lipgloss.Width(left)+lipgloss.Width(right) > m.width-2 && len([]rune(kube)) > 8 {
		kube = truncateRunes(kube, len([]rune(kube))-4)
		right = renderRight(model)
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

// formatTokens renders a token count compactly: plain below 1000 ("42"),
// one-decimal thousands above ("1.2k").
func formatTokens(n int) string {
	if n < 1000 {
		return strconv.Itoa(n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
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

// viewBottomDivider renders the divider that sits between the transcript and
// the input. When the user has scrolled up in the transcript, it carries a
// right-aligned scroll-position indicator ("↓ NN%  PgDn for latest") so it's
// obvious there is newer content below and how far down the view is — like
// Claude Code's scroll cue. At the bottom it is a plain divider.
func (m model) viewBottomDivider() string {
	if m.viewport.TotalLineCount() <= m.viewport.Height || m.viewport.AtBottom() {
		return m.viewDivider()
	}
	pct := int(m.viewport.ScrollPercent() * 100)
	cue := fmt.Sprintf(" ↓ %d%%  PgDn for latest ", pct)
	cueWidth := lipgloss.Width(cue)
	avail := m.width - cueWidth
	if avail < 2 {
		// Too narrow for a divider + cue: show just the cue.
		return dimStyle.Render(cue)
	}
	line := strings.Repeat("─", avail)
	return dimStyle.Render(line) + warnText.Render(cue)
}

// slashCommands is the static list offered for autocomplete when the input
// starts with "/".
var slashCommands = []string{
	"/model", "/models", "/tools", "/sessions", "/session", "/new", "/save",
	"/rename", "/resume", "/delete", "/delete-session", "/clear", "/exit",
	"/quit", "/compact", "/context", "/namespace", "/ns", "/help",
}

// slashCompletions returns the commands matching the input prefix, or nil
// when the input isn't a slash command.
func slashCompletions(input string) []string {
	if !strings.HasPrefix(input, "/") {
		return nil
	}
	var matches []string
	for _, c := range slashCommands {
		if strings.HasPrefix(c, input) {
			matches = append(matches, c)
		}
	}
	return matches
}

// helpText returns the on-demand reference printed to the transcript by the
// /help (and /?) command: the key bindings and slash commands, grouped for
// quick scanning. It's plain markdown so glamour renders it inline.
func helpText() string {
	return `## Keyboard shortcuts

| Key | Action |
| --- | --- |
| **Enter** | send message |
| **Ctrl+J** / **Alt+Enter** | insert newline |
| **Ctrl+P** | command palette |
| **↑ / ↓** | input history (or move cursor in multi-line) |
| **Shift+Tab** | toggle auto-accept mode |
| **Esc** | clear input / interrupt agent / decline prompt |
| **Ctrl+C** | quit |
| **Ctrl+L** | clear screen (scroll up to reveal) |
| **Ctrl+Y** | copy last reply |
| **Ctrl+O** | expand/collapse tool results |
| **PgUp / PgDn** | scroll transcript |
| **Tab** | autocomplete (commands, @files, /context, /namespace) |

## Slash commands

| Command | Action |
| --- | --- |
| **/sessions** | browse & resume past sessions |
| **/context** | switch kube context (Tab to autocomplete) |
| **/namespace** / **/ns** | switch namespace (Tab to autocomplete) |
| **/model** | switch model |
| **/rename** | rename this session |
| **/compact** | summarize & free context |
| **/clear** | clear the transcript |
| **/help** / **/?** | show this reference |
| **/exit** | quit |

Type **!command** to run a shell command, or **@path** to attach a file.
`
}

// lastToken returns the input's trailing whitespace-separated token
// (what an autocomplete would complete).
func (m model) lastToken() string {
	v := m.input.Value()
	if i := strings.LastIndexAny(v, " \t\n"); i >= 0 {
		return v[i+1:]
	}
	return v
}

// shellMode reports whether the input is a `!` shell escape command.
func (m model) shellMode() bool {
	return strings.HasPrefix(m.input.Value(), "!")
}

// completionHintVisible reports whether a hint line (slash completions,
// file mentions, context names, namespaces, or the shell marker) is shown
// under the input.
func (m *model) completionHintVisible() bool {
	return m.shellMode() ||
		len(slashCompletions(m.input.Value())) > 0 ||
		len(m.fileMatches()) > 0 ||
		len(m.contextMatches()) > 0 ||
		len(m.namespaceMatches()) > 0
}

// namespaceMatches returns cluster namespaces matching the partial after
// "/namespace " or "/ns " in the input, or nil otherwise. Names come from a
// live cluster query and are cached briefly.
func (m *model) namespaceMatches() []string {
	v := m.input.Value()
	var prefix string
	switch {
	case strings.HasPrefix(v, "/namespace "):
		prefix = strings.TrimPrefix(v, "/namespace ")
	case strings.HasPrefix(v, "/ns "):
		prefix = strings.TrimPrefix(v, "/ns ")
	default:
		return nil
	}
	if m.agent == nil {
		return nil
	}
	if time.Since(m.namespaceNamesLast) > 30*time.Second || m.namespaceNamesLast.IsZero() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		names, err := m.agent.ListNamespaces(ctx)
		if err != nil {
			return nil
		}
		m.namespaceNames = names
		m.namespaceNamesLast = time.Now()
	}
	var matches []string
	for _, n := range m.namespaceNames {
		if strings.HasPrefix(n, prefix) {
			matches = append(matches, n)
		}
	}
	return matches
}

// contextMatches returns kube context names matching the partial after
// "/context " in the input, or nil when the input isn't a context command.
// Names are cached briefly to avoid reloading the kubeconfig on every frame.
func (m *model) contextMatches() []string {
	v := m.input.Value()
	const cmd = "/context "
	if !strings.HasPrefix(v, cmd) {
		return nil
	}
	prefix := strings.TrimPrefix(v, cmd)
	if time.Since(m.contextNamesLast) > 10*time.Second || m.contextNamesLast.IsZero() {
		if m.agent == nil {
			return nil
		}
		names, err := kube.ListContexts(m.agent.Kubeconfig)
		if err != nil {
			return nil
		}
		m.contextNames = names
		m.contextNamesLast = time.Now()
	}
	var matches []string
	for _, n := range m.contextNames {
		if strings.HasPrefix(n, prefix) {
			matches = append(matches, n)
		}
	}
	return matches
}

// fileMatches returns filesystem path completions for the current `@`
// mention token (directories get a trailing "/" so completion can continue).
// Returns nil when the last token isn't a mention prefix.
func (m model) fileMatches() []string {
	tok := m.lastToken()
	if !strings.HasPrefix(tok, "@") || tok == "@" {
		return nil
	}
	prefix := tok[1:]
	if prefix == "" {
		return nil
	}
	glob, err := filepath.Glob(prefix + "*")
	if err != nil {
		return nil
	}
	var matches []string
	for _, g := range glob {
		if info, err := os.Stat(g); err == nil && info.IsDir() {
			g += "/"
		}
		matches = append(matches, g)
	}
	const maxMatches = 8
	if len(matches) > maxMatches {
		matches = matches[:maxMatches]
	}
	return matches
}

// completionHint renders the dim hint line shown under the input: the shell
// marker, context names, namespaces, file mention matches, or slash-command
// completions.
func (m *model) completionHint() string {
	if m.shellMode() {
		return dimStyle.Render("  shell command")
	}
	if matches := m.contextMatches(); len(matches) > 0 {
		hint := "  " + strings.Join(matches, "  ")
		return dimStyle.Render(truncateRunes(hint, max(m.input.Width(), 20)))
	}
	if matches := m.namespaceMatches(); len(matches) > 0 {
		hint := "  " + strings.Join(matches, "  ")
		return dimStyle.Render(truncateRunes(hint, max(m.input.Width(), 20)))
	}
	if matches := m.fileMatches(); len(matches) > 0 {
		hint := "  " + strings.Join(matches, "  ")
		return dimStyle.Render(truncateRunes(hint, max(m.input.Width(), 20)))
	}
	matches := slashCompletions(m.input.Value())
	if len(matches) == 0 {
		return ""
	}
	hint := "  " + strings.Join(matches, "  ")
	return dimStyle.Render(truncateRunes(hint, max(m.input.Width(), 20)))
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
		return lipgloss.NewStyle().Padding(0, 1).Render(m.runningBox().Width(m.width - 4).Render(content))
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
	// Slash-command completions, file mentions, or the shell marker are
	// hinted inside the input box.
	if hint := m.completionHint(); hint != "" {
		content += "\n" + hint
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

	return lipgloss.NewStyle().Padding(0, 1).Render(m.inputBox().Width(m.width - 4).Render(content))
}

// inputBox returns the input box style. The border is warning-colored when
// auto-accept mode is on (so it's unmistakable that commands will run without
// approval — like Claude Code's yellow auto border) or for `!` shell escape
// commands, primary otherwise.
func (m model) inputBox() lipgloss.Style {
	if m.shellMode() {
		return inputBox.BorderForeground(colorWarning)
	}
	if m.agent != nil && m.agent.SkipPermissionsEnabled() {
		return inputBox.BorderForeground(colorWarning)
	}
	return inputBox
}

// runningBox returns the dimmed box style shown while the agent is running.
// Like inputBox, it uses the warning border when auto-accept mode is on, so
// the "commands run without approval" cue stays visible during execution
// rather than reverting to a plain dim border.
func (m model) runningBox() lipgloss.Style {
	if m.agent != nil && m.agent.SkipPermissionsEnabled() {
		return inputBoxDim.BorderForeground(colorWarning)
	}
	return inputBoxDim
}

func (m model) viewHelp(state api.AgentState) string {
	// Each entry is a key hint. They are ordered by importance: the most
	// essential bindings come first, so as the terminal narrows the less
	// critical ones drop off and the bar never wraps into an ugly
	// multi-line mess.
	var hints []string
	switch {
	case m.browserOpen:
		return "" // the browser renders its own key hints
	case m.inChoiceMode:
		hints = []string{"↑/↓: navigate", "Enter: select", "Ctrl+C: quit"}
	case state == api.AgentStateRunning:
		hints = []string{"Ctrl+C: cancel", "Esc: interrupt"}
	default:
		hints = []string{
			"Enter: send", "Ctrl+P: commands", "↑/↓: history",
			"Ctrl+J: newline", "Ctrl+L: clear", "Ctrl+Y: copy",
			"Ctrl+O: expand", "Shift+Tab: auto", "Esc: clear/stop",
			"Ctrl+C: quit",
		}
		if m.viewport.TotalLineCount() > m.viewport.Height {
			hints = append(hints, "PgUp/PgDn: scroll")
		}
	}
	// Fit the hints to the available width, dropping the least-important
	// trailing ones when they would cause a wrap. The padding (left+right=4)
	// and the separators (" • ") are accounted for.
	sep := " • "
	avail := max(m.width-4, 0)
	fitted := fitHints(hints, sep, avail)
	if len(fitted) == 0 && len(hints) > 0 {
		// Even the first hint doesn't fit: show it truncated rather than an
		// empty bar, so the primary action is always discoverable.
		fitted = []string{truncateRunes(hints[0], max(avail, 1))}
	}
	return dimStyle.Padding(0, 2, 1, 2).Render(strings.Join(fitted, sep))
}

// fitHints greedily selects hints from the front of the slice so the joined
// string (with sep between entries) fits within width. It preserves priority
// order: the first/most-important hints are kept and trailing ones drop off.
func fitHints(hints []string, sep string, width int) []string {
	if width <= 0 || len(hints) == 0 {
		return nil
	}
	var out []string
	used := 0
	for _, h := range hints {
		extra := lipgloss.Width(h)
		if len(out) > 0 {
			extra += lipgloss.Width(sep)
		}
		if used+extra > width {
			break
		}
		used += extra
		out = append(out, h)
	}
	return out
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
