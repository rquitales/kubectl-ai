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
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/kube"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/mcp"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
	"github.com/google/uuid"
	"k8s.io/klog/v2"
)

//go:embed systemprompt_template_default.txt
var defaultSystemPromptTemplate string

type Agent struct {
	// Input is the channel to receive user input.
	Input chan any

	// Output is the channel to send messages to the UI.
	Output chan any

	// RunOnce indicates if the agent should run only once.
	// If true, the agent will run only once and then exit.
	// If false, the agent will run in a loop until the context is done.
	RunOnce bool

	// InitialQuery is the initial query to the agent.
	// If provided, the agent will run only once and then exit.
	InitialQuery string

	// tool calls that are pending execution
	// These will typically be all the tool calls suggested by the LLM in the
	// previous iteration of the agentic loop.
	pendingFunctionCalls []ToolCallAnalysis

	// currChatContent tracks chat content that needs to be sent
	// to the LLM in the current iteration of the agentic loop.
	currChatContent []any

	// currIteration tracks the current iteration of the agentic loop.
	currIteration int

	LLM gollm.Client

	// PromptTemplateFile allows specifying a custom template file
	PromptTemplateFile string
	// ExtraPromptPaths allows specifying additional prompt templates
	// to be combined with PromptTemplateFile
	ExtraPromptPaths []string
	Model            string
	Provider         string

	RemoveWorkDir bool

	MaxIterations int

	// Kubeconfig is the path to the kubeconfig file.
	Kubeconfig string
	// kubeconfigOverride is the session-scoped kubeconfig override path
	// (set by /context or /namespace), applied via KUBECONFIG for this
	// process without mutating the base file.
	kubeconfigOverride string
	// pinnedKubeconfig is a snapshot of the effective kubeconfig taken at
	// session start, so external edits to the global kubeconfig (e.g.
	// `kubectl config use-context` in another terminal) do not change this
	// session's context. Empty when no kubeconfig was available to pin.
	pinnedKubeconfig string
	// Sandbox indicates whether to execute tools in a sandbox environment
	Sandbox string

	// SandboxImage is the container image to use for the sandbox
	SandboxImage string

	SkipPermissions bool

	// allowedTools is the in-memory set of tools the user chose to always
	// allow (via the "Always allow <tool>" permission option), keyed by tool
	// name. It is per-process and not persisted.
	allowedTools map[string]bool

	Tools tools.Tools

	EnableToolUseShim bool

	// MCPClientEnabled indicates whether MCP client mode is enabled
	MCPClientEnabled bool

	// Recorder captures events for diagnostics
	Recorder journal.Recorder

	llmChat gollm.Chat

	workDir string

	// executor is the executor for tool execution
	Executor sandbox.Executor

	// Session optionally provides a session to use.
	// This is used by the UI to track the state of the agent and the conversation.
	Session *api.Session

	// protects session from concurrent access
	sessionMu sync.Mutex

	// cached list of available models
	availableModels []string

	// mcpManager manages MCP client connections
	mcpManager *mcp.Manager

	// ChatMessageStore is the underlying session persistence layer.
	ChatMessageStore api.ChatMessageStore

	// SessionBackend is the configured backend for session persistence (e.g., memory, filesystem).
	SessionBackend string

	// lastErr is the most recent error run into, for use across the stack
	lastErr error

	// pendingModelChoice, when non-nil, holds the model IDs offered by an
	// open /model picker; the next UserChoiceResponse selects among them.
	pendingModelChoice []string

	// pendingIterationContinue is true while a max-iterations continue/stop
	// choice request is outstanding; the next UserChoiceResponse resolves it.
	pendingIterationContinue bool

	// titleAttempted ensures an LLM-generated session title is requested at
	// most once per agent lifetime.
	titleAttempted bool

	// titleGenerated is set once an LLM-generated title was applied to the
	// session; exit-time content-derived naming must not override it.
	titleGenerated bool

	// runCtx is the context of the current agentic run (a fresh one is
	// created per user query); runCancel interrupts it. The parent ctx
	// governs the agent's whole lifetime; runCtx governs a single run so
	// the user can interrupt a run without killing the agent.
	runCtx    context.Context
	runCancel context.CancelFunc

	// cancel is the function to cancel the agent's context
	cancel context.CancelFunc
}

// startRun begins a new interruptible run context derived from ctx.
func (c *Agent) StartRun(ctx context.Context) context.Context {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.runCancel != nil {
		c.runCancel()
	}
	c.runCtx, c.runCancel = context.WithCancel(ctx)
	return c.runCtx
}

// endRun cancels and clears the current run context.
func (c *Agent) endRun() {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.runCancel != nil {
		c.runCancel()
		c.runCancel = nil
	}
}

// CancelRun interrupts the currently running agentic run (e.g. the user
// pressed Esc). It returns false when no run is in progress. Safe to call
// from UI goroutines.
func (c *Agent) CancelRun() bool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.runCancel == nil {
		return false
	}
	c.runCancel()
	return true
}

// interruptRequested reports whether err results from a CancelRun interrupt
// (as opposed to the parent context being done or a real failure).
func (c *Agent) interruptRequested(err error) bool {
	if !errors.Is(err, context.Canceled) {
		return false
	}
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	return c.runCtx != nil && c.runCtx.Err() != nil
}

// Assert InMemoryChatStore implements ChatMessageStore
var _ api.ChatMessageStore = &sessions.InMemoryChatStore{}

func (s *Agent) GetSession() *api.Session {
	s.sessionMu.Lock()
	defer s.sessionMu.Unlock()

	// Create a shallow copy of the session struct. The Messages slice header
	// is also copied, providing the caller with a snapshot of the messages
	// at this point in time. The UI should treat the messages as read-only
	// to avoid race conditions.
	sessionCopy := *s.Session
	return &sessionCopy
}

// addMessage creates a new message, adds it to the session, and sends it to the output channel
func (c *Agent) addMessage(source api.MessageSource, messageType api.MessageType, payload any) *api.Message {
	return c.addMessageWithID(uuid.New().String(), source, messageType, payload)
}

// addMessageWithID is addMessage with a caller-chosen message ID. It is used
// for the final text of a streamed iteration so it carries the same ID as the
// live text-delta messages that preceded it (UIs then replace the streaming
// entry in place).
func (c *Agent) addMessageWithID(id string, source api.MessageSource, messageType api.MessageType, payload any) *api.Message {
	return c.sendMessage(&api.Message{
		ID:        id,
		Source:    source,
		Type:      messageType,
		Payload:   payload,
		Timestamp: time.Now(),
	})
}

// addModelTextMessage adds a model text message to the session, recording the
// token usage reported by the provider (0 when none was reported).
func (c *Agent) addModelTextMessage(text string, tokens int) *api.Message {
	return c.sendMessage(&api.Message{
		ID:        uuid.New().String(),
		Source:    api.MessageSourceModel,
		Type:      api.MessageTypeText,
		Payload:   text,
		Timestamp: time.Now(),
		Tokens:    tokens,
	})
}

// sendMessage adds a fully-formed message to the session and sends it to the
// output channel.
func (c *Agent) sendMessage(message *api.Message) *api.Message {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()

	// Don't store UI control signals - they're not part of the conversation
	if message.Type != api.MessageTypeUserInputRequest {
		c.Session.ChatMessageStore.AddChatMessage(message)
		c.Session.LastModified = time.Now()
	}
	c.Output <- message
	return message
}

// setAgentState updates the agent state and ensures LastModified is updated
func (c *Agent) setAgentState(newState api.AgentState) {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	currentState := c.agentState()
	if currentState != newState {
		klog.Infof("Agent state changing from %s to %s", currentState, newState)
		c.Session.AgentState = newState
		c.Session.LastModified = time.Now()
	}
}
func (c *Agent) AgentState() api.AgentState {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	return c.agentState()
}

// agentState returns the agent state without locking.
// The caller is responsible for locking.
func (c *Agent) agentState() api.AgentState {
	return c.Session.AgentState
}

func (s *Agent) Init(ctx context.Context) error {
	log := klog.FromContext(ctx)

	s.Input = make(chan any, 10)
	s.Output = make(chan any, 64)
	s.currIteration = 0
	s.allowedTools = map[string]bool{}
	// when we support session, we will need to initialize this with the
	// current history of the conversation.
	s.currChatContent = []any{}

	if s.InitialQuery == "" && s.RunOnce {
		return fmt.Errorf("RunOnce mode requires an initial query to be provided")
	}

	if s.Session != nil {
		if s.Session.ChatMessageStore == nil {
			s.Session.ChatMessageStore = sessions.NewInMemoryChatStore()
		}
		s.ChatMessageStore = s.Session.ChatMessageStore
		if s.Session.ID == "" {
			s.Session.ID = uuid.New().String()
		}
		if s.Session.CreatedAt.IsZero() {
			s.Session.CreatedAt = time.Now()
		}
		if s.Session.LastModified.IsZero() {
			s.Session.LastModified = time.Now()
		}
		s.Session.Messages = s.Session.ChatMessageStore.ChatMessages()

	} else {
		return fmt.Errorf("agent requires a session to be provided")
	}

	// Create a temporary working directory
	workDir, err := os.MkdirTemp("", "agent-workdir-*")
	if err != nil {
		log.Error(err, "Failed to create temporary working directory")
		return err
	}

	log.Info("Created temporary working directory", "workDir", workDir)

	switch s.Sandbox {
	case "k8s":
		sandboxName := fmt.Sprintf("kubectl-ai-sandbox-%s", uuid.New().String()[:8])

		// Use default image if not specified
		sandboxImage := s.SandboxImage
		if sandboxImage == "" {
			sandboxImage = "bitnami/kubectl:latest"
		}

		// Create sandbox with kubeconfig
		sb, err := sandbox.NewKubernetesSandbox(sandboxName,
			sandbox.WithKubeconfig(s.Kubeconfig),
			sandbox.WithImage(sandboxImage),
		)
		if err != nil {
			return fmt.Errorf("failed to create sandbox: %w", err)
		}

		s.Executor = sb
		log.Info("Created sandbox", "name", sandboxName, "image", sandboxImage)

	case "seatbelt":
		if runtime.GOOS != "darwin" {
			return fmt.Errorf("seatbelt sandbox is only supported on macOS")
		}
		s.Executor = sandbox.NewSeatbeltExecutor()
		log.Info("Using Seatbelt executor")

	case "":
		// No sandbox, use local executor
		s.Executor = sandbox.NewLocalExecutor()

	default:
		return fmt.Errorf("unknown sandbox type: %s", s.Sandbox)
	}

	s.workDir = workDir

	// Pin a session-scoped snapshot of the kubeconfig so external edits to
	// the global config do not leak into this session.
	s.pinSessionKubeconfig()

	// Register tools with executor if none registered yet
	// We clone existing tools (e.g. custom tools) to ensure we have a fresh map
	// This avoids polluting the global default tools and ensures thread safety.
	s.Tools = s.Tools.CloneWithExecutor(s.Executor)

	s.Tools.RegisterTool(tools.NewBashTool(s.Executor))
	s.Tools.RegisterTool(tools.NewKubectlTool(s.Executor))

	// MCP tools must be registered BEFORE rebuildChat so the chat's
	// function definitions include them.
	if s.MCPClientEnabled {
		if err := s.InitializeMCPClient(ctx); err != nil {
			klog.Errorf("Failed to initialize MCP client: %v", err)
			return fmt.Errorf("failed to initialize MCP client: %w", err)
		}

		// Update MCP status in session
		if err := s.UpdateMCPStatus(ctx, s.MCPClientEnabled); err != nil {
			klog.Warningf("Failed to update MCP status: %v", err)
		}
	}

	if err := s.rebuildChat(ctx); err != nil {
		return err
	}

	return nil
}

// rebuildChat (re)creates the LLM chat for the current model with the full
// system prompt and the session's conversation history. Used at Init time
// and when switching models mid-session.
func (s *Agent) rebuildChat(ctx context.Context) error {
	systemPrompt, err := s.generatePrompt(ctx, defaultSystemPromptTemplate, PromptData{
		Tools:             s.Tools,
		EnableToolUseShim: s.EnableToolUseShim,
		// RunOnce is a good proxy to indicate the agentic session is non-interactive mode.
		SessionIsInteractive: !s.RunOnce,
	})
	if err != nil {
		return fmt.Errorf("generating system prompt: %w", err)
	}

	// Start a new chat session
	s.llmChat = gollm.NewRetryChat(
		s.LLM.StartChat(systemPrompt, s.Model),
		gollm.RetryConfig{
			MaxAttempts:    3,
			InitialBackoff: 10 * time.Second,
			MaxBackoff:     60 * time.Second,
			BackoffFactor:  2,
			Jitter:         true,
		},
	)
	if err := s.llmChat.Initialize(s.Session.ChatMessageStore.ChatMessages()); err != nil {
		return fmt.Errorf("initializing chat session: %w", err)
	}

	if !s.EnableToolUseShim {
		var functionDefinitions []*gollm.FunctionDefinition
		for _, tool := range s.Tools.AllTools() {
			functionDefinitions = append(functionDefinitions, tool.FunctionDefinition())
		}
		// Sort function definitions to help KV cache reuse
		sort.Slice(functionDefinitions, func(i, j int) bool {
			return functionDefinitions[i].Name < functionDefinitions[j].Name
		})
		if err := s.llmChat.SetFunctionDefinitions(functionDefinitions); err != nil {
			return fmt.Errorf("setting function definitions: %w", err)
		}
	}

	return nil
}

// streamDeltaInterval is the minimum spacing between live text-delta
// emissions. Fast providers can produce a chunk every few milliseconds;
// unthrottled, that backpressures the buffered UI output channel and stalls
// the whole run (the TUI renders a frame per message received).
const streamDeltaInterval = 150 * time.Millisecond

// deriveSessionName builds a short session name from the first real user
// message (skipping slash commands and bare meta commands), so the name
// says what the session is actually about. Returns "" when there's nothing
// to derive from.
func deriveSessionName(messages []*api.Message) string {
	metaWords := map[string]bool{
		"clear": true, "reset": true, "exit": true, "quit": true,
		"session": true, "sessions": true, "tools": true, "model": true,
		"models": true, "new-session": true, "save-session": true,
		"compact": true,
	}
	for _, msg := range messages {
		if msg.Source != api.MessageSourceUser || msg.Type != api.MessageTypeText {
			continue
		}
		p, ok := msg.Payload.(string)
		if !ok {
			continue
		}
		p = strings.Join(strings.Fields(p), " ") // collapse whitespace/newlines
		if p == "" || strings.HasPrefix(p, "/") || metaWords[strings.ToLower(p)] {
			continue
		}
		p = sessions.SanitizeSessionName(p)
		const maxLen = 48
		if r := []rune(p); len(r) > maxLen {
			p = string(r[:maxLen]) + "…"
		}
		return p
	}
	return ""
}

// llmTitleGenerated reports whether an LLM-generated title was applied to
// the session. Safe to call from any goroutine.
func (c *Agent) llmTitleGenerated() bool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	return c.titleGenerated
}

func (c *Agent) Close() error {
	// Exit checks for the session: delete it if empty, otherwise give it a
	// content-derived name (unless the user named it manually).
	if c.Session != nil {
		messages := c.Session.AllMessages()
		if !sessions.HasConversationMessages(messages) {
			if manager, err := sessions.NewSessionManager(c.SessionBackend); err == nil {
				if err := manager.DeleteSession(c.Session.ID); err != nil {
					klog.Warningf("failed to delete empty session %s: %v", c.Session.ID, err)
				} else {
					klog.Infof("Deleted empty session %s on exit", c.Session.ID)
				}
			}
		} else if !c.Session.ManuallyNamed && !c.llmTitleGenerated() {
			if name := deriveSessionName(messages); name != "" && name != c.Session.Name {
				if manager, err := sessions.NewSessionManager(c.SessionBackend); err == nil {
					if err := manager.SetSessionName(c.Session.ID, name, false); err != nil {
						klog.Warningf("failed to auto-name session %s: %v", c.Session.ID, err)
					} else {
						klog.Infof("Named session %s as %q on exit", c.Session.ID, name)
					}
				}
			}
		}
	}

	if c.workDir != "" {
		if c.RemoveWorkDir {
			if err := os.RemoveAll(c.workDir); err != nil {
				klog.Warningf("error cleaning up directory %q: %v", c.workDir, err)
			}
		}
	}
	// Close MCP client connections
	if err := c.CloseMCPClient(); err != nil {
		klog.Warningf("error closing MCP client: %v", err)
	}

	// Close sandbox if enabled
	// Close executor if it exists
	if c.Executor != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := c.Executor.Close(ctx); err != nil {
			klog.Warningf("error cleaning up executor: %v", err)
		} else {
			klog.Info("Executor cleaned up successfully")
		}
	}
	// Cancel the agent's context
	if c.cancel != nil {
		c.cancel()
	}
	// Close the LLM client
	if c.LLM != nil {
		if err := c.LLM.Close(); err != nil {
			klog.Warningf("error closing LLM client: %v", err)
		}
	}
	return nil
}

func (c *Agent) LastErr() error {
	return c.lastErr
}

func (c *Agent) Run(ctx context.Context, initialQuery string) error {
	log := klog.FromContext(ctx)

	if c.Recorder != nil {
		ctx = journal.ContextWithRecorder(ctx, c.Recorder)
	}

	// Save unexpected error and return it in for RunOnce mode
	log.Info("Starting agent loop", "initialQuery", initialQuery, "runOnce", c.RunOnce)
	go func() {
		runCtx := ctx
		// If initialQuery is empty, try to use the one from the struct
		if initialQuery == "" {
			initialQuery = c.InitialQuery
		}

		if initialQuery != "" {
			c.addMessage(api.MessageSourceUser, api.MessageTypeText, initialQuery)
			answer, handled, err := c.handleMetaQuery(ctx, initialQuery)
			if err != nil {
				log.Error(err, "error handling meta query")
				c.setAgentState(api.AgentStateDone)
				c.pendingFunctionCalls = []ToolCallAnalysis{}
				c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
			} else if handled {
				// initialQuery is the 'exit' or 'quit' metaquery
				if c.AgentState() == api.AgentStateExited {
					c.addMessage(api.MessageSourceAgent, api.MessageTypeText, answer)
					close(c.Output)
					return
				}
				// we handled the meta query, so we don't need to run the agentic loop
				c.setAgentState(api.AgentStateDone)
				c.pendingFunctionCalls = []ToolCallAnalysis{}
				c.addMessage(api.MessageSourceAgent, api.MessageTypeText, answer)
			} else {
				// Start the agentic loop with the initial query
				c.setAgentState(api.AgentStateRunning)
				runCtx = c.StartRun(ctx)
				c.currIteration = 0
				c.currChatContent = []any{initialQuery}
				c.pendingFunctionCalls = []ToolCallAnalysis{}
			}
		}
		c.lastErr = nil
		for {
			var userInput any
			log.Info("Agent loop iteration", "state", c.AgentState())
			switch c.AgentState() {
			case api.AgentStateIdle, api.AgentStateDone:
				// In RunOnce mode, we are done, so exit
				if c.RunOnce {
					log.Info("RunOnce mode, exiting agent loop")
					c.setAgentState(api.AgentStateExited)
					return
				}
				log.Info("initiating user input")
				c.addMessage(api.MessageSourceAgent, api.MessageTypeUserInputRequest, ">>>")
				select {
				case <-ctx.Done():
					log.Info("Agent loop done")
					return
				case userInput = <-c.Input:
					log.Info("Received input from channel", "userInput", userInput)
					if userInput == io.EOF {
						log.Info("Agent loop done, EOF received")
						c.setAgentState(api.AgentStateExited)
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "It has been a pleasure assisting you. Have a great day!")
						return
					}

					if sessionPickerResp, ok := userInput.(*api.SessionPickerResponse); ok {
						if sessionPickerResp.Cancelled {
							continue
						}
						if err := c.LoadSession(sessionPickerResp.SessionID); err != nil {
							log.Error(err, "error loading session")
							c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error loading session: "+err.Error())
						} else {
							c.addMessage(api.MessageSourceAgent, api.MessageTypeText, fmt.Sprintf("Switched to session %s", sessionPickerResp.SessionID))
						}
						continue
					}

					if _, ok := userInput.(*api.NewSessionRequest); ok {
						if err := c.createAndSwitchSession(); err != nil {
							log.Error(err, "error creating new session")
							c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error creating new session: "+err.Error())
						}
						continue
					}

					query, ok := userInput.(*api.UserInputResponse)
					if !ok {
						// Ignore unexpected input rather than dying: a dead loop
						// leaves every UI producer blocked on the input channel.
						log.Error(nil, "Received unexpected input from channel; ignoring", "userInput", userInput)
						continue
					}
					if strings.TrimSpace(query.Query) == "" {
						log.Info("No query provided, skipping agentic loop")
						continue
					}
					// Shell escape: "!<command>" runs locally via the executor
					// instead of being sent to the LLM.
					if command, ok := shellEscapeCommand(query.Query); ok {
						c.addMessage(api.MessageSourceUser, api.MessageTypeText, query.Query)
						c.runShellEscape(ctx, command)
						continue
					}
					// Inline @file mentions: the transcript shows compact
					// "[+path]" chips, while the LLM gets the file contents.
					llmQuery, attachments := expandFileMentions(query.Query)
					c.addMessage(api.MessageSourceUser, api.MessageTypeText, displayQueryWithChips(query.Query, attachments))
					// we don't need the agentic loop for meta queries
					// for ex. model, tools, etc.
					answer, handled, err := c.handleMetaQuery(ctx, query.Query)
					if err != nil {
						log.Error(err, "error handling meta query")
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}
						c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
						continue
					}
					if handled {
						// metaquery set the state to 'Exited', so we should exit
						if c.AgentState() == api.AgentStateExited {
							c.addMessage(api.MessageSourceAgent, api.MessageTypeText, answer)
							close(c.Output)
							return
						}
						// metaquery set up an interactive picker, wait for response
						if c.AgentState() == api.AgentStateWaitingForInput {
							continue
						}
						// we handled the meta query, so we don't need to run the agentic loop
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}
						if answer != "" {
							c.addMessage(api.MessageSourceAgent, api.MessageTypeText, answer)
						}
						continue
					}

					c.setAgentState(api.AgentStateRunning)
					runCtx = c.StartRun(ctx)
					c.currIteration = 0
					// Preserve any pending observations (e.g. shell escape
					// output) so the LLM sees them with this query.
					c.currChatContent = append(c.currChatContent, llmQuery)
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					log.Info("Set agent state to running, will process agentic loop", "currIteration", c.currIteration, "currChatContent", len(c.currChatContent))
				}
			case api.AgentStateWaitingForInput:
				// In RunOnce mode, if we need user choice, exit with error
				if c.RunOnce {
					log.Error(nil, "RunOnce mode cannot handle user choice requests")
					c.setAgentState(api.AgentStateExited)
					c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: RunOnce mode cannot handle user choice requests")
					return
				}
				select {
				case <-ctx.Done():
					log.Info("Agent loop done")
					return
				case userInput = <-c.Input:
					if userInput == io.EOF {
						log.Info("Agent loop done, EOF received")
						c.setAgentState(api.AgentStateExited)
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "It has been a pleasure assisting you. Have a great day!")
						return
					}

					switch response := userInput.(type) {
					case *api.SessionPickerResponse:
						if response.Cancelled {
							c.setAgentState(api.AgentStateDone)
							continue
						}
						if err := c.LoadSession(response.SessionID); err != nil {
							log.Error(err, "error loading session")
							c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error loading session: "+err.Error())
						} else {
							c.addMessage(api.MessageSourceAgent, api.MessageTypeText, fmt.Sprintf("Switched to session %s", response.SessionID))
						}
						c.setAgentState(api.AgentStateDone)
						continue

					case *api.NewSessionRequest:
						if err := c.createAndSwitchSession(); err != nil {
							log.Error(err, "error creating new session")
							c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error creating new session: "+err.Error())
						}
						c.setAgentState(api.AgentStateDone)
						continue

					case *api.UserChoiceResponse:
						if c.pendingIterationContinue {
							c.pendingIterationContinue = false
							if response.Choice == 1 {
								// Continue: reset the budget and resume the
								// turn; tool results already in
								// currChatContent carry the context forward.
								c.currIteration = 0
								c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "Continuing (iteration budget reset).")
								c.setAgentState(api.AgentStateRunning)
							} else {
								// Stop (2), Esc/decline (3), or anything else.
								c.setAgentState(api.AgentStateDone)
								c.pendingFunctionCalls = []ToolCallAnalysis{}
								c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "Stopped at the iteration limit.")
								c.endRun()
							}
							continue
						}
						if c.pendingModelChoice != nil {
							c.handleModelChoice(ctx, response)
							continue
						}
						dispatchToolCalls := c.handleChoice(runCtx, response)
						if dispatchToolCalls {
							if err := c.DispatchToolCalls(runCtx); err != nil {
								log.Error(err, "error dispatching tool calls")
								c.setAgentState(api.AgentStateDone)
								c.pendingFunctionCalls = []ToolCallAnalysis{}
								c.Session.LastModified = time.Now()
								if c.interruptRequested(err) {
									c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "⚠ Interrupted.")
								} else {
									c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
								}
								// In RunOnce mode, exit on tool execution error
								if c.RunOnce {
									c.setAgentState(api.AgentStateExited)
									c.lastErr = err
									return
								}
								continue
							}
							// Clear pending function calls after execution
							c.pendingFunctionCalls = []ToolCallAnalysis{}
							c.setAgentState(api.AgentStateRunning)
							c.currIteration = c.currIteration + 1
						} else {
							// if user has declined, we are done with this iteration
							c.currIteration = c.currIteration + 1
							c.pendingFunctionCalls = []ToolCallAnalysis{}
							c.setAgentState(api.AgentStateRunning)
							c.Session.LastModified = time.Now()
						}

					default:
						// Ignore unexpected input rather than dying: a dead loop
						// leaves every UI producer blocked on the input channel.
						log.Error(nil, "Received unexpected input from channel; ignoring", "userInput", userInput)
						continue
					}
				}
			case api.AgentStateRunning:
				// Agent is running, don't wait for input, just continue to process the agentic loop
				log.Info("Agent is in running state, processing agentic loop")
			case api.AgentStateExited:
				log.Info("Agent exited in RunOnce mode")
				return
			}

			if c.AgentState() == api.AgentStateRunning {
				log.Info("Processing agentic loop", "currIteration", c.currIteration, "maxIterations", c.MaxIterations, "currChatContentLen", len(c.currChatContent))

				if c.currIteration >= c.MaxIterations {
					if c.RunOnce {
						// No interactive prompt is available in RunOnce mode.
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "Maximum number of iterations reached.")
						c.endRun()
						continue
					}
					// Ask instead of dying: long debug sessions legitimately
					// need more iterations, while the prompt still catches
					// runaway loops. The run context stays alive so the turn
					// can resume exactly where it hit the limit.
					c.pendingIterationContinue = true
					c.setAgentState(api.AgentStateWaitingForInput)
					c.addMessage(api.MessageSourceAgent, api.MessageTypeUserChoiceRequest, &api.UserChoiceRequest{
						Prompt: fmt.Sprintf("Reached the maximum of %d iterations for this turn.\n\nContinue working on it?", c.MaxIterations),
						Options: []api.UserChoiceOption{
							{Value: "continue", Label: "Continue"},
							{Value: "stop", Label: "Stop"},
						},
						Kind: "continue",
					})
					continue
				}

				// we run the agentic loop for one iteration
				stream, err := c.llmChat.SendStreaming(runCtx, c.currChatContent...)
				if err != nil {
					log.Error(err, "error sending streaming LLM response")
					c.setAgentState(api.AgentStateDone)
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					if c.interruptRequested(err) {
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "⚠ Interrupted.")
					} else {
						c.lastErr = err
						c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
					}
					continue
				}

				// Clear our "response" now that we sent the last response
				c.currChatContent = nil

				if c.EnableToolUseShim {
					// convert the candidate response into a gollm.ChatResponse
					stream, err = candidateToShimCandidate(stream)
					if err != nil {
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}

						// In RunOnce mode, exit on shim conversion error
						if c.RunOnce {
							c.setAgentState(api.AgentStateExited)
							return
						}

						continue
					}
				}
				// Process each part of the response
				var functionCalls []gollm.FunctionCall

				// accumulator for streamed text
				var streamedText string
				// accumulator for streamed model reasoning/thinking
				var streamedThinking string
				var llmError error

				// lastUsage holds the usage metadata of the most recent chunk
				// that reported any (providers report cumulative totals, so the
				// last one wins).
				var lastUsage any

				// All live text-delta messages of this iteration and the final
				// text message share one ID, so UIs can update the streaming
				// entry in place and finally replace it with the stored message.
				streamID := uuid.New().String()
				// Live thinking-delta messages share their own ID, separate from
				// the text stream, so the UI can render a distinct collapsible
				// thinking block alongside the streaming text.
				thinkStreamID := uuid.New().String()
				lastDeltaEmit := time.Time{}

				for response, err := range stream {
					if err != nil {
						log.Error(err, "error reading streaming LLM response")
						llmError = err
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}
						c.lastErr = llmError
						break
					}
					if response == nil {
						// end of streaming response
						break
					}
					// klog.Infof("response: %+v", response)

					if metadata := response.UsageMetadata(); metadata != nil {
						lastUsage = metadata
					}

					if len(response.Candidates()) == 0 {
						llmError = fmt.Errorf("no candidates in response")
						log.Error(nil, "No candidates in response")
						c.setAgentState(api.AgentStateDone)
						c.pendingFunctionCalls = []ToolCallAnalysis{}
						break
					}

					candidate := response.Candidates()[0]

					for _, part := range candidate.Parts() {
						// Check if it's a text response
						if text, ok := part.AsText(); ok {
							log.Info("text response", "text", text)
							streamedText += text
							// Stream the accumulated text live, throttled to
							// streamDeltaInterval so a fast provider can't
							// backpressure the UI channel and stall the run.
							// Deltas are ephemeral: they go straight to the
							// output channel and are never stored.
							if time.Since(lastDeltaEmit) >= streamDeltaInterval {
								lastDeltaEmit = time.Now()
								c.Output <- &api.Message{
									ID:        streamID,
									Source:    api.MessageSourceModel,
									Type:      api.MessageTypeTextDelta,
									Payload:   streamedText,
									Timestamp: time.Now(),
								}
							}
						}

						// Check if it's the model's reasoning/thinking.
						if thinking, ok := part.AsThinking(); ok {
							log.Info("thinking response", "thinkingLen", len(thinking))
							streamedThinking += thinking
							// Stream thinking live (throttled like text deltas) so
							// the UI can show a collapsible thinking block updating
							// in real time. Thinking deltas are ephemeral: they go
							// to the output channel and are never stored.
							if time.Since(lastDeltaEmit) >= streamDeltaInterval {
								lastDeltaEmit = time.Now()
								c.Output <- &api.Message{
									ID:        thinkStreamID,
									Source:    api.MessageSourceModel,
									Type:      api.MessageTypeThinkingDelta,
									Payload:   streamedThinking,
									Timestamp: time.Now(),
								}
							}
						}

						// Check if it's a function call
						if calls, ok := part.AsFunctionCalls(); ok && len(calls) > 0 {
							log.Info("function calls", "calls", calls)
							functionCalls = append(functionCalls, calls...)
						}
					}
				}
				if llmError != nil {
					log.Error(llmError, "error streaming LLM response")
					c.setAgentState(api.AgentStateDone)
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					if c.interruptRequested(llmError) {
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "⚠ Interrupted.")
					} else {
						c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+llmError.Error())
						c.lastErr = llmError
					}
					continue
				}

				log.Info("streamedText", "streamedText", streamedText)

				if streamedText != "" {
					// The final text message reuses the stream ID so it replaces
					// the live delta entry in UIs, and carries token usage.
					c.sendMessage(&api.Message{
						ID:        streamID,
						Source:    api.MessageSourceModel,
						Type:      api.MessageTypeText,
						Payload:   streamedText,
						Timestamp: time.Now(),
						Tokens:    usageTotalTokens(lastUsage),
					})
					c.maybeGenerateSessionTitle()
				}
				// The final thinking message replaces the live thinking-delta
				// entry in UIs. It is not stored (reasoning is ephemeral and is
				// already kept in the provider's history via the accumulator).
				if streamedThinking != "" {
					c.Output <- &api.Message{
						ID:        thinkStreamID,
						Source:    api.MessageSourceModel,
						Type:      api.MessageTypeThinking,
						Payload:   streamedThinking,
						Timestamp: time.Now(),
					}
				}
				// If no function calls to be made, we're done
				if len(functionCalls) == 0 {
					log.Info("No function calls to be made, so most likely the task is completed, so we're done.")
					c.setAgentState(api.AgentStateDone)
					c.endRun()
					c.currChatContent = []any{}
					c.currIteration = 0
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					log.Info("Agent task completed, transitioning to done state")
					if streamedText == "" {
						// If no tool calls to be made and we do not have a response from the LLM
						// we should let the user know for better diagnostics.
						// IMPORTANT: This also prevents UIs from getting blocked on reading from the output channel.
						log.Info("Empty response with no tool calls from LLM.")
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "Empty response from LLM")
					}
					continue
				}

				toolCallAnalysisResults, err := c.analyzeToolCalls(ctx, functionCalls)
				if err != nil {
					log.Error(err, "error analyzing tool calls")
					c.setAgentState(api.AgentStateDone)
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					c.Session.LastModified = time.Now()
					c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
					c.lastErr = err
					continue
				}

				// mark the tools for dispatching
				c.pendingFunctionCalls = toolCallAnalysisResults

				interactiveToolCallIndex := -1
				modifiesResourceToolCallIndex := -1
				for i, result := range toolCallAnalysisResults {
					if result.ModifiesResourceStr != "no" {
						modifiesResourceToolCallIndex = i
					}
					if result.IsInteractive {
						interactiveToolCallIndex = i
					}
				}

				if interactiveToolCallIndex >= 0 {
					// Show error block for both shim enabled and disabled modes
					errorMessage := fmt.Sprintf("  %s\n", toolCallAnalysisResults[interactiveToolCallIndex].IsInteractiveError.Error())
					c.addMessage(api.MessageSourceAgent, api.MessageTypeError, errorMessage)

					if c.EnableToolUseShim {
						// Add the error as an observation
						observation := fmt.Sprintf("Result of running %q:\n%v",
							toolCallAnalysisResults[interactiveToolCallIndex].FunctionCall.Name,
							toolCallAnalysisResults[interactiveToolCallIndex].IsInteractiveError.Error())
						c.currChatContent = append(c.currChatContent, observation)
					} else {
						// For models with tool-use support (shim disabled), use proper FunctionCallResult
						// Note: This assumes the model supports sending FunctionCallResult
						c.currChatContent = append(c.currChatContent, gollm.FunctionCallResult{
							ID:     toolCallAnalysisResults[interactiveToolCallIndex].FunctionCall.ID,
							Name:   toolCallAnalysisResults[interactiveToolCallIndex].FunctionCall.Name,
							Result: map[string]any{"error": toolCallAnalysisResults[interactiveToolCallIndex].IsInteractiveError.Error()},
						})
					}
					c.pendingFunctionCalls = []ToolCallAnalysis{} // reset pending function calls
					c.currIteration = c.currIteration + 1
					continue // Skip execution for interactive commands
				}

				if !c.SkipPermissionsEnabled() && modifiesResourceToolCallIndex >= 0 && !c.modifyingCallsAllowed(toolCallAnalysisResults) {
					// In RunOnce mode, exit with error if permission is required
					if c.RunOnce {
						var commandDescriptions []string
						for _, call := range c.pendingFunctionCalls {
							commandDescriptions = append(commandDescriptions, call.ParsedToolCall.Description())
						}
						errorMessage := "RunOnce mode cannot handle permission requests. The following commands require approval:\n* " + strings.Join(commandDescriptions, "\n* ")
						errorMessage += "\nUse --skip-permissions flag to bypass permission checks in RunOnce mode."

						log.Error(nil, "RunOnce mode cannot handle permission requests", "commands", commandDescriptions)
						c.setAgentState(api.AgentStateExited)
						c.addMessage(api.MessageSourceAgent, api.MessageTypeError, errorMessage)
						c.lastErr = fmt.Errorf("%s", errorMessage)
						return
					}

					choiceRequest := permissionChoiceRequest(c.pendingFunctionCalls)
					c.setAgentState(api.AgentStateWaitingForInput)
					c.addMessage(api.MessageSourceAgent, api.MessageTypeUserChoiceRequest, choiceRequest)
					// Request input from the user by sending a message on the output channel.
					// Remaining part of the loop will be now resumed when we receive a choice input
					// from the user.
					continue
				}

				// we are here means we are in the clear to dispatch the tool calls
				if err := c.DispatchToolCalls(runCtx); err != nil {
					log.Error(err, "error dispatching tool calls")
					c.setAgentState(api.AgentStateDone)
					c.pendingFunctionCalls = []ToolCallAnalysis{}
					c.Session.LastModified = time.Now()
					if c.interruptRequested(err) {
						c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "⚠ Interrupted.")
					} else {
						c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: "+err.Error())
						c.lastErr = err
					}
					continue
				}
				c.currIteration = c.currIteration + 1
				c.pendingFunctionCalls = []ToolCallAnalysis{}
				log.Info("Tool calls dispatched successfully", "currIteration", c.currIteration, "currChatContentLen", len(c.currChatContent), "agentState", c.AgentState())
			}
		}
	}()

	return nil
}

// slashCommands maps slash-command names (without the leading slash) to
// their canonical meta query form.
var slashCommands = map[string]string{
	"clear":     "clear",
	"reset":     "reset",
	"compact":   "compact",
	"exit":      "exit",
	"quit":      "quit",
	"model":     "model",
	"models":    "models",
	"tools":     "tools",
	"session":   "session",
	"sessions":  "sessions",
	"new":       "new-session",
	"save":      "save-session",
	"rename":    "rename-session",
	"resume":    "resume-session",
	"delete":    "delete-session",
	"context":   "context",
	"contexts":  "context",
	"namespace": "namespace",
	"ns":        "namespace",
}

// normalizeSlashCommand translates a slash-prefixed command (e.g. "/rename
// my session") into the canonical meta query form ("rename-session my
// session"). ok is false when the command name is not recognized.
func normalizeSlashCommand(query string) (normalized string, ok bool) {
	body := strings.TrimSpace(strings.TrimPrefix(query, "/"))
	head, rest := body, ""
	if i := strings.IndexAny(body, " \t"); i >= 0 {
		head, rest = body[:i], strings.TrimSpace(body[i+1:])
	}
	canonical, known := slashCommands[strings.ToLower(head)]
	if !known {
		return "", false
	}
	if rest != "" {
		return canonical + " " + rest, true
	}
	return canonical, true
}

// unknownCommandMessage is returned for unrecognized slash commands, so they
// are never sent to the LLM as regular prompts.
func unknownCommandMessage(query string) string {
	names := make([]string, 0, len(slashCommands))
	for name := range slashCommands {
		names = append(names, "/"+name)
	}
	sort.Strings(names)
	return fmt.Sprintf("Unknown command `%s`. Available commands: %s", query, strings.Join(names, " "))
}

// shellEscapeTimeout bounds how long a "!<command>" shell escape may run.
const shellEscapeTimeout = 30 * time.Second

// shellEscapeCommand extracts the command from a shell escape query
// ("!kubectl get pods"). ok is false when the query is not a shell escape.
func shellEscapeCommand(query string) (command string, ok bool) {
	trimmed := strings.TrimSpace(query)
	if !strings.HasPrefix(trimmed, "!") {
		return "", false
	}
	return strings.TrimSpace(strings.TrimPrefix(trimmed, "!")), true
}

// runShellEscape executes a "!<command>" query locally via the agent's
// executor, records the output in the transcript like a tool result, and
// appends it to the chat context as an observation so the LLM sees it on the
// next turn.
func (c *Agent) runShellEscape(ctx context.Context, command string) {
	log := klog.FromContext(ctx)
	if c.Executor == nil {
		c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error: cannot run shell command: no executor configured")
		return
	}

	execCtx, cancel := context.WithTimeout(ctx, shellEscapeTimeout)
	defer cancel()

	log.Info("Running shell escape command", "command", command)
	result, err := c.Executor.Execute(execCtx, command, nil, "")

	var output string
	if err != nil {
		output = fmt.Sprintf("Error: %v", err)
	} else {
		output = result.Stdout + result.Stderr
		if result.Error != "" {
			output += fmt.Sprintf("\nError: %s", result.Error)
		}
	}
	if execCtx.Err() == context.DeadlineExceeded {
		output += fmt.Sprintf("\n(timed out after %s)", shellEscapeTimeout)
	}
	output = strings.TrimRight(output, "\n")
	if output == "" {
		output = "(no output)"
	}

	c.addMessage(api.MessageSourceAgent, api.MessageTypeText, fmt.Sprintf("$ %s\n%s", command, output))

	observation := fmt.Sprintf("User ran: %s\nOutput:\n%s", command, output)
	c.currChatContent = append(c.currChatContent, observation)
}

func (c *Agent) handleMetaQuery(ctx context.Context, query string) (answer string, handled bool, err error) {
	// UIs may forward the query with surrounding whitespace/newlines.
	query = strings.TrimSpace(query)
	// Resolve slash commands before anything else; unknown ones are
	// rejected here and never reach the LLM.
	if strings.HasPrefix(query, "/") {
		normalized, ok := normalizeSlashCommand(query)
		if !ok {
			return unknownCommandMessage(query), true, nil
		}
		query = normalized
	}
	switch query {
	case "clear", "reset":
		c.sessionMu.Lock()
		// TODO: Remove this check when session persistence is default
		if err := c.Session.ChatMessageStore.ClearChatMessages(); err != nil {
			c.sessionMu.Unlock()
			return "Failed to clear the conversation", false, err
		}
		c.llmChat.Initialize(c.Session.ChatMessageStore.ChatMessages())
		c.sessionMu.Unlock()
		return "Cleared the conversation.", true, nil
	case "compact":
		return c.compactConversation(ctx)
	case "exit", "quit":
		c.setAgentState(api.AgentStateExited)
		return "It has been a pleasure assisting you. Have a great day!", true, nil
	case "model":
		// Bare "model" opens an interactive model picker.
		return c.openModelPicker(ctx)
	case "models":
		models, err := c.listModels(ctx)
		if err != nil {
			return "", false, fmt.Errorf("listing models: %w", err)
		}
		return "Available models:\n\n  - " + strings.Join(models, "\n  - ") + "\n\n", true, nil
	case "tools":
		return "Available tools:\n\n  - " + strings.Join(c.Tools.Names(), "\n  - ") + "\n\n", true, nil
	case "context":
		return c.handleContextQuery("")
	case "namespace", "ns":
		return c.handleNamespaceQuery(ctx, "")
	case "session":
		if c.SessionBackend != "filesystem" {
			return "Ephemeral session (memory backed). No persistent info available.", true, nil
		}
		return fmt.Sprintf("Current session:\n\n%s", c.Session.String()), true, nil

	case "save-session":
		savedSessionID, err := c.SaveSession()
		if err != nil {
			return "", false, fmt.Errorf("failed to save session: %w", err)
		}
		return "Saved session as " + savedSessionID, true, nil

	case "new-session":
		if err := c.createAndSwitchSession(); err != nil {
			return "", false, fmt.Errorf("failed to create new session: %w", err)
		}
		return fmt.Sprintf("Created and switched to new session %s.", c.Session.ID), true, nil

	case "sessions":
		sessions, err := c.ListSessions()
		if err != nil {
			return "", false, err
		}
		if len(sessions) == 0 {
			return "No sessions found.", true, nil
		}
		// Add ```text so markdown doesn't wreck the format
		availableSessions := "```text"
		availableSessions += "Available sessions:\n\n"
		availableSessions += "ID\t\t\tName\t\t\tCreated\t\t\tLast Accessed\t\tModel\t\tProvider\n"
		availableSessions += "--\t\t\t----\t\t\t-------\t\t\t-------------\t\t-----\t\t--------\n"

		for _, session := range sessions {
			name := session.Name
			if name == "" {
				name = "-"
			}
			availableSessions += fmt.Sprintf("%s\t%s\t%s\t%s\t%s\t%s\n",
				session.ID,
				name,
				session.CreatedAt.Format("2006-01-02 15:04"),
				session.LastModified.Format("2006-01-02 15:04"),
				session.ModelID,
				session.ProviderID)
		}
		// close the ```text box
		availableSessions += "```"
		return availableSessions, true, nil
	}

	if query == "resume-session" || strings.HasPrefix(query, "resume-session ") {
		parts := strings.Split(query, " ")
		if len(parts) != 2 {
			return "Invalid command. Usage: resume-session <session_id>", true, nil
		}
		sessionID := parts[1]
		if err := c.LoadSession(sessionID); err != nil {
			return "", false, err
		}
		return fmt.Sprintf("Resumed session %s.", sessionID), true, nil
	}

	if strings.HasPrefix(query, "model ") {
		modelID := strings.TrimSpace(strings.TrimPrefix(query, "model "))
		if modelID == "" {
			return "Invalid command. Usage: model <model_id>", true, nil
		}
		if err := c.switchModel(ctx, modelID); err != nil {
			return "", false, err
		}
		return fmt.Sprintf("Switched to model `%s`.", modelID), true, nil
	}

	if query == "context" || strings.HasPrefix(query, "context ") {
		return c.handleContextQuery(strings.TrimSpace(strings.TrimPrefix(query, "context")))
	}

	if query == "namespace" || query == "ns" || strings.HasPrefix(query, "namespace ") || strings.HasPrefix(query, "ns ") {
		name := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(query, "namespace"), "ns"))
		return c.handleNamespaceQuery(ctx, name)
	}

	if query == "delete-session" || strings.HasPrefix(query, "delete-session ") {
		parts := strings.SplitN(query, " ", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[1]) == "" {
			return "Invalid command. Usage: delete-session <session_id>", true, nil
		}
		sessionID := strings.TrimSpace(parts[1])
		if err := c.DeleteSession(sessionID); err != nil {
			return "", false, err
		}
		return fmt.Sprintf("Deleted session %s.", sessionID), true, nil
	}

	if query == "rename-session" || strings.HasPrefix(query, "rename-session ") {
		parts := strings.SplitN(query, " ", 2)
		if len(parts) != 2 || strings.TrimSpace(parts[1]) == "" {
			return "Invalid command. Usage: rename-session <name>", true, nil
		}
		name := strings.TrimSpace(parts[1])
		if err := c.RenameSession(c.Session.ID, name); err != nil {
			return "", false, err
		}
		return fmt.Sprintf("Renamed session to %q.", sessions.SanitizeSessionName(name)), true, nil
	}

	return "", false, nil
}

// compactConversation summarizes the session's conversation into a fresh
// context: the LLM condenses the history, the stored messages are replaced
// by a single summary message, and the chat is re-initialized with the new
// history. On LLM error the history is left intact.
func (c *Agent) compactConversation(ctx context.Context) (answer string, handled bool, err error) {
	if c.RunOnce {
		return "The compact command is not supported in quiet mode.", true, nil
	}

	c.sessionMu.Lock()
	messages := c.Session.ChatMessageStore.ChatMessages()
	c.sessionMu.Unlock()
	if len(messages) == 0 {
		return "Nothing to compact yet.", true, nil
	}

	// Build a plain-text transcript of the conversation, keeping at most the
	// last ~100KB so the summarization request stays a reasonable size.
	const maxTranscriptBytes = 100 * 1024
	var transcript strings.Builder
	for _, msg := range messages {
		text, ok := msg.Payload.(string)
		if !ok {
			text = fmt.Sprintf("%v", msg.Payload)
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		fmt.Fprintf(&transcript, "[%s] %s\n\n", msg.Source, text)
	}
	conversation := transcript.String()
	if len(conversation) > maxTranscriptBytes {
		conversation = conversation[len(conversation)-maxTranscriptBytes:]
	}

	// Surface an ephemeral status message so the user sees the compact is
	// in progress (the summarization LLM call can take a few seconds and
	// otherwise appears to hang with no feedback). It goes straight to the
	// output channel and is NOT stored, so an LLM error leaves the history
	// intact and the message vanishes on the next snapshot.
	c.Output <- &api.Message{
		Source:    api.MessageSourceAgent,
		Type:      api.MessageTypeText,
		Payload:   "⏳ Compacting conversation…",
		Timestamp: time.Now(),
	}
	// Flip to Running so the TUI shows the spinner (and the bottom-divider
	// scroll cue) during the summarization call, then back to Done so the
	// input box returns.
	c.setAgentState(api.AgentStateRunning)
	defer func() {
		c.setAgentState(api.AgentStateDone)
	}()

	completion, err := c.LLM.GenerateCompletion(ctx, &gollm.CompletionRequest{
		Model:  c.Model,
		Prompt: "Summarize this conversation concisely for continuing the task. Keep key facts, decisions, and current state:\n\n" + conversation,
	})
	if err != nil {
		return "", false, fmt.Errorf("summarizing conversation: %w", err)
	}
	summary := strings.TrimSpace(completion.Response())
	if summary == "" {
		return "", false, fmt.Errorf("summarizing conversation: empty response from LLM")
	}

	c.sessionMu.Lock()
	if err := c.Session.ChatMessageStore.ClearChatMessages(); err != nil {
		c.sessionMu.Unlock()
		return "Failed to compact the conversation", false, err
	}
	c.sessionMu.Unlock()

	// Seed the compacted history with the summary as a model message.
	c.addMessage(api.MessageSourceModel, api.MessageTypeText, "Previous conversation summary:\n\n"+summary)

	c.sessionMu.Lock()
	if err := c.llmChat.Initialize(c.Session.ChatMessageStore.ChatMessages()); err != nil {
		c.sessionMu.Unlock()
		return "", false, fmt.Errorf("re-initializing chat after compact: %w", err)
	}
	c.sessionMu.Unlock()

	return fmt.Sprintf("Conversation compacted (~%d messages summarized).", len(messages)), true, nil
}

// createAndSwitchSession creates a new session, switches the agent to it,
// and announces the switch. It is used both by the "new-session" meta query
// and by NewSessionRequest messages from UIs.
func (c *Agent) createAndSwitchSession() error {
	newSessionID, err := c.NewSession()
	if err != nil {
		return err
	}
	name := newSessionID
	c.sessionMu.Lock()
	if c.Session != nil && c.Session.Name != "" {
		name = c.Session.Name
	}
	c.sessionMu.Unlock()
	c.addMessage(api.MessageSourceAgent, api.MessageTypeText, fmt.Sprintf("Created and switched to new session %s (%s)", name, newSessionID))
	return nil
}

func (c *Agent) NewSession() (string, error) {
	if _, err := c.SaveSession(); err != nil {
		return "", fmt.Errorf("failed to save current session: %w", err)
	}

	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return "", fmt.Errorf("failed to create session manager: %w", err)
	}

	metadata := sessions.Metadata{
		ModelID:    c.Model,
		ProviderID: c.Provider,
	}

	newSession, err := manager.NewSession(metadata)
	if err != nil {
		return "", fmt.Errorf("failed to create new session: %w", err)
	}

	// If we are using a sandbox, we should spin up a new one for the new session
	if c.Sandbox == "k8s" {
		sandboxName := fmt.Sprintf("kubectl-ai-sandbox-%s", uuid.New().String()[:8])
		sandboxImage := c.SandboxImage

		sb, err := sandbox.NewKubernetesSandbox(sandboxName,
			sandbox.WithKubeconfig(c.Kubeconfig),
			sandbox.WithImage(sandboxImage),
		)

		if err != nil {
			return "", fmt.Errorf("failed to create new sandbox: %w", err)
		}

		c.sessionMu.Lock()
		if c.Executor != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			if err := c.Executor.Close(ctx); err != nil {
				klog.Warningf("error closing old executor: %v", err)
			}
			cancel()
		}

		c.Executor = sb
		klog.Info("Created new sandbox for new session", "name", sandboxName)

		// Re-bind all tools to the new executor
		c.Tools = c.Tools.CloneWithExecutor(c.Executor)

		c.Tools.RegisterTool(tools.NewBashTool(c.Executor))
		c.Tools.RegisterTool(tools.NewKubectlTool(c.Executor))
		c.sessionMu.Unlock()
	}

	if err := c.LoadSession(newSession.ID); err != nil {
		return "", fmt.Errorf("failed to load new session: %w", err)
	}

	return newSession.ID, nil
}

func (c *Agent) SaveSession() (string, error) {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()

	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return "", fmt.Errorf("failed to create session manager: %w", err)
	}

	if c.Session != nil {
		foundSession, _ := manager.FindSessionByID(c.Session.ID)
		if foundSession != nil {
			return foundSession.ID, nil
		}
	}

	metadata := sessions.Metadata{
		CreatedAt:    c.Session.CreatedAt,
		LastAccessed: time.Now(),
		ModelID:      c.Model,
		ProviderID:   c.Provider,
	}

	newSession, err := manager.NewSession(metadata)
	if err != nil {
		return "", fmt.Errorf("failed to create new session: %w", err)
	}

	messages := c.ChatMessageStore.ChatMessages()
	if err := newSession.ChatMessageStore.SetChatMessages(messages); err != nil {
		return "", fmt.Errorf("failed to save chat messages to new session: %w", err)
	}

	c.ChatMessageStore = newSession.ChatMessageStore
	c.Session = newSession
	c.Session.Messages = messages

	if c.llmChat != nil {
		_ = c.llmChat.Initialize(c.Session.ChatMessageStore.ChatMessages())
	}

	return newSession.ID, nil
}

// LoadSession loads a session by ID (or latest), updates the agent's state, and re-initializes the chat.
func (c *Agent) LoadSession(sessionID string) error {
	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return fmt.Errorf("failed to create session manager: %w", err)
	}

	var session *api.Session
	if sessionID == "" || sessionID == "latest" {
		s, err := manager.GetLatestSession()
		if err != nil {
			return fmt.Errorf("failed to get latest session: %w", err)
		}
		if s == nil {
			return fmt.Errorf("no sessions found to resume")
		}
		session = s
	} else {
		s, err := manager.FindSessionByID(sessionID)
		if err != nil {
			return fmt.Errorf("failed to get session %q: %w", sessionID, err)
		}
		session = s
	}

	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()

	if session.ChatMessageStore == nil {
		session.ChatMessageStore = sessions.NewInMemoryChatStore()
	}

	c.Session = session
	c.ChatMessageStore = session.ChatMessageStore
	c.Session.Messages = session.ChatMessageStore.ChatMessages()
	c.Session.LastModified = time.Now()

	// Reset state if it was left running (e.g. from a crash)
	if c.Session.AgentState == api.AgentStateRunning || c.Session.AgentState == api.AgentStateInitializing {
		c.Session.AgentState = api.AgentStateIdle
	}

	if err := manager.UpdateLastAccessed(session); err != nil {
		return fmt.Errorf("failed to update session metadata: %w", err)
	}

	if c.llmChat != nil {
		if err := c.llmChat.Initialize(c.Session.ChatMessageStore.ChatMessages()); err != nil {
			return fmt.Errorf("failed to re-initialize chat with new session: %w", err)
		}
	}

	return nil
}

// ToggleSkipPermissions flips auto-accept mode (skipping permission
// prompts) and returns the new state. Safe to call from UI goroutines.
func (c *Agent) ToggleSkipPermissions() bool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	c.SkipPermissions = !c.SkipPermissions
	return c.SkipPermissions
}

// SkipPermissionsEnabled reports whether auto-accept mode is on.
// Safe to call from any goroutine.
func (c *Agent) SkipPermissionsEnabled() bool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	return c.skipPermissions()
}

// setSkipPermissionsEnabled sets auto-accept mode under the lock.
func (c *Agent) setSkipPermissionsEnabled(enabled bool) {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	c.SkipPermissions = enabled
}

// skipPermissions reads the flag without locking; the caller must hold
// sessionMu.
func (c *Agent) skipPermissions() bool {
	return c.SkipPermissions
}

// DeleteSession deletes a session from the store. It refuses to delete the
// agent's current session. Safe to call from UI goroutines.
func (c *Agent) DeleteSession(sessionID string) error {
	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return fmt.Errorf("failed to create session manager: %w", err)
	}

	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.Session != nil && c.Session.ID == sessionID {
		return fmt.Errorf("cannot delete the current session")
	}

	return manager.DeleteSession(sessionID)
}

// kubeconfigOverride is the path of the session-scoped kubeconfig override
// the agent has applied via /context or /namespace (empty when none).
// ActiveKubeconfig returns it when set, else the agent's base kubeconfig.
func (c *Agent) ActiveKubeconfig() string {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.kubeconfigOverride != "" {
		return c.kubeconfigOverride
	}
	if c.pinnedKubeconfig != "" {
		return c.pinnedKubeconfig
	}
	return c.Kubeconfig
}

// pinSessionKubeconfig snapshots the effective kubeconfig into the session
// working directory and points KUBECONFIG at it for this process, so edits
// made to the global kubeconfig in another terminal do not change this
// session's context. No-op when no usable kubeconfig is available.
func (c *Agent) pinSessionKubeconfig() {
	if c.workDir == "" {
		return
	}
	pinnedPath := filepath.Join(c.workDir, "kubeconfig-pinned")
	if err := kube.WriteOverride(c.Kubeconfig, pinnedPath, "", ""); err != nil {
		klog.V(2).Infof("not pinning session kubeconfig: %v", err)
		return
	}
	c.sessionMu.Lock()
	c.pinnedKubeconfig = pinnedPath
	c.sessionMu.Unlock()
	_ = os.Setenv("KUBECONFIG", pinnedPath)
}

// ListAllTools returns every registered tool (built-in and MCP), safe to call
// from UI goroutines. It snapshots under the session lock so a concurrent
// tool registration can't race the caller.
func (c *Agent) ListAllTools() []tools.Tool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	return c.Tools.AllTools()
}

// applyKubeOverride builds (or rebuilds) the session-scoped kubeconfig
// override with the given context and/or namespace and points KUBECONFIG at
// it for this process. The base kubeconfig file is never mutated.
func (c *Agent) applyKubeOverride(context, namespace string) error {
	if c.workDir == "" {
		return fmt.Errorf("no working directory for the override kubeconfig")
	}
	outPath := filepath.Join(c.workDir, "kubeconfig-override")
	// Derive from the pinned snapshot when present, so /context never picks
	// up external edits made to the base file after session start.
	c.sessionMu.Lock()
	base := c.pinnedKubeconfig
	c.sessionMu.Unlock()
	if base == "" {
		base = c.Kubeconfig
	}
	if err := kube.WriteOverride(base, outPath, context, namespace); err != nil {
		return err
	}

	c.sessionMu.Lock()
	c.kubeconfigOverride = outPath
	c.sessionMu.Unlock()
	return os.Setenv("KUBECONFIG", outPath)
}

// resetKubeOverride clears the session-scoped override and restores the base
// kubeconfig for the process.
func (c *Agent) resetKubeOverride() {
	c.sessionMu.Lock()
	path := c.kubeconfigOverride
	c.kubeconfigOverride = ""
	c.sessionMu.Unlock()

	if path != "" {
		_ = os.Remove(path)
	}
	// Restore the pinned snapshot (or base) — not whatever the base file
	// currently says, which may have been edited externally.
	if active := c.ActiveKubeconfig(); active != "" {
		_ = os.Setenv("KUBECONFIG", active)
	} else {
		_ = os.Unsetenv("KUBECONFIG")
	}
}

// handleContextQuery answers the "context" meta query: bare shows the
// current and available kube contexts; "context <name>" applies a
// session-scoped override (the global kubeconfig is untouched);
// "context --reset" clears the override.
func (c *Agent) handleContextQuery(name string) (answer string, handled bool, err error) {
	if name == "--reset" {
		c.resetKubeOverride()
		return "Cleared the session context override.", true, nil
	}

	if name == "" {
		current, _, ok := kube.CurrentContext(c.ActiveKubeconfig())
		names, err := kube.ListContexts(c.ActiveKubeconfig())
		if err != nil {
			return "", false, fmt.Errorf("listing contexts: %w", err)
		}
		var b strings.Builder
		if ok {
			if c.kubeconfigOverride != "" {
				b.WriteString("Current context (session override): `" + current + "`\n\n")
			} else {
				b.WriteString("Current context: `" + current + "`\n\n")
			}
		}
		b.WriteString("Available contexts:\n\n")
		for _, n := range names {
			if n == current {
				b.WriteString("  - " + n + " (current)\n")
			} else {
				b.WriteString("  - " + n + "\n")
			}
		}
		b.WriteString("\nSwitch with `/context <name>` (session-scoped, global kubeconfig untouched).")
		return b.String(), true, nil
	}

	if err := c.applyKubeOverride(name, ""); err != nil {
		names, _ := kube.ListContexts(c.ActiveKubeconfig())
		if len(names) > 0 {
			return "", false, fmt.Errorf("%v. Available contexts: %s", err, strings.Join(names, ", "))
		}
		return "", false, err
	}
	return fmt.Sprintf("Switched to context `%s` (session only — global kubeconfig unchanged).", name), true, nil
}

// handleNamespaceQuery answers the "namespace" meta query: bare shows the
// current namespace and live namespaces from the cluster; "namespace <name>"
// applies a session-scoped namespace override; "namespace --reset" clears it.
func (c *Agent) handleNamespaceQuery(ctx context.Context, name string) (answer string, handled bool, err error) {
	if name == "--reset" {
		c.resetKubeOverride()
		return "Cleared the session namespace override.", true, nil
	}

	if name == "" {
		_, current, ok := kube.CurrentContext(c.ActiveKubeconfig())
		names, err := c.ListNamespaces(ctx)
		var b strings.Builder
		if ok {
			if c.kubeconfigOverride != "" {
				b.WriteString("Current namespace (session override): `" + current + "`\n\n")
			} else {
				b.WriteString("Current namespace: `" + current + "`\n\n")
			}
		}
		if err != nil {
			b.WriteString("Could not list namespaces from the cluster: " + err.Error())
		} else {
			b.WriteString("Namespaces in this cluster:\n\n")
			for _, n := range names {
				if n == current {
					b.WriteString("  - " + n + " (current)\n")
				} else {
					b.WriteString("  - " + n + "\n")
				}
			}
		}
		b.WriteString("\nSwitch with `/namespace <name>` (session-scoped).")
		return b.String(), true, nil
	}

	if err := c.applyKubeOverride("", name); err != nil {
		return "", false, err
	}
	return fmt.Sprintf("Switched to namespace `%s` (session only — global kubeconfig unchanged).", name), true, nil
}

// ListNamespaces returns live namespaces from the cluster through the
// executor (which inherits the session override via KUBECONFIG). Safe to
// call from UI goroutines.
func (c *Agent) ListNamespaces(ctx context.Context) ([]string, error) {
	if c.Executor == nil {
		return nil, fmt.Errorf("no executor available to list namespaces")
	}
	execCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	res, err := c.Executor.Execute(execCtx, "kubectl get namespaces -o custom-columns=NAME:.metadata.name --no-headers", os.Environ(), c.workDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, line := range strings.Split(res.Stdout, "\n") {
		if n := strings.TrimSpace(line); n != "" {
			names = append(names, n)
		}
	}
	return names, nil
}

// openModelPicker presents an interactive picker of the provider's models
// (used by the bare "model" meta query).
func (c *Agent) openModelPicker(ctx context.Context) (answer string, handled bool, err error) {
	models, err := c.listModels(ctx)
	if err != nil {
		return "", false, fmt.Errorf("listing models: %w", err)
	}
	if len(models) == 0 {
		return fmt.Sprintf("Current model is `%s` (no models reported by the provider).", c.Model), true, nil
	}

	options := make([]api.UserChoiceOption, len(models))
	for i, m := range models {
		label := m
		if m == c.Model {
			label = m + " (current)"
		}
		options[i] = api.UserChoiceOption{Label: label, Value: m}
	}

	c.pendingModelChoice = models
	c.setAgentState(api.AgentStateWaitingForInput)
	c.addMessage(api.MessageSourceAgent, api.MessageTypeUserChoiceRequest, &api.UserChoiceRequest{
		Prompt:  "Select a model:",
		Options: options,
		Kind:    "model",
	})
	return "", true, nil
}

// handleModelChoice applies the selection from an open /model picker.
func (c *Agent) handleModelChoice(ctx context.Context, response *api.UserChoiceResponse) {
	models := c.pendingModelChoice
	c.pendingModelChoice = nil
	c.setAgentState(api.AgentStateDone)

	idx := response.Choice - 1
	if response.Choice == 0 {
		// Esc/cancel: leave the model unchanged.
		c.addMessage(api.MessageSourceAgent, api.MessageTypeText, "Model switch cancelled.")
		return
	}
	if idx < 0 || idx >= len(models) {
		c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Invalid model selection.")
		return
	}
	if err := c.switchModel(ctx, models[idx]); err != nil {
		c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Error switching model: "+err.Error())
		return
	}
	c.addMessage(api.MessageSourceAgent, api.MessageTypeText, fmt.Sprintf("Switched to model `%s`.", models[idx]))
}

// switchModel changes the current model, preserving the conversation: the
// session's ModelID is updated and persisted, and the chat is rebuilt with
// the full history and tool definitions.
func (c *Agent) switchModel(ctx context.Context, modelID string) error {
	models, err := c.listModels(ctx)
	if err != nil {
		return fmt.Errorf("listing models: %w", err)
	}
	found := false
	for _, m := range models {
		if m == modelID {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("unknown model %q. Available models: %s", modelID, strings.Join(models, ", "))
	}

	c.sessionMu.Lock()
	c.Model = modelID
	c.Session.ModelID = modelID
	c.Session.LastModified = time.Now()
	session := c.Session
	c.sessionMu.Unlock()

	// Persist the model change so resumed sessions keep it.
	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return fmt.Errorf("failed to create session manager: %w", err)
	}
	if err := manager.UpdateLastAccessed(session); err != nil {
		klog.Warningf("Failed to persist model change for session %q: %v", session.ID, err)
	}

	if err := c.rebuildChat(ctx); err != nil {
		return fmt.Errorf("rebuilding chat for model %q: %w", modelID, err)
	}
	return nil
}

// RenameSession renames a session. If the renamed session is the agent's
// current session, the in-memory state is updated as well so UIs reflect
// the new name immediately.
func (c *Agent) RenameSession(sessionID, name string) error {
	if sessions.SanitizeSessionName(name) == "" {
		return fmt.Errorf("session name cannot be empty")
	}

	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return fmt.Errorf("failed to create session manager: %w", err)
	}

	// Hold the lock across the whole operation: with the memory backend the
	// store hands out shared *api.Session pointers, so the rename can
	// otherwise race the agent loop and UI readers.
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()

	if err := manager.SetSessionName(sessionID, name, true); err != nil {
		return fmt.Errorf("failed to rename session %q: %w", sessionID, err)
	}

	if c.Session != nil && c.Session.ID == sessionID {
		c.Session.Name = sessions.SanitizeSessionName(name)
		c.Session.ManuallyNamed = true
	}

	return nil
}

// ListSessions returns available sessions for UI pickers
func (c *Agent) ListSessions() ([]api.SessionInfo, error) {
	manager, err := sessions.NewSessionManager(c.SessionBackend)
	if err != nil {
		return nil, fmt.Errorf("failed to create session manager: %w", err)
	}

	// Hold the lock while reading session fields: with the memory backend
	// the store hands out shared *api.Session pointers which the agent loop
	// mutates under sessionMu.
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()

	sessionList, err := manager.ListSessions()
	if err != nil {
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}

	sessionInfos := make([]api.SessionInfo, len(sessionList))
	for i, session := range sessionList {
		var messages []*api.Message
		if session.ChatMessageStore != nil {
			messages = session.ChatMessageStore.ChatMessages()
		}
		sessionInfos[i] = api.SessionInfo{
			ID:           session.ID,
			Name:         session.Name,
			ModelID:      session.ModelID,
			ProviderID:   session.ProviderID,
			CreatedAt:    session.CreatedAt,
			LastModified: session.LastModified,
			MessageCount: len(messages),
			FirstMessage: firstUserMessage(messages),
		}
	}
	return sessionInfos, nil
}

// firstUserMessage returns a short preview of the first user message, for
// displaying as a fallback session title.
func firstUserMessage(messages []*api.Message) string {
	const maxLen = 80
	for _, msg := range messages {
		if msg.Source != api.MessageSourceUser || msg.Type != api.MessageTypeText {
			continue
		}
		p, ok := msg.Payload.(string)
		if !ok {
			continue
		}
		p = strings.Join(strings.Fields(p), " ") // collapse whitespace/newlines
		if p == "" {
			continue
		}
		if r := []rune(p); len(r) > maxLen {
			p = string(r[:maxLen]) + "…"
		}
		return p
	}
	return ""
}

func (c *Agent) listModels(ctx context.Context) ([]string, error) {
	if c.availableModels == nil {
		// Surface an ephemeral status so the user sees the model list is being
		// fetched (ListModels can be a network call). Not stored, so it leaves
		// no trace in the history. Guarded for tests that construct an agent
		// without an Output channel.
		if c.Output != nil {
			c.Output <- &api.Message{
				Source:    api.MessageSourceAgent,
				Type:      api.MessageTypeText,
				Payload:   "⏳ Fetching available models…",
				Timestamp: time.Now(),
			}
		}
		modelNames, err := c.LLM.ListModels(ctx)
		if err != nil {
			return nil, fmt.Errorf("listing models: %w", err)
		}
		c.availableModels = modelNames
	}
	return c.availableModels, nil
}

func (c *Agent) DispatchToolCalls(ctx context.Context) error {
	log := klog.FromContext(ctx)
	// execute all pending function calls
	for _, call := range c.pendingFunctionCalls {
		// Only show "Running" message and proceed with execution for non-interactive commands
		toolDescription := call.ParsedToolCall.Description()

		c.addMessage(api.MessageSourceModel, api.MessageTypeToolCallRequest, toolDescription)

		output, err := call.ParsedToolCall.InvokeTool(ctx, tools.InvokeToolOptions{
			Kubeconfig: c.ActiveKubeconfig(),
			WorkDir:    c.workDir,
			Executor:   c.Executor,
		})

		if err != nil {
			log.Error(err, "error executing action", "output", output)
			c.addMessage(api.MessageSourceAgent, api.MessageTypeToolCallResponse, err.Error())
			return err
		}

		// Handle timeout message using UI blocks
		if execResult, ok := output.(*sandbox.ExecResult); ok && execResult != nil && execResult.StreamType == "timeout" {
			c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "\nTimeout reached after 7 seconds\n")
		}
		// Add the tool call result to maintain conversation flow
		var payload any
		if c.EnableToolUseShim {
			// Add the error as an observation
			observation := fmt.Sprintf("Result of running %q:\n%v",
				call.FunctionCall.Name,
				output)
			c.currChatContent = append(c.currChatContent, observation)
			payload = observation
		} else {
			// If shim is disabled, convert the result to a map and append FunctionCallResult
			result, err := tools.ToolResultToMap(output)
			if err != nil {
				log.Error(err, "error converting tool result to map", "output", output)
				return err
			}
			payload = result
			c.currChatContent = append(c.currChatContent, gollm.FunctionCallResult{
				ID:     call.FunctionCall.ID,
				Name:   call.FunctionCall.Name,
				Result: result,
			})
		}
		// Cap oversized tool output before it enters the session history so a
		// massive stdout (e.g. kubectl get -A across many clusters) can't
		// bloat the context and brick the session past the model's context
		// limit. The full output was already delivered to the LLM via
		// currChatContent above; the stored/Ui copy only needs to be
		// representative.
		payload = capToolResultOutput(payload)
		c.addMessage(api.MessageSourceAgent, api.MessageTypeToolCallResponse, payload)
	}
	return nil
}

// maxToolOutputBytes is the cap on a single tool-call result's stdout/stderr
// before it is stored in the session history. 16KB is enough to read the
// head of a table/list and diagnose a failure without flooding context.
const maxToolOutputBytes = 16 * 1024

// capToolResultOutput truncates oversized stdout/stderr in a tool-call
// result payload (a map[string]any from tools.ToolResultToMap) to
// maxToolOutputBytes, appending a "[+N bytes truncated]" marker. String
// payloads (shim observations) are capped too. Other payload shapes pass
// through unchanged.
func capToolResultOutput(payload any) any {
	switch p := payload.(type) {
	case string:
		if len(p) > maxToolOutputBytes {
			return p[:maxToolOutputBytes] + fmt.Sprintf("\n[+%d bytes truncated]\n", len(p)-maxToolOutputBytes)
		}
		return p
	case map[string]any:
		for _, k := range []string{"stdout", "stderr"} {
			if s, ok := p[k].(string); ok && len(s) > maxToolOutputBytes {
				p[k] = s[:maxToolOutputBytes] + fmt.Sprintf("\n[+%d bytes truncated]\n", len(s)-maxToolOutputBytes)
			}
		}
	}
	return payload
}

// The key idea is to treat all tool calls to be executed atomically or not
// If all tool calls are readonly call, it is straight forward
// if some of the tool calls are not readonly, then the interesting question is should the permission
// be asked for each of the tool call or only once for all the tool calls.
// I think treating all tool calls as atomic is the right thing to do.

type ToolCallAnalysis struct {
	FunctionCall        gollm.FunctionCall
	ParsedToolCall      *tools.ToolCall
	IsInteractive       bool
	IsInteractiveError  error
	ModifiesResourceStr string
}

func (c *Agent) analyzeToolCalls(ctx context.Context, toolCalls []gollm.FunctionCall) ([]ToolCallAnalysis, error) {
	toolCallAnalysis := make([]ToolCallAnalysis, len(toolCalls))
	for i, call := range toolCalls {
		toolCallAnalysis[i].FunctionCall = call
		toolCall, err := c.Tools.ParseToolInvocation(ctx, call.Name, call.Arguments)
		if err != nil {
			return nil, fmt.Errorf("error parsing tool call: %w", err)
		}
		toolCallAnalysis[i].IsInteractive, err = toolCall.GetTool().IsInteractive(call.Arguments)
		if err != nil {
			toolCallAnalysis[i].IsInteractiveError = err
		}
		toolCallAnalysis[i].ModifiesResourceStr = toolCall.GetTool().CheckModifiesResource(call.Arguments)
		toolCallAnalysis[i].ParsedToolCall = toolCall
	}
	return toolCallAnalysis, nil
}

// modifyingToolNames returns the distinct, sorted names of the tools behind
// the pending calls that modify resources (and therefore need permission).
func modifyingToolNames(calls []ToolCallAnalysis) []string {
	seen := make(map[string]bool)
	var names []string
	for _, call := range calls {
		if call.ModifiesResourceStr == "no" || call.ParsedToolCall == nil {
			continue
		}
		name := call.ParsedToolCall.GetTool().Name()
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// permissionChoiceRequest builds the confirmation prompt for the pending
// resource-modifying tool calls. When every modifying call is for the same
// tool, a fourth "Always allow <tool>" option is offered, which remembers
// the tool for the rest of the process.
func permissionChoiceRequest(calls []ToolCallAnalysis) *api.UserChoiceRequest {
	var commandDescriptions []string
	var dryRunPreviews []string
	for _, call := range calls {
		commandDescriptions = append(commandDescriptions, call.ParsedToolCall.Description())
		// For kubectl mutating commands, show a safe dry-run preview the user
		// can eyeball before approving: the same command with --dry-run=server
		// (or --dry-run=client). No cluster side effect (dry-run validates but
		// does not persist); non-kubectl/unparseable commands produce no preview.
		if preview := tools.KubectlDryRunPreview(call.ParsedToolCall.Description()); preview != "" {
			dryRunPreviews = append(dryRunPreviews, preview)
		}
	}
	confirmationPrompt := "The following commands require your approval to run:\n* " + strings.Join(commandDescriptions, "\n* ")
	if len(dryRunPreviews) > 0 {
		confirmationPrompt += "\n\nDry-run preview (safe, not applied):\n* " + strings.Join(dryRunPreviews, "\n* ")
	}
	confirmationPrompt += "\n\nDo you want to proceed ?"

	options := []api.UserChoiceOption{
		{Value: "yes", Label: "Yes"},
		{Value: "yes_and_dont_ask_me_again", Label: "Yes, and don't ask me again"},
		{Value: "no", Label: "No"},
	}
	if names := modifyingToolNames(calls); len(names) == 1 {
		options = append(options, api.UserChoiceOption{Value: "always_allow_" + names[0], Label: "Always allow " + names[0]})
	}
	return &api.UserChoiceRequest{Prompt: confirmationPrompt, Options: options, Kind: "permission"}
}

// allowTools adds the given tool names to the in-memory always-allow set.
func (c *Agent) allowTools(names []string) {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	if c.allowedTools == nil {
		c.allowedTools = make(map[string]bool)
	}
	for _, name := range names {
		c.allowedTools[name] = true
	}
}

// modifyingCallsAllowed reports whether every resource-modifying call uses
// a tool the user has previously marked as always allowed; such batches are
// dispatched without prompting.
func (c *Agent) modifyingCallsAllowed(calls []ToolCallAnalysis) bool {
	c.sessionMu.Lock()
	defer c.sessionMu.Unlock()
	for _, call := range calls {
		if call.ModifiesResourceStr == "no" || call.ParsedToolCall == nil {
			continue
		}
		if !c.allowedTools[call.ParsedToolCall.GetTool().Name()] {
			return false
		}
	}
	return true
}

func (c *Agent) handleChoice(ctx context.Context, choice *api.UserChoiceResponse) (dispatchToolCalls bool) {
	log := klog.FromContext(ctx)
	// if user input is a choice and use has declined the operation,
	// we need to abort all pending function calls.
	// update the currChatContent with the choice and keep the agent loop running.

	// Normalize the input
	switch choice.Choice {
	case 1:
		dispatchToolCalls = true
	case 2:
		c.setSkipPermissionsEnabled(true)
		dispatchToolCalls = true
	case 3:
		c.currChatContent = append(c.currChatContent, gollm.FunctionCallResult{
			ID:   c.pendingFunctionCalls[0].FunctionCall.ID,
			Name: c.pendingFunctionCalls[0].FunctionCall.Name,
			Result: map[string]any{
				"error":     "User declined to run this operation.",
				"status":    "declined",
				"retryable": false,
			},
		})
		c.pendingFunctionCalls = []ToolCallAnalysis{}
		dispatchToolCalls = false
		c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Operation was skipped. User declined to run this operation.")
	case 4:
		// "Always allow <tool>": remember the modifying tool(s) for the rest
		// of the process so future calls skip the prompt, then dispatch.
		c.allowTools(modifyingToolNames(c.pendingFunctionCalls))
		dispatchToolCalls = true
	default:
		// This case should technically not be reachable due to AskForConfirmation loop
		err := fmt.Errorf("invalid confirmation choice: %q", choice.Choice)
		log.Error(err, "Invalid choice received from AskForConfirmation")
		c.pendingFunctionCalls = []ToolCallAnalysis{}
		dispatchToolCalls = false
		c.addMessage(api.MessageSourceAgent, api.MessageTypeError, "Invalid choice received. Cancelling operation.")
	}
	return dispatchToolCalls
}

// generateFromTemplate generates a prompt for LLM. It uses the prompt from the provides template file or default.
func (a *Agent) generatePrompt(_ context.Context, defaultPromptTemplate string, data PromptData) (string, error) {
	promptTemplate := defaultPromptTemplate
	if a.PromptTemplateFile != "" {
		content, err := os.ReadFile(a.PromptTemplateFile)
		if err != nil {
			return "", fmt.Errorf("error reading template file: %v", err)
		}
		promptTemplate = string(content)
	}

	for _, extraPromptPath := range a.ExtraPromptPaths {
		content, err := os.ReadFile(extraPromptPath)
		if err != nil {
			return "", fmt.Errorf("error reading extra prompt path: %v", err)
		}
		promptTemplate += "\n" + string(content)
	}

	tmpl, err := template.New("promptTemplate").Parse(promptTemplate)
	if err != nil {
		return "", fmt.Errorf("building template for prompt: %w", err)
	}

	var result strings.Builder
	err = tmpl.Execute(&result, &data)
	if err != nil {
		return "", fmt.Errorf("evaluating template for prompt: %w", err)
	}
	return result.String(), nil
}

// PromptData represents the structure of the data to be filled into the template.
type PromptData struct {
	Query string
	Tools tools.Tools

	EnableToolUseShim    bool
	SessionIsInteractive bool
}

func (a *PromptData) ToolsAsJSON() string {
	var toolDefinitions []*gollm.FunctionDefinition

	for _, tool := range a.Tools.AllTools() {
		toolDefinitions = append(toolDefinitions, tool.FunctionDefinition())
	}

	json, err := json.MarshalIndent(toolDefinitions, "", "  ")
	if err != nil {
		return ""
	}
	return string(json)
}

func (a *PromptData) ToolNames() string {
	return strings.Join(a.Tools.Names(), ", ")
}

type ReActResponse struct {
	Thought string  `json:"thought"`
	Answer  string  `json:"answer,omitempty"`
	Action  *Action `json:"action,omitempty"`
}

type Action struct {
	Name             string `json:"name"`
	Reason           string `json:"reason"`
	Command          string `json:"command"`
	ModifiesResource string `json:"modifies_resource"`
}

func extractJSON(s string) (string, bool) {
	const jsonBlockMarker = "```json"

	first := strings.Index(s, jsonBlockMarker)
	last := strings.LastIndex(s, "```")
	if first == -1 || last == -1 || first == last {
		return "", false
	}
	data := s[first+len(jsonBlockMarker) : last]

	return data, true
}

// parseReActResponse parses the LLM response into a ReActResponse struct
// This function assumes the input contains exactly one JSON code block
// formatted with ```json and ``` markers. The JSON block is expected to
// contain a valid ReActResponse object.
func parseReActResponse(input string) (*ReActResponse, error) {
	cleaned, found := extractJSON(input)
	if !found {
		return nil, fmt.Errorf("no JSON code block found in %q", cleaned)
	}

	cleaned = strings.ReplaceAll(cleaned, "\n", "")
	cleaned = strings.TrimSpace(cleaned)

	var reActResp ReActResponse
	if err := json.Unmarshal([]byte(cleaned), &reActResp); err != nil {
		return nil, fmt.Errorf("parsing JSON %q: %w", cleaned, err)
	}
	return &reActResp, nil
}

// toMap converts the value to a map, going via JSON
func toMap(v any) (map[string]any, error) {
	j, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("converting %T to json: %w", v, err)
	}
	m := make(map[string]any)
	if err := json.Unmarshal(j, &m); err != nil {
		return nil, fmt.Errorf("converting json to map: %w", err)
	}
	return m, nil
}

func candidateToShimCandidate(iterator gollm.ChatResponseIterator) (gollm.ChatResponseIterator, error) {
	return func(yield func(gollm.ChatResponse, error) bool) {
		buffer := ""
		for response, err := range iterator {
			if err != nil {
				yield(nil, err)
				return
			}

			if len(response.Candidates()) == 0 {
				yield(nil, fmt.Errorf("no candidates in LLM response"))
				return
			}

			candidate := response.Candidates()[0]

			for _, part := range candidate.Parts() {
				if text, ok := part.AsText(); ok {
					buffer += text
					klog.Infof("text is %q", text)
				} else {
					yield(nil, fmt.Errorf("no text part found in candidate"))
					return
				}
			}
		}

		if buffer == "" {
			yield(nil, nil)
			return
		}

		parsedReActResp, err := parseReActResponse(buffer)
		if err != nil {
			yield(nil, fmt.Errorf("parsing ReAct response %q: %w", buffer, err))
			return
		}
		buffer = "" // TODO: any trailing text?
		yield(&ShimResponse{candidate: parsedReActResp}, nil)
	}, nil
}

type ShimResponse struct {
	candidate *ReActResponse
}

func (r *ShimResponse) UsageMetadata() any {
	return nil
}

func (r *ShimResponse) Candidates() []gollm.Candidate {
	return []gollm.Candidate{&ShimCandidate{candidate: r.candidate}}
}

type ShimCandidate struct {
	candidate *ReActResponse
}

func (c *ShimCandidate) String() string {
	return fmt.Sprintf("Thought: %s\nAnswer: %s\nAction: %s", c.candidate.Thought, c.candidate.Answer, c.candidate.Action)
}

func (c *ShimCandidate) Parts() []gollm.Part {
	var parts []gollm.Part
	if c.candidate.Thought != "" {
		parts = append(parts, &ShimPart{text: c.candidate.Thought})
	}
	if c.candidate.Answer != "" {
		parts = append(parts, &ShimPart{text: c.candidate.Answer})
	}
	if c.candidate.Action != nil {
		parts = append(parts, &ShimPart{action: c.candidate.Action})
	}
	return parts
}

type ShimPart struct {
	text   string
	action *Action
}

func (p *ShimPart) AsText() (string, bool) {
	return p.text, p.text != ""
}

func (p *ShimPart) AsFunctionCalls() ([]gollm.FunctionCall, bool) {
	if p.action != nil {
		functionCallArgs, err := toMap(p.action)
		if err != nil {
			return nil, false
		}
		delete(functionCallArgs, "name") // passed separately
		// delete(functionCallArgs, "reason")
		// delete(functionCallArgs, "modifies_resource")
		return []gollm.FunctionCall{
			{
				Name:      p.action.Name,
				Arguments: functionCallArgs,
			},
		}, true
	}
	return nil, false
}

func (p *ShimPart) AsThinking() (string, bool) {
	return "", false
}
