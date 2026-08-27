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

package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
)

// StreamDetector determines if a command is a streaming command and returns the stream type.
// It returns (true, streamType) if it is a streaming command, and (false, "") otherwise.
type StreamDetector func(command string) (isStreaming bool, streamType string)

// StreamingCommandTimeout bounds streaming commands (watch, logs -f, attach);
// the partial output captured so far is returned when it fires.
const StreamingCommandTimeout = 7 * time.Second

// DefaultCommandTimeout bounds every OTHER tool command: without it, a wedged
// apiserver call, 'tail -f', or a hung MCP server blocks the turn forever
// (only user interrupt could recover, and RunOnce mode has no user).
const DefaultCommandTimeout = 2 * time.Minute

// ExecuteWithStreamingHandling executes a command using the provided executor,
// handling streaming commands (watch, logs -f, attach) by applying a timeout
// and capturing partial output. All other commands still get a generous
// default deadline so nothing hangs the agent forever.
func ExecuteWithStreamingHandling(ctx context.Context, executor sandbox.Executor, command string, workDir string, env []string, detector StreamDetector) (*sandbox.ExecResult, error) {
	isStreaming := false
	if detector != nil {
		isStreaming, _ = detector(command)
	}

	timeout := DefaultCommandTimeout
	if isStreaming {
		timeout = StreamingCommandTimeout
	}
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	result, err := executor.Execute(cmdCtx, command, env, workDir)

	// If executor returns nil result on error (it shouldn't, but let's be safe), create one
	if result == nil {
		result = &sandbox.ExecResult{Command: command}
	}

	if isStreaming && cmdCtx.Err() == context.DeadlineExceeded {
		// Timeout is expected for streaming commands; return partial output.
		// StreamType is set to "timeout" (and NOT overwritten with the
		// detected kind — that bug made the agent's timeout notice dead
		// code) so the transcript can note it after the response.
		result.StreamType = "timeout"
		result.Error = "Timeout reached after 7 seconds (partial output returned)"
		return result, nil
	}
	if !isStreaming && cmdCtx.Err() == context.DeadlineExceeded {
		result.Error = fmt.Sprintf("command timed out after %s", timeout)
		err = nil // the result carries the error; don't kill the turn
	}

	return result, err
}
