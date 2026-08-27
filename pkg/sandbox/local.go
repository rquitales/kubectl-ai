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

package sandbox

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"time"

	"k8s.io/klog/v2"
)

const (
	defaultBashBin = "/bin/bash"
)

// Local executes commands locally.
type Local struct{}

// NewLocalExecutor creates a new LocalExecutor.
func NewLocalExecutor() *Local {
	return &Local{}
}

// Execute executes the command locally.
func (e *Local) Execute(ctx context.Context, command string, env []string, workDir string) (*ExecResult, error) {
	// Use the provided context directly
	cmdCtx := ctx

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(cmdCtx, os.Getenv("COMSPEC"), "/c", command)
	} else {
		cmd = exec.CommandContext(cmdCtx, lookupBashBin(), "-c", command)
		// Run the command in its own process group and kill the WHOLE group
		// on cancel/timeout: exec.CommandContext's default kill targets only
		// the shell, so pipelined grandchildren (kubectl logs -f | grep x)
		// survive — and cmd.Wait() then blocks forever on their pipe writes.
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		cmd.Cancel = func() error {
			if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
				return cmd.Process.Kill() // fall back to just the shell
			}
			return nil
		}
		// Belt and braces: if a grandchild somehow keeps the pipes open even
		// after the group kill, Wait returns after this delay instead of
		// blocking forever.
		cmd.WaitDelay = 3 * time.Second
	}
	cmd.Dir = workDir
	cmd.Env = env

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err := cmd.Run()

	result := &ExecResult{
		Command: command,
		Stdout:  stdoutBuf.String(),
		Stderr:  stderrBuf.String(),
	}

	if err != nil {
		// If it wasn't a timeout (or not a streaming command), it's a real error
		if exitError, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitError.ExitCode()
			result.Error = exitError.Error()
			// Stderr is already captured in result.Stderr
		} else {
			return nil, err
		}
	}

	return result, nil
}

// Close is a no-op for Local executor.
func (e *Local) Close(ctx context.Context) error {
	return nil
}

// Find the bash executable path using exec.LookPath.
func lookupBashBin() string {
	actualBashPath, err := exec.LookPath("bash")
	if err != nil {
		klog.Warningf("'bash' not found in PATH, defaulting to %s: %v", defaultBashBin, err)
		return defaultBashBin
	}
	return actualBashPath
}
