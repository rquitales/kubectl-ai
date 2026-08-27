// Copyright 2026 Google LLC
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
	"context"
	"runtime"
	"testing"
	"time"
)

func TestLocalExecutorKillsWholePipelineOnTimeout(t *testing.T) {
	// Regression: exec.CommandContext's default cancel kills only the shell;
	// pipelined grandchildren survived, and Wait blocked forever on the pipe
	// they held open — a streaming timeout never actually returned.
	if runtime.GOOS == "windows" {
		t.Skip("unix-only process-group semantics")
	}
	e := NewLocalExecutor()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	start := time.Now()
	result, err := e.Execute(ctx, "sleep 30 | cat", nil, "")
	elapsed := time.Since(start)
	if err != nil {
		t.Logf("execute error (expected on timeout): %v", err)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("pipeline survived the timeout: Execute returned after %v", elapsed)
	}
	_ = result
}
