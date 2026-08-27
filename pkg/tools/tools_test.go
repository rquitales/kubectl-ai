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

package tools

import (
	"context"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sandbox"
)

func TestBashToolMalformedArgsDoNotPanic(t *testing.T) {
	// Regression: a model emitting a bash call with a missing or non-string
	// "command" used to panic the whole process on an unchecked assertion.
	tool := NewBashTool(nil)
	for _, args := range []map[string]any{
		{},
		{"command": nil},
		{"command": 42},
		{"command": []string{"rm", "-rf"}},
	} {
		result, err := tool.Run(context.Background(), args)
		if err != nil {
			t.Fatalf("args %v: unexpected Go error: %v", args, err)
		}
		er, ok := result.(*sandbox.ExecResult)
		if !ok || er.Error == "" {
			t.Errorf("args %v: expected an ExecResult carrying the error, got %+v", args, result)
		}
	}
}

func TestToolCallDescriptionSafeOnMalformedArgs(t *testing.T) {
	// Description is called for every tool call before dispatch; malformed
	// arguments must not panic it.
	tool := NewBashTool(nil)
	tc := &ToolCall{tool: tool, name: "bash", arguments: map[string]any{"command": 42}}
	_ = tc.Description()
}
