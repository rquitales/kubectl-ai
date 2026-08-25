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

package agent

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/mcp"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/tools"
)

func TestRegisterMCPToolRegistersIntoAgentToolset(t *testing.T) {
	// Regression: MCP tools were registered into the package-global registry
	// after the agent cloned its own toolset and built the chat, so the LLM
	// never received MCP function definitions and dispatch would have failed
	// with "tool not recognized".
	var toolset tools.Tools
	toolset.Init()
	a := &Agent{Tools: toolset}

	toolInfo := mcp.Tool{Name: "search", Description: "searches things"}
	if err := a.registerMCPTool("github", toolInfo, nil); err != nil {
		t.Fatalf("registerMCPTool: %v", err)
	}

	got := a.Tools.Lookup("github_search")
	if got == nil {
		t.Fatal("MCP tool not found in the agent's toolset under its unique name")
	}
	if def := got.FunctionDefinition(); def == nil || def.Name != "github_search" {
		t.Errorf("FunctionDefinition name = %v, want github_search", def)
	}

	// A second registration (e.g. a second agent Init in the same process)
	// must be a no-op, not a panic.
	if err := a.registerMCPTool("github", toolInfo, nil); err != nil {
		t.Fatalf("duplicate registerMCPTool: %v", err)
	}
	if n := len(a.Tools.AllTools()); n != 1 {
		t.Errorf("tools registered = %d, want 1 (duplicate skipped)", n)
	}
}
