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

package mcp

import (
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
)

func TestConvertMCPToolsSkipsBadSchemas(t *testing.T) {
	// Regression: one malformed or unusual schema used to abort the whole
	// server's tool list — and agent startup with --mcp-client.
	tools := []mcp.Tool{
		{Name: "good", Description: "fine", InputSchema: mcp.ToolInputSchema{Type: "object"}},
		{Name: "no-schema", Description: "missing schema"},
		{Name: "weird", Description: "unusual type", InputSchema: mcp.ToolInputSchema{Type: "something-exotic"}},
		{Name: "numeric", Description: "number param", InputSchema: mcp.ToolInputSchema{Type: "number"}},
	}
	got, err := convertMCPToolsToTools(tools)
	if err != nil {
		t.Fatalf("convertMCPToolsToTools: %v", err)
	}
	var names []string
	for _, tool := range got {
		names = append(names, tool.Name)
	}
	if len(names) != 2 || names[0] != "good" || names[1] != "numeric" {
		t.Errorf("converted tools = %v, want [good numeric] (bad ones skipped)", names)
	}
}
