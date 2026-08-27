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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
)

// WriteFileTool lets the model author files (typically Kubernetes manifests)
// inside the session working directory, so it can `kubectl apply -f` them
// instead of fighting heredoc quoting through the shell tool.
type WriteFileTool struct{}

func NewWriteFileTool() *WriteFileTool { return &WriteFileTool{} }

func (t *WriteFileTool) Name() string { return "write_file" }

func (t *WriteFileTool) Description() string {
	return "Writes a file (e.g. a Kubernetes manifest) in the session working directory. Use write_file to author YAML, then apply it with `kubectl apply -f <path>`."
}

func (t *WriteFileTool) FunctionDefinition() *gollm.FunctionDefinition {
	return &gollm.FunctionDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"path": {
					Type:        gollm.TypeString,
					Description: "File path, relative to the session working directory (or absolute within it).",
				},
				"content": {
					Type:        gollm.TypeString,
					Description: "The full file content to write.",
				},
			},
			Required: []string{"path", "content"},
		},
	}
}

func (t *WriteFileTool) Run(ctx context.Context, args map[string]any) (any, error) {
	pathVal, ok := args["path"]
	if !ok || pathVal == nil {
		return map[string]any{"error": "path not provided"}, nil
	}
	path, ok := pathVal.(string)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("path must be a string, got %T", pathVal)}, nil
	}
	contentVal, ok := args["content"]
	if !ok || contentVal == nil {
		return map[string]any{"error": "content not provided"}, nil
	}
	content, ok := contentVal.(string)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("content must be a string, got %T", contentVal)}, nil
	}

	workDir, _ := ctx.Value(WorkDirKey).(string)
	if workDir == "" {
		return map[string]any{"error": "no working directory configured"}, nil
	}

	// Constrain to the working directory: the model must not write outside
	// the session scratch space.
	abs := path
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(workDir, path)
	}
	abs = filepath.Clean(abs)
	if abs != workDir && !strings.HasPrefix(abs, workDir+string(filepath.Separator)) {
		return map[string]any{"error": "path must be inside the session working directory"}, nil
	}

	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	return map[string]any{"result": fmt.Sprintf("wrote %d bytes to %s", len(content), abs)}, nil
}

func (t *WriteFileTool) IsInteractive(map[string]any) (bool, error) { return false, nil }

// Writing a file is a mutation: require approval.
func (t *WriteFileTool) CheckModifiesResource(map[string]any) string { return "yes" }

// ReadFileTool lets the model read files from the session working directory.
type ReadFileTool struct{}

func NewReadFileTool() *ReadFileTool { return &ReadFileTool{} }

func (t *ReadFileTool) Name() string { return "read_file" }

func (t *ReadFileTool) Description() string {
	return "Reads a file from the session working directory."
}

func (t *ReadFileTool) FunctionDefinition() *gollm.FunctionDefinition {
	return &gollm.FunctionDefinition{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &gollm.Schema{
			Type: gollm.TypeObject,
			Properties: map[string]*gollm.Schema{
				"path": {
					Type:        gollm.TypeString,
					Description: "File path, relative to the session working directory (or absolute within it).",
				},
			},
			Required: []string{"path"},
		},
	}
}

func (t *ReadFileTool) Run(ctx context.Context, args map[string]any) (any, error) {
	pathVal, ok := args["path"]
	if !ok || pathVal == nil {
		return map[string]any{"error": "path not provided"}, nil
	}
	path, ok := pathVal.(string)
	if !ok {
		return map[string]any{"error": fmt.Sprintf("path must be a string, got %T", pathVal)}, nil
	}

	workDir, _ := ctx.Value(WorkDirKey).(string)
	if workDir == "" {
		return map[string]any{"error": "no working directory configured"}, nil
	}

	abs := path
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(workDir, path)
	}
	abs = filepath.Clean(abs)
	if abs != workDir && !strings.HasPrefix(abs, workDir+string(filepath.Separator)) {
		return map[string]any{"error": "path must be inside the session working directory"}, nil
	}

	data, err := os.ReadFile(abs)
	if err != nil {
		return map[string]any{"error": err.Error()}, nil
	}
	return map[string]any{"content": string(data)}, nil
}

func (t *ReadFileTool) IsInteractive(map[string]any) (bool, error) { return false, nil }

// Reading is read-only.
func (t *ReadFileTool) CheckModifiesResource(map[string]any) string { return "no" }
