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
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAndReadFileTool(t *testing.T) {
	workDir := t.TempDir()
	ctx := context.WithValue(context.Background(), WorkDirKey, workDir)

	w := NewWriteFileTool()
	res, err := w.Run(ctx, map[string]any{"path": "manifests/pod.yaml", "content": "apiVersion: v1\n"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if m, _ := res.(map[string]any); m["error"] != nil {
		t.Fatalf("write error: %v", m["error"])
	}
	data, err := os.ReadFile(filepath.Join(workDir, "manifests/pod.yaml"))
	if err != nil || string(data) != "apiVersion: v1\n" {
		t.Fatalf("file content = %q, err=%v", data, err)
	}

	r := NewReadFileTool()
	res, err = r.Run(ctx, map[string]any{"path": "manifests/pod.yaml"})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if m, _ := res.(map[string]any); m["content"] != "apiVersion: v1\n" {
		t.Fatalf("read content = %v", m)
	}
}

func TestFileToolsConfineToWorkDir(t *testing.T) {
	workDir := t.TempDir()
	ctx := context.WithValue(context.Background(), WorkDirKey, workDir)

	for _, p := range []string{"../escape.yaml", "/etc/passwd", filepath.Join(workDir, "..", "escape2.yaml")} {
		res, _ := NewWriteFileTool().Run(ctx, map[string]any{"path": p, "content": "x"})
		if m, _ := res.(map[string]any); m["error"] == nil {
			t.Errorf("write to %q escaped the workdir", p)
		}
		res, _ = NewReadFileTool().Run(ctx, map[string]any{"path": p})
		if m, _ := res.(map[string]any); m["error"] == nil {
			t.Errorf("read of %q escaped the workdir", p)
		}
	}
}

func TestFileToolsMalformedArgs(t *testing.T) {
	ctx := context.WithValue(context.Background(), WorkDirKey, t.TempDir())
	for _, tool := range []Tool{NewWriteFileTool(), NewReadFileTool()} {
		if _, err := tool.Run(ctx, map[string]any{"path": 42}); err != nil {
			t.Fatalf("%s: Go error instead of tool error", tool.Name())
		}
	}
}
