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

package kube

import (
	"os"
	"path/filepath"
	"testing"
)

func writeKubeConfig(t *testing.T, currentContext string, contexts map[string]string) string {
	t.Helper()
	content := "apiVersion: v1\nkind: Config\ncurrent-context: " + currentContext + "\nclusters: []\ncontexts:\n"
	for name, ns := range contexts {
		content += "- context:\n    cluster: c\n    user: u\n"
		if ns != "" {
			content += "    namespace: " + ns + "\n"
		}
		content += "  name: " + name + "\n"
	}
	content += "users: []\n"
	path := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCurrentContext(t *testing.T) {
	path := writeKubeConfig(t, "prod", map[string]string{"prod": "payments", "staging": ""})

	context, namespace, ok := CurrentContext(path)
	if !ok {
		t.Fatal("expected ok")
	}
	if context != "prod" {
		t.Errorf("context = %q, want %q", context, "prod")
	}
	if namespace != "payments" {
		t.Errorf("namespace = %q, want %q", namespace, "payments")
	}

	if _, _, ok := CurrentContext(filepath.Join(t.TempDir(), "missing")); ok {
		t.Error("expected ok=false for a missing kubeconfig")
	}
}

func TestListContexts(t *testing.T) {
	path := writeKubeConfig(t, "prod", map[string]string{"staging": "", "prod": "", "dev": ""})

	names, err := ListContexts(path)
	if err != nil {
		t.Fatalf("ListContexts failed: %v", err)
	}
	want := []string{"dev", "prod", "staging"}
	if len(names) != len(want) {
		t.Fatalf("ListContexts = %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("ListContexts = %v, want %v", names, want)
		}
	}
}

func TestWriteOverride(t *testing.T) {
	base := writeKubeConfig(t, "prod", map[string]string{"prod": "", "staging": ""})
	out := filepath.Join(t.TempDir(), "override")

	// Context + namespace override.
	if err := WriteOverride(base, out, "staging", "payments"); err != nil {
		t.Fatalf("WriteOverride failed: %v", err)
	}
	context, namespace, ok := CurrentContext(out)
	if !ok || context != "staging" {
		t.Errorf("override context = %q (ok=%v), want staging", context, ok)
	}
	if namespace != "payments" {
		t.Errorf("override namespace = %q, want payments", namespace)
	}
	// The base file is untouched.
	if context, _, _ := CurrentContext(base); context != "prod" {
		t.Errorf("base context changed to %q, want prod untouched", context)
	}

	// Namespace-only override keeps the base's current context.
	out2 := filepath.Join(t.TempDir(), "override2")
	if err := WriteOverride(base, out2, "", "dev"); err != nil {
		t.Fatalf("WriteOverride failed: %v", err)
	}
	if context, namespace, _ := CurrentContext(out2); context != "prod" || namespace != "dev" {
		t.Errorf("override2 = %q/%q, want prod/dev", context, namespace)
	}

	// Unknown context errors.
	if err := WriteOverride(base, filepath.Join(t.TempDir(), "x"), "nope", ""); err == nil {
		t.Error("expected error for an unknown context")
	}
}
