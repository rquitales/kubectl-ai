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

package sessions

import (
	"strings"
	"testing"
)

func TestGenerateSessionNameFormat(t *testing.T) {
	adj := make(map[string]bool, len(sessionAdjectives))
	for _, a := range sessionAdjectives {
		adj[a] = true
	}
	noun := make(map[string]bool, len(sessionNouns))
	for _, n := range sessionNouns {
		noun[n] = true
	}

	for i := 0; i < 200; i++ {
		name := generateSessionName()
		parts := strings.Split(name, "-")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			t.Fatalf("name %q is not adjective-noun", name)
		}
		if !adj[parts[0]] {
			t.Errorf("name %q: %q is not a known adjective", name, parts[0])
		}
		if !noun[parts[1]] {
			t.Errorf("name %q: %q is not a known noun", name, parts[1])
		}
	}
}

func TestNewSessionGetsFriendlyName(t *testing.T) {
	manager := &SessionManager{store: newMemoryStore()}
	s, err := manager.NewSession(Metadata{ModelID: "m"})
	if err != nil {
		t.Fatalf("NewSession failed: %v", err)
	}
	if s.Name == "" || strings.HasPrefix(s.Name, "Session ") {
		t.Errorf("expected friendly name, got %q", s.Name)
	}
	if strings.Contains(s.Name, " ") {
		t.Errorf("expected no spaces in session name, got %q", s.Name)
	}
}
