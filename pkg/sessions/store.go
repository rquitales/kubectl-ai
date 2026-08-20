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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
)

const sessionsDirName = "sessions"

// ErrSessionExists is returned when creating a session whose ID is already
// taken; callers may retry with a fresh ID.
var ErrSessionExists = errors.New("session already exists")

// maxSessionNameLen caps session names for display purposes.
const maxSessionNameLen = 128

// SanitizeSessionName makes a session name safe for display and storage: it
// strips control characters (so e.g. newlines can't break single-line UI
// elements and terminal escape sequences can't be injected), trims
// whitespace, and truncates to maxSessionNameLen runes.
func SanitizeSessionName(name string) string {
	name = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, name)
	name = strings.TrimSpace(name)
	if r := []rune(name); len(r) > maxSessionNameLen {
		name = string(r[:maxSessionNameLen])
	}
	return name
}

type Metadata struct {
	Name          string    `json:"name,omitempty"`
	ManuallyNamed bool      `json:"manuallyNamed,omitempty"`
	ProviderID    string    `json:"providerID"`
	ModelID       string    `json:"modelID"`
	CreatedAt     time.Time `json:"createdAt"`
	LastAccessed  time.Time `json:"lastAccessed"`
}

var defaultMemoryStore Store = newMemoryStore()

type Store interface {
	GetSession(id string) (*api.Session, error)
	CreateSession(session *api.Session) error
	UpdateSession(session *api.Session) error
	ListSessions() ([]*api.Session, error)
	DeleteSession(id string) error
}

func NewStore(backend string) (Store, error) {
	switch backend {
	case "memory":
		return defaultMemoryStore, nil
	case "filesystem":
		basePath, err := defaultFilesystemBasePath()
		if err != nil {
			return nil, err
		}
		if err := os.MkdirAll(basePath, 0o755); err != nil {
			return nil, err
		}
		return newFilesystemStore(basePath), nil
	default:
		return nil, fmt.Errorf("unsupported sessions backend: %s", backend)
	}
}

func defaultFilesystemBasePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".kubectl-ai", sessionsDirName), nil
}
