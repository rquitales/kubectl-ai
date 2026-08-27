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
	"math/rand"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"k8s.io/klog/v2"
)

type SessionManager struct {
	store Store
}

func NewSessionManager(backend string) (*SessionManager, error) {
	var store Store
	var err error

	if backend == "" {
		// Try filesystem first
		store, err = NewStore("filesystem")
		if err != nil {
			// Fallback to memory
			store, err = NewStore("memory")
		}
	} else {
		store, err = NewStore(backend)
	}

	if err != nil {
		return nil, err
	}
	return &SessionManager{store: store}, nil
}

func (sm *SessionManager) NewSession(meta Metadata) (*api.Session, error) {
	// Session IDs embed random suffixes; retry a few times on the (rare)
	// birthday-paradox collision instead of clobbering or failing.
	var lastErr error
	for attempt := 0; attempt < 10; attempt++ {
		suffix := fmt.Sprintf("%04d", rand.Intn(10000))
		sessionID := time.Now().Format("20060102") + "-" + suffix

		now := time.Now()
		session := &api.Session{
			ID:           sessionID,
			Name:         "", // unnamed until content-derived naming on exit
			ProviderID:   meta.ProviderID,
			ModelID:      meta.ModelID,
			AgentState:   api.AgentStateIdle,
			CreatedAt:    now,
			LastModified: now,
		}

		if err := sm.store.CreateSession(session); err != nil {
			if errors.Is(err, ErrSessionExists) {
				lastErr = err
				continue
			}
			return nil, err
		}

		return session, nil
	}
	return nil, fmt.Errorf("failed to allocate a unique session ID: %w", lastErr)
}

func (sm *SessionManager) ListSessions() ([]*api.Session, error) {
	return sm.store.ListSessions()
}

// ForkSession copies a session's metadata and full message history under a
// new ID and returns the new session — try a risky approach, then go back.
func (sm *SessionManager) ForkSession(sourceID string) (*api.Session, error) {
	src, err := sm.FindSessionByID(sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to find session %q to fork: %w", sourceID, err)
	}

	fork, err := sm.NewSession(Metadata{
		ProviderID: src.ProviderID,
		ModelID:    src.ModelID,
	})
	if err != nil {
		return nil, err
	}

	if src.Name != "" {
		if err := sm.SetSessionName(fork.ID, src.Name+" (fork)", false); err != nil {
			klog.Warningf("failed to name forked session: %v", err)
		}
	}

	// Copy the message history.
	if src.ChatMessageStore != nil && fork.ChatMessageStore != nil {
		for _, msg := range src.ChatMessageStore.ChatMessages() {
			if err := fork.ChatMessageStore.AddChatMessage(msg); err != nil {
				return nil, fmt.Errorf("failed to copy message history: %w", err)
			}
		}
	}

	return fork, nil
}

func (sm *SessionManager) FindSessionByID(id string) (*api.Session, error) {
	return sm.store.GetSession(id)
}

func (sm *SessionManager) DeleteSession(id string) error {
	return sm.store.DeleteSession(id)
}

// RenameSession sets a new display name for the session with the given ID.
// The name is sanitized before being stored. It does not change the
// ManuallyNamed flag (used by content-derived auto-naming).
func (sm *SessionManager) RenameSession(id, name string) error {
	name = SanitizeSessionName(name)
	if name == "" {
		return errors.New("session name cannot be empty")
	}
	session, err := sm.store.GetSession(id)
	if err != nil {
		return err
	}
	session.Name = name
	return sm.store.UpdateSession(session)
}

// SetSessionName sets a display name and records whether the name was
// chosen manually (true) or derived automatically from content (false).
func (sm *SessionManager) SetSessionName(id, name string, manuallyNamed bool) error {
	name = SanitizeSessionName(name)
	if name == "" {
		return errors.New("session name cannot be empty")
	}
	session, err := sm.store.GetSession(id)
	if err != nil {
		return err
	}
	session.Name = name
	session.ManuallyNamed = manuallyNamed
	return sm.store.UpdateSession(session)
}

func (sm *SessionManager) GetLatestSession() (*api.Session, error) {
	sessions, err := sm.store.ListSessions()
	if err != nil {
		return nil, err
	}

	if len(sessions) == 0 {
		return nil, nil
	}

	latest := sessions[0]
	for _, session := range sessions[1:] {
		if session.LastModified.After(latest.LastModified) {
			latest = session
		}
	}

	return latest, nil
}

func (sm *SessionManager) UpdateLastAccessed(session *api.Session) error {
	session.LastModified = time.Now()
	return sm.store.UpdateSession(session)
}

// HasConversationMessages reports whether the messages contain any real
// conversation (at least one model-sourced message). Sessions with only
// meta/slash commands (or nothing) are considered empty.
func HasConversationMessages(messages []*api.Message) bool {
	for _, m := range messages {
		if m.Source == api.MessageSourceModel {
			return true
		}
	}
	return false
}

// PruneEmptySessions deletes every session that has no real conversation
// (no model-sourced messages), returning the number deleted. Used to sweep
// accumulated empty sessions on startup.
func (sm *SessionManager) PruneEmptySessions(excludeIDs ...string) (int, error) {
	exclude := make(map[string]bool, len(excludeIDs))
	for _, id := range excludeIDs {
		exclude[id] = true
	}
	sessionList, err := sm.store.ListSessions()
	if err != nil {
		return 0, err
	}
	pruned := 0
	for _, s := range sessionList {
		if exclude[s.ID] {
			continue
		}
		if HasConversationMessages(s.ChatMessageStore.ChatMessages()) {
			continue
		}
		// Never delete a session whose on-disk history has content: a
		// history that fails to parse (or parses to only non-conversation
		// messages after skipping torn lines) must survive the sweep —
		// deleting it would be unrecoverable data loss.
		if fs, ok := s.ChatMessageStore.(interface{ HistoryPath() string }); ok {
			if info, err := os.Stat(fs.HistoryPath()); err == nil && info.Size() > 0 {
				klog.Warningf("Keeping session %s: history file exists (%d bytes) but no conversation messages parsed", s.ID, info.Size())
				continue
			}
		}
		if err := sm.store.DeleteSession(s.ID); err != nil {
			return pruned, fmt.Errorf("failed to delete session %s: %w", s.ID, err)
		}
		pruned++
	}
	return pruned, nil
}
