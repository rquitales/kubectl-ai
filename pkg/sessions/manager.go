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
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
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
			Name:         generateSessionName(),
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

func (sm *SessionManager) FindSessionByID(id string) (*api.Session, error) {
	return sm.store.GetSession(id)
}

func (sm *SessionManager) DeleteSession(id string) error {
	return sm.store.DeleteSession(id)
}

// RenameSession sets a new display name for the session with the given ID.
// The name is sanitized before being stored.
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
