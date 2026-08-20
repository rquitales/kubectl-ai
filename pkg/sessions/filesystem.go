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
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"sigs.k8s.io/yaml"
)

type filesystemStore struct {
	basePath string
}

func newFilesystemStore(basePath string) Store {
	return &filesystemStore{basePath: basePath}
}

func (f *filesystemStore) GetSession(id string) (*api.Session, error) {
	sessionPath := filepath.Join(f.basePath, id)
	metadataPath := filepath.Join(sessionPath, "metadata.yaml")

	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, errors.New("session not found")
		}
		return nil, err
	}

	var meta Metadata
	if err := yaml.Unmarshal(metadataBytes, &meta); err != nil {
		return nil, err
	}

	chatStore := NewFileChatMessageStore(sessionPath)
	return &api.Session{
		ID:               id,
		Name:             meta.Name,
		ManuallyNamed:    meta.ManuallyNamed,
		ProviderID:       meta.ProviderID,
		ModelID:          meta.ModelID,
		AgentState:       api.AgentStateIdle,
		CreatedAt:        meta.CreatedAt,
		LastModified:     meta.LastAccessed,
		ChatMessageStore: chatStore,
	}, nil
}

func (f *filesystemStore) CreateSession(session *api.Session) error {
	sessionPath := filepath.Join(f.basePath, session.ID)
	if _, err := os.Stat(sessionPath); err == nil {
		// Refuse to clobber an existing session's metadata.
		return fmt.Errorf("%w: %s", ErrSessionExists, session.ID)
	}
	if err := os.MkdirAll(sessionPath, 0o755); err != nil {
		return err
	}

	chatStore := NewFileChatMessageStore(sessionPath)
	session.ChatMessageStore = chatStore

	meta := Metadata{
		Name:          session.Name,
		ManuallyNamed: session.ManuallyNamed,
		ProviderID:    session.ProviderID,
		ModelID:       session.ModelID,
		CreatedAt:     session.CreatedAt,
		LastAccessed:  session.LastModified,
	}

	data, err := yaml.Marshal(meta)
	if err != nil {
		return err
	}

	return writeFileAtomic(filepath.Join(sessionPath, "metadata.yaml"), data, 0o644)
}

func (f *filesystemStore) UpdateSession(session *api.Session) error {
	sessionPath := filepath.Join(f.basePath, session.ID)
	metadataPath := filepath.Join(sessionPath, "metadata.yaml")

	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return errors.New("session not found")
		}
		return err
	}

	var meta Metadata
	if err := yaml.Unmarshal(metadataBytes, &meta); err != nil {
		return err
	}

	meta.Name = session.Name
	meta.ManuallyNamed = session.ManuallyNamed
	meta.ProviderID = session.ProviderID
	meta.ModelID = session.ModelID
	meta.LastAccessed = session.LastModified

	data, err := yaml.Marshal(meta)
	if err != nil {
		return err
	}

	return writeFileAtomic(metadataPath, data, 0o644)
}

// writeFileAtomic writes data to path atomically via a temp file + rename, so
// concurrent readers never observe a truncated/partial metadata file.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func (f *filesystemStore) ListSessions() ([]*api.Session, error) {
	entries, err := os.ReadDir(f.basePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []*api.Session{}, nil
		}
		return nil, err
	}

	sessions := make([]*api.Session, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		session, err := f.GetSession(entry.Name())
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].LastModified.After(sessions[j].LastModified)
	})

	return sessions, nil
}

func (f *filesystemStore) DeleteSession(id string) error {
	sessionPath := filepath.Join(f.basePath, id)
	return os.RemoveAll(sessionPath)
}

// FileChatMessageStore implements api.ChatMessageStore by persisting history to disk.
type FileChatMessageStore struct {
	Path string
	mu   sync.Mutex
}

// NewFileChatMessageStore creates a new file-backed chat message store.
func NewFileChatMessageStore(path string) *FileChatMessageStore {
	return &FileChatMessageStore{Path: path}
}

// HistoryPath returns the location of the history file for this session.
func (s *FileChatMessageStore) HistoryPath() string {
	return filepath.Join(s.Path, "history.json")
}

// AddChatMessage appends a message to the existing history on disk.
func (s *FileChatMessageStore) AddChatMessage(record *api.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Ensure directory exists
	if err := os.MkdirAll(s.Path, 0o755); err != nil {
		return err
	}

	path := s.HistoryPath()

	// Check for legacy format and migrate if needed
	isLegacy := false
	if f, err := os.Open(path); err == nil {
		buf := make([]byte, 1)
		if _, err := f.Read(buf); err == nil && buf[0] == '[' {
			isLegacy = true
		}
		f.Close()
	}

	if isLegacy {
		// Read all messages (handles legacy format)
		messages, err := s.readMessages()
		if err != nil {
			return err
		}
		messages = append(messages, record)
		return s.writeMessages(messages)
	}

	// Normal append for JSONL or new files. The record and its trailing
	// newline go out in a single Write so concurrent readers of the file
	// never observe a torn line.
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.Write(data)
	return err
}

// SetChatMessages replaces the history file with the provided messages.
func (s *FileChatMessageStore) SetChatMessages(newHistory []*api.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.writeMessages(newHistory)
}

// ChatMessages returns all persisted chat messages.
func (s *FileChatMessageStore) ChatMessages() []*api.Message {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, err := s.readMessages()
	if err != nil {
		return []*api.Message{}
	}
	return messages
}

// ClearChatMessages truncates the history file, leaving an empty array.
func (s *FileChatMessageStore) ClearChatMessages() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.writeMessages([]*api.Message{})
}

func (s *FileChatMessageStore) readMessages() ([]*api.Message, error) {
	path := s.HistoryPath()
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return []*api.Message{}, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Check if the file is empty
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if stat.Size() == 0 {
		return []*api.Message{}, nil
	}

	// Peek at the first byte to determine format
	// If it starts with '[', it's a legacy JSON array
	// Otherwise, assume JSONL
	buf := make([]byte, 1)
	if _, err := f.Read(buf); err != nil {
		return nil, err
	}
	// Reset file pointer
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}

	var messages []*api.Message

	if buf[0] == '[' {
		// Legacy JSON array format
		decoder := json.NewDecoder(f)
		if err := decoder.Decode(&messages); err != nil {
			return nil, err
		}
		return messages, nil
	}

	// JSONL format
	scanner := bufio.NewScanner(f)
	// LLM responses and tool outputs can be large; the default 64KB token
	// limit would make the whole history unreadable once a single message
	// exceeds it. Allow up to 10MB per line.
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var msg api.Message
		if err := json.Unmarshal(line, &msg); err != nil {
			return nil, err
		}
		messages = append(messages, &msg)
	}

	if err := scanner.Err(); err != nil {
		// A partial history is strictly better than no history: return what
		// was read so far instead of wiping the whole transcript.
		return messages, nil
	}

	return messages, nil
}

func (s *FileChatMessageStore) writeMessages(messages []*api.Message) error {
	if err := os.MkdirAll(s.Path, 0o755); err != nil {
		return err
	}

	f, err := os.OpenFile(s.HistoryPath(), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, msg := range messages {
		data, err := json.Marshal(msg)
		if err != nil {
			return err
		}
		if _, err := f.Write(data); err != nil {
			return err
		}
		if _, err := f.WriteString("\n"); err != nil {
			return err
		}
	}
	return nil
}
