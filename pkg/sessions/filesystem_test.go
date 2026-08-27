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
	"os"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
)

func TestFilesystemStoreSessionNameRoundTrip(t *testing.T) {
	store := newFilesystemStore(t.TempDir())

	now := time.Now().Truncate(time.Second)
	session := &api.Session{
		ID:           "20260101-0001",
		Name:         "my debug session",
		ProviderID:   "gemini",
		ModelID:      "gemini-2.5-pro",
		AgentState:   api.AgentStateIdle,
		CreatedAt:    now,
		LastModified: now,
	}
	if err := store.CreateSession(session); err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	loaded, err := store.GetSession(session.ID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if loaded.Name != session.Name {
		t.Errorf("loaded session name = %q, want %q", loaded.Name, session.Name)
	}

	// Update the name and verify it persists.
	loaded.Name = "renamed session"
	if err := store.UpdateSession(loaded); err != nil {
		t.Fatalf("UpdateSession failed: %v", err)
	}
	reloaded, err := store.GetSession(session.ID)
	if err != nil {
		t.Fatalf("GetSession after update failed: %v", err)
	}
	if reloaded.Name != "renamed session" {
		t.Errorf("reloaded session name = %q, want %q", reloaded.Name, "renamed session")
	}
}

func TestSessionManagerRenameSession(t *testing.T) {
	store := newFilesystemStore(t.TempDir())
	manager := &SessionManager{store: store}

	session := &api.Session{
		ID:           "20260101-0002",
		Name:         "before",
		CreatedAt:    time.Now(),
		LastModified: time.Now(),
	}
	if err := store.CreateSession(session); err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	if err := manager.RenameSession(session.ID, "after"); err != nil {
		t.Fatalf("RenameSession failed: %v", err)
	}

	loaded, err := store.GetSession(session.ID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if loaded.Name != "after" {
		t.Errorf("session name = %q, want %q", loaded.Name, "after")
	}
	// Rename must preserve the other metadata fields.
	if !loaded.CreatedAt.Equal(session.CreatedAt) {
		t.Errorf("CreatedAt changed by rename: %v -> %v", session.CreatedAt, loaded.CreatedAt)
	}

	if err := manager.RenameSession("does-not-exist", "x"); err == nil {
		t.Error("expected error renaming non-existent session, got nil")
	}
	if err := manager.RenameSession(session.ID, "   "); err == nil {
		t.Error("expected error renaming to empty name, got nil")
	}
}

func TestRenameSessionSanitizesName(t *testing.T) {
	store := newFilesystemStore(t.TempDir())
	manager := &SessionManager{store: store}

	session := &api.Session{
		ID:           "20260101-0003",
		CreatedAt:    time.Now(),
		LastModified: time.Now(),
	}
	if err := store.CreateSession(session); err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	// Newlines and ANSI escape sequences must not survive a rename.
	if err := manager.RenameSession(session.ID, "foo\nbar\x1b[31m"); err != nil {
		t.Fatalf("RenameSession failed: %v", err)
	}
	loaded, err := store.GetSession(session.ID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if loaded.Name != "foobar[31m" {
		t.Errorf("session name = %q, want %q", loaded.Name, "foobar[31m")
	}
}

func TestSanitizeSessionName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", "plain"},
		{"  padded  ", "padded"},
		{"with\nnewline", "withnewline"},
		{"with\ttab", "withtab"},
		{"\x1b[2Jescape", "[2Jescape"},
		{"", ""},
		{"\n\t ", ""},
	}
	for _, c := range cases {
		if got := SanitizeSessionName(c.in); got != c.want {
			t.Errorf("SanitizeSessionName(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	long := strings.Repeat("a", 200)
	if got := SanitizeSessionName(long); len([]rune(got)) != 128 {
		t.Errorf("expected truncation to 128 runes, got %d", len([]rune(got)))
	}
	// Multi-byte runes must not be split mid-sequence.
	mb := strings.Repeat("界", 200)
	got := SanitizeSessionName(mb)
	if len([]rune(got)) != 128 || !utf8.ValidString(got) {
		t.Errorf("expected 128 valid runes, got %d (valid=%v)", len([]rune(got)), utf8.ValidString(got))
	}
}

func TestCreateSessionRefusesToClobberExisting(t *testing.T) {
	store := newFilesystemStore(t.TempDir())

	session := &api.Session{
		ID:           "20260101-0004",
		Name:         "original",
		CreatedAt:    time.Now(),
		LastModified: time.Now(),
	}
	if err := store.CreateSession(session); err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}

	dupe := &api.Session{
		ID:           session.ID,
		Name:         "clobbered",
		CreatedAt:    time.Now(),
		LastModified: time.Now(),
	}
	err := store.CreateSession(dupe)
	if !errors.Is(err, ErrSessionExists) {
		t.Fatalf("expected ErrSessionExists, got %v", err)
	}

	loaded, err := store.GetSession(session.ID)
	if err != nil {
		t.Fatalf("GetSession failed: %v", err)
	}
	if loaded.Name != "original" {
		t.Errorf("existing session was clobbered: name = %q", loaded.Name)
	}
}

func TestFileChatMessageStoreLargeMessages(t *testing.T) {
	store := NewFileChatMessageStore(t.TempDir())

	small := &api.Message{ID: "1", Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "hi", Timestamp: time.Now()}
	if err := store.AddChatMessage(small); err != nil {
		t.Fatal(err)
	}

	// A message whose JSON line exceeds bufio's default 64KB token limit:
	// previously this made the whole history unreadable.
	big := &api.Message{ID: "2", Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: strings.Repeat("x", 200*1024), Timestamp: time.Now()}
	if err := store.AddChatMessage(big); err != nil {
		t.Fatal(err)
	}

	small2 := &api.Message{ID: "3", Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "after big", Timestamp: time.Now()}
	if err := store.AddChatMessage(small2); err != nil {
		t.Fatal(err)
	}

	msgs := store.ChatMessages()
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d (history wiped by oversized line?)", len(msgs))
	}
	if msgs[1].Payload.(string) != big.Payload {
		t.Errorf("big message payload corrupted")
	}
	if msgs[2].Payload != "after big" {
		t.Errorf("message after the big one = %q", msgs[2].Payload)
	}
}

func TestHasConversationMessages(t *testing.T) {
	cases := []struct {
		name string
		msgs []*api.Message
		want bool
	}{
		{"none", nil, false},
		{"user only", []*api.Message{{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "hi"}}, false},
		{"model text", []*api.Message{{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "hi"}}, true},
		{"model tool call", []*api.Message{{Source: api.MessageSourceModel, Type: api.MessageTypeToolCallRequest, Payload: "kubectl get pods"}}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := HasConversationMessages(c.msgs); got != c.want {
				t.Errorf("HasConversationMessages() = %v, want %v", got, c.want)
			}
		})
	}
}

func TestPruneEmptySessions(t *testing.T) {
	manager := &SessionManager{store: newMemoryStore()}

	empty1, _ := manager.NewSession(Metadata{ModelID: "m"})
	empty2, _ := manager.NewSession(Metadata{ModelID: "m"})
	withConv, _ := manager.NewSession(Metadata{ModelID: "m"})
	_ = withConv.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "real"})
	metaOnly, _ := manager.NewSession(Metadata{ModelID: "m"})
	_ = metaOnly.ChatMessageStore.AddChatMessage(&api.Message{Source: api.MessageSourceUser, Type: api.MessageTypeText, Payload: "/sessions"})

	pruned, err := manager.PruneEmptySessions()
	if err != nil {
		t.Fatalf("PruneEmptySessions failed: %v", err)
	}
	if pruned != 3 {
		t.Errorf("pruned = %d, want 3", pruned)
	}

	remaining, err := manager.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(remaining) != 1 || remaining[0].ID != withConv.ID {
		t.Errorf("expected only the conversation session to remain, got %v", remaining)
	}
	_ = empty1
	_ = empty2
}

func TestPruneEmptySessionsExcludesResumeTarget(t *testing.T) {
	// Regression: the startup sweep ran before resume resolution, so
	// --resume-session of a session with no model reply yet (e.g. an
	// LLM-error-only first turn) failed with "session not found".
	manager := &SessionManager{store: newMemoryStore()}

	empty, _ := manager.NewSession(Metadata{ModelID: "m"})
	resumeTarget, _ := manager.NewSession(Metadata{ModelID: "m"})

	pruned, err := manager.PruneEmptySessions(resumeTarget.ID)
	if err != nil {
		t.Fatalf("PruneEmptySessions failed: %v", err)
	}
	if pruned != 1 {
		t.Errorf("pruned = %d, want 1 (only the non-excluded empty session)", pruned)
	}
	if got, err := manager.FindSessionByID(resumeTarget.ID); err != nil || got == nil {
		t.Errorf("resume target must survive the sweep: %v", err)
	}
	if got, _ := manager.FindSessionByID(empty.ID); got != nil {
		t.Error("non-excluded empty session should have been pruned")
	}
}

func TestTornHistoryLineDoesNotWipeSession(t *testing.T) {
	// Regression: one torn JSONL line (crash mid-append) used to fail the
	// whole history parse, the session read as empty, and the startup prune
	// then deleted it — permanent loss of the full transcript.
	manager := &SessionManager{store: newMemoryStore()}
	_ = manager // (memory store covered below via the fs store)

	dir := t.TempDir()
	store := &FileChatMessageStore{Path: dir}
	good := &api.Message{Source: api.MessageSourceModel, Type: api.MessageTypeText, Payload: "real answer"}
	if err := store.AddChatMessage(good); err != nil {
		t.Fatalf("AddChatMessage: %v", err)
	}
	// Simulate a crash mid-append: torn final line.
	f, err := os.OpenFile(store.HistoryPath(), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"ID":"x","Source":"model","Type":"text","Paylo`); err != nil {
		t.Fatal(err)
	}
	f.Close()

	msgs := store.ChatMessages()
	if len(msgs) != 1 || msgs[0].Payload != "real answer" {
		t.Fatalf("torn line wiped the history: got %d messages", len(msgs))
	}
}

func TestPruneNeverDeletesUnreadableHistory(t *testing.T) {
	// A session whose history.json has content but parses to zero
	// conversation messages (fully torn/corrupt) must survive the sweep.
	manager := &SessionManager{store: newFilesystemStore(t.TempDir())}

	sess, err := manager.NewSession(Metadata{ModelID: "m"})
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	// Write garbage directly into the history file.
	fs, ok := sess.ChatMessageStore.(interface{ HistoryPath() string })
	if !ok {
		t.Fatalf("session store does not expose HistoryPath")
	}
	if err := os.WriteFile(fs.HistoryPath(), []byte("not-json\n{\"partial\":"), 0o644); err != nil {
		t.Fatal(err)
	}

	pruned, err := manager.PruneEmptySessions()
	if err != nil {
		t.Fatalf("PruneEmptySessions: %v", err)
	}
	if pruned != 0 {
		t.Errorf("pruned = %d, want 0 (corrupt-history session must survive)", pruned)
	}
	if got, _ := manager.FindSessionByID(sess.ID); got == nil {
		t.Error("corrupt-history session was pruned — data loss")
	}
}
