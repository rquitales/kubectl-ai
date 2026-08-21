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

package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExpandFileMentions(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	if err := os.WriteFile("small.txt", []byte("hello from small"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("other.txt", []byte("hello from other"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir("somedir", 0755); err != nil {
		t.Fatal(err)
	}

	t.Run("no mentions", func(t *testing.T) {
		expanded, attachments := expandFileMentions("just a plain query")
		if expanded != "just a plain query" {
			t.Errorf("expanded = %q, want query unchanged", expanded)
		}
		if attachments != nil {
			t.Errorf("attachments = %v, want nil", attachments)
		}
	})

	t.Run("non-file tokens left alone", func(t *testing.T) {
		query := "check @missing.txt and user@example.com and @somedir and @ please"
		expanded, attachments := expandFileMentions(query)
		if expanded != query {
			t.Errorf("expanded = %q, want query unchanged %q", expanded, query)
		}
		if len(attachments) != 0 {
			t.Errorf("attachments = %v, want none", attachments)
		}
	})

	t.Run("relative path mention", func(t *testing.T) {
		expanded, attachments := expandFileMentions("explain @small.txt please")
		want := "explain \n\nsmall.txt:\n```\nhello from small\n```\n please"
		if expanded != want {
			t.Errorf("expanded = %q, want %q", expanded, want)
		}
		if len(attachments) != 1 || attachments[0].Path != "small.txt" || attachments[0].Content != "hello from small" || attachments[0].Truncated {
			t.Errorf("attachments = %+v, want one small.txt attachment", attachments)
		}
	})

	t.Run("absolute path mention", func(t *testing.T) {
		abs := filepath.Join(dir, "small.txt")
		expanded, attachments := expandFileMentions("explain @" + abs)
		if !strings.Contains(expanded, abs+":\n```\nhello from small\n```") {
			t.Errorf("expanded = %q, want fenced block for %q", expanded, abs)
		}
		if len(attachments) != 1 || attachments[0].Path != abs {
			t.Errorf("attachments = %+v, want one attachment for %q", attachments, abs)
		}
	})

	t.Run("multiple mentions preserve whitespace", func(t *testing.T) {
		query := "first @small.txt\nsecond\t@other.txt"
		expanded, attachments := expandFileMentions(query)
		if len(attachments) != 2 {
			t.Fatalf("attachments = %+v, want two", attachments)
		}
		if !strings.Contains(expanded, "\nsecond\t\n\nother.txt:") {
			t.Errorf("expanded = %q, want original whitespace preserved", expanded)
		}
	})

	t.Run("large file is capped and marked truncated", func(t *testing.T) {
		if err := os.WriteFile("large.bin", []byte(strings.Repeat("x", maxFileMentionBytes+100)), 0644); err != nil {
			t.Fatal(err)
		}
		expanded, attachments := expandFileMentions("see @large.bin")
		if len(attachments) != 1 {
			t.Fatalf("attachments = %+v, want one", attachments)
		}
		att := attachments[0]
		if !att.Truncated {
			t.Error("expected Truncated to be true")
		}
		if len(att.Content) != maxFileMentionBytes {
			t.Errorf("content length = %d, want %d", len(att.Content), maxFileMentionBytes)
		}
		if !strings.Contains(expanded, "[truncated]") {
			t.Errorf("expanded missing [truncated] marker: %q", expanded[:100])
		}
	})
}

func TestDisplayQueryWithChips(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	if err := os.WriteFile("a.txt", []byte("aaa"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("b.txt", []byte("bbb"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Run("no attachments leaves query unchanged", func(t *testing.T) {
		if got := displayQueryWithChips("plain @nothing-here", nil); got != "plain @nothing-here" {
			t.Errorf("got %q, want query unchanged", got)
		}
	})

	t.Run("mentions become chips, unresolved tokens stay", func(t *testing.T) {
		query := "compare @a.txt\nwith @b.txt and @missing.txt"
		_, attachments := expandFileMentions(query)
		got := displayQueryWithChips(query, attachments)
		want := "compare [+a.txt]\nwith [+b.txt] and @missing.txt"
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("repeated mention of same file", func(t *testing.T) {
		query := "@a.txt vs @a.txt"
		_, attachments := expandFileMentions(query)
		if len(attachments) != 2 {
			t.Fatalf("attachments = %+v, want two", attachments)
		}
		got := displayQueryWithChips(query, attachments)
		if want := "[+a.txt] vs [+a.txt]"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}
