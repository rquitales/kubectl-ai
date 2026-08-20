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
	"io"
	"os"
	"regexp"
	"strings"
)

// maxFileMentionBytes caps how much of an @-mentioned file is inlined into
// the prompt sent to the LLM.
const maxFileMentionBytes = 200 * 1024

// wordRe matches maximal runs of non-whitespace (i.e. the tokens that may be
// @file mentions). Replacement preserves all other text verbatim.
var wordRe = regexp.MustCompile(`\S+`)

// fileAttachment records a file inlined into a query via an @path mention.
type fileAttachment struct {
	Path      string
	Content   string
	Truncated bool
}

// fencedBlock renders the attachment for the LLM-bound query.
func (a fileAttachment) fencedBlock() string {
	block := "\n\n" + a.Path + ":\n```\n" + a.Content + "\n```\n"
	if a.Truncated {
		block += "[truncated]\n"
	}
	return block
}

// expandFileMentions expands @path tokens in the query: each token that
// resolves to an existing readable regular file (absolute or relative to the
// process working directory) is replaced with a fenced block of the file's
// content. Tokens that do not resolve to a file are left as-is.
func expandFileMentions(query string) (expanded string, attachments []fileAttachment) {
	if !strings.Contains(query, "@") {
		return query, nil
	}
	expanded = wordRe.ReplaceAllStringFunc(query, func(token string) string {
		attachment, ok := readFileMention(token)
		if !ok {
			return token
		}
		attachments = append(attachments, attachment)
		return attachment.fencedBlock()
	})
	return expanded, attachments
}

// displayQueryWithChips builds the user-visible form of the query, replacing
// each resolved @path token with a compact "[+path]" chip. attachments must
// be the ones returned by expandFileMentions for the same query (they are
// recorded in token order).
func displayQueryWithChips(query string, attachments []fileAttachment) string {
	if len(attachments) == 0 {
		return query
	}
	next := 0
	return wordRe.ReplaceAllStringFunc(query, func(token string) string {
		if next < len(attachments) && token == "@"+attachments[next].Path {
			chip := "[+" + attachments[next].Path + "]"
			next++
			return chip
		}
		return token
	})
}

// readFileMention reads the file referenced by an @path token. ok is false
// when the token is not an @mention or does not resolve to a readable
// regular file.
func readFileMention(token string) (attachment fileAttachment, ok bool) {
	if !strings.HasPrefix(token, "@") || len(token) == 1 {
		return fileAttachment{}, false
	}
	path := token[1:]
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return fileAttachment{}, false
	}
	f, err := os.Open(path)
	if err != nil {
		return fileAttachment{}, false
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxFileMentionBytes+1))
	if err != nil {
		return fileAttachment{}, false
	}
	truncated := len(data) > maxFileMentionBytes
	if truncated {
		data = data[:maxFileMentionBytes]
	}
	return fileAttachment{Path: path, Content: string(data), Truncated: truncated}, true
}
