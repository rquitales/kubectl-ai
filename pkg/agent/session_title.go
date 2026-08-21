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
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubectl-ai/gollm"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/sessions"
	"k8s.io/klog/v2"
)

// titleSnippetLen caps each conversation excerpt embedded in the title prompt.
const titleSnippetLen = 300

// maybeGenerateSessionTitle asks the LLM for a short session title, at most
// once per agent lifetime, right after the first model reply — but only while
// the session is still unnamed and was not named manually. It runs in the
// background; on failure nothing happens and the exit-time deriveSessionName
// fallback still applies.
func (c *Agent) maybeGenerateSessionTitle() {
	c.sessionMu.Lock()
	if c.titleAttempted || c.Session.Name != "" || c.Session.ManuallyNamed {
		c.sessionMu.Unlock()
		return
	}
	c.titleAttempted = true
	sessionID := c.Session.ID
	userSnippet, modelSnippet := titleSnippets(c.Session.ChatMessageStore.ChatMessages())
	c.sessionMu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		title, err := c.generateSessionTitle(ctx, userSnippet, modelSnippet)
		if err != nil {
			klog.Warningf("failed to generate session title: %v", err)
			return
		}
		if title == "" {
			return
		}

		// Hold the lock across the check and the persist: with the memory
		// backend the store hands out shared *api.Session pointers, so the
		// update can otherwise race a manual rename (see RenameSession).
		c.sessionMu.Lock()
		defer c.sessionMu.Unlock()
		if c.Session == nil || c.Session.ID != sessionID || c.Session.Name != "" || c.Session.ManuallyNamed {
			return
		}
		manager, err := sessions.NewSessionManager(c.SessionBackend)
		if err != nil {
			klog.Warningf("failed to create session manager: %v", err)
			return
		}
		if err := manager.SetSessionName(sessionID, title, false); err != nil {
			klog.Warningf("failed to set generated session title: %v", err)
			return
		}
		c.Session.Name = title
		c.titleGenerated = true
	}()
}

// generateSessionTitle asks the LLM to summarize the start of the
// conversation as a short title. Returns "" when nothing usable comes back.
func (c *Agent) generateSessionTitle(ctx context.Context, userSnippet, modelSnippet string) (string, error) {
	prompt := fmt.Sprintf("Summarize this conversation as a short title of at most 6 words, no quotes or punctuation:\nUser: %s\nAssistant: %s\nTitle:", userSnippet, modelSnippet)
	resp, err := c.LLM.GenerateCompletion(ctx, &gollm.CompletionRequest{Model: c.Model, Prompt: prompt})
	if err != nil {
		return "", err
	}
	if resp == nil {
		return "", nil
	}
	return sessionTitleFromResponse(resp.Response()), nil
}

// sessionTitleFromResponse cleans an LLM completion into a session title:
// trims whitespace (and the quotes LLMs add despite instructions), caps at 60
// runes, and sanitizes for display and storage.
func sessionTitleFromResponse(response string) string {
	title := strings.TrimSpace(response)
	title = strings.Trim(title, `"'`)
	const maxLen = 60
	if r := []rune(title); len(r) > maxLen {
		title = strings.TrimSpace(string(r[:maxLen]))
	}
	return sessions.SanitizeSessionName(title)
}

// titleSnippets returns excerpts of the first user message and the first
// model text reply, for the title prompt.
func titleSnippets(messages []*api.Message) (user, model string) {
	for _, msg := range messages {
		if msg.Type != api.MessageTypeText {
			continue
		}
		p, ok := msg.Payload.(string)
		if !ok || strings.TrimSpace(p) == "" {
			continue
		}
		switch msg.Source {
		case api.MessageSourceUser:
			if user == "" {
				user = titleSnippet(p)
			}
		case api.MessageSourceModel:
			if model == "" {
				model = titleSnippet(p)
			}
		}
	}
	return user, model
}

// titleSnippet collapses whitespace and caps s at titleSnippetLen runes.
func titleSnippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > titleSnippetLen {
		return string(r[:titleSnippetLen])
	}
	return s
}
