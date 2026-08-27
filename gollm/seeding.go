// Copyright 2026 Google LLC
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

package gollm

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/api"
)

// SeedMessage is a normalized text-only conversation entry for history
// seeding. Role is "user" or "assistant" (model-authored text).
type SeedMessage struct {
	Role    string
	Content string
}

// SeedableMessages filters persisted session messages down to what belongs
// in a provider's conversation history on resume/compact/clear: text
// content only, never ephemeral display-only messages (thinking blocks,
// local /help output), and never tool-call request/response records (they
// lack the call IDs native tool pairing needs). Agent text (shell-escape
// results, notices) is treated as user-authored context — attributing it
// to the assistant both misleads the model and risks role-alternation API
// errors on strict providers.
func SeedableMessages(messages []*api.Message) []SeedMessage {
	var out []SeedMessage
	for _, msg := range messages {
		if msg == nil || msg.Ephemeral || msg.Type != api.MessageTypeText || msg.Payload == nil {
			continue
		}
		var content string
		if textPayload, ok := msg.Payload.(string); ok {
			content = textPayload
		} else {
			content = fmt.Sprintf("%v", msg.Payload)
		}
		if content == "" {
			continue
		}
		var role string
		switch msg.Source {
		case api.MessageSourceModel:
			role = "assistant"
		case api.MessageSourceUser, api.MessageSourceAgent:
			role = "user"
		default:
			continue
		}
		out = append(out, SeedMessage{Role: role, Content: content})
	}
	return out
}
