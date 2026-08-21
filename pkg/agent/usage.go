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
	"encoding/json"
	"reflect"
	"strings"
)

// Token-count key names, normalized (lowercase, no underscores) so both
// JSON-ish ("total_tokens") and Go-style ("TotalTokenCount") spellings match.
var (
	usageTotalKeys      = []string{"totaltokens", "totaltokencount"}
	usagePromptKeys     = []string{"prompttokens", "prompttokencount", "inputtokens"}
	usageCompletionKeys = []string{"completiontokens", "candidatestokencount", "outputtokens"}
)

// usageTotalTokens extracts a total (prompt + completion) token count from a
// provider's UsageMetadata() value. Shapes differ per provider: some return a
// map[string]any (JSON-ish keys like "total_tokens"/"totalTokenCount"),
// others SDK structs (openai's TotalTokens, genai's TotalTokenCount). When no
// total is present, prompt and completion counts are summed. Returns 0 when
// nothing usable is found.
func usageTotalTokens(metadata any) int {
	if metadata == nil {
		return 0
	}

	lookup := usageKeyLookup(metadata)
	if lookup == nil {
		return 0
	}
	if total, ok := lookup(usageTotalKeys); ok && total > 0 {
		return total
	}
	prompt, _ := lookup(usagePromptKeys)
	completion, _ := lookup(usageCompletionKeys)
	return prompt + completion
}

// usageKeyLookup returns a function resolving normalized key names against
// metadata, or nil when metadata has a shape we cannot read.
func usageKeyLookup(metadata any) func(keys []string) (int, bool) {
	switch v := metadata.(type) {
	case map[string]any:
		return func(keys []string) (int, bool) {
			for k, val := range v {
				if n, ok := tokenCount(val); ok && matchUsageKey(keys, k) {
					return n, true
				}
			}
			return 0, false
		}
	}
	rv := reflect.Indirect(reflect.ValueOf(metadata))
	if rv.Kind() != reflect.Struct {
		return nil
	}
	rt := rv.Type()
	return func(keys []string) (int, bool) {
		for i := 0; i < rt.NumField(); i++ {
			field := rt.Field(i)
			if field.PkgPath != "" { // unexported
				continue
			}
			if n, ok := tokenCount(rv.Field(i).Interface()); ok && matchUsageKey(keys, field.Name) {
				return n, true
			}
		}
		return 0, false
	}
}

func matchUsageKey(keys []string, name string) bool {
	name = normalizeUsageKey(name)
	for _, k := range keys {
		if k == name {
			return true
		}
	}
	return false
}

func normalizeUsageKey(name string) string {
	return strings.ToLower(strings.ReplaceAll(name, "_", ""))
}

// tokenCount converts a numeric value of any common JSON/Go numeric type.
func tokenCount(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
	}
	return 0, false
}
