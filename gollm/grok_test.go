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
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

func chunkFromJSON(t *testing.T, s string) openai.ChatCompletionChunk {
	t.Helper()
	var c openai.ChatCompletionChunk
	if err := json.Unmarshal([]byte(s), &c); err != nil {
		t.Fatalf("chunk JSON: %v", err)
	}
	return c
}

func TestGrokYieldsOnlyCompletedToolCalls(t *testing.T) {
	// Regression: grok streaming converted each raw tool-call DELTA into a
	// function call, so the agent received partial JSON arguments (parsed as
	// empty args) — tools executed with no command. Only fully accumulated
	// calls may be yielded.
	acc := openai.ChatCompletionAccumulator{}

	// Chunk 1: tool call begins (name + first args fragment).
	c1 := chunkFromJSON(t, `{"id":"x","object":"chat.completion.chunk","created":1,"model":"grok","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"kubectl","arguments":"{\"command\":\"get"}}]}}]}`)
	acc.AddChunk(c1)
	r1 := &grokChatStreamResponse{streamChunk: c1, accumulator: acc}
	for _, cand := range r1.Candidates() {
		for _, part := range cand.Parts() {
			if calls, ok := part.AsFunctionCalls(); ok {
				t.Fatalf("chunk 1 yielded a partial function call: %+v", calls)
			}
		}
	}

	// Chunk 2: args fragment continues; the call is still incomplete.
	c2 := chunkFromJSON(t, `{"id":"x","object":"chat.completion.chunk","created":1,"model":"grok","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" pods\"}"}}]}}]}`)
	acc.AddChunk(c2)

	// Chunk 3: finish_reason closes the tool call; now it may be yielded,
	// with the complete arguments.
	c3 := chunkFromJSON(t, `{"id":"x","object":"chat.completion.chunk","created":1,"model":"grok","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
	acc.AddChunk(c3)
	completed := []openai.ChatCompletionMessageToolCall{}
	if tool, ok := acc.JustFinishedToolCall(); ok {
		completed = append(completed, openai.ChatCompletionMessageToolCall{
			ID: tool.ID,
			Function: openai.ChatCompletionMessageToolCallFunction{
				Name:      tool.Name,
				Arguments: tool.Arguments,
			},
		})
	}
	r3 := &grokChatStreamResponse{streamChunk: c3, accumulator: acc, completedToolCalls: completed}

	found := false
	for _, cand := range r3.Candidates() {
		for _, part := range cand.Parts() {
			if calls, ok := part.AsFunctionCalls(); ok {
				found = true
				if len(calls) != 1 || calls[0].Name != "kubectl" {
					t.Fatalf("unexpected calls: %+v", calls)
				}
				if calls[0].Arguments["command"] != "get pods" {
					t.Errorf("arguments = %v, want complete args {command: get pods}", calls[0].Arguments)
				}
			}
		}
	}
	if !found {
		t.Fatal("the completed tool call was never yielded")
	}
}
