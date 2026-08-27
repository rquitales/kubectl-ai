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
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type captureRecorder struct{ events []*journal.Event }

func (c *captureRecorder) Write(_ context.Context, e *journal.Event) error {
	c.events = append(c.events, e)
	return nil
}
func (c *captureRecorder) Close() error { return nil }

func TestJournalRedactsAuthHeaders(t *testing.T) {
	rec := &captureRecorder{}
	jrt := &journalingRoundTripper{next: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			Status: "200 OK",
			Header: http.Header{},
			Body:   io.NopCloser(strings.NewReader(`{}`)),
		}, nil
	})}

	req, _ := http.NewRequest("POST", "https://api.example.com/v1/chat", strings.NewReader("{}"))
	req.Header.Set("Authorization", "Bearer sk-secret")
	req.Header.Set("X-Api-Key", "sk-secret")
	ctx := journal.ContextWithRecorder(req.Context(), rec)
	req = req.WithContext(ctx)

	resp, err := jrt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	resp.Body.Close()

	if len(rec.events) == 0 {
		t.Fatal("no events recorded")
	}
	for _, e := range rec.events {
		payload, _ := e.Payload.(map[string]any)
		if s, ok := payload["request"].(string); ok {
			if strings.Contains(s, "sk-secret") {
				t.Errorf("request log leaked credentials:\n%s", s)
			}
			if !strings.Contains(s, "REDACTED") {
				t.Errorf("expected redaction marker in request log:\n%s", s)
			}
		}
	}
}

func TestJournalTeesStreamInsteadOfBuffering(t *testing.T) {
	rec := &captureRecorder{}
	jrt := &journalingRoundTripper{next: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			Status: "200 OK",
			Header: http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:   io.NopCloser(strings.NewReader("data: one\n\ndata: two\n\n")),
		}, nil
	})}

	req, _ := http.NewRequest("POST", "https://api.example.com/stream", nil)
	req = req.WithContext(journal.ContextWithRecorder(req.Context(), rec))

	resp, err := jrt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	// The body must stream through untouched...
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(body) != "data: one\n\ndata: two\n\n" {
		t.Errorf("stream body altered: %q", body)
	}
	// ...and the journal receives the assembled body at stream end.
	if len(rec.events) == 0 {
		t.Fatal("no journal event after stream end")
	}
	found := false
	for _, e := range rec.events {
		payload, _ := e.Payload.(map[string]any)
		if s, ok := payload["body"].(string); ok && strings.Contains(s, "data: two") {
			found = true
		}
	}
	if !found {
		t.Error("streamed body was not journaled after completion")
	}
}
