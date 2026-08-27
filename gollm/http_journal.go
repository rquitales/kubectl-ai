// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gollm

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httputil"
	"strings"

	"github.com/GoogleCloudPlatform/kubectl-ai/pkg/journal"

	"k8s.io/klog/v2"
)

// journalingRoundTripper wraps an existing http.RoundTripper to record requests and responses.
type journalingRoundTripper struct {
	next http.RoundTripper // The actual transport that does the network call
}

// RoundTrip satisfies the http.RoundTripper interface. It intercepts an HTTP request,
// logs it, passes it to the next handler, and then logs the response.
// It includes special handling to correctly parse and summarize streaming responses.
func (jrt *journalingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	recorder := journal.RecorderFromContext(req.Context())

	// Log the outgoing request — with credentials redacted: trace files and
	// V(2) logs previously recorded Authorization/x-api-key headers in
	// plaintext.
	//
	// Read and restore the body ourselves: req.Clone shares the Body reader,
	// so DumpRequestOut on a clone drains the REAL request's body (the actual
	// call then went out with ContentLength set but an empty body → API 400s).
	var bodyBytes []byte
	if req.Body != nil && req.Body != http.NoBody {
		var err error
		bodyBytes, err = io.ReadAll(req.Body)
		req.Body.Close()
		if err != nil {
			klog.Errorf("Error reading request body (for logging): %v", err)
		}
		// Restore the body for the real network call.
		req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		req.ContentLength = int64(len(bodyBytes))
	}
	reqForLog := req.Clone(req.Context())
	reqForLog.Header = req.Header.Clone()
	if bodyBytes != nil {
		reqForLog.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		reqForLog.ContentLength = int64(len(bodyBytes))
	}
	for _, h := range []string{"Authorization", "X-Api-Key", "X-Goog-Api-Key", "Api-Key", "X-Amz-Security-Token"} {
		if reqForLog.Header.Get(h) != "" {
			reqForLog.Header.Set(h, "REDACTED")
		}
	}
	reqBytes, err := httputil.DumpRequestOut(reqForLog, true)
	if err == nil {
		err = recorder.Write(req.Context(), &journal.Event{
			Action:  journal.ActionHTTPRequest,
			Payload: map[string]any{"request": string(reqBytes)},
		})
		if err != nil {
			klog.Errorf("Error writing outgoing request to journal: %v", err)
		}
	}

	// Pass the request to the next RoundTripper to make the actual network call.
	resp, err := jrt.next.RoundTrip(req)
	if err != nil {
		writeErr := recorder.Write(req.Context(), &journal.Event{
			Action:  journal.ActionHTTPError,
			Payload: map[string]any{"error": "http transport failed", "detail": err.Error()},
		})
		if writeErr != nil {
			klog.Errorf("Error writing RoundTripper error to journal: %v", writeErr)
		}
		klog.Errorf("RoundTripper error: %v", err)
		return nil, err
	}

	// Streaming (SSE) responses must NOT be buffered here: buffering delays
	// the stream until generation completes (time-to-first-token becomes the
	// full generation time) and trips the client's total timeout on long
	// generations. Tee the body instead: record chunks as they pass through
	// and journal the assembled body when the stream finishes.
	if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		resp.Body = &teeReadCloser{
			ReadCloser: resp.Body,
			onDone: func(body []byte) {
				payload := map[string]any{
					"status":  resp.Status,
					"headers": resp.Header,
					"body":    string(body),
				}
				if err := recorder.Write(req.Context(), &journal.Event{Action: journal.ActionHTTPResponse, Payload: payload}); err != nil {
					klog.Errorf("Error writing streamed response to journal: %v", err)
				}
			},
		}
		return resp, nil
	}

	// Read the entire response body so we can log it and then pass it along.
	respBodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		// handle error
		klog.Errorf("Error reading response body (for logging): %v", err)
		return nil, err
	}
	resp.Body.Close() // Close the original body

	// Default payload is the raw body, for non-streaming responses.
	logPayload := map[string]any{
		"status":  resp.Status,
		"headers": resp.Header,
		"body":    string(respBodyBytes),
	}

	// Write the final event to the journal.
	err = recorder.Write(req.Context(), &journal.Event{
		Action:  journal.ActionHTTPResponse,
		Payload: logPayload,
	})
	if err != nil {
		// Log the error and continue
		klog.Errorf("Error writing to journal: %v", err)
	}

	// IMPORTANT: Return the original, untouched body to the client.
	resp.Body = io.NopCloser(bytes.NewBuffer(respBodyBytes))
	return resp, nil
}

// withJournaling is a decorator function that wraps an http.Client's transport
// with the journalingRoundTripper, but only if a recorder is found in the context.
func withJournaling(client *http.Client) *http.Client {
	// wrap the transport
	client.Transport = &journalingRoundTripper{
		next: client.Transport,
	}

	return client
}

// teeReadCloser accumulates response bytes as they are read and invokes
// onDone with the assembled body when the stream reaches EOF or is closed.
type teeReadCloser struct {
	io.ReadCloser
	buf    bytes.Buffer
	onDone func(body []byte)
	done   bool
}

func (t *teeReadCloser) Read(p []byte) (int, error) {
	n, err := t.ReadCloser.Read(p)
	if n > 0 {
		t.buf.Write(p[:n])
	}
	if err == io.EOF {
		t.finish()
	}
	return n, err
}

func (t *teeReadCloser) Close() error {
	err := t.ReadCloser.Close()
	t.finish()
	return err
}

func (t *teeReadCloser) finish() {
	if t.done {
		return
	}
	t.done = true
	t.onDone(t.buf.Bytes())
}
