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

package tools

import "testing"

func TestDetectKubectlStreamingLongForms(t *testing.T) {
	// Regression: substring matching missed the long flag forms, so these
	// were classified non-streaming AND read-only — no prompt, no timeout,
	// hanging the turn forever (unrecoverable in RunOnce).
	streaming := []string{
		"kubectl get pods --watch",
		"kubectl get pods --watch=true",
		"kubectl get pods -w",
		"kubectl get pods -w -o wide",
		"kubectl logs --follow mypod",
		"kubectl logs -f mypod",
		"kubectl attach mypod -c app",
		"kubectl proxy",
		"kubectl wait --for=condition=Ready pod/x",
	}
	for _, cmd := range streaming {
		if ok, _ := DetectKubectlStreaming(cmd); !ok {
			t.Errorf("%q not detected as streaming", cmd)
		}
	}

	notStreaming := []string{
		"kubectl get pods",
		"kubectl get pods --watch=false",
		"kubectl logs --follow=false mypod",
		"kubectl wait --for=condition=Ready --timeout=30s pod/x",
		"kubectl describe pod x",
		"kubectl rollout status deployment/web",
		"kubectl get svc -o jsonpath='{.spec.ports[?(@.name==\"-w\")]}'",
	}
	for _, cmd := range notStreaming {
		if ok, st := DetectKubectlStreaming(cmd); ok {
			t.Errorf("%q falsely detected as streaming (%s)", cmd, st)
		}
	}
}
