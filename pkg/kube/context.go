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

// Package kube provides small kubeconfig helpers shared by the agent and
// the TUI (current context, context listing, and context switching) built
// on client-go — no kubectl subprocess.
package kube

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// loadingRules returns the loading rules for path, or the KUBECONFIG env /
// ~/.kube/config default when path is empty.
func loadingRules(path string) *clientcmd.ClientConfigLoadingRules {
	if path != "" {
		return &clientcmd.ClientConfigLoadingRules{ExplicitPath: path}
	}
	return clientcmd.NewDefaultClientConfigLoadingRules()
}

// LoadConfig loads the kubeconfig at path (or the default when empty).
func LoadConfig(path string) (*clientcmdapi.Config, error) {
	return loadingRules(path).Load()
}

// CurrentContext returns the current context and its namespace ("default"
// when unset) from the kubeconfig at path (or the default when empty).
// ok is false when no config or no current context exists.
func CurrentContext(path string) (context, namespace string, ok bool) {
	cfg, err := LoadConfig(path)
	if err != nil || cfg.CurrentContext == "" {
		return "", "", false
	}
	namespace = "default"
	if ctx, exists := cfg.Contexts[cfg.CurrentContext]; exists && ctx.Namespace != "" {
		namespace = ctx.Namespace
	}
	return cfg.CurrentContext, namespace, true
}

// ListContexts returns the sorted names of all contexts in the kubeconfig
// at path (or the default when empty).
func ListContexts(path string) ([]string, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(cfg.Contexts))
	for name := range cfg.Contexts {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

// WriteOverride writes a session-scoped kubeconfig to outPath: a copy of
// the base kubeconfig (path, or the default when empty) with the current
// context set to context (validated) and/or the current context's namespace
// set to namespace. Pointing KUBECONFIG at outPath applies the override to
// every kubectl invocation in the process without mutating the base file.
func WriteOverride(basePath, outPath, context, namespace string) error {
	cfg, err := LoadConfig(basePath)
	if err != nil {
		return err
	}

	current := cfg.CurrentContext
	if context != "" {
		if _, exists := cfg.Contexts[context]; !exists {
			return fmt.Errorf("context %q does not exist in the kubeconfig", context)
		}
		current = context
	}
	if current == "" {
		return fmt.Errorf("no context available to override (base kubeconfig has no current context)")
	}
	cfg.CurrentContext = current

	if namespace != "" {
		ctx, exists := cfg.Contexts[current]
		if !exists {
			return fmt.Errorf("context %q does not exist in the kubeconfig", current)
		}
		ctx.Namespace = namespace
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	return clientcmd.WriteToFile(*cfg, outPath)
}
