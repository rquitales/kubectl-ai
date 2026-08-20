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

// UseContext switches the current context in the kubeconfig at path (or the
// default when empty) to name, validating that the context exists.
func UseContext(path, name string) error {
	rules := loadingRules(path)
	cfg, err := rules.Load()
	if err != nil {
		return err
	}
	if _, exists := cfg.Contexts[name]; !exists {
		return fmt.Errorf("context %q does not exist in the kubeconfig", name)
	}
	cfg.CurrentContext = name
	return clientcmd.WriteToFile(*cfg, rules.GetDefaultFilename())
}
