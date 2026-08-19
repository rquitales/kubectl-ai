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

package sessions

import "math/rand"

// Memorable session names instead of bare timestamp IDs (like Docker's
// random container names).
var sessionAdjectives = []string{
	"amber", "bold", "brave", "bright", "calm", "clever", "cool", "cosmic",
	"crisp", "daring", "eager", "fancy", "gentle", "golden", "happy", "jolly",
	"keen", "kind", "lively", "lucky", "merry", "mighty", "noble", "proud",
	"quick", "quiet", "rapid", "sharp", "sunny", "swift", "witty", "zesty",
}

var sessionNouns = []string{
	"badger", "bison", "cobra", "condor", "crane", "falcon", "ferret", "finch",
	"fox", "gecko", "heron", "ibex", "jaguar", "koala", "lemur", "lynx",
	"marten", "moose", "newt", "otter", "panda", "puma", "raven", "salmon",
	"sparrow", "tiger", "toucan", "viper", "walrus", "wombat", "yak", "zebra",
}

// generateSessionName returns a short memorable session name of the form
// "adjective-noun", e.g. "swift-heron".
func generateSessionName() string {
	return sessionAdjectives[rand.Intn(len(sessionAdjectives))] + "-" + sessionNouns[rand.Intn(len(sessionNouns))]
}
