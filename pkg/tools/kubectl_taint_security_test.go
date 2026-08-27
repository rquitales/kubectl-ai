package tools

import "testing"

func TestKubectlTaintDryRunAndSeparatorHoles(t *testing.T) {
	cases := []struct{ cmd, want string }{
		{"kubectl delete pod x --dry-run=none", "yes"},
		{"kubectl delete pod x --dry-run=false", "yes"},
		{"kubectl delete pod x --dry-run=client", "no"},
		{"kubectl delete pod x --dry-run=server", "no"},
		{"kubectl exec mypod -- touch /tmp/x --dry-run=server", "yes"},
		{"kubectl exec mypod -- rm -rf /data --dry-run=client", "yes"},
		{"/tmp/evil-kubectl get secrets", "unknown"},
		{"./kubectl get pods", "unknown"},
		{"kubectl-1.28 get pods", "no"},
		{"kubectl get pods", "no"},
		{"kubectl delete pod x", "yes"},
	}
	for _, tc := range cases {
		if got := kubectlModifiesResource(tc.cmd); got != tc.want {
			t.Errorf("%q => %s, want %s", tc.cmd, got, tc.want)
		}
	}
}
