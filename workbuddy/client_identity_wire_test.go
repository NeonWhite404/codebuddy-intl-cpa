package main

import (
	"net/http"
	"strings"
	"testing"
)

// The Intl realm used to be stamped as the desktop "IDE" identity. The desktop
// wire shape is the one the upstream validates most strictly and the one this
// plugin cannot reproduce faithfully (the reference gateway's desktop alignment
// was withdrawn by its own author), so both realms now send the CLI family that
// was reverse engineered from the published `@tencent-ai/codebuddy-code`.
//
// This test pins BOTH halves of that decision: the Intl realm must not drift
// back to IDE by accident, and the CN realm must keep the identity it already
// had.
func TestIdentityIsCLIForEveryRealm(t *testing.T) {
	stubClientVersionRefresh(t)
	resetClientVersionCache(t)
	t.Setenv("WB_CLIENT_VERSION", "")

	// The identity is realm-independent now; a regression here means the IDE
	// branch came back for one realm only, which is the exact mismatch the
	// channel check looks for.
	for _, tc := range []struct {
		reason string
		domain string
	}{
		{"intl workbuddy host", "https://www.workbuddy.ai"},
		{"intl codebuddy host", "https://www.codebuddy.ai"},
		{"cn copilot gateway", "https://copilot.tencent.com"},
		{"legacy empty domain", ""},
	} {
		sa := &storedAuth{}
		sa.Auth.Domain = tc.domain
		if got := identityFor(sa); got != "CLI" {
			t.Errorf("%s: identityFor = %q, want CLI", tc.reason, got)
		}
	}
	if got := identityFor(nil); got != "CLI" {
		t.Errorf("identityFor(nil) = %q, want CLI", got)
	}

	// The UA must carry the CLI prefix on a Global account, since that is the
	// header the version gate reads first.
	intl := &storedAuth{}
	intl.Auth.Domain = "https://www.workbuddy.ai"
	req, _ := http.NewRequest(http.MethodPost, "https://www.codebuddy.ai/v2/chat/completions", nil)
	applyIdentityHeaders(req, intl)
	ua := wireHeader(req, "User-Agent")
	if !strings.HasPrefix(ua, "CLI/") {
		t.Errorf("Global User-Agent = %q, want a CLI/ prefix", ua)
	}
	if strings.Contains(ua, "IDE/") {
		t.Errorf("Global User-Agent = %q, must not claim the desktop identity", ua)
	}
}

// Every conversation/tracing id must be UUID-shaped, because a bare 32-char hex
// string is itself a fingerprint the upstream channel check can key on.
func TestConversationIDsAreUUIDShaped(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4.1-flash","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}`)
	cid := stableConversationID(body)
	if !isUUID(cid) {
		t.Fatalf("stableConversationID = %q, want UUID shape", cid)
	}

	// Determinism is the prompt-cache contract: the same conversation prefix
	// must keep the same id, or every turn re-writes the upstream KV cache.
	if again := stableConversationID(body); again != cid {
		t.Errorf("conversation id is not stable: %q != %q", again, cid)
	}
	// A different conversation must not collide.
	other := []byte(`{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"other"}]}`)
	if stableConversationID(other) == cid {
		t.Error("different conversations share one id")
	}

	req, _ := http.NewRequest(http.MethodPost, "https://www.codebuddy.ai/v2/chat/completions", nil)
	turnID, messageID := applyConversationHeaders(req, cid)
	for name, id := range map[string]string{
		"X-Conversation-ID":         cid,
		"X-Conversation-Request-ID": turnID,
		"X-Conversation-Message-ID": messageID,
		"X-Request-ID":              wireHeader(req, "X-Request-ID"),
		"X-Root-Request-ID":         wireHeader(req, "X-Root-Request-ID"),
	} {
		if !isUUID(id) {
			t.Errorf("%s = %q, want UUID shape", name, id)
		}
	}

	// B3 keeps its own spec shape: 32 hex trace, 16 hex span, derived from the
	// message id's canonical hex. A dashed trace id would be 36 chars and the
	// upstream drops the pair.
	trace, span := wireHeader(req, "X-B3-TraceId"), wireHeader(req, "X-B3-SpanId")
	bare := strings.ReplaceAll(messageID, "-", "")
	if len(trace) != 32 || trace != bare {
		t.Errorf("X-B3-TraceId = %q, want the 32-hex form of %q", trace, messageID)
	}
	if len(span) != 16 || span != bare[:16] {
		t.Errorf("X-B3-SpanId = %q, want the first 16 hex of %q", span, messageID)
	}
}
