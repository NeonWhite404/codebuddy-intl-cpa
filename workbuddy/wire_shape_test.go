package main

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"strings"
	"testing"
)

// Chat headers must ship under the exact spelling an official client sends.
//
// Go's http.Header.Set canonicalizes keys, and the canonical form is wrong for
// several of these names: `X-Conversation-ID` becomes `X-Conversation-Id`,
// `X-B3-TraceId` becomes `X-B3-Traceid`, `X-IDE-Type` becomes `X-Ide-Type`.
// The reference gateway lists header-name fidelity as its first transport-layer
// fix (workbuddy2api-hub #429), so a regression here reintroduces exactly the
// "not the real client" signal those changes removed.
func TestChatHeadersKeepOfficialSpelling(t *testing.T) {
	sa := &storedAuth{}
	sa.Auth.Domain = "https://www.workbuddy.ai"
	sa.Auth.AccessToken = "tok"
	sa.Account.UID = "uid-1"

	req, err := http.NewRequest(http.MethodPost, endpointChatFor(sa), strings.NewReader(`{"model":"m","messages":[]}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	backendHeaders(req, sa)

	for _, name := range []string{
		"X-Conversation-ID", "X-Conversation-Request-ID", "X-Conversation-Message-ID",
		"X-Root-Request-ID", "X-Request-ID", "X-B3-TraceId", "X-B3-SpanId",
		"X-IDE-Type", "X-IDE-Name", "X-Product", "X-User-Id", "X-Domain",
		"x-codebuddy-request", "x-trace-id",
	} {
		if _, ok := req.Header[name]; !ok {
			t.Errorf("header %q is not stored under its official spelling", name)
		}
		if canonical := http.CanonicalHeaderKey(name); canonical != name {
			if _, dup := req.Header[canonical]; dup {
				t.Errorf("header %q also exists as canonical %q (would ship twice)", name, canonical)
			}
		}
	}

	// Re-stamping must not leave a duplicate behind: two entries with the same
	// key would serialize as two header lines.
	applyConversationHeaders(req, "11111111-2222-8333-8444-555555555555")
	if n := len(req.Header["X-Conversation-ID"]); n != 1 {
		t.Errorf("X-Conversation-ID has %d values after re-stamp, want 1", n)
	}
}

// Chat is not a browser request: no official client sends X-Requested-With on a
// completions call, and a browser-shaped chat request is one more fingerprint.
func TestChatDropsBrowserHeaders(t *testing.T) {
	sa := &storedAuth{}
	sa.Auth.Domain = "https://www.workbuddy.ai"

	req, _ := http.NewRequest(http.MethodPost, endpointChatFor(sa), strings.NewReader(`{}`))
	backendHeaders(req, sa)

	for _, h := range []string{"X-Requested-With"} {
		if v := wireHeader(req, h); v != "" {
			t.Errorf("chat request must not send %s (got %q)", h, v)
		}
	}
	// Accept is re-stamped by the identity layer, so the browser-ish
	// "application/json, text/plain, */*" must not survive either.
	if got := wireHeader(req, "Accept"); got != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", got)
	}

	// The login/refresh endpoints really are browser-driven, so commonHeaders
	// must keep stamping the browser family there.
	loginReq, _ := http.NewRequest(http.MethodPost, "https://www.codebuddy.ai/v2/plugin/auth/token/refresh", nil)
	commonHeaders(loginReq)
	if got := wireHeader(loginReq, "X-Requested-With"); got != "XMLHttpRequest" {
		t.Errorf("non-chat request lost X-Requested-With: %q", got)
	}
}

// The chat body must go out gzipped, and the bytes on the wire must be what the
// Content-Encoding header claims — a mismatch would be worse than no compression.
func TestChatBodyIsGzipped(t *testing.T) {
	sa := &storedAuth{}
	sa.Auth.Domain = "https://www.workbuddy.ai"
	plain := []byte(`{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"hi"}]}`)

	req, _ := http.NewRequest(http.MethodPost, endpointChatFor(sa), bytes.NewReader(plain))
	wire := backendHeaders(req, sa)
	if wire == nil {
		t.Fatal("backendHeaders returned no wire body")
	}
	if got := wireHeader(req, "Content-Encoding"); got != "gzip" {
		t.Fatalf("Content-Encoding = %q, want gzip", got)
	}
	// The request body must hold the same bytes that were returned, because the
	// host bridge reads req.Body itself.
	body, _ := io.ReadAll(req.Body)
	if !bytes.Equal(body, wire) {
		t.Error("req.Body and the returned wire body differ")
	}
	// And those bytes must actually decode to the original payload.
	zr, err := gzip.NewReader(bytes.NewReader(wire))
	if err != nil {
		t.Fatalf("gzip reader: %v", err)
	}
	defer zr.Close()
	got, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("gunzip: %v", err)
	}
	if !bytes.Equal(got, plain) {
		t.Errorf("decoded body = %q, want %q", got, plain)
	}
	// Content-Length must describe the compressed bytes, not the plain ones.
	if cl := wireHeader(req, "Content-Length"); cl == "" || cl == string(rune(len(plain))) {
		t.Errorf("Content-Length = %q, want the compressed length", cl)
	}
}
