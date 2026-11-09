// Client identity stamped on every upstream CodeBuddy request.
//
// The upstream validates the caller's client identity, not just the bearer
// token: a request whose identity family does not match what the gateway
// expects is refused with `11128 "Illegal API invocation from an unapproved
// channel"` — the same code that a blocked system prompt draws.
//
// Both realms now present the CLI identity. The Intl realm used to be stamped as
// IDE (the desktop client), but the desktop wire shape is the one nobody has
// been able to pin down: the reference gateway's desktop-alignment change was
// withdrawn by its own author because too many fields could only be inferred, so
// a half-copied desktop identity is a recognizable "not the real client" signal.
// The CLI shape, by contrast, was derived from the published package
// `@tencent-ai/codebuddy-code` and is the same family the CN realm already
// serves. When the upstream's desktop validation is understood well enough to
// match, the IDE branch can come back.
//
// Both identities send the full header family — `X-IDE-Type`/`X-IDE-Name`,
// `x-codebuddy-request`, `X-Root-Request-ID`/`X-Request-ID` and the B3 trace
// pair — and both stamp a client version tracked from the publisher's release
// feed rather than pinned, because a stale version is itself a signal the
// upstream can reject. Earlier builds of this plugin sent only
// `CLI/<pinned version>` plus `X-Product`, which is the gap this file closes.
//
// Version resolution is deliberately non-blocking: a request always gets the
// cached (or fallback) version immediately, and the feed is refreshed in the
// background at most once per TTL. A registry outage must never stall traffic.
//
// Every header is written through setWireHeader, which keeps the exact case the
// official client uses. Go's http.Header.Set canonicalizes the key, and
// canonicalization is lossy in a way the upstream can see: `X-Conversation-ID`
// becomes `X-Conversation-Id`, `X-B3-TraceId` becomes `X-B3-Traceid`, `X-IDE-Type`
// becomes `X-Ide-Type`. Those spellings are not what the official clients send,
// so they are one more "not the real client" signal on a request the upstream
// already scrutinizes.
package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// codeBuddyVersionURL is the npm release feed for the CodeBuddy CLI. Both
	// realms are stamped with a version from this feed.
	codeBuddyVersionURL = "https://registry.npmjs.org/@tencent-ai/codebuddy-code/latest"

	// clientIdentity is the identity family sent for every account. The Intl
	// realm used to send IDE; see the package comment for why that changed.
	clientIdentity = "CLI"

	// Pinned fallbacks, used until the first successful feed fetch. The Intl
	// fallback matches the version the reference gateways ship; the CN fallback
	// keeps the long-standing pinned value so existing CN traffic is unchanged
	// if the feed is unreachable.
	fallbackIntlClientVersion = "2.168.2"
	fallbackCNClientVersion   = "2.63.2"

	clientVersionTTL     = 6 * time.Hour
	clientVersionTimeout = 8 * time.Second
)

// clientVersionHTTPClient is deliberately separate from sharedHTTPClient():
// the upstream client is stubbed in tests and carries the chat transport, and a
// background version poll must never ride the same connection pool (or a test
// mock) as live traffic.
var clientVersionHTTPClient = &http.Client{Timeout: clientVersionTimeout}

var (
	clientVersionMu        sync.Mutex
	clientVersionValue     string
	clientVersionFetchedAt time.Time
	clientVersionFetching  int32 // single-flight guard for the background refresh

	// clientVersionNow is a seam for tests.
	clientVersionNow = time.Now

	// clientVersionRefresh is a seam for tests: the resolver schedules the
	// background fetch through it so a unit test never opens a socket.
	clientVersionRefresh = refreshClientVersion
)

// identityFor returns the client identity stamped for the account. Every realm
// gets the CLI family: the desktop/IDE family is the one the upstream validates
// most strictly and the one this plugin cannot reproduce faithfully, so present
// the identity that was reverse engineered from the published CLI package.
func identityFor(sa *storedAuth) string {
	return clientIdentity
}

// clientVersionEnv returns an operator override, checked per realm first so a
// single test can pin one realm without touching the other.
func clientVersionEnv(global bool) string {
	if global {
		if v := strings.TrimSpace(os.Getenv("WB_CLIENT_VERSION_INTL")); v != "" {
			return v
		}
	} else if v := strings.TrimSpace(os.Getenv("WB_CLIENT_VERSION_CN")); v != "" {
		return v
	}
	return strings.TrimSpace(os.Getenv("WB_CLIENT_VERSION"))
}

// fallbackClientVersion is the pinned version for a realm.
func fallbackClientVersion(global bool) string {
	if global {
		return fallbackIntlClientVersion
	}
	return fallbackCNClientVersion
}

// clientVersionFor returns the version to stamp for a realm. It never blocks on
// the network: a stale cache schedules a background refresh and the caller gets
// the cached value (or the pinned fallback) right away.
func clientVersionFor(global bool) string {
	if v := clientVersionEnv(global); v != "" {
		return v
	}
	clientVersionMu.Lock()
	v := clientVersionValue
	fetchedAt := clientVersionFetchedAt
	clientVersionMu.Unlock()
	if v == "" || clientVersionNow().Sub(fetchedAt) > clientVersionTTL {
		go clientVersionRefresh()
	}
	if v == "" {
		return fallbackClientVersion(global)
	}
	return v
}

// refreshClientVersion fetches the published CodeBuddy CLI version. Failures
// are silent and leave the previous value in place; the next caller retries
// after the TTL, and the single-flight guard keeps a burst from fanning out.
func refreshClientVersion() {
	if !atomic.CompareAndSwapInt32(&clientVersionFetching, 0, 1) {
		return
	}
	defer atomic.StoreInt32(&clientVersionFetching, 0)

	ctx, cancel := context.WithTimeout(context.Background(), clientVersionTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, codeBuddyVersionURL, nil)
	if err != nil {
		return
	}
	// A plain browser-ish UA is fine here: the registry is public and not gated.
	req.Header.Set("User-Agent", "Mozilla/5.0 (CodeBuddy client version check)")
	req.Header.Set("Accept", "application/json")

	resp, err := clientVersionHTTPClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if err != nil {
		return
	}
	var payload struct {
		Version string `json:"version"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return
	}
	version := strings.TrimSpace(payload.Version)
	if !semverish(version) {
		return
	}
	clientVersionMu.Lock()
	clientVersionValue = version
	clientVersionFetchedAt = clientVersionNow()
	clientVersionMu.Unlock()
}

// semverish accepts the `x.y.z` shape the upstream versions use, so a feed
// hiccup that returns HTML or an empty field cannot replace a good version.
func semverish(v string) bool {
	if v == "" {
		return false
	}
	parts := strings.Split(v, ".")
	if len(parts) < 2 {
		return false
	}
	if len(parts) > 3 {
		parts = parts[:3]
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
		for _, r := range p {
			if r < '0' || r > '9' {
				return false
			}
		}
	}
	return true
}

// buildClientUA renders the identity-qualified user agent, e.g.
// `IDE/2.168.2 CodeBuddy/2.168.2`.
func buildClientUA(identity, version string) string {
	return identity + "/" + version + " CodeBuddy/" + version
}

// deviceTokenFor returns the device token to forward, if the account carries
// one. Imported CodeBuddy credentials usually do not, and the upstream accepts
// a request without it; the header is simply omitted rather than invented.
// WB_DEVICE_TOKEN overrides every account (the same escape hatch the reference
// gateway offers through its shared device-token file).
func deviceTokenFor(sa *storedAuth) string {
	if v := strings.TrimSpace(os.Getenv("WB_DEVICE_TOKEN")); v != "" {
		return v
	}
	if sa == nil || len(sa.rawJSON) == 0 {
		return ""
	}
	var doc map[string]any
	if json.Unmarshal(sa.rawJSON, &doc) != nil {
		return ""
	}
	for _, path := range [][]string{
		{"auth", "deviceToken"},
		{"auth", "device_token"},
		{"deviceToken"},
		{"device_token"},
	} {
		var cur any = doc
		for _, key := range path {
			m, ok := cur.(map[string]any)
			if !ok {
				cur = nil
				break
			}
			cur = m[key]
		}
		if s, ok := cur.(string); ok && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

// setWireHeader writes a header under the exact spelling given, bypassing the
// canonicalization http.Header.Set applies.
//
// This matters on the wire, not just cosmetically. Go canonicalizes header keys
// to MIME form when serializing, and that form is wrong for several of the names
// an official CodeBuddy client sends: `X-Conversation-ID` would ship as
// `X-Conversation-Id`, `X-B3-TraceId` as `X-B3-Traceid`, `X-IDE-Type` as
// `X-Ide-Type`, `x-codebuddy-request` as `X-Codebuddy-Request`. The reference
// gateway (workbuddy2api-hub #429) calls this out as its first transport-layer
// fix: send the header names the official client sends, not Go's re-spelling.
//
// The previous value is deleted from both the canonical and the exact key so a
// re-stamp cannot leave a stale duplicate behind; Go's textproto writer emits a
// duplicate key as two header lines, which would be worse than either spelling.
func setWireHeader(req *http.Request, name, value string) {
	if req == nil {
		return
	}
	if req.Header == nil {
		req.Header = make(http.Header, 8)
	}
	canonical := http.CanonicalHeaderKey(name)
	if canonical != name {
		delete(req.Header, canonical)
	}
	delete(req.Header, name)
	req.Header[name] = []string{value}
}

// wireHeader reads a header the way setWireHeader wrote it.
//
// Header.Get canonicalizes the key it is asked for, so it misses a header stored
// under an exact non-canonical spelling like `X-Conversation-ID` (which
// canonicalizes to `X-Conversation-Id`). Internal readers must use this so an
// exact-case wire name never becomes invisible to our own code.
func wireHeader(req *http.Request, name string) string {
	if req == nil {
		return ""
	}
	return getWireHeader(req.Header, name)
}

// getWireHeader is wireHeader for a bare header map (a captured request's
// headers, a transport stub's copy, and so on).
func getWireHeader(h http.Header, name string) string {
	if h == nil {
		return ""
	}
	if vs, ok := h[name]; ok {
		if len(vs) > 0 {
			return vs[0]
		}
		return ""
	}
	return h.Get(name)
}

// applyIdentityHeaders stamps the per-request client identity the upstream
// validates. It is the single place that decides what an official client looks
// like, so a future upstream change has one edit site.
func applyIdentityHeaders(req *http.Request, sa *storedAuth) {
	global := sa != nil && isGlobalDomain(sa.Auth.Domain)
	identity := identityFor(sa)
	setWireHeader(req, "User-Agent", buildClientUA(identity, clientVersionFor(global)))
	setWireHeader(req, "X-Product", "SaaS")
	setWireHeader(req, "X-IDE-Type", identity)
	setWireHeader(req, "X-IDE-Name", identity)
	setWireHeader(req, "x-codebuddy-request", "1")
	if tok := deviceTokenFor(sa); tok != "" {
		setWireHeader(req, "X-Device-Token", tok)
	}
	// The upstream is always streamed by this plugin; say so on the wire.
	setWireHeader(req, "Accept", "text/event-stream")
}

// applyConversationHeaders stamps the conversation/tracing family.
//
// `X-Conversation-ID` is cache identity — stable per conversation, which is
// what makes prompt caching hit. The rest are correlation ids: the upstream's
// channel check reads the turn-level `X-Conversation-Request-ID` and refuses a
// chat call that arrives without it (`11128 "conversation id missing"`).
//
// Every id is a UUID string, the shape an official client sends. They used to be
// bare 32-char hex, which is valid enough to be accepted but is itself a
// fingerprint the channel check can key on.
//
// The B3 pair is derived from the message UUID's canonical 32-hex form rather
// than from the dashed string, because B3 requires exactly 32 hex chars for the
// trace id and 16 for the span id: the bare hex of the same 16 bytes is that
// 32-hex trace id, and its first half is the 16-hex span id. Streaming the
// dashed form would make the trace id 36 chars and the upstream drops the pair.
func applyConversationHeaders(req *http.Request, cid string) (turnID, messageID string) {
	if strings.TrimSpace(cid) == "" {
		cid = randomUUID()
	}
	turnID = randomUUID()    // turn-scoped aggregation key
	messageID = randomUUID() // message-scoped id
	traceHex := strings.ReplaceAll(messageID, "-", "")

	setWireHeader(req, "X-Conversation-ID", cid)
	setWireHeader(req, "X-Conversation-Request-ID", turnID)
	setWireHeader(req, "X-Root-Request-ID", turnID)
	setWireHeader(req, "X-Conversation-Message-ID", messageID)
	setWireHeader(req, "X-Request-ID", messageID)
	setWireHeader(req, "x-trace-id", turnID)
	setWireHeader(req, "X-B3-TraceId", traceHex)
	setWireHeader(req, "X-B3-SpanId", traceHex[:16])
	setWireHeader(req, "X-B3-Sampled", "1")
	return turnID, messageID
}
