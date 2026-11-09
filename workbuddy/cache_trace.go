// cache_trace.go records bounded, content-free request fingerprints alongside
// accounting. It never changes the request/CID and never stores prompt text,
// headers, credentials, tool names/arguments, or raw upstream errors.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

const cacheTraceMaxMessages = 512

// Hashes are SHA-256 truncated to 128 bits. Ordered message/prefix hashes let
// us find the first changed message, even for histories containing secret text.
// Lists are capped; full messages_hash still covers messages beyond the cap.
type cacheBodyFingerprint struct {
	BodyBytes     int    `json:"body_bytes"`
	BodyHash      string `json:"body_hash"`
	ValidJSON     bool   `json:"valid_json"`
	SystemHash    string `json:"system_hash,omitempty"`
	FirstUserHash string `json:"first_user_hash,omitempty"`
	ToolsHash     string `json:"tools_hash,omitempty"`
	OptionsHash   string `json:"options_hash,omitempty"`
	// Fixed option names only: values remain digests, never raw settings.
	OptionHashes    map[string]string `json:"option_hashes,omitempty"`
	MessageCount    int               `json:"message_count"`
	MessagesHash    string            `json:"messages_hash,omitempty"`
	MessageHashes   []string          `json:"message_hashes,omitempty"`
	PrefixHashes    []string          `json:"prefix_hashes,omitempty"`
	HashesTruncated bool              `json:"hashes_truncated,omitempty"`
}

type cacheTraceAttempt struct {
	AuthID            string `json:"auth_id"`
	CID               string `json:"cid"`
	UpstreamRequestID string `json:"upstream_request_id"`
	StartedAt         string `json:"started_at"`
	Status            int    `json:"status,omitempty"`
}

type requestCacheTrace struct {
	Version       int                  `json:"version"`
	RequestID     string               `json:"request_id"` // plugin-generated; safe log correlation
	HostRequestID string               `json:"host_request_id,omitempty"`
	StartedAt     string               `json:"started_at"`
	Inbound       cacheBodyFingerprint `json:"inbound"`
	Upstream      cacheBodyFingerprint `json:"upstream"`
	Attempts      []cacheTraceAttempt  `json:"attempts,omitempty"`
}

func traceHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:16])
}

// cacheTraceEnabled reports whether per-request cache fingerprints are collected.
//
// Off by default. The fingerprints dominate every usage record — measured at
// 99.4% of the JSONL line, i.e. ~976 message/prefix hashes per request — and cost
// a SHA-256 pass over the whole prompt plus two log lines per request. The usage
// query also has to parse all of it to show five fields (330ms vs 90ms with the
// log at 52MB). So they are opt-in diagnostics for the prompt-cache
// investigations, not always-on telemetry. Set WB_CACHE_TRACE=1 to collect them.
//
// Atomic because tests flip it while request paths read it under -race.
var cacheTraceEnabled atomic.Bool

// cacheTraceEnabledFromEnv parses the WB_CACHE_TRACE switch. Pure so the
// default (empty → off) can be tested without depending on the ambient
// environment of the test process.
func cacheTraceEnabledFromEnv(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// Reads the switch once at process start. This is a plain assignment, not a
// goroutine or a timer: nothing may run at library load (see
// TestNothingRunsAtLibraryLoad).
func init() {
	if cacheTraceEnabledFromEnv(os.Getenv("WB_CACHE_TRACE")) {
		cacheTraceEnabled.Store(true)
	}
}

// cacheTraceEnvDefault reports the deployment default from WB_CACHE_TRACE.
func cacheTraceEnvDefault() bool {
	return cacheTraceEnabledFromEnv(os.Getenv("WB_CACHE_TRACE"))
}

// applyCacheTracePreference resolves the effective switch. An explicit panel
// choice persisted in workbuddy-state.json wins over the environment default,
// so a toggle flipped from the Usage card survives plugin restarts. Called from
// configure() on every register/reconfigure.
func applyCacheTracePreference() {
	if st, err := loadState(); err == nil && st.CacheTrace != nil {
		cacheTraceEnabled.Store(*st.CacheTrace)
		return
	}
	cacheTraceEnabled.Store(cacheTraceEnvDefault())
}

// handleCacheTraceStatus reports the effective switch and where it came from:
// an explicit panel choice (has_override) or the WB_CACHE_TRACE default.
func handleCacheTraceStatus() map[string]any {
	override := false
	if st, err := loadState(); err == nil {
		override = st.CacheTrace != nil
	}
	return map[string]any{
		"enabled":      cacheTraceEnabled.Load(),
		"env_default":  cacheTraceEnvDefault(),
		"has_override": override,
		"persistent":   true,
	}
}

// handleCacheTraceConfig flips tracing at runtime and persists the choice in
// workbuddy-state.json (POST /cachetrace/config, body {"enabled": bool}). The
// flip applies to new requests immediately; rows already written keep whatever
// they had. Unlike the scheduler toggles, a restart does not silently turn the
// setting back off — this is a debugging session the user started on purpose.
func handleCacheTraceConfig(req pluginapi.ManagementRequest) map[string]any {
	var body struct {
		Enabled *bool `json:"enabled"`
	}
	if err := json.Unmarshal(req.Body, &body); err != nil || body.Enabled == nil {
		return map[string]any{"error": `expected {"enabled": true|false}`}
	}
	on := *body.Enabled
	cacheTraceEnabled.Store(on)
	persistent := true
	if err := mutateState(func(st *wbStateFile) error {
		st.CacheTrace = cloneBoolPtr(&on)
		return nil
	}); err != nil {
		persistent = false
		guardLogf("cache-trace: state write failed, keeping runtime-only: %v", err)
	}
	return map[string]any{
		"enabled":      on,
		"env_default":  cacheTraceEnvDefault(),
		"has_override": persistent,
		"persistent":   persistent,
	}
}

// newCacheTrace returns a fingerprint for the request, or nil when tracing is
// off. Callers pass the result straight through; every consumer tolerates nil
// (the trace is an optional field on the usage record).
func newCacheTrace(payload, original, upstream []byte, metadata map[string]any, started time.Time) *requestCacheTrace {
	if !cacheTraceEnabled.Load() {
		return nil
	}
	return newRequestCacheTrace(payload, original, upstream, metadata, started)
}

func traceDigest(h hash.Hash) string { return hex.EncodeToString(h.Sum(nil)[:16]) }

// Hash length-framed parts to avoid ambiguous concatenations. JSON field values
// remain exact bytes (wire serialization differences must remain observable).
func tracePart(h hash.Hash, b []byte) {
	fmt.Fprintf(h, "%d:", len(b))
	_, _ = h.Write(b)
}

func fingerprintCacheBody(body []byte) cacheBodyFingerprint {
	out := cacheBodyFingerprint{BodyBytes: len(body), BodyHash: traceHash(body)}
	var obj map[string]json.RawMessage
	if json.Unmarshal(body, &obj) != nil || obj == nil {
		return out
	}
	out.ValidJSON = true
	var messages []json.RawMessage
	_ = json.Unmarshal(obj["messages"], &messages)
	out.MessageCount = len(messages)
	system, prefix := sha256.New(), sha256.New()
	for i, m := range messages {
		tracePart(prefix, m)
		if i < cacheTraceMaxMessages {
			out.MessageHashes = append(out.MessageHashes, traceHash(m))
			out.PrefixHashes = append(out.PrefixHashes, traceDigest(prefix))
		}
		var msg struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(m, &msg) != nil {
			continue
		}
		if msg.Role == "system" || msg.Role == "developer" {
			tracePart(system, msg.Content)
		}
		if msg.Role == "user" && out.FirstUserHash == "" {
			out.FirstUserHash = traceHash(msg.Content)
		}
	}
	out.SystemHash = traceDigest(system)
	out.MessagesHash = traceDigest(prefix)
	out.HashesTruncated = len(messages) > cacheTraceMaxMessages
	// Tools and legacy functions affect the provider prefix independently of CID.
	tools, _ := json.Marshal(map[string]json.RawMessage{"tools": obj["tools"], "functions": obj["functions"]})
	out.ToolsHash = traceHash(tools)
	delete(obj, "messages")
	delete(obj, "tools")
	delete(obj, "functions")
	options, _ := json.Marshal(obj)
	out.OptionsHash = traceHash(options)
	for _, key := range []string{"model", "temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning", "reasoning_effort", "tool_choice", "parallel_tool_calls", "stream", "stream_options", "seed", "response_format", "stop", "logprobs", "top_logprobs", "presence_penalty", "frequency_penalty", "n", "service_tier"} {
		if value, ok := obj[key]; ok {
			if out.OptionHashes == nil {
				out.OptionHashes = make(map[string]string)
			}
			out.OptionHashes[key] = traceHash(value)
		}
	}
	return out
}

// Host-supplied metadata/IDs may be arbitrary client input. Only short hexadecimal
// correlation IDs/UUIDs are emitted verbatim; everything else is hashed.
func traceSafeID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	if len(id) <= 64 {
		safe := true
		for _, c := range id {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F' || c == '-') {
				safe = false
				break
			}
		}
		if safe {
			return id
		}
	}
	return "hash:" + traceHash([]byte(id))
}

func newRequestCacheTrace(payload, original, upstream []byte, metadata map[string]any, started time.Time) *requestCacheTrace {
	if len(payload) == 0 {
		payload = original
	}
	t := &requestCacheTrace{Version: 1, RequestID: randomHex(16), StartedAt: started.UTC().Format(time.RFC3339Nano), Inbound: fingerprintCacheBody(payload), Upstream: fingerprintCacheBody(upstream)}
	for _, k := range []string{"request_id", "requestId"} {
		if s, ok := metadata[k].(string); ok && s != "" {
			t.HostRequestID = traceSafeID(s)
			break
		}
	}
	return t
}

// One trace belongs to exactly one inbound request/goroutine. Fingerprints are
// built once, while failover appends only the actual outgoing header identities.
func (t *requestCacheTrace) beginAttempt(req *http.Request, authID string) int {
	if t == nil {
		return -1
	}
	a := cacheTraceAttempt{AuthID: traceSafeID(authID), CID: traceSafeID(wireHeader(req, "X-Conversation-ID")), UpstreamRequestID: traceSafeID(wireHeader(req, "X-Request-ID")), StartedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	t.Attempts = append(t.Attempts, a)
	fmt.Fprintf(os.Stderr, "[workbuddy] cache-trace id=%s attempt=%d auth=%s cid=%s upstream_request=%s bytes=%d messages=%d\n", t.RequestID, len(t.Attempts), a.AuthID, a.CID, a.UpstreamRequestID, t.Upstream.BodyBytes, t.Upstream.MessageCount)
	return len(t.Attempts) - 1
}

func (t *requestCacheTrace) finishAttempt(index, status int) {
	if t != nil && index >= 0 && index < len(t.Attempts) {
		t.Attempts[index].Status = status
	}
}

func firstCacheTrace(traces []*requestCacheTrace) *requestCacheTrace {
	if len(traces) > 0 {
		return traces[0]
	}
	return nil
}
