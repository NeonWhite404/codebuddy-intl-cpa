package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func traceBody(system, user string, extra ...string) []byte {
	messages := []map[string]any{{"role": "system", "content": system}, {"role": "user", "content": user}}
	for _, text := range extra {
		messages = append(messages, map[string]any{"role": "assistant", "content": text})
	}
	body, _ := json.Marshal(map[string]any{"model": "test-model", "messages": messages, "tools": []any{map[string]any{"type": "function", "function": map[string]any{"name": "fake-secret-tool", "parameters": map[string]any{"type": "object"}}}}})
	return body
}

func TestCacheFingerprintAppendOnlyAndMutation(t *testing.T) {
	a := fingerprintCacheBody(traceBody("sys", "user", "answer"))
	b := fingerprintCacheBody(traceBody("sys", "user", "answer", "new"))
	if a.BodyHash == b.BodyHash || a.MessagesHash == b.MessagesHash {
		t.Fatal("full hashes must notice append")
	}
	if a.SystemHash != b.SystemHash || a.FirstUserHash != b.FirstUserHash || a.ToolsHash != b.ToolsHash || a.OptionsHash != b.OptionsHash {
		t.Fatal("append changed stable hashes")
	}
	for i, h := range a.PrefixHashes {
		if h != b.PrefixHashes[i] {
			t.Fatalf("prefix %d differs", i)
		}
	}
	c := fingerprintCacheBody(traceBody("sys", "user", "changed"))
	if a.MessageHashes[0] != c.MessageHashes[0] || a.MessageHashes[1] != c.MessageHashes[1] || a.MessageHashes[2] == c.MessageHashes[2] {
		t.Fatal("cannot isolate changed message")
	}
	if a.PrefixHashes[2] == c.PrefixHashes[2] {
		t.Fatal("changed prefix not detected")
	}
}

func TestCacheFingerprintSchemaOptionsAndCID(t *testing.T) {
	body := traceBody("sys", "user")
	var obj map[string]any
	_ = json.Unmarshal(body, &obj)
	a := fingerprintCacheBody(body)
	obj["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "changed-tool"}}}
	changed, _ := json.Marshal(obj)
	b := fingerprintCacheBody(changed)
	if a.ToolsHash == b.ToolsHash || a.MessagesHash != b.MessagesHash {
		t.Fatal("tool change attribution failed")
	}
	if stableConversationID(body) != stableConversationID(changed) {
		t.Fatal("schema change should not alter current CID algorithm")
	}
	obj["temperature"] = 0.5
	changed, _ = json.Marshal(obj)
	c := fingerprintCacheBody(changed)
	if c.OptionsHash == b.OptionsHash || c.ToolsHash != b.ToolsHash {
		t.Fatal("options change attribution failed")
	}
	if c.OptionHashes["temperature"] == "" || c.OptionHashes["temperature"] != traceHash([]byte("0.5")) || c.OptionHashes["model"] != b.OptionHashes["model"] {
		t.Fatal("per-option attribution failed")
	}
	spaced := bytes.ReplaceAll(body, []byte(`,"`), []byte(`, "`))
	if got := fingerprintCacheBody(spaced); got.BodyHash == a.BodyHash {
		t.Fatal("exact body serialization not observed")
	}
}

func TestCacheTraceDoesNotStorePromptOrCredentials(t *testing.T) {
	secret := "Bearer private-key-material-test"
	body := traceBody(secret, secret, "private-history-text")
	var options map[string]any
	_ = json.Unmarshal(body, &options)
	options["private-key-material-test"] = secret // arbitrary option names must not be logged
	options["response_format"] = map[string]any{"private": secret}
	body, _ = json.Marshal(options)
	tr := newRequestCacheTrace(body, nil, body, map[string]any{"request_id": secret}, time.Now())
	req, _ := http.NewRequest(http.MethodPost, "https://example.invalid", bytes.NewReader(body))
	backendHeaders(req, &storedAuth{Auth: storedTokens{AccessToken: secret}, Account: storedAccount{UID: secret}})
	idx := tr.beginAttempt(req, secret)
	tr.finishAttempt(idx, 200)
	raw, _ := json.Marshal(tr)
	for _, bad := range []string{secret, "private-key-material-test", "private-history-text", "fake-secret-tool", "Authorization", "parameters"} {
		if bytes.Contains(raw, []byte(bad)) {
			t.Fatalf("trace leaked %s", bad)
		}
	}
	if !strings.HasPrefix(tr.HostRequestID, "hash:") {
		t.Fatal("untrusted metadata ID not hashed")
	}
	if tr.Attempts[0].CID != wireHeader(req, "X-Conversation-ID") || tr.Attempts[0].Status != 200 {
		t.Fatal("actual CID/status missing")
	}
	// The body on the wire is now gzipped by backendHeaders, so compare against
	// the decoded form: the invariant under test is that tracing did not ALTER
	// the payload, not that it left it uncompressed.
	after := readWireBody(t, req)
	if !bytes.Equal(body, after) {
		t.Fatal("trace changed body")
	}
	for _, s := range []string{tr.Upstream.BodyHash, tr.Upstream.SystemHash, tr.Upstream.ToolsHash} {
		if len(s) != 32 {
			t.Fatalf("invalid digest %s", s)
		}
	}
}

func TestCacheTraceBoundsInvalidBodiesAndDeterminism(t *testing.T) {
	invalid := []byte("not JSON secret text")
	a := fingerprintCacheBody(invalid)
	if a.ValidJSON || a.BodyHash == "" || len(a.MessageHashes) > 0 {
		t.Fatalf("invalid fingerprint %+v", a)
	}
	text := []string{}
	for i := 0; i < cacheTraceMaxMessages+5; i++ {
		text = append(text, fmt.Sprint(i))
	}
	body := traceBody("sys", "user", text...)
	b := fingerprintCacheBody(body)
	if !b.HashesTruncated || len(b.PrefixHashes) != cacheTraceMaxMessages || b.MessageCount != cacheTraceMaxMessages+7 {
		t.Fatalf("unbounded hashes %+v", b)
	}
	c := fingerprintCacheBody(body)
	rawB, _ := json.Marshal(b)
	rawC, _ := json.Marshal(c)
	if !bytes.Equal(rawB, rawC) {
		t.Fatal("fingerprints must survive restart deterministically")
	}
	if len(rawB) > 40000 {
		t.Fatalf("trace too large: %d", len(rawB))
	}
}

func TestTraceIDsAndOptionalNil(t *testing.T) {
	if traceSafeID("92dd944e") != "92dd944e" || traceSafeID("abc-secret") == "abc-secret" {
		t.Fatal("unsafe correlation IDs")
	}
	var tr *requestCacheTrace
	if tr.beginAttempt(nil, "") != -1 {
		t.Fatal("nil trace not optional")
	}
	tr.finishAttempt(-1, 0)
}

func TestCacheTracePersistAndReload(t *testing.T) {
	t.Setenv("WORKBUDDY_DATA_DIR", t.TempDir())
	body := traceBody("sys", "user")
	trace := newRequestCacheTrace(body, nil, body, nil, time.Now())
	store := &requestUsageStore{perDay: map[string]map[string]*dayAgg{}}
	store.record(requestUsageRecord{Timestamp: time.Now().UTC().Format(time.RFC3339), AuthID: "abcd", Credit: 0.1, CacheTrace: trace})
	loaded := &requestUsageStore{perDay: map[string]map[string]*dayAgg{}}
	loaded.loadFromDisk()
	records := loaded.ringSnapshot()
	if len(records) != 1 || records[0].CacheTrace == nil || records[0].CacheTrace.Upstream.BodyHash != trace.Upstream.BodyHash {
		t.Fatalf("trace not persisted/reloaded: %+v", records)
	}
}

func TestUsageRotationPreservesEarlierMonthlyEvidence(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("WORKBUDDY_DATA_DIR", dir)
	old := filepath.Join(dir, "request-usage-2026-10.jsonl")
	if err := os.WriteFile(old, []byte("preserve this older evidence"), 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, usageLogBaseName)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(usageLogMaxBytes + 1); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()
	store := &requestUsageStore{perDay: map[string]map[string]*dayAgg{}}
	store.record(requestUsageRecord{Timestamp: "2026-10-13T05:00:00Z", AuthID: "abcd"})
	if raw, _ := os.ReadFile(old); string(raw) != "preserve this older evidence" {
		t.Fatal("overwrote older monthly log")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "request-usage-2026-10.part-*.jsonl"))
	if len(files) != 1 {
		t.Fatalf("no unique rotated log: %v", files)
	}
}

func BenchmarkFingerprintLargeHistory(b *testing.B) {
	extra := make([]string, 180)
	for i := range extra {
		extra[i] = strings.Repeat("synthetic text ", 250)
	}
	body := traceBody("sys", "user", extra...)
	b.SetBytes(int64(len(body)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fingerprintCacheBody(body)
	}
}

// TestCacheTraceEnvParsing pins the switch itself. Parsing is a pure function so
// this does not depend on whether the test process happens to have
// WB_CACHE_TRACE set.
func TestCacheTraceEnvParsing(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{
		{"", false},      // the default: fingerprints off
		{"   ", false},   // whitespace only is still off
		{"0", false},     // an explicit off
		{"false", false}, // an explicit off
		{"1", true},
		{"true", true},
		{"YES", true},      // case-insensitive
		{" on ", true},     // trimmed
		{"garbage", false}, // anything unrecognised stays off
	} {
		if got := cacheTraceEnabledFromEnv(tc.value); got != tc.want {
			t.Errorf("cacheTraceEnabledFromEnv(%q) = %v, want %v", tc.value, got, tc.want)
		}
	}
}

// TestCacheTraceGateControlsRecording proves the switch actually gates the work:
// off means no fingerprint and no per-request log, on means a full fingerprint.
func TestCacheTraceGateControlsRecording(t *testing.T) {
	original := cacheTraceEnabled.Load()
	t.Cleanup(func() { cacheTraceEnabled.Store(original) })

	body := traceBody("sys", "user")

	cacheTraceEnabled.Store(false)
	if tr := newCacheTrace(body, nil, body, nil, time.Now()); tr != nil {
		t.Fatal("disabled gate produced a fingerprint")
	}
	// Consumers must tolerate a nil trace: this is what the request path passes
	// when tracing is off.
	var nilTrace *requestCacheTrace
	if got := firstCacheTrace([]*requestCacheTrace{nilTrace}); got != nil {
		t.Fatal("firstCacheTrace should pass nil through")
	}
	if idx := nilTrace.beginAttempt(nil, "auth"); idx != -1 {
		t.Fatalf("beginAttempt on a nil trace returned %d, want -1", idx)
	}
	nilTrace.finishAttempt(0, 200) // must not panic

	cacheTraceEnabled.Store(true)
	tr := newCacheTrace(body, nil, body, nil, time.Now())
	if tr == nil {
		t.Fatal("enabled gate produced no fingerprint")
	}
	if tr.Upstream.BodyHash == "" || len(tr.Upstream.MessageHashes) == 0 {
		t.Fatalf("enabled fingerprint is empty: %+v", tr.Upstream)
	}
}
