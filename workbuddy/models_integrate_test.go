package main

import (
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/pluginapi"
)

// The model tester in the panel must send the same client identity a real chat
// call sends: it exists to answer "does this model work on this account", so a
// different header set would make it answer a different question (and it used
// to: it stamped the stale WorkBuddy UA and none of the identity family).
func TestModelsTestUsesChatIdentity(t *testing.T) {
	t.Setenv("WORKBUDDY_DATA_DIR", t.TempDir())

	prevTarget, prevDo := mmFetchTargetFn, mmChatHTTPDo
	t.Cleanup(func() { mmFetchTargetFn, mmChatHTTPDo = prevTarget, prevDo })

	mmFetchTargetFn = func(string) (*mmFetchTarget, error) {
		return &mmFetchTarget{
			AuthID: "idx-1",
			Realm:  realmGlobal,
			Token:  "test-token",
			UID:    "uid-1",
			Origin: originRefererGlobal,
		}, nil
	}

	var got http.Header
	mmChatHTTPDo = func(req *http.Request) (*hostHTTPResponse, error) {
		got = req.Header.Clone()
		return &hostHTTPResponse{
			StatusCode: http.StatusOK,
			Body:       []byte("data: {\"model\":\"deepseek-v4.1-flash\",\"choices\":[{\"delta\":{\"content\":\"OK\"}}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"credit\":0.0}}\n\ndata: [DONE]\n\n"),
		}, nil
	}

	out := handleModelsTest(pluginapi.ManagementRequest{
		Body: []byte(`{"realm":"global","auth":"idx-1","model":"deepseek-v4.1-flash"}`),
	})
	if ok, _ := out["ok"].(bool); !ok {
		t.Fatalf("handleModelsTest = %#v, want ok", out)
	}

	if ua := getWireHeader(got, "User-Agent"); !strings.HasPrefix(ua, "CLI/") {
		t.Errorf("User-Agent = %q, want the Intl CLI identity", ua)
	}
	for _, h := range []string{"X-IDE-Type", "X-IDE-Name"} {
		if v := getWireHeader(got, h); v != "CLI" {
			t.Errorf("%s = %q, want IDE", h, v)
		}
	}
	if v := getWireHeader(got, "x-codebuddy-request"); v != "1" {
		t.Errorf("x-codebuddy-request = %q, want 1", v)
	}
	if getWireHeader(got, "X-Product") != "SaaS" {
		t.Errorf("X-Product = %q, want SaaS", getWireHeader(got, "X-Product"))
	}
	if turn, root := getWireHeader(got, "X-Conversation-Request-ID"), getWireHeader(got, "X-Root-Request-ID"); turn == "" || turn != root {
		t.Errorf("conversation ids = %q / %q, want the root id to mirror the turn id", turn, root)
	}
	if trace, span := getWireHeader(got, "X-B3-TraceId"), getWireHeader(got, "X-B3-SpanId"); len(trace) != 32 || len(span) != 16 {
		t.Errorf("B3 pair = %q/%q, want 32- and 16-hex ids", trace, span)
	} else if span != trace[:16] {
		t.Errorf("X-B3-SpanId %q must derive from X-B3-TraceId %q", span, trace)
	}
	if v := got.Get("X-User-Id"); v != "uid-1" {
		t.Errorf("X-User-Id = %q, want the account UID", v)
	}
	if v := got.Get("Accept"); v != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", v)
	}
}

func TestModelsTestUsesCLIIdentityForCN(t *testing.T) {
	t.Setenv("WORKBUDDY_DATA_DIR", t.TempDir())

	prevTarget, prevDo := mmFetchTargetFn, mmChatHTTPDo
	t.Cleanup(func() { mmFetchTargetFn, mmChatHTTPDo = prevTarget, prevDo })

	mmFetchTargetFn = func(string) (*mmFetchTarget, error) {
		return &mmFetchTarget{AuthID: "idx-cn", Realm: realmCN, Token: "test-token", Origin: originReferer}, nil
	}
	var got http.Header
	mmChatHTTPDo = func(req *http.Request) (*hostHTTPResponse, error) {
		got = req.Header.Clone()
		return &hostHTTPResponse{StatusCode: http.StatusOK, Body: []byte("data: [DONE]\n\n")}, nil
	}

	_ = handleModelsTest(pluginapi.ManagementRequest{Body: []byte(`{"realm":"cn","auth":"idx-cn","model":"glm-5.3"}`)})
	if ua := getWireHeader(got, "User-Agent"); !strings.HasPrefix(ua, "CLI/") {
		t.Errorf("User-Agent = %q, want the CN CLI identity", ua)
	}
	if v := getWireHeader(got, "X-IDE-Type"); v != "CLI" {
		t.Errorf("X-IDE-Type = %q, want CLI", v)
	}
}
