package main

import (
	"bytes"
	"compress/gzip"
	"io"
	"net/http"
	"testing"
)

// readWireBody drains a request body, transparently gunzipping it when the
// request carries Content-Encoding: gzip.
//
// Chat bodies are gzipped in backendHeaders (see gzipChatBody), so a transport
// stub that wants to inspect the JSON must decode first. Keeping this in one
// helper means a future body encoding is a one-file change instead of six.
func readWireBody(t *testing.T, r *http.Request) []byte {
	t.Helper()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	if wireHeader(r, "Content-Encoding") != "gzip" {
		return raw
	}
	zr, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("gzip header: %v", err)
	}
	defer zr.Close()
	plain, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("gunzip request body: %v", err)
	}
	return plain
}
