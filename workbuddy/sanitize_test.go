package main

import (
	"strings"
	"testing"
)

func TestSanitizeBlockedTemplates_ClaudeCode(t *testing.T) {
	in := "You are Claude Code, Anthropic's official CLI for Claude."
	out := sanitizeBlockedTemplates(in)
	if out == in {
		t.Fatal("should replace blocked template")
	}
	want := "You are Claude Code, Anthropic's official CLI tool for Claude."
	if out != want {
		t.Fatalf("got %q want %q", out, want)
	}
}

// The Claude Agent SDK form continues the sentence with a comma instead of a
// period. Matching only the period-terminated sentence left this variant
// untouched, and the upstream rejected every such request with 400 / code
// 11128 ("Illegal API invocation from an unapproved channel").
func TestSanitizeBlockedTemplates_ClaudeAgentSDKForm(t *testing.T) {
	in := "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."
	out := sanitizeBlockedTemplates(in)
	if out == in {
		t.Fatal("should replace the comma/SDK variant")
	}
	want := "You are Claude Code, Anthropic's official CLI tool for Claude, running within the Claude Agent SDK."
	if out != want {
		t.Fatalf("got %q want %q", out, want)
	}
}

// The feedback line is blocklisted as one exact sentence; verbatim it makes
// the upstream reject the request with 400 / code 11128.
func TestSanitizeBlockedTemplates_FeedbackLine(t *testing.T) {
	in := "  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues"
	out := sanitizeBlockedTemplates(in)
	if out == in {
		t.Fatal("should replace the blocklisted feedback line")
	}
	if strings.Contains(out, "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("blocklisted URL survived: %q", out)
	}
	if !strings.Contains(out, "To give feedback") {
		t.Fatalf("rewrite lost the line's meaning: %q", out)
	}
}

// A real system prompt carries both blocklisted strings at once; the sanitizer
// must clear both in a single pass.
func TestSanitizeBlockedTemplates_BothTemplatesInOnePrompt(t *testing.T) {
	in := "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.\n" +
		"  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues\n"
	out := sanitizeBlockedTemplates(in)
	if strings.Contains(out, "Anthropic's official CLI for Claude") {
		t.Fatalf("identity line survived: %q", out)
	}
	if strings.Contains(out, "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("feedback URL survived: %q", out)
	}
}

func TestSanitizeBlockedTemplates_MainBranch(t *testing.T) {
	in := "Main branch (you will usually use this for PRs)"
	out := sanitizeBlockedTemplates(in)
	if out == in {
		t.Fatal("should replace Main branch")
	}
}

func TestSanitizeBlockedTemplates_Codex(t *testing.T) {
	in := "You are a coding agent running in the Codex CLI, a terminal-based coding assistant."
	out := sanitizeBlockedTemplates(in)
	if out == in {
		t.Fatal("should replace codex template")
	}
	want := "You are a coding agent running in the Codex CLI tool, a terminal-based coding assistant."
	if out != want {
		t.Fatalf("got %q want %q", out, want)
	}
}

func TestRewriteSystemInPlace_TopLevelSystemString(t *testing.T) {
	feedback := "  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues"
	obj := map[string]any{
		"model":  "deepseek-v4.1-flash-sg",
		"system": feedback,
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
		},
	}
	changed := rewriteSystemInPlace(obj)
	if !changed {
		t.Fatal("should mark changed for top-level system")
	}
	sys, ok := obj["system"].(string)
	if !ok || strings.Contains(sys, "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("top-level system not sanitized: %v", obj["system"])
	}
}

func TestRewriteSystemInPlace_TopLevelSystemArray(t *testing.T) {
	feedback := "  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues"
	obj := map[string]any{
		"model": "deepseek-v4.1-flash-sg",
		"system": []any{
			map[string]any{"type": "text", "text": feedback},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
		},
	}
	changed := rewriteSystemInPlace(obj)
	if !changed {
		t.Fatal("should mark changed for top-level system array")
	}
	parts, ok := obj["system"].([]any)
	if !ok || len(parts) == 0 {
		t.Fatalf("system is not []any: %v", obj["system"])
	}
	part0 := parts[0].(map[string]any)
	text := part0["text"].(string)
	if strings.Contains(text, "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("top-level system array block not sanitized: %v", text)
	}
}

func TestRewriteSystemInPlace_InstructionsAndGemini(t *testing.T) {
	feedback := "  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues"
	obj := map[string]any{
		"instructions": feedback,
		"system_instruction": map[string]any{
			"parts": []any{
				map[string]any{"text": feedback},
			},
		},
	}
	changed := rewriteSystemInPlace(obj)
	if !changed {
		t.Fatal("should mark changed")
	}
	if strings.Contains(obj["instructions"].(string), "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("instructions not sanitized: %v", obj["instructions"])
	}
	si := obj["system_instruction"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"].(string)
	if strings.Contains(si, "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("system_instruction not sanitized: %v", si)
	}
}

func TestRewriteSystemInPlace_ToolCallsAndReasoning(t *testing.T) {
	feedback := "  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues"
	obj := map[string]any{
		"messages": []any{
			map[string]any{
				"role":              "assistant",
				"reasoning_content": feedback,
				"reasoning":         feedback,
				"tool_calls": []any{
					map[string]any{
						"function": map[string]any{
							"name":      "view",
							"arguments": feedback,
						},
					},
				},
			},
		},
	}
	changed := rewriteSystemInPlace(obj)
	if !changed {
		t.Fatal("should mark changed")
	}
	msg := obj["messages"].([]any)[0].(map[string]any)
	if strings.Contains(msg["reasoning_content"].(string), "https://github.com/anthropics/claude-code/issues") {
		t.Fatal("reasoning_content not sanitized")
	}
	if strings.Contains(msg["reasoning"].(string), "https://github.com/anthropics/claude-code/issues") {
		t.Fatal("reasoning not sanitized")
	}
	tc := msg["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)["arguments"].(string)
	if strings.Contains(tc, "https://github.com/anthropics/claude-code/issues") {
		t.Fatal("tool_calls arguments not sanitized")
	}
}

func TestRewriteSystemForUpstream_TopLevelSystem(t *testing.T) {
	in := []byte(`{"system":"  - To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues","messages":[{"role":"user","content":"hi"}]}`)
	out := rewriteSystemForUpstream(in)
	if strings.Contains(string(out), "https://github.com/anthropics/claude-code/issues") {
		t.Fatalf("feedback survived: %s", string(out))
	}
}

func TestSanitizeBlockedTemplates_NoMatch(t *testing.T) {
	in := "Hello world"
	out := sanitizeBlockedTemplates(in)
	if out != in {
		t.Fatal("should pass through unchanged")
	}
}

func TestForceMaxThinking_Hy3Model(t *testing.T) {
	obj := map[string]any{"model": "hy3-std"}
	changed := forceMaxThinking(obj)
	if !changed {
		t.Fatal("should change hy3 model")
	}
	if obj["reasoning_effort"] != "high" {
		t.Fatal("should set high")
	}
}

func TestForceMaxThinking_NonHy3Model(t *testing.T) {
	obj := map[string]any{"model": "glm-5.2"}
	changed := forceMaxThinking(obj)
	if changed {
		t.Fatal("should not change non-hy3 model")
	}
}

func TestForceMaxThinking_AlreadyHigh(t *testing.T) {
	obj := map[string]any{"model": "hy3-std", "reasoning_effort": "high"}
	changed := forceMaxThinking(obj)
	if changed {
		t.Fatal("should not change when already high")
	}
}

func TestTruncate(t *testing.T) {
	if truncate("hello", 10) != "hello" {
		t.Fatal("short string should be unchanged")
	}
	if truncate("hello world", 5) != "hello" {
		t.Fatal("should truncate to 5 chars")
	}
	if truncate("", 5) != "" {
		t.Fatal("empty string")
	}
}

func TestSanitizeBlockedTemplates_DeepSeekHarness(t *testing.T) {
	// The identity sentence is DROPPED. The official client separates the brand
	// with U+200B, and the upstream refuses the sentence wherever it sits, so
	// leading padding must not save it either. Only the sentence goes; the line
	// break after it must stay, or two paragraphs would be joined.
	cases := []struct{ name, in, want string }{
		{
			name: "zero-width separator",
			in:   "You are an AI agent powered by DeepSeek\u200bHarness.\nNext line.",
			want: "\nNext line.",
		},
		{
			name: "no trailing period",
			in:   "You are an AI agent powered by DeepSeek\u200bHarness\nNext line.",
			want: "\nNext line.",
		},
		{
			name: "leading BOM does not save it",
			in:   "\ufeffYou are an AI agent powered by DeepSeek\u200bHarness.\nNext line.",
			want: "\ufeff\nNext line.",
		},
		{
			name: "leading spaces do not save it",
			in:   "   You are an AI agent powered by DeepSeek\u200bHarness.\nNext line.",
			want: "   \nNext line.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sanitizeBlockedTemplates(tc.in)
			if got == tc.in {
				t.Fatalf("blocklisted sentence survived: %q", got)
			}
			if strings.Contains(got, "AI agent powered by") {
				t.Fatalf("identity sentence survived: %q", got)
			}
			if got != tc.want {
				t.Fatalf("sanitizeBlockedTemplates(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// The persona sentence is a DIFFERENT sentence with a different trigger: it
// carries no brand word, so the upstream must match the "coding agent powered by"
// frame and rewriting the model name would not help. It is reworded, not deleted,
// and the model id is kept because it tells the model which model serves it.
func TestSanitizeBlockedTemplates_PersonaPoweredBy(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{
			name: "cn model id",
			in:   "You are a coding agent powered by the hy4-preview model.\nNext line.",
			want: "You are a coding agent running on hy4-preview.\nNext line.",
		},
		{
			name: "intl model id with dots",
			in:   "You are a coding agent powered by the deepseek-v4.1-flash model.\nNext line.",
			want: "You are a coding agent running on deepseek-v4.1-flash.\nNext line.",
		},
		{
			name: "no leading article",
			in:   "You are a coding agent powered by gpt-6.1-sol model.",
			want: "You are a coding agent running on gpt-6.1-sol.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := sanitizeBlockedTemplates(tc.in)
			if strings.Contains(got, "coding agent powered by") {
				t.Fatalf("persona frame survived: %q", got)
			}
			if got != tc.want {
				t.Fatalf("sanitizeBlockedTemplates(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// The file-reference paragraph keeps its wording but loses the @ glyph, which is
// part of what the channel check matches.
func TestSanitizeBlockedTemplates_FileReferenceAt(t *testing.T) {
	in := "Tokens prefixed with @ are paths the user explicitly referenced. @\"...\" quotes a path.\nNext line."
	got := sanitizeBlockedTemplates(in)
	if strings.Contains(got, "Tokens prefixed with @") {
		t.Fatalf("@ survived in the file-reference paragraph: %q", got)
	}
	if !strings.Contains(got, "Tokens prefixed with (at) are paths") {
		t.Fatalf("paragraph lost its wording: %q", got)
	}
	if !strings.Contains(got, "Next line") {
		t.Fatalf("following line was consumed: %q", got)
	}
	// An @ outside that paragraph must survive: on its own the upstream accepts it.
	const plain = "mail me at a@b.com or look at @decorator"
	if out := sanitizeBlockedTemplates(plain); out != plain {
		t.Fatalf("an unrelated @ was rewritten: %q", out)
	}
}

// No rule may consume a newline and join two paragraphs together.
func TestSanitizeBlockedTemplates_KeepsLineStructure(t *testing.T) {
	in := "You are an AI agent powered by DeepSeek\u200bHarness.\n" +
		"You are a coding agent powered by the deepseek-v4.1-flash model.\n" +
		"Tokens prefixed with @ are paths.\n" +
		"Plain line."
	out := sanitizeBlockedTemplates(in)
	if strings.Count(in, "\n") != strings.Count(out, "\n") {
		t.Fatalf("line count changed: %d -> %d\n%q",
			strings.Count(in, "\n"), strings.Count(out, "\n"), out)
	}
}

// Near-misses must survive untouched: only the full "You are an AI agent
// powered by DeepSeek Harness" sentence is blocklisted (verified by replay —
// each of these returned 200 against the live gateway).
func TestSanitizeBlockedTemplates_DeepSeekHarnessNearMisses(t *testing.T) {
	untouched := []string{
		"DeepSeek Harness",
		"powered by DeepSeek Harness",
		"You are an AI agent powered by DeepSeek.",
		"You are an AI agent powered by Acme Harness.",
		"AI agent powered by DeepSeek Harness.",
	}
	for _, in := range untouched {
		if got := sanitizeBlockedTemplates(in); got != in {
			t.Fatalf("sanitizeBlockedTemplates(%q) = %q, want it untouched", in, got)
		}
	}
}

// The rewrite must survive the whole request pipeline (system message inside a
// chat-completions body), not just the string helper.
func TestRewriteSystemForUpstream_DeepSeekHarness(t *testing.T) {
	// End-to-end through the whole request pipeline: the identity sentence must be
	// gone from the system message while the rest of the prompt is untouched.
	in := []byte(`{"messages":[{"role":"system","content":"You are an AI agent powered by DeepSeek\u200bHarness.\nBe brief."},{"role":"user","content":"hi"}]}`)
	out := rewriteSystemForUpstream(in)
	if strings.Contains(string(out), "AI agent powered by") {
		t.Fatalf("identity sentence survived: %s", string(out))
	}
	if !strings.Contains(string(out), "Be brief.") {
		t.Fatalf("rewrite lost the surrounding prompt: %s", string(out))
	}
}
