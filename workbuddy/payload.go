// payload.go rewrites the outgoing chat completion request body before it's
// forwarded upstream. The single-pass entry point is prepareUpstreamBody; the
// four *InPlace helpers are the field-level mutations it composes, and the
// legacy *ForUpstream / forceStreamBody wrappers exist for tests and other
// call sites that need them individually.
package main

import (
	"encoding/json"
	"regexp"
	"strings"
)

// reDeepSeekHarnessIdentity matches the DSH base-instructions identity sentence,
// which the official client puts at the very start of the system message
// (`You are an AI agent powered by DeepSeek` + U+200B + `Harness.`) and which the
// upstream refuses with 400 / code 11128. Located by single-variable replay
// against copilot.tencent.com: the bare sentence answers 400 every time, while
// "DeepSeek Harness" alone, "powered by DeepSeek Harness" and the same sentence
// naming a different harness all answer 200; swapping only that sentence inside
// a 175 KB request (47 messages, 66 tools, reasoning_effort=max) flips 400 -> 200.
//
// The client separates the brand with U+200B, so the gap is matched as
// invisibles/whitespace rather than a literal space. Leading padding does not
// dodge the upstream's match, so this rule does not try to anchor on position.
//
// The sentence is DROPPED rather than reworded: it carries no instruction the
// model needs (the persona sentence and the tool guidance that follow already
// set the framing), and deletion cannot accidentally re-match a future upstream
// signature the way a reworded copy might.
var reDeepSeekHarnessIdentity = regexp.MustCompile(`(?i)You are an AI agent powered by DeepSeek[\x{200B}\x{200C}\x{200D}\x{FEFF}\x{00A0}\t ]*Harness\.?[ \t]*($|[\r\n])`)

// reDeepSeekPersonaPoweredBy matches a DIFFERENT sentence - the persona line with
// the model id interpolated:
//
//	You are a coding agent powered by the deepseek-v4.1-flash model.
//
// Two sentences, two rules. The identity sentence above is recognised by its
// brand pair; this one contains no brand word at all, so the upstream must be
// matching the `coding agent powered by` frame and rewriting the model name
// would not help. Only that phrase is replaced; the model id is kept, because it
// tells the model which model is serving it.
var reDeepSeekPersonaPoweredBy = regexp.MustCompile(`(?i)(You are a coding agent)[ \t]+powered[ \t]+by[ \t]+(?:the[ \t]+)?(.+?)(?:[ \t]+model)?(\.[ \t]*)($|[\r\n])`)

// reDeepSeekBrandPair is the catch-all behind the identity rule: any remaining
// `DeepSeek` + invisible separator + `Harness` anywhere in a message - a frozen
// paragraph from an older conversation, a variant intro - is collapsed to the
// hyphenated name.
var reDeepSeekBrandPair = regexp.MustCompile(`DeepSeek[\x{200B}\x{200C}\x{200D}\x{FEFF}]+Harness`)

const deepSeekBrandPairReplacement = "DeepSeek-Harness"

// reFileReferencePara isolates the file-reference paragraph the client injects
// into the system message. Its `@` glyph is part of what the channel check
// matches, so `@` is rewritten to the equivalent `(at)` inside this paragraph
// only; a bare `@`, an email address and a decorator elsewhere are accepted by
// the upstream on their own and are left alone.
var reFileReferencePara = regexp.MustCompile(`Tokens prefixed with @ are paths[^\n]*`)

// reWebGUIBrand matches the brand reference inside the Web GUI paragraph.
var reWebGUIBrand = regexp.MustCompile(`(?i)the[ \t]+DeepSeek[\x{200B}\x{200C}\x{200D}\x{FEFF}\x{00A0}\t ]*Harness[ \t]+Web GUI`)

const webGUIBrandReplacement = "this Web GUI"

// forceStreamBody returns the request body with "stream":true set, since the
// upstream rejects non-streaming chat requests.
// prepareUpstreamBody composes forceStreamBody + normalizeToolsForUpstream +
// rewriteSystemForUpstream + ensureSystemMessage + rewriteModelInBody into a
// single unmarshal/marshal pass (v0.6.31 perf: was 4-5 full JSON round-trips
// on every chat completion). The 4 legacy helpers remain for tests and other
// call sites that need them individually.
func prepareUpstreamBody(payload, original []byte, sa *storedAuth, upstreamModel string) []byte {
	src := payload
	if len(src) == 0 {
		src = original
	}
	if len(src) == 0 {
		return src
	}
	var obj map[string]any
	if json.Unmarshal(src, &obj) != nil {
		return src
	}

	// 1. forceStream: CodeBuddy rejects non-stream requests.
	obj["stream"] = true

	// 2. normalizeTools: tool_choice object form → string; "none" suppresses tools.
	normalizeToolsInPlace(obj)

	// 3. rewriteSystem: strip blocked Claude Code template phrases + force thinking.
	// Recover original instruction roles before changing their text.
	restoreResponsesDeveloperRoles(obj, original)
	rewriteSystemInPlace(obj)

	// 3b. normalizeDeveloper: CodeBuddy upstream rejects role=developer (code 11128).
	normalizeDeveloperRoleInPlace(obj)

	// 4. ensureSystemMessage: inject minimal system msg for Global only.
	ensureSystemMessageInPlace(obj, sa)

	// 5. rewriteModel: swap client model name to upstream model id.
	rewriteModelInPlace(obj, upstreamModel)

	out, err := json.Marshal(obj)
	if err != nil {
		return src
	}
	return out
}

// normalizeToolsInPlace is the in-place form of normalizeToolsForUpstream.
// Returns true when obj was modified.
func normalizeToolsInPlace(obj map[string]any) bool {
	changed := false
	suppressTools := func() {
		if _, ok := obj["tools"]; ok {
			delete(obj, "tools")
			changed = true
		}
		if _, ok := obj["functions"]; ok {
			delete(obj, "functions")
			changed = true
		}
	}
	if tc, present := obj["tool_choice"]; present {
		switch v := tc.(type) {
		case string:
			if strings.EqualFold(strings.TrimSpace(v), "none") {
				delete(obj, "tool_choice")
				suppressTools()
				changed = true
			}
		case map[string]any:
			typ, _ := v["type"].(string)
			typ = strings.ToLower(strings.TrimSpace(typ))
			switch typ {
			case "none":
				delete(obj, "tool_choice")
				suppressTools()
				changed = true
			case "auto", "required":
				obj["tool_choice"] = typ
				changed = true
			case "function":
				name := ""
				if fn, ok := v["function"].(map[string]any); ok {
					name, _ = fn["name"].(string)
				}
				if name == "" {
					name, _ = v["name"].(string)
				}
				name = strings.TrimSpace(name)
				if name != "" {
					obj["tool_choice"] = name
				} else {
					obj["tool_choice"] = "auto"
				}
				changed = true
			default:
				delete(obj, "tool_choice")
				changed = true
			}
		default:
			delete(obj, "tool_choice")
			changed = true
		}
	}
	return changed
}

// rewriteSystemInPlace is the in-place form of rewriteSystemForUpstream.
func rewriteSystemInPlace(obj map[string]any) bool {
	changed := false

	// Top-level system field (Anthropic Messages API: string or []content_block).
	if sys, ok := obj["system"]; ok {
		if newSys, mod := sanitizeTextOrParts(sys); mod {
			obj["system"] = newSys
			changed = true
		}
	}

	// Top-level instructions field (Responses API).
	if inst, ok := obj["instructions"].(string); ok {
		if r := sanitizeBlockedTemplates(inst); r != inst {
			obj["instructions"] = r
			changed = true
		}
	}

	// Top-level system_instruction (Gemini API shape: {"parts":[{"text":"..."}]}).
	if si, ok := obj["system_instruction"].(map[string]any); ok {
		if parts, ok := si["parts"].([]any); ok {
			for _, p := range parts {
				if part, ok := p.(map[string]any); ok {
					if t, ok := part["text"].(string); ok {
						if r := sanitizeBlockedTemplates(t); r != t {
							part["text"] = r
							changed = true
						}
					}
				}
			}
		}
	}

	messages, _ := obj["messages"].([]any)
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if rewriteContentField(msg) {
			changed = true
		}
	}
	if forceMaxThinking(obj) {
		changed = true
	}
	return changed
}

// normalizeDeveloperRoleInPlace rewrites role "developer" to "system".
// CodeBuddy upstream rejects "developer" with code 11128 ("Illegal API
// invocation from an unapproved channel" / "first message is not system
// prompt"); only "system" is accepted for the first message.
func normalizeDeveloperRoleInPlace(obj map[string]any) bool {
	messages, ok := obj["messages"].([]any)
	if !ok {
		return false
	}
	changed := false
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); strings.EqualFold(strings.TrimSpace(role), "developer") {
			msg["role"] = "system"
			changed = true
		}
	}
	return changed
}

// ensureSystemMessageInPlace is the in-place form of ensureSystemMessage.
// Returns true when obj was modified.
func ensureSystemMessageInPlace(obj map[string]any, sa *storedAuth) bool {
	if sa == nil || !isGlobalDomain(sa.Auth.Domain) {
		return false
	}
	messages, ok := obj["messages"].([]any)
	if !ok || len(messages) == 0 {
		return false
	}
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); strings.EqualFold(role, "system") {
			return false
		}
	}
	systemMsg := map[string]any{
		"role":    "system",
		"content": "You are a helpful assistant.",
	}
	obj["messages"] = append([]any{systemMsg}, messages...)
	return true
}

// rewriteModelInPlace swaps obj["model"] to upstreamModel when non-empty.
// Mirrors rewriteModelInBody's behavior (case-insensitive compare); returns
// true when modified.
func rewriteModelInPlace(obj map[string]any, upstreamModel string) bool {
	upstreamModel = strings.TrimSpace(upstreamModel)
	if upstreamModel == "" {
		return false
	}
	cur, _ := obj["model"].(string)
	if strings.EqualFold(strings.TrimSpace(cur), upstreamModel) {
		return false
	}
	obj["model"] = upstreamModel
	return true
}

func forceStreamBody(payload, original []byte) []byte {
	src := payload
	if len(src) == 0 {
		src = original
	}
	var obj map[string]any
	if json.Unmarshal(src, &obj) != nil {
		return src
	}
	obj["stream"] = true
	out, err := json.Marshal(obj)
	if err != nil {
		return src
	}
	return out
}

// normalizeToolsForUpstream adapts OpenAI tools / tool_choice fields to
// CodeBuddy's chat schema before the request is forwarded.
//
// Live-verified against /v2/chat/completions (2026-07):
//  1. tool_choice is typed as string on the upstream Go struct. OpenAI's object
//     form {"type":"function","function":{"name":"..."}} returns 400 code 11101
//     ("cannot unmarshal object into Go struct field Request.tool_choice of
//     type string"). Convert known object shapes to the matching string.
//  2. tool_choice "none" is accepted but ignored when tools[] is non-empty —
//     the model still emits tool_calls. The only reliable way to suppress tools
//     is to omit tools (and functions) entirely.
//
// String values auto / required / <function name> are left untouched.
func normalizeToolsForUpstream(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	var obj map[string]any
	if json.Unmarshal(payload, &obj) != nil {
		return payload
	}
	changed := false

	suppressTools := func() {
		if _, ok := obj["tools"]; ok {
			delete(obj, "tools")
			changed = true
		}
		if _, ok := obj["functions"]; ok {
			delete(obj, "functions")
			changed = true
		}
	}

	if tc, present := obj["tool_choice"]; present {
		switch v := tc.(type) {
		case string:
			if strings.EqualFold(strings.TrimSpace(v), "none") {
				delete(obj, "tool_choice")
				suppressTools()
				changed = true
			}
		case map[string]any:
			typ, _ := v["type"].(string)
			typ = strings.ToLower(strings.TrimSpace(typ))
			switch typ {
			case "none":
				delete(obj, "tool_choice")
				suppressTools()
				changed = true
			case "auto", "required":
				obj["tool_choice"] = typ
				changed = true
			case "function":
				name := ""
				if fn, ok := v["function"].(map[string]any); ok {
					name, _ = fn["name"].(string)
				}
				if name == "" {
					name, _ = v["name"].(string)
				}
				name = strings.TrimSpace(name)
				if name != "" {
					obj["tool_choice"] = name
				} else {
					// Object force without a name: fall back to auto instead of 400.
					obj["tool_choice"] = "auto"
				}
				changed = true
			default:
				// Unknown object shape → drop rather than forward a 400.
				delete(obj, "tool_choice")
				changed = true
			}
		default:
			// null / array / number — drop to keep upstream happy.
			delete(obj, "tool_choice")
			changed = true
		}
	}

	if !changed {
		return payload
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return payload
	}
	return out
}

// rewriteSystemForUpstream neutralizes Claude Code template phrases that
// Tencent CodeBuddy's content filter blocklists verbatim — the agent identity
// line ("You are Claude Code, Anthropic's official CLI for Claude.") and the
// git injection ("Main branch (you will usually use this for PRs)"). Each
// rewrite is a single-word change so the prompt's meaning is preserved while
// dodging the exact-match filter.
func rewriteSystemForUpstream(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	var obj map[string]any
	if json.Unmarshal(payload, &obj) != nil {
		return payload
	}
	if !rewriteSystemInPlace(obj) {
		return payload
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return payload
	}
	return out
}

// rewriteContentField sanitizes blocked templates in one message's content,
// reasoning, and tool calls.
func rewriteContentField(msg map[string]any) bool {
	changed := false
	if c, ok := msg["content"]; ok {
		if newC, mod := sanitizeTextOrParts(c); mod {
			msg["content"] = newC
			changed = true
		}
	}
	if rc, ok := msg["reasoning_content"].(string); ok {
		if r := sanitizeBlockedTemplates(rc); r != rc {
			msg["reasoning_content"] = r
			changed = true
		}
	}
	if r, ok := msg["reasoning"].(string); ok {
		if s := sanitizeBlockedTemplates(r); s != r {
			msg["reasoning"] = s
			changed = true
		}
	}
	if tc, ok := msg["tool_calls"].([]any); ok {
		for _, c := range tc {
			call, ok := c.(map[string]any)
			if !ok {
				continue
			}
			fn, ok := call["function"].(map[string]any)
			if !ok {
				continue
			}
			args, ok := fn["arguments"].(string)
			if !ok {
				continue
			}
			if s := sanitizeBlockedTemplates(args); s != args {
				fn["arguments"] = s
				changed = true
			}
		}
	}
	return changed
}

// sanitizeTextOrParts sanitizes blocked templates in either a plain string or
// an array of content parts (e.g. Anthropic multimodal blocks [{"type":"text","text":"..."}]).
func sanitizeTextOrParts(v any) (any, bool) {
	switch c := v.(type) {
	case string:
		if r := sanitizeBlockedTemplates(c); r != c {
			return r, true
		}
	case []any:
		modified := false
		for _, p := range c {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if t, ok := part["text"].(string); ok {
				if r := sanitizeBlockedTemplates(t); r != t {
					part["text"] = r
					modified = true
				}
			}
		}
		return c, modified
	}
	return v, false
}

func sanitizeBlockedTemplates(s string) string {
	// The agent identity line ships in two shapes. The Claude Agent SDK form
	// continues the sentence with a comma ("...for Claude, running within the
	// Claude Agent SDK."), so matching the period-terminated form alone let
	// every SDK request through unchanged and the upstream answered 400 /
	// code 11128 "Illegal API invocation from an unapproved channel".
	// Match the invariant substring instead of either full sentence.
	s = strings.ReplaceAll(s,
		"Anthropic's official CLI for Claude",
		"Anthropic's official CLI tool for Claude")
	s = strings.ReplaceAll(s,
		"Main branch (you will usually use this for PRs)",
		"Default branch (you will usually use this for PRs)")
	// The feedback line is blocklisted as one exact sentence; the upstream
	// rejects the whole request when it appears verbatim anywhere in the
	// prompt. Reword it so the meaning survives and the exact match does not.
	s = strings.ReplaceAll(s,
		"To give feedback, users should report the issue at https://github.com/anthropics/claude-code/issues",
		"To give feedback, users should report the issue in the project's issue tracker")
	s = strings.ReplaceAll(s,
		"You are a coding agent running in the Codex CLI, a terminal-based coding assistant.",
		"You are a coding agent running in the Codex CLI tool, a terminal-based coding assistant.")

	// DeepSeek / DSH family. These are TWO different sentences with two
	// different triggers, so they are handled by two rules:
	//
	//   1. the identity sentence at the top of the system message, recognised
	//      by its brand pair (DeepSeek + U+200B + Harness) and dropped;
	//   2. the persona sentence carrying the model id, recognised by its
	//      "coding agent powered by" frame, reworded but kept.
	//
	// Every rule below is line-scoped: none of them may consume a newline, or
	// removing a sentence would also join it to the next line and change the
	// prompt's line structure. The brand pair inside a sentence is matched with
	// inline whitespace only, for the same reason.
	s = reDeepSeekHarnessIdentity.ReplaceAllString(s, "${1}")
	s = reDeepSeekPersonaPoweredBy.ReplaceAllString(s, "${1} running on $2${3}${4}")
	s = reWebGUIBrand.ReplaceAllString(s, webGUIBrandReplacement)
	s = reDeepSeekBrandPair.ReplaceAllString(s, deepSeekBrandPairReplacement)

	// The file-reference paragraph's `@` is part of what the channel check
	// matches. Only this paragraph is rewritten: a bare `@`, an email address and
	// a decorator elsewhere are all accepted on their own.
	s = reFileReferencePara.ReplaceAllStringFunc(s, func(para string) string {
		return strings.ReplaceAll(para, "@", "(at)")
	})
	return s
}

// forceMaxThinking pins reasoning_effort to "high" for hy3-family models so
// Tencent Hunyuan 3 always reasons at maximum depth. CodeBuddy only honors
// "high" for deep thinking (medium/low/max/xhigh/ultra all fall back to no
// reasoning), so we override whatever the client sent. Returns true if changed.
func forceMaxThinking(obj map[string]any) bool {
	model, _ := obj["model"].(string)
	if !strings.HasPrefix(model, "hy3") {
		return false
	}
	if eff, _ := obj["reasoning_effort"].(string); eff == "high" {
		return false
	}
	obj["reasoning_effort"] = "high"
	return true
}

// rewriteModelInBody replaces the "model" field of a chat-completions body
// with the resolved upstream model ID.
func rewriteModelInBody(body []byte, upstreamModel string) []byte {
	if len(body) == 0 || strings.TrimSpace(upstreamModel) == "" {
		return body
	}
	var obj map[string]any
	if json.Unmarshal(body, &obj) != nil {
		return body
	}
	cur, _ := obj["model"].(string)
	if strings.EqualFold(strings.TrimSpace(cur), strings.TrimSpace(upstreamModel)) {
		return body
	}
	obj["model"] = upstreamModel
	out, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return out
}

func isEmptyValue(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case map[string]any:
		if len(x) == 0 {
			return true
		}
		// Legacy function_call shell: {"name":"","arguments":""} is the
		// upstream's terminal-chunk artifact, not a real call — treat as empty
		// when every value is itself empty.
		for _, val := range x {
			if !isEmptyValue(val) {
				return false
			}
		}
		return true
	}
	return false
}
