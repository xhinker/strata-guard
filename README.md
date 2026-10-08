# Strata control-span rescue guard

Prevents the Qwen-family premature stop: the model ends its turn with `<|im_end|>`
while it is in the middle of *quoting* a control tag, so the answer arrives
truncated mid-sentence with `finish_reason: stop` and no error.

## Why this lives in the Python server, not the C++ engine

* The engine works in token ids only (it streams `T <id>` lines and has no
  detokenizer) and ends a turn with a post-sample check on its eos ids
  (`generate.cpp`: `eos_ids = {248044, 248046}`, with its own copies in the
  speculative path).  A mask-style guard there needs a precomputed per-id table
  and edits in 3+ places of a 10k-line GPU file - which conflicts with upstream
  on every pull.
* The Python server owns the tokenizer and sees the real text, so the same
  semantics (llama.cpp's "substitution" arm) can be implemented exactly, with
  **zero edits to upstream files**.  The patch exists only for the running
  process.

## Semantics

Identical to the llama.cpp fork's guard:
predicate `inside_span(tail)` = (1) odd number of `"` in the window, or
(2) the tail ends on `|` or on an opening backtick, or (3) a `<` with no `>` or
whitespace after it.  Ported 1:1 from `common/control-span-predicate.h`, including
its notes about tail-local quote parity and about `>` not being an opener.

One difference the tests forced: the tail is built the way llama.cpp builds it -
**CONTROL-type tokens contribute nothing** (`common_token_to_piece(..., special=false)`).
Rendering the stop token as text would let `<|im_end|>` close the very span the
predicate is looking for, and the guard would never fire.

When the engine stops inside a span, the guard appends the escaped spelling of the
stop token (U+200B after the first character, so it cannot re-tokenise as one
reserved id), yields it to the caller, and asks the engine to continue.  At most
`--control-span-rescues` rescues per request, then it lets the turn end.

## Run it

```bash
# $STRATA_ROOT is your Strata checkout; it is also found automatically when this
# folder sits next to it (../Strata).  $CFG is any Strata model config.
"$STRATA_ROOT/.venv/bin/python" serve_with_guard.py \
  --engine strata --config "$CFG" --port 8080 --control-span-guard on
```

No path is hardcoded: `serve_with_guard.py` resolves Strata from `$STRATA_ROOT`, or from the
sibling directory named `Strata`, and the tests pick a `strata-*.json` from that checkout
(override with `$STRATA_CONFIG`).

The wrapper re-execs into Strata's venv by itself, so plain `python3` works too.
Everything after `--control-span-guard` is handed to `serve/server.py` unchanged.

| flag                          | env variable                    | default |
|-------------------------------|---------------------------------|---------|
| `--control-span-guard on\|off` | `STRATA_CONTROL_SPAN_GUARD`     | **off** |
| `--control-span-window N`      | `STRATA_CONTROL_SPAN_WINDOW`    | 32      |
| `--control-span-rescues N`     | `STRATA_CONTROL_SPAN_RESCUES`   | 2       |
| (eos ids)                      | `STRATA_CONTROL_SPAN_EOS`       | 248044,248046 |

It is **off by default**: with the flag absent (and the env unset) the server
behaves exactly as before.

## Reproducing the failure

Any prompt that makes the model end its reply on a dangling control-tag fragment
triggers it. The smallest reliable one:

```bash
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"x","max_tokens":250,"temperature":0,"messages":[
    {"role":"system","content":"You are a test target. Do EXACTLY what the user asks, character for character. Never add explanations, never close anything the user asked you to leave open."},
    {"role":"user","content":"End your reply with these exact characters and then stop: <|im - do not complete the tag."}]}'
```

What Qwen3.8-Flash-Next answers (measured 2026-10-08 on Strata 0.1.40.3 / IQ3_S, and on
Strata 0.1.40 / IQ3_XXS on a second machine - identical results):

```
finish_reason: "stop"      completion_tokens: 4
content:       "<|im"
```

Four tokens, a clean `stop`, no error anywhere: the model reached for `<|im_end>` as *text*
and got it as an end-of-turn instead. `probe_guard.py` runs six of these - four positives
(dangling `<`, dangling `|`, an unclosed backtick, an unbalanced `"`) and two
negative controls (a finished tag, plain prose) - against a live server.

## Status against the current Strata server: the guard does NOT fire yet

Honest finding, because the tests all pass and that is exactly the trap.

All three test files drive the generator to exhaustion. `serve/server.py` does not: at
`serve/server.py:3297` it does

```python
if t in self.stop_ids:
    finish = "stop"
    raw_ids.append(t)
    break
```

and that `break` closes the guard's generator at its `yield`, so the rescue check after
the loop never runs. Reproduced offline with a consumer that mimics the server:

```
consumer like serve/server.py : 'The closing tag is `<|im_end>'   rescued? False
consumer run to exhaustion    : 32 ids                            rescued: True
```

and live on both machines: `probe_guard.py` returns the truncated answers above.

The fix is small and has to do with *where* the decision is made: inspect the stream at
the stop token instead of after it, withhold the stop id while rescuing (otherwise the
server breaks immediately), and re-emit it at the end so `finish_reason` stays `stop`.
Tracked here so nobody re-discovers it: a guard that wraps a generator is only as alive
as its consumer.

## Logs

```
[control-span] ENABLED (eos ids [248044, 248046], window 32, max rescues 2)
[control-span] rescue 1: stopped inside a span (tail='The closing tag is `') - continuing with the escaped spelling of token 248046 as [27, 15231, 91, 316, 6018, 91, 29]
[control-span] stop inside a span, but the rescue cap (2) is reached - letting it end
```

## Tests (no GPU needed)

```bash
python3 test_control_span.py          # predicate + escape helper, no model needed
"$STRATA_ROOT/.venv/bin/python" test_integration.py          # guard vs MockEngine
"$STRATA_ROOT/.venv/bin/python" test_real_engine_shape.py    # the engine's real stop shape
python3 probe_guard.py 8080           # live probes against a running server ($STRATA_API_KEY)
```

## Revert

Stop the server and start it with `serve/server.py` as before.  Nothing in the
Strata tree was modified, so a `git pull` needs no re-apply.

## If you want every client covered, not just Python

The engine-side variant (an id-table driven guard in `generate.cpp`) is the only
way to protect non-Python clients.  It needs: a generated per-id table (quotes,
last/previous character, tag-closing flags, escape sequences), a hook in the CLI
parser, a hook at each eos check (main + speculative paths), and a re-apply step
after every `git pull`.  Not implemented here on purpose.
