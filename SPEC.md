# Strata control-span rescue guard — specification

## 1. Problem

The Qwen chat format makes the end-of-turn marker a *single vocabulary id* whose
characters the model must also **print** when it writes about its own format.  While a
generation is inside an open quote, an open code span, or a half-typed tag, the model
often samples that id instead of the characters it meant to write.  The engine treats it
as "turn over", the caller receives a sentence cut mid-word, and the API reports
`finish_reason: stop` — a truncation with no error.

## 2. What the guard does

It watches the *end* of a generation.  If the turn ended while the tail looks like an open
span, it refuses to let the turn die: it appends the **escaped spelling** of the turn-end
token (U+200B inserted so the text can never re-tokenise as the reserved id), hands those
ids to the caller, and asks the engine to keep going from there.

It lives entirely in the Python server layer (`Strata-guard/`), patched in at runtime.
No upstream file is modified, so a `git pull` needs no re-apply.

## 3. Decision table

| step | condition | result |
|---|---|---|
| 1 | a substitution is in progress | force its next id (abandon after 32 steps) |
| 2 | stream ended **short of the cap** and not cancelled, **or** last id is an eos | candidate stop |
| 3 | otherwise | return — the turn ends normally |
| 4 | candidate stop **and** `inside_span(tail)` | rescue |
| 5 | rescues so far >= `max_rescues` | log the cap and let it end |

## 4. The predicate (`inside_span`)

| rule | condition | why |
|---|---|---|
| 1 | odd number of `"` in the window | an unterminated quoted string |
| 2 | tail ends on `\|`, or on a backtick whose previous character is one of `space tab newline ( [ : , - = ;` | an opening code span. `>` is deliberately NOT an opener, so a finished `` `<tag>` `` is a legitimate ending |
| 3 | the last `<` has no `>` and no whitespace after it | mid-tag |

The window is the last 32 **characters** of the generated text (the llama.cpp original
trims 32 *bytes* — a difference worth knowing in CJK output).

## 5. The tail — one load-bearing detail

The tail is built from the streamed ids, **excluding CONTROL-type tokens**
(`token_type.json` type 3), mirroring llama.cpp's
`common_token_to_piece(..., special=false)`.  Rendering the stop token as text instead
would let `<|im_end|>` *close* the very span the predicate is looking for, and the guard
would never fire.  The first integration test failed for exactly this reason.

## 6. Interface

| flag | env | default |
|---|---|---|
| `--control-span-guard on\|off` | `STRATA_CONTROL_SPAN_GUARD` | **off** |
| `--control-span-window N` | `STRATA_CONTROL_SPAN_WINDOW` | 32 |
| `--control-span-rescues N` | `STRATA_CONTROL_SPAN_RESCUES` | 2 |
| — | `STRATA_CONTROL_SPAN_EOS` | `248044,248046` |

Run-script form: `CONTROL_SPAN_GUARD=on ./run-3090x3-iq3s.sh <CTX> <EC> <GPUS> <PORT> <VISION>`

The turn-end token escaped on a rescue: the last id in the configured eos list (248046,
`<|im_end|>`), because the resident engine never yields the eos itself.

## 7. Log lines

```
[control-span] ENABLED (eos ids [248044, 248046], window 32, max rescues 2)
[control-span] rescue 1: the turn ended inside a span (short of the cap, so a turn-end token, tail='The closing tag is `') - continuing with the escaped spelling of token 248046 as [...]
[control-span] stop inside a span, but the rescue cap (2) is reached - letting it end
```

## 8. Scope

| covered | not covered |
|---|---|
| a turn that ends inside an open span (the premature stop) | the reasoning-splitter leak: a tag spelled *inside thinking* closing the block early — a separate, parser-side defect |
| any Qwen-family pack whose eos ids are configured | a corrupted forward pass (e.g. draft-MTP at n_max >= 4 putting ~92% on im_end) |
| clients using the Python server | clients that talk to the engine directly (that needs the engine-side id-table variant) |
| | max_tokens truncation — by design: a stream that hits the cap is not treated as an early stop |

## 9. Cost

One extra engine pass per rescue, at most `max_rescues` per request.  The rescue only
happens at a turn end, so there is no per-token overhead.  A false positive costs at most
two bounded continuations, and it is logged.

## 10. Tests

| test | what it pins |
|---|---|
| `test_control_span.py` | 14 predicate/escape cases |
| `test_integration.py` | guard vs Strata's own Service + MockEngine |
| `test_real_engine_shape.py` | the production shape: an engine that returns on DONE **without** yielding the eos (the bug that made the first version unable to fire), the escape target, the no-rescue controls |
| `probe_guard.py` | live A/B prompts against a running instance |
