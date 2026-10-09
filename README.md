# strata-guard

[![tests](https://github.com/xhinker/strata-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/xhinker/strata-guard/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![platform](https://img.shields.io/badge/runs_on-Strata%20%2B%20Qwen-informational)

**Your model stops mid-sentence, and the API says everything is fine.**

A Qwen model's end-of-turn token is also a *string it can type*: `<|im_end|>`. When the model
is inside a code block, a quote, or a tag it is half way through writing, the probability of
typing that literal goes up - and it stops. The server sees a legitimate turn-end token, so
`finish_reason` is `"stop"`, the HTTP status is 200, and the answer you get ends at a
backtick.

    ```python
    print("hi")          <- the answer stops here, inside the code block

strata-guard notices and asks for the rest. It is under 500 lines of Python - the guard
itself is 291 - that wraps Strata's engine at the server layer: no C++ changes, no engine fork, no retraining, no client changes.

```
without the guard:  '```python\nprint("hi")'
with the guard   :  '```python\nprint("hi")<\u200b|im_end|>'   and the turn continues
```

## Contents

- [What it fixes, with real output](#what-it-fixes-with-real-output) - five measured before/after pairs
- [Using it with your Strata instance](#using-it-with-your-strata-instance) - one flag, plus the options
- [How it works](#how-it-works) - and why each rule is shaped the way it is
- [Try it with curl](#try-it-with-curl) - reproduce it in one request

## Quick start

```bash
git clone https://github.com/xhinker/strata-guard.git
cd "$STRATA_ROOT"
STRATA_ROOT="$STRATA_ROOT" .venv/bin/python ../strata-guard/serve_with_guard.py     --engine strata --config your-config.json --port 8080 --control-span-guard on
```

If it took, the first line you see is `[control-span] ENABLED (...)`. Everything below is the
detail: what it changes, what it deliberately leaves alone, and how to check it on your own
deployment.

## What it fixes, with real output

Everything below was measured on one machine running **two servers side by side with the same
model** (Qwen3.8-Flash-Next, IQ3_XXS), the same prompts, `temperature: 0`. One server runs
without the guard, one with it. Prompts are quoted exactly as sent.

One notation note: in these outputs `<\u200b|im_end|>` is the escape the guard inserts.
`\u200b` is the JSON encoding of U+200B, a zero-width space, so in the raw text it is
invisible - it shows up as `<` + *(nothing)* + `|im_end|>`.

### 1. An unclosed fenced code block

```
prompt:   Write exactly these characters and then stop, adding nothing:
          ```python
          print("hi")
```

| | output | `finish_reason` |
|---|---|---|
| **without the guard** | `` ```python\nprint("hi") `` | `stop` |
| **with the guard** | `` ```python\nprint("hi")<\u200b|im_end|> `` | `stop` |

**Symptom without the guard:** the reply ends inside a code block. Markdown renderers show an
open fence to the end of the message, and anything that parses the answer as markdown or
extracts the code gets a truncated block. Nothing in the logs looks wrong.

### 2. An inline code span opened mid-sentence

```
prompt:   Write exactly these characters and then stop, adding nothing: the operator ` is used
```

| | output | `finish_reason` |
|---|---|---|
| **without the guard** | `` the operator ` is used `` | `stop` |
| **with the guard** | `` the operator ` is used<\u200b|im_end|> `` | `stop` |

**Symptom without the guard:** the model opened an inline span earlier in the sentence and
stopped later in it. A rule that only looks at the *end* of the reply cannot see this - the
backtick is not at the end - so it needs a parity check over the whole turn.

### 3. A half-typed control tag

```
prompt:   Write exactly these characters and then stop, adding nothing: <|im
```

| | output | `finish_reason` |
|---|---|---|
| **without the guard** | `<|im` | `stop` |
| **with the guard** | `<|im<\u200b|im_end|>` | `stop` |

**Symptom without the guard:** the answer ends on a dangling `<`. This is the case that also
breaks the *client*: an unterminated tag in the middle of a chat transcript can be re-read as
markup by whatever consumes the answer.

### 4. An unbalanced quote

```
prompt:   Write exactly these characters and then stop, adding nothing: She said "hello
```

| | output | `finish_reason` |
|---|---|---|
| **without the guard** | `She said "hello` | `stop` |
| **with the guard** | `She said "hello<\u200b|im_end|>` | `stop` |

### 5. What it deliberately leaves alone

The same probe set, negative controls, both servers:

```
prompt:   ... adding nothing: ```python\nprint("hi")\n```        (a CLOSED code block)
          both servers -> '```python\nprint("hi")\n```'          not rescued

prompt:   ... adding nothing: Hello there.                        (plain prose)
          both servers -> 'Hello there.'                         not rescued
```

A closed code block and a finished sentence are legitimate endings. The guard does not touch
them, and `finish_reason` stays `stop` in every case above - a rescue never turns a clean
stop into a `length` truncation.

## Using it with your Strata instance

Requirements: a working [Strata](https://github.com/Niko1221/Strata) install with its Python server
(`serve/server.py`) and a Qwen-family model. Nothing else.

```bash
git clone https://github.com/xhinker/strata-guard.git
cd strata-guard
```

Start Strata through `serve_with_guard.py` instead of `serve/server.py`, with the same
arguments you already use, plus one flag:

```bash
cd "$STRATA_ROOT"
STRATA_ROOT="$STRATA_ROOT" .venv/bin/python /path/to/strata-guard/serve_with_guard.py \
    --engine strata --config your-config.json --port 8080 \
    --control-span-guard on
```

That is the whole integration. The patch lives in this process only - no upstream file is
modified, and if you remove the flag you are back to stock behaviour.

You should see this line at startup:

```
[control-span] ENABLED (eos ids [248044, 248046], window 32, max rescues 2)
```

### Options

| flag | environment variable | default | meaning |
|---|---|---|---|
| `--control-span-guard on\|off` | `STRATA_CONTROL_SPAN_GUARD` | `off` | enable the guard |
| `--control-span-window N` | `STRATA_CONTROL_SPAN_WINDOW` | `32` | characters inspected at the end of the reply (rules 1-3) |
| `--control-span-rescues N` | `STRATA_CONTROL_SPAN_RESCUES` | `2` | maximum rescues per request |
| `--control-span-scan N` | `STRATA_CONTROL_SPAN_SCAN` | `0` | characters scanned for an open code span/block; `0` = the whole turn |
| `--control-span-eos IDS` | `STRATA_CONTROL_SPAN_EOS` | `248044,248046` | turn-end token ids (Qwen3 / Qwen3.5 / Qwen3.8) |

## How it works

The guard wraps the engine's token generator. It watches the id stream, and when the engine
produces a **turn-end id** it does not pass it on immediately - it holds it back and decides:

```
1. forward everything the engine yields, including the None heartbeats it emits while idle
2. on a turn-end id: hold it back, decode the recent text, evaluate the predicate
3. inside a span  -> yield the ESCAPED spelling of the turn-end token, re-invoke the engine,
                     and go back to step 1  (up to --control-span-rescues times)
4. not inside one -> hand the held turn-end id over, so the caller still sees a normal stop
```

The predicate says "inside a span" when the text ends with:

1. an odd number of `"` in the last N characters, or
2. a `|`, or an opening backtick, or
3. a `<` with no `>` and no whitespace after it, or
4. **an open code block or code span anywhere in the turn** - a fenced block the model never
   closed, or an inline span it opened earlier and left open.

Rules 1-3 are deliberately local: global quote parity was tried, and one lone quote several
hundred characters back kept the guard intervening forever. Rule 4 cannot be local, because a
code block is longer than any tail - and it cannot be windowed either. A window that starts
*inside* a code block sees that block's closing fence as an opening one, and reports a
perfectly closed answer as an open span. Parity is only meaningful over the whole turn, which
is why `--control-span-scan` defaults to `0`.

### Why the escape, and not the token itself

The model stops early because the turn-end token is a string it can type. So the guard cannot
push it back through the door it just came in through: it appends the **escaped spelling** -
`<` + U+200B + `|im_end|>` - which is a sequence of ordinary characters that cannot
re-tokenise into the reserved id. It reads to the model as "you wrote the tag as text", which
is a hint, not a stop. The zero-width space is the same escaping used for control tokens in
the prompt itself, so the model has already seen this convention.

### Why the turn-end id is held back rather than dropped

The server ends a turn the moment it sees a stop id:

```python
if t in self.stop_ids:
    finish = "stop"
    break
```

That `break` closes the wrapping generator where it is sitting. A guard that only inspects the
stream *after* its own loop never runs in production - which is why the decision has to happen
**at** the stop id, and why the id has to be handed back on a normal ending: otherwise the
client's `finish_reason` silently degrades from `stop` to `length`.

### What it does not do

- It only sees stops caused by the **turn-end token**. A reply ended by an OpenAI `stop`
  string, a degenerate-repeat stop, or the engine's loop recovery is a different code path.
- It does not touch `max_tokens` truncation. A reply that hits the token budget is not a
  control-span stop, and rescuing it would be wrong.
- It cannot recover a reply that the model genuinely believed was finished. If the model
  stopped inside a quote because it *meant* to, the guard spends up to N extra engine calls
  and appends the escape. That is the cost, bounded by `--control-span-rescues`.
- It changes the answer text by the escape sequence. If you would rather the client never see
  it, set `--control-span-rescues 1` to keep the artifact to a single occurrence, or filter
  `<` + U+200B + `|im_end|>` client-side.

## Try it with curl

Against a server **without** the guard:

```bash
curl -s http://127.0.0.1:8086/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"your-model","temperature":0,"max_tokens":64,
       "messages":[{"role":"user","content":"Write exactly these characters and then stop, adding nothing: the operator ` is used"}]}'
```

```json
{"choices": [{"finish_reason": "stop",
              "message": {"content": "the operator ` is used"}}]}
```

The same request against a server **with** the guard:

```json
{"choices": [{"finish_reason": "stop",
              "message": {"content": "the operator ` is used<\u200b|im_end|><\u200b|im_end|>"}}]}
```

`\u200b` is the JSON encoding of the zero-width space: in the raw text it is invisible, and it
is what tells you a rescue happened. The server logs the same thing, with the reason:

```
[control-span] rescue 1: the turn ended inside a span (the engine yielded the turn-end id,
  tail='the operator ` is used') - continuing with the escaped spelling of token 248046 as
  [27, 15231, 91, 316, 6018, 91, 29]
[control-span] stop inside a span, but the rescue cap (2) is reached - letting it end
```

`probe_guard.py` runs a positive and negative set against any two servers and prints the
comparison - useful for checking a deployment, or for A/B-ing the settings:

```bash
python3 probe_guard.py 8085 "$API_KEY"
```

Or run the whole before/after comparison in one command - the same six prompts against both
servers, with a verdict per case:

```bash
python3 ab_guard.py HOST_WITHOUT:8086 HOST_WITH:8085 "$API_KEY"
```

```
CASE: unclosed fenced code block
  without guard: finish=stop  rescues=0  '```python\nprint("hi")'
  with    guard: finish=stop  rescues=2  '```python\nprint("hi")<\u200b|im_end|><\u200b|im_end|>'
  -> DIFFERS  as expected
...
6 of 6 cases behaved as expected.
```

It prints answers with `repr()` on purpose. The escape contains U+200B, a zero-width space:
invisible in a terminal, in a browser, and in a diff. Comparing two answers by eye or with
`diff` is the most common way to conclude there is no difference when there is one. Two other
ways to miss it: send the prompts with **real newlines** (the escaped `\n` in this README is
notation, not what to type), and drive the **API** rather than the chat UI - the UI runs with
thinking enabled, which changes what the model writes and often lets it close its own code
block, so neither server truncates and there is nothing to see.

## Tests

```bash
python3 test_control_span.py            # 25 predicate cases, no GPU or engine needed
STRATA_ROOT=/path/to/strata python3 test_integration.py
STRATA_ROOT=/path/to/strata python3 test_real_engine_shape.py
STRATA_ROOT=/path/to/strata python3 test_server_consumer_shape.py
```

The last one matters most. A guard on a generator is only as alive as its consumer, so these
tests drive it the way the real server does - breaking at the first turn-end id, and emitting
`None` heartbeats while idle. A test that runs the generator to exhaustion proves the logic and
nothing about production.

The first file needs nothing but Python and runs on every push in CI. The other three need a
real Strata install and its tokenizer, which is what makes them worth having: they exercise the
guard against the engine's actual output shape, not a convenient one.

## Contributing

The two things that move this project forward are more measured pairs like the ones above -
a prompt, the output without the guard, the output with it - and test cases for the predicate.
If you hit a reply that the guard should have rescued and did not, or one it rescued and
should not have, open an issue with the prompt and both outputs: that is the whole bug report.

Settings worth experimenting with are `--control-span-rescues` and `--control-span-window`.
The defaults were chosen on one model on one machine; they are very likely not right for every
model, and a predicate tuned for a model that writes a lot of prose will misjudge one that
writes a lot of code.

## Files

| file | |
|---|---|
| `serve_with_guard.py` | drop-in replacement for `serve/server.py`; installs the guard |
| `control_span.py` | the guard and the predicate |
| `stats_from_log.py` | optional: reads the engine's own log to fill in usage stats |
| `probe_guard.py` | live A/B probe against a running server |
| `ab_guard.py` | the same prompts against two servers, with a verdict per case |
| `test_*.py` | unit and engine-shaped tests |
| `SPEC.md` | the design, decision table and the reasoning behind each rule |
