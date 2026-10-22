"""Regression: a stop while the thinking block is still open is a truncation, not an end of turn.

2026-10-09, profile Qwen3.8-Flash-Next-strata-3090-63 on the .63 server: three premature
thinking stops with the guard enabled.  The span predicate (quotes / code spans / mid-tag)
cannot see a model that samples the turn-end id in PLAIN PROSE inside the thinking block -
no open quote, no code span, no half-typed tag.  The fix tracks the think tags themselves:
the open tag is 248068 and the close tag is 248069 (both USER_DEFINED, so they appear in
the stream as real ids).  If the most recent of the two in the generated ids is the OPEN
tag, the turn is still inside thinking and any stop is rescued.  Id-based, so a model
QUOTING the tag (character pieces) never triggers it.

    "$STRATA_ROOT/.venv/bin/python" test_thinking_open.py
"""
import json, os, sys, threading
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_span as CS                                  # noqa: E402
import strata_tokenizer as ST                              # noqa: E402


def _pack_dir():
    import glob
    named = os.environ.get("STRATA_CONFIG")
    cand = (STRATA / named) if named else None
    if cand is None or not cand.exists():
        hits = sorted(glob.glob(str(STRATA / "strata-*.json")))
        if not hits:
            raise SystemExit("no strata-*.json found in %s - set STRATA_ROOT / STRATA_CONFIG" % STRATA)
        cand = Path(hits[0])
    args = json.loads(cand.read_text())["args"]
    return Path(args[args.index("--pack") + 1]) / "tokenizer"


PACK = _pack_dir()
vocab = json.loads((PACK / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
TYPES = json.loads((PACK / "token_type.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(tokens, (PACK / "merges.txt").read_text(encoding="utf-8").split("\n"), TYPES)
TURN_ID = 248046
THINK_OPEN, THINK_CLOSE = 248068, 248069


class CursorEngine:
    """Streams a fixed id list from a cursor (a real engine continues where it stopped),
    then yields the turn-end id - the production shape (the consumer breaks at it)."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.cursor = 0
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for t in self.ids[self.cursor:self.cursor + max_new]:
            self.cursor += 1
            yield t
        yield TURN_ID


class GrowingEngine:
    """Every call writes one more token, then the turn-end id: there is always new output,
    so every rescue is legitimate and only the cap can end it (the one-shot rule does not
    apply - see test_one_shot_rescue.py for the nothing-new case)."""

    def __init__(self, first):
        self.first = list(first)
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        seq = self.first if self.calls == 1 else tok.encode("y" * self.calls, parse_special=False)
        for t in seq[:max_new]:
            yield t
        yield TURN_ID


class TwoPhaseEngine:
    """Call 1: stream the thinking then the stop id (a mid-thinking stop).  Call 2 (after the
    guard's rescue): stream the close tag + the answer, then a genuine stop - the model
    continues where the guard left it and finishes the turn."""

    def __init__(self, thinking, answer):
        self.thinking, self.answer, self.calls = thinking, answer, 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        if self.calls == 1:
            for t in self.thinking:
                yield t
            yield TURN_ID
        else:
            for t in [THINK_CLOSE] + self.answer:
                yield t
            yield TURN_ID


def consume_like_server(gen):
    """serve/server.py's loop: take tokens, and BREAK at the first turn-end id."""
    seen, ended_on_turn_id = [], False
    for t in gen:
        seen.append(t)
        if t in CS.DEFAULT_EOS:
            ended_on_turn_id = True
            break
    return seen, ended_on_turn_id


def run(ids, max_new=512, max_rescues=5):
    eng = CursorEngine(ids)
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=max_rescues, log=sys.stderr)
    seen, ended = consume_like_server(
        g.guard(eng.generate, tok.encode("hi", parse_special=False), max_new, {}, threading.Event()))
    return seen, ended, eng


fail = 0
total = 0
def check(label, cond):
    global fail, total
    total += 1
    fail += 0 if cond else 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


prose = "Let me check the log file to see what happened last night."

print("stop in plain prose inside an open thinking block (the 2026-10-09 case):")
seen, ended, eng = run([THINK_OPEN] + tok.encode(prose, parse_special=False))
out = tok.decode(seen)
check(f"rescued, not truncated ({len(seen)} ids, {eng.calls} engine calls)",
      "\u200b" in out and eng.calls >= 2)
check("the caller still receives a turn-end id, so finish_reason stays stop", ended)

print("\nstop AFTER the thinking block is closed (must NOT rescue):")
closed = ([THINK_OPEN] + tok.encode(prose, parse_special=False) + [THINK_CLOSE]
          + tok.encode(" The answer is 4.", parse_special=False))
seen2, ended2, eng2 = run(closed)
out2 = tok.decode(seen2)
check(f"not rescued ({len(seen2)} ids, {eng2.calls} engine call)",
      "\u200b" not in out2 and eng2.calls == 1 and ended2)

print("\nthree quoted tags inside one thinking block (request 4 from 2026-10-09):")
# the model quotes the tags as single reserved ids (that is the bug: copying the tag
# samples the stop id).  248045 (im_start) is not a stop id, so only 248044/248046 stop.
ids3 = ([THINK_OPEN]
        + tok.encode("the ids are `", parse_special=False) + [248044]
        + tok.encode("`, `", parse_special=False) + [248045]
        + tok.encode("`, and `", parse_special=False) + [248046]
        + tok.encode("` - endoftext, im_start, im_end.", parse_special=False)
        + [THINK_CLOSE] + tok.encode(" Table done.", parse_special=False))
seen3, ended3, eng3 = run(ids3)
out3 = tok.decode(seen3)
check(f"both stop-quotes rescued, turn ends cleanly ({eng3.calls} engine calls)",
      eng3.calls == 3 and ended3)
check("two escapes in the answer, no bare turn-end id leaked mid-answer",
      out3.count("\u200b") == 2 and seen3.count(TURN_ID) == 1)

print("\nrescue cap still enforced (a model that keeps writing new output, cap 5):")
eng4 = GrowingEngine([THINK_OPEN] + tok.encode("x", parse_special=False))
g4 = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=5, log=sys.stderr)
seen4, ended4 = consume_like_server(
    g4.guard(eng4.generate, tok.encode("hi", parse_special=False), 256, {}, threading.Event()))
check(f"5 rescues, then the turn is allowed to end ({eng4.calls} engine calls)",
      eng4.calls == 6 and ended4)

print()
print("think-open tag in the PROMPT, not the generated ids (the real 2026-10-09 case):")
# In production the chat template injects the think-OPEN tag (248068) into the PROMPT - the
# model does not generate it. The old guard scanned only the generated ids, so it was blind to
# the prompt's open tag and let the mid-thinking stop through as a "genuine end" -> content:
# None.  Now it scans prompt + generated, so it sees the open tag and rescues.
eng5 = TwoPhaseEngine(tok.encode(prose, parse_special=False),
                      tok.encode("The answer is 4.", parse_special=False))
g5 = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=5, log=sys.stderr)
seen5, ended5 = consume_like_server(
    g5.guard(eng5.generate, [THINK_OPEN] + tok.encode("q", parse_special=False), 512, {}, threading.Event()))
out5 = tok.decode(seen5)
check(f"rescued even though the open tag is in the prompt ({eng5.calls} engine calls)",
      eng5.calls == 2 and ended5)
check("the turn completed with the answer, not truncated", "The answer is 4." in out5)

print()
print(f"{total} checks, {fail} failures")
raise SystemExit(1 if fail else 0)
