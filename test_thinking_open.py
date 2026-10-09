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
def check(label, cond):
    global fail
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

print("\nrescue cap still enforced (a model that never closes thinking, cap 5):")
seen4, ended4, eng4 = run([THINK_OPEN] + tok.encode("x", parse_special=False), max_rescues=5)
check(f"5 rescues, then the turn is allowed to end ({eng4.calls} engine calls)",
      eng4.calls == 6 and ended4)

print()
print(f"7 checks, {fail} failures")
raise SystemExit(1 if fail else 0)
