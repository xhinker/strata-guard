"""Regression: the guard must work against the REAL consumer, not a cooperative one.

serve/server.py:3297 ends a turn like this:

    if t in self.stop_ids:
        finish = "stop"
        raw_ids.append(t)
        break

The consumer stops pulling at the first turn-end id.  A guard that inspects the stream
*after* its own loop never gets that far - which is exactly how the guard shipped while
doing nothing in production, with every other test file green (they all run the generator
to exhaustion).  This file pins the production shape: the engine yields the turn-end id and
the consumer breaks at it.

    "$STRATA_ROOT/.venv/bin/python" test_server_consumer_shape.py
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
TURN_ID = 248046                     # the turn-end id the engine emits in this shape


class YieldingEosEngine:
    """The current shape: streams tokens, then YIELDS the turn-end id, then returns."""

    def __init__(self, text):
        self.ids = tok.encode(text, parse_special=False)
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for t in self.ids[:max_new]:
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


def run(text, max_new=64, max_rescues=2):
    eng = YieldingEosEngine(text)
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=max_rescues, log=sys.stderr)
    seen, ended = consume_like_server(g.guard(eng.generate, tok.encode("hi", parse_special=False),
                                             max_new, {}, threading.Event()))
    return seen, ended, eng


class HeartbeatEngine(YieldingEosEngine):
    """The engine also yields None as a heartbeat while it is quiet (serve/server.py does
    `if t is None:  ... continue`).  A guard that records those as generated tokens dies with
    TypeError inside tail_text - which is what happened on the live server on 2026-10-08."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for n, t in enumerate(self.ids[:max_new]):
            if n == 1:
                yield None
            yield t
        yield None
        yield TURN_ID


def run_heartbeat(text, max_new=64):
    eng = HeartbeatEngine(text)
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=2, log=sys.stderr)
    seen, ended = consume_like_server(g.guard(eng.generate, tok.encode("hi", parse_special=False),
                                             max_new, {}, threading.Event()))
    return seen, ended, eng


fail = 0
def check(label, cond):
    global fail
    fail += 0 if cond else 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


print("engine yields the turn-end id, consumer breaks at it (the production shape):")
seen, ended, eng = run("The closing tag is `")
out = tok.decode(seen)
check(f"it was rescued against a breaking consumer ({len(seen)} ids, {eng.calls} engine calls)",
      "\u200b" in out and eng.calls >= 2)
check("the caller still receives a turn-end id, so finish_reason stays stop", ended)
check("the escape is the escaped spelling (U+200B), never a bare turn-end id mid-answer",
      out.count("\u200b") >= 1 and seen.count(TURN_ID) == 1)

print("\nplain ending, same consumer (must NOT rescue):")
pseen, pended, peng = run("The answer is 4.")
check(f"not rescued ({len(pseen)} ids, 1 engine call)",
      "\u200b" not in tok.decode(pseen) and peng.calls == 1 and pended)

print("\nrescue cap, same consumer:")
cseen, cended, ceng = run("The closing tag is `", max_new=64, max_rescues=1)
check(f"at most 1 rescue, then the turn is allowed to end ({ceng.calls} engine calls)",
      ceng.calls == 2 and cended and tok.decode(cseen).count("\u200b") == 1)

print("\nengine emits None heartbeats mid-stream (must not crash, must still rescue):")
try:
    hseen, hended, heng = run_heartbeat("The closing tag is `")
    hok = True
except TypeError as exc:
    print("  TypeError:", exc)
    hseen, hended, heng, hok = [], False, None, False
check("no TypeError from a heartbeat id (the live crash of 2026-10-08)", hok)
# heartbeats ARE forwarded (the server handles them: `if t is None:  yield "ping"`); what must
# never happen is the guard recording them as generated ids - that is what crashed tail_text.
check("heartbeats are forwarded, never recorded as ids; the rescue still happened",
      hok and None in hseen and "\u200b" in tok.decode([t for t in hseen if t is not None])
      and hended and heng.calls >= 2)

print(f"\n  8 checks, {fail} failures")
raise SystemExit(1 if fail else 0)
