#!/usr/bin/env python3
"""demo_rescue.py - the mechanism, exactly, without asking a model to cooperate.

An engine writes a code block and then yields the turn-end id in the middle of it.
Left alone the client receives an unterminated block; with the guard it receives the
block plus the escaped tag, and the engine's next call finishes what it started.

    python3 demo_rescue.py
"""
from __future__ import annotations
import json, os, sys, threading
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(STRATA / "tools")); sys.path.insert(0, ".")
import control_span as CS      # noqa: E402
import strata_tokenizer as ST  # noqa: E402

F3 = chr(96) * 3
TURN = 248046
FIRST = F3 + 'python\nprint("hi")\n'    # the block the model is writing
SECOND = F3 + "\n"                       # what it still had to write: the closing fence


def pack() -> Path:
    import glob
    named = os.environ.get("STRATA_CONFIG")
    cand = Path(named) if named else None
    if cand is None or not cand.exists():
        hits = sorted(glob.glob(str(STRATA / "strata-*.json")))
        if not hits:
            raise SystemExit("no strata-*.json - set STRATA_CONFIG")
        cand = Path(hits[0])
    a = json.loads(cand.read_text())["args"]
    return Path(a[a.index("--pack") + 1]) / "tokenizer"


P = pack()
vocab = json.loads((P / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
types = json.loads((P / "token_type.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(tokens, (P / "merges.txt").read_text(encoding="utf-8").split("\n"), types)


class Engine:
    """Call 1: the block, then the turn-end id.  Call 2 (only reached after a rescue): the fence."""
    def __init__(self):
        self.calls = 0
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        text = FIRST if self.calls == 1 else SECOND
        for t in tok.encode(text, parse_special=False):
            yield t
        yield TURN


def client(gen):
    out, ended = [], False
    for t in gen:                        # serve/server.py's loop: break at the first turn-end id
        out.append(t)
        if t in CS.DEFAULT_EOS:
            ended = True
            break
    return out, ended


base = tok.encode("hi", parse_special=False)

eng = Engine(); raw, _ = client(eng.generate(base, 64, {}, threading.Event()))
eng2 = Engine(); g = CS.Guard(tok, CS.DEFAULT_EOS, types=types, log=sys.stderr)
guarded, ended = client(g.guard(eng2.generate, base, 64, {}, threading.Event()))

print("\nwithout the guard:")
print("  client content:", repr(tok.decode([t for t in raw if t not in CS.DEFAULT_EOS])))
print("  engine calls   :", eng.calls)
print("\nwith the guard:")
print("  client content:", repr(tok.decode([t for t in guarded if t not in CS.DEFAULT_EOS])))
print("  engine calls   :", eng2.calls, "| ended on a turn-end id:", ended)
print("\nthe difference is the closing fence the block was missing, plus one invisible escape")
