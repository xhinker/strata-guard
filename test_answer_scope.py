"""Regression: the span check reads the ANSWER, never the reasoning block, and never a
64-id slice of it.

2026-10-10, profile Qwen3.8-Flash-Next-strata-3090-63 on the .63 server: an answer ended
normally, four rescues fired, and the answer printed four escaped turn-end tokens.  The
cause was the input to the predicate, not the predicate: guard() fed inside_span()
`tail_text(produced)`, which caps at the LAST 64 ids.  That window began inside a fenced
block the answer had already closed, so the block's CLOSING fence read as an OPENING one
and a complete answer looked truncated - the exact artifact inside_span's rule 4 docstring
says a window creates.  For that answer: keep=64 -> True, keep=128/256/full -> False.
The reasoning block is a second hazard: it is prose ABOUT code (inline backticks, nested
quoting), and its parity says nothing about whether the answer is complete.

    "$STRATA_ROOT/.venv/bin/python" test_answer_scope.py
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
        if cand is None or not hits:
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
TURN_ID, THINK_OPEN, THINK_CLOSE = 248046, 248068, 248069
F3 = chr(96) * 3


class CursorEngine:
    def __init__(self, ids):
        self.ids, self.cursor, self.calls = list(ids), 0, 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for t in self.ids[self.cursor:self.cursor + max_new]:
            self.cursor += 1
            yield t
        yield TURN_ID


def consume_like_server(gen):
    seen, ended = [], False
    for t in gen:
        seen.append(t)
        if t in CS.DEFAULT_EOS:
            ended = True
            break
    return seen, ended


def run(ids, max_new=1024):
    eng = CursorEngine(ids)
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=5, log=sys.stderr)
    seen, ended = consume_like_server(
        g.guard(eng.generate, tok.encode("hi", parse_special=False), max_new, {}, threading.Event()))
    return seen, ended, eng, g


fail = total = 0
def check(label, cond):
    global fail, total
    total += 1
    fail += 0 if cond else 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


# A complete answer: a closed fenced block, then a short closing paragraph.  The last 64
# ids land inside that block - the window that produced the false positive.
BODY = "".join("    value_%d = %d\n" % (i, i) for i in range(40))
ANSWER = ("Here is the code.\n\n" + F3 + "python\n" + BODY + F3 +
          "\n\nWant me to add that instrumentation to the guard? Say go and I will apply it.")
NOISE = ("The trace shows `inside_span` on `'quoted'` text, then ``` runs, "
         "and `tail_text(produced)` capped at 64 ids.")

print("a complete answer whose last 64 ids start inside its own closed code block:")
ids = [THINK_OPEN] + tok.encode(NOISE, parse_special=False) + [THINK_CLOSE] + tok.encode(ANSWER, parse_special=False)
seen, ended, eng, g = run(ids)
out = tok.decode(seen)
capped = g.tail_text(ids, keep=64)
check("the 64-id window does start inside the block (the artifact, pinned: it says True)",
      CS.inside_span(capped) is True)
check("the answer itself is closed (inside_span over the answer says False)",
      CS.inside_span(g.answer_text(ids)) is False)
check(f"not rescued ({len(seen)} ids, {eng.calls} engine call)", "\u200b" not in out and eng.calls == 1)
check("the caller still receives a turn-end id, so finish_reason stays stop", ended)
check("the answer came through complete, with no spliced markers in it",
      "Say go and I will apply it." in out and "\u200b" not in out)

print("\nreasoning full of backtick noise, answer clean (must NOT rescue):")
seen2, ended2, eng2, g2 = run(ids)
check(f"not rescued ({eng2.calls} engine call)", "\u200b" not in tok.decode(seen2) and eng2.calls == 1)

print("\na genuine truncation inside an open code block (must STILL rescue):")
TRUNC = "Here is the code.\n\n" + F3 + "python\n" + BODY
ids3 = [THINK_OPEN] + tok.encode("thinking.", parse_special=False) + [THINK_CLOSE] + tok.encode(TRUNC, parse_special=False)
seen3, ended3, eng3, _ = run(ids3)
out3 = tok.decode(seen3)
check(f"rescued ({len(seen3)} ids, {eng3.calls} engine calls)", "\u200b" in out3 and eng3.calls >= 2)

print(f"\n{total} checks, {fail} failures")
sys.exit(1 if fail else 0)
