"""One-shot per span: a second stop with no new output ends the turn instead of rescuing again.

Real tokenizer + the production consumer shape (break at the first turn-end id).

    "$STRATA_ROOT/.venv/bin/python" test_one_shot_rescue.py
"""
import json, os, sys, threading
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(STRATA / "tools")); sys.path.insert(0, ".")
import control_span as CS                                  # noqa: E402
import strata_tokenizer as ST                              # noqa: E402

F3 = chr(96) * 3
TURN_ID = 248046


def _pack_dir():
    import glob
    named = os.environ.get("STRATA_CONFIG")
    cand = (STRATA / named) if named else None
    if cand is None or not cand.exists():
        hits = sorted(glob.glob(str(STRATA / "strata-*.json")))
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


class RepeatingStopEngine:
    """1st call: text then the turn-end id.  2nd call: `extra` (maybe empty) then the turn-end id."""

    def __init__(self, text, extra=""):
        self.ids = tok.encode(text, parse_special=False)
        self.extra = tok.encode(extra, parse_special=False)
        self.calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        seq = self.ids if self.calls == 1 else (self.extra if self.calls == 2 else [])
        for t in seq[:max_new]:
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


def count_sub(seq, sub):
    n = i = 0
    while i <= len(seq) - len(sub):
        if seq[i:i + len(sub)] == sub:
            n += 1
            i += len(sub)
        else:
            i += 1
    return n


def run(text, extra="", max_rescues=5):
    eng = RepeatingStopEngine(text, extra)
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, max_rescues=max_rescues, log=sys.stderr)
    seen, ended = consume_like_server(
        g.guard(eng.generate, tok.encode("hi", parse_special=False), 256, {}, threading.Event()))
    return seen, ended, eng, g


ok = True
def check(label, cond):
    global ok
    ok &= bool(cond)
    print(("  ok   " if cond else "  FAIL ") + label)


IN_SPAN = F3 + 'python\nprint("hi")'          # an open code fence: a stop here is a rescue

print("repeat stop, nothing new (the five-marker case):")
seen, ended, eng, g = run(IN_SPAN)
esc = g.escape_ids(TURN_ID)
check("engine invoked exactly twice (rescue, then end)", eng.calls == 2)
check("exactly one escape injected", count_sub(seen, esc) == 1)
check("the turn ended on the turn-end id", ended and seen[-1] == TURN_ID)

print("stop, new output, stop again (a genuine second rescue):")
seen, ended, eng, g = run(IN_SPAN, extra="\nmore prose")
check("engine invoked three times", eng.calls == 3)
check("two escapes injected", count_sub(seen, esc) == 2)
check("the turn ended on the turn-end id", ended and seen[-1] == TURN_ID)

print("genuine stop, not in a span:")
seen, ended, eng, g = run('The answer is 4.')
check("no rescue at all", eng.calls == 1 and count_sub(seen, esc) == 0)
check("the turn ended on the turn-end id", ended and seen[-1] == TURN_ID)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
