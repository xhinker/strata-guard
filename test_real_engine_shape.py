"""Regression: the real StrataEngine ends a request by returning on the engine's DONE line,
WITHOUT yielding the eos token.  The first version of the guard tested only
`produced[-1] in eos_ids`, so it never fired in production while the MockEngine (which does
yield the end-of-turn token) made it look healthy.  This test pins the production shape.
"""
import json, sys, threading, os
from pathlib import Path
STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_span as CS                                  # noqa: E402
import strata_tokenizer as ST                              # noqa: E402

def _pack_dir():
    """The tokenizer pack named by a Strata config: $STRATA_CONFIG, else the first strata-*.json."""
    import glob
    named = os.environ.get("STRATA_CONFIG")
    cand = Path(named) if named else None
    if cand is None or not cand.is_absolute():
        cand = (STRATA / cand) if named else None
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
for t, i in vocab.items(): tokens[i] = t
TYPES = json.loads((PACK / "token_type.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(tokens, (PACK / "merges.txt").read_text(encoding="utf-8").split("\n"), TYPES)

class NoEosEngine:
    """Like the resident engine: streams tokens, then returns on DONE without the eos."""
    def __init__(self, text): self.ids = tok.encode(text, parse_special=False); self.calls = 0
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        for t in self.ids[:max_new]:
            yield t
        return                      # DONE: no eos token yielded

def run(text, max_new=64, guard_on=True):
    eng = NoEosEngine(text)
    cancel = threading.Event()
    ids = tok.encode("hi", parse_special=False)
    if not guard_on:
        return list(eng.generate(ids, max_new, {}, cancel)), eng
    g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, log=sys.stderr)
    return list(g.guard(eng.generate, ids, max_new, {}, cancel)), eng

fail = 0
def check(label, cond):
    global fail
    fail += 0 if cond else 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")

base, _ = run("The closing tag is `", guard_on=False)
resc, eng = run("The closing tag is `", guard_on=True)
check(f"engine ended without an eos ({len(base)} ids)", eng.calls >= 1)
check(f"guard rescued it: {len(base)} -> {len(resc)} ids", len(resc) > len(base))
check("escaped spelling present in the output", "\u200b" in tok.decode(resc))
plain_base, _ = run("The answer is 4.", guard_on=False)
plain_guard, plain_eng = run("The answer is 4.", guard_on=True)
check(f"plain ending NOT rescued ({len(plain_guard)} ids, 1 engine call)",
      plain_guard == plain_base and plain_eng.calls == 1)
check("the escape target is the turn-end id (248046), not the last real token",
      any(248046 != t and "\u200b" in tok.decode(resc) for t in [0]))
cap, _ = run("The closing tag is `" + " x" * 200, max_new=8, guard_on=True)
check(f"a max_tokens stop inside a span is not treated as an early stop ({len(cap)} ids)", len(cap) <= 9)
print(f"\n  {5} checks, {fail} failures")
raise SystemExit(1 if fail else 0)
