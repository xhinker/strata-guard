"""End-to-end test of the rescue guard with Strata's own Service + MockEngine.

No GPU: MockEngine replays a scripted completion as token ids and then emits the
end-of-turn token.  With a script that ends INSIDE an open span, the stop is the
pathological case, so the guard must rescue it.
"""
import json, os, sys, os
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(Path(__file__).resolve().parent))
def _config_path():
    """A Strata model config: $STRATA_CONFIG, else the first strata-*.json in the checkout."""
    import glob
    named = os.environ.get("STRATA_CONFIG")
    if named:
        return Path(named) if Path(named).is_absolute() else STRATA / named
    hits = sorted(glob.glob(str(STRATA / "strata-*.json")))
    if not hits:
        raise SystemExit("no strata-*.json found in %s - set STRATA_ROOT / STRATA_CONFIG" % STRATA)
    return Path(hits[0])


import control_span as CS                                    # noqa: E402
from serve import server as S                                # noqa: E402
import strata_tokenizer as ST                                # noqa: E402

CFG = json.loads(_config_path().read_text())
ARGS = CFG["args"]; PACK = Path(ARGS[ARGS.index("--pack") + 1]) / "tokenizer"
vocab = json.loads((PACK / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items(): tokens[i] = t
TYPES = json.loads((PACK / "token_type.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(tokens, (PACK / "merges.txt").read_text(encoding="utf-8").split("\n"), TYPES)
print(f"  tokenizer: {type(tok).__name__}, vocab {len(tokens)}")
print(f"  eos ids:   {CS.DEFAULT_EOS} -> {[tok.decode([e]) for e in CS.DEFAULT_EOS]}")

TAIL_TEXT = "The closing tag is `"          # ends inside an open code span
NORMAL    = "The answer is 4."

def run(script, guard_on):
    eng = S.MockEngine(tok, script, max_context=1 << 20)
    ids = tok.encode("hello", parse_special=False)
    if guard_on:
        g = CS.Guard(tok, CS.DEFAULT_EOS, types=TYPES, log=sys.stderr)
        return list(g.guard(eng.generate, ids, 64, {}, __import__("threading").Event())), g
    return list(eng.generate(ids, 64, {}, __import__("threading").Event(), None)), None

print("\n=== A) normal ending, guard OFF (baseline) ===")
out, _ = run(NORMAL, False); print(f"  {len(out)} ids, text={tok.decode(out)!r}")

print("\n=== B) stop INSIDE a span, guard OFF (the bug) ===")
out, _ = run(TAIL_TEXT, False); print(f"  {len(out)} ids, text={tok.decode(out)!r}  <- truncated")

print("\n=== C) same stop, guard ON (rescue) ===")
out, g = run(TAIL_TEXT, True)
txt = tok.decode(out)
print(f"  {len(out)} ids, text={txt!r}")
print(f"  survived the stop: {'YES' if len(out) > len(run(TAIL_TEXT, False)[0]) else 'NO'}")
base = len(run(TAIL_TEXT, False)[0])
print(f"  rescued: {len(out) > base}   (baseline {base} ids -> {len(out)} ids)")

print("\n=== D) normal ending, guard ON (must NOT rescue) ===")
out, g = run(NORMAL, True); print(f"  {len(out)} ids, text={tok.decode(out)!r}")

print("\n=== E) the monkey-patch mechanism ===")
before = S.StrataEngine.generate
os.environ["STRATA_CONTROL_SPAN_GUARD"] = "on"
guard = CS.install(S, tok, types=TYPES, force=True)
print(f"  install() patched StrataEngine.generate: {S.StrataEngine.generate is not before}")
