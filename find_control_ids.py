#!/usr/bin/env python3
"""find_control_ids.py - the control ids of YOUR pack, and the flags to run the guard with.

The guard needs two facts about the model it is protecting: which ids end a turn, and which
ids open/close the thinking block. Those ids belong to the pack, not to the guard - a Qwen3.8
pack puts the end of turn at 248046, another pack will put it somewhere else entirely.

    python3 find_control_ids.py --config strata-your-model.json
    python3 find_control_ids.py --pack /path/to/pack/tokenizer

Prints what each control string encodes to, its token type, and the two flags to pass.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT") or Path(__file__).resolve().parent.parent / "Strata")
sys.path.insert(0, str(STRATA)); sys.path.insert(0, str(STRATA / "tools"))

LT = "<"
# built by concatenation: no raw control tag lives in this source file
STRINGS = {
    "end of turn (im_end)":      LT + "|im_end|>",
    "end of text (endoftext)":   LT + "|endoftext|>",
    "start of message":          LT + "|im_start|>",
    "thinking open":             LT + "think>",
    "thinking close":            LT + "/think>",
    "tool call open":            LT + "tool_call>",
    "tool call close":           LT + "/tool_call>",
}
TYPE_NAMES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}


def pack_from(args) -> Path:
    if args.pack:
        p = Path(args.pack)
        return p / "tokenizer" if (p / "tokenizer").is_dir() else p
    named = args.config or os.environ.get("STRATA_CONFIG")
    cand = None
    if named:
        for c in (Path(named), STRATA / named):
            if c.exists():
                cand = c; break
    if cand is None:
        hits = sorted(glob.glob(str(STRATA / "strata-*.json")))
        if not hits:
            raise SystemExit("no config found - pass --config or --pack")
        cand = Path(hits[0])
    a = json.loads(cand.read_text())["args"]
    return Path(a[a.index("--pack") + 1]) / "tokenizer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config"); ap.add_argument("--pack")
    a = ap.parse_args()
    pack = pack_from(a)
    import strata_tokenizer as ST
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    types = json.loads((pack / "token_type.json").read_text(encoding="utf-8"))
    tok = ST.Tokenizer(tokens, (pack / "merges.txt").read_text(encoding="utf-8").split("\n"), types)

    print(f"pack: {pack}\n")
    single, names = {}, {}
    for label, text in STRINGS.items():
        ids = tok.encode(text, parse_special=True)
        one = len(ids) == 1
        kind = TYPE_NAMES.get(types[ids[0]], types[ids[0]]) if one else "-"
        print(f"  {label:26s} -> {str(ids):24s} {'one token' if one else f'{len(ids)} pieces'}  {kind}")
        if one:
            single[label] = ids[0]
            names[ids[0]] = text
    eos = [single[k] for k in ("end of turn (im_end)", "end of text (endoftext)") if k in single]
    think = [single[k] for k in ("thinking open", "thinking close") if k in single]
    print("\nrun the guard with:")
    print(f"  --control-span-eos {','.join(map(str, eos)) or 'NONE FOUND'}")
    print(f"  --control-span-think {','.join(map(str, think)) or '(none - model without a thinking channel)'}")
    print("\nor as environment:")
    print(f"  STRATA_CONTROL_SPAN_EOS={','.join(map(str, eos))}")
    print(f"  STRATA_CONTROL_SPAN_THINK={','.join(map(str, think))}")
    print("\nsanity check after starting: probe_guard.py <port> <api-key>  (every positive must be rescued)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
