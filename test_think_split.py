"""Offline test: no GPU, no server. Real OutputParser, a byte tokenizer, a fake Detokenizer."""
import io, os, sys, contextlib, pathlib
sys.path.insert(0, ".")
sys.path.insert(0, os.environ.get("STRATA_ROOT") or str(pathlib.Path(__file__).resolve().parent.parent / "Strata"))
from pathlib import Path  # noqa
import pathlib
from serve.frontend import OutputParser
import think_split as TS

THINK_END = "<" + "/think>"


class ByteTok:
    def __init__(self, table): self.table = table
    def token_bytes(self, t): return self.table[t].encode()
    def decode(self, ids): return "".join(self.table[i] for i in ids)
    def encode(self, text, parse_special=True): return [k for k, v in self.table.items() if v == text][:1] or [99]


class Detok:
    def __init__(self, tok): self.tok = tok
    def push(self, t): return self.tok.token_bytes(t).decode()


class S:
    Detokenizer = Detok
    OutputParser = OutputParser


def trial(reserved: bool):
    if reserved:
        table = {7: "<" + "/think>", 9: "answer"}
        pieces = [7]
    else:
        table = {1: "<", 2: "/think>", 9: "answer"}
        pieces = [1, 2]
    tok = ByteTok(table)
    S.Detokenizer = type("D", (Detok,), {})
    S.OutputParser = type("P", (OutputParser,), {})
    TS.install(S, tok, mode="log")
    d = S.Detokenizer(tok)
    p = S.OutputParser(thinking=True, tools=None)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        for t in pieces:
            p.feed(d.push(t))
        p.feed(d.push(9))
    return buf.getvalue()


for label, reserved, want in (("reserved id", True, "RESERVED"), ("spelled pieces", False, "SPELLED")):
    out = trial(reserved)
    ok = want in out
    print(("  ok   " if ok else "  FAIL ") + f"{label}: log says {want}")
    for line in out.splitlines():
        if "reasoning -> content" in line:
            print("        " + line)
