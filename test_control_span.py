"""Unit tests for the predicate + escape helper (no GPU, no engine needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_span import inside_span, escape_text, ZWSP

CASES = [
    # (tail, expected, label)   -- ported from the llama.cpp predicate test set
    ('The answer is `', True, "open backtick after a space"),
    ('write it as `<', True, "mid-tag: '<' with nothing after it"),
    ('write it as `<|im_', True, "half-typed tag"),
    ('say "hello', True, "unbalanced double quote"),
    ('The answer is 4.', False, "plain prose"),
    ('see `<\\u200b/think>`, then continue', False, "escaped tag inside a closed span"),
    ('`<'+'think> and <'+'think>`', False, "closed span, finished tag"),
    ('the two markers are <'+'think> <'+'think>', False, "tags at the very end"),
    ('7 x 3 = 21', False, "math with no '<'"),
    ('(`', True, "backtick after an opening paren"),
    ('x = `', True, "backtick after '='"),
    ('done`', False, "backtick preceded by a letter is closing"),
    ('text `<|endoftext|>`', False, "closed tag at the end"),
]

def run():
    bad = 0
    for tail, want, label in CASES:
        got = inside_span(tail)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:5s} got={got!s:5s}  {label}")
    assert escape_text("</think>") == "<" + ZWSP + "/think>", escape_text("</think>")
    print(f"  PASS  escape_text inserts U+200B after the first character")
    print(f"\n  {len(CASES)+1} cases, {bad} failures")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(run())
