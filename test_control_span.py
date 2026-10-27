"""Unit tests for the predicate + escape helper (no GPU, no engine needed)."""
F3 = chr(96) * 3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_span import (inside_span, escape_text, ZWSP,
                         ALL_RULES, DEFAULT_RULES, parse_rules)

CASES = [
    # (tail, expected, label)   -- ported from the llama.cpp predicate test set
    ('The answer is `', True, "open backtick after a space"),
    ('write it as `<', True, "mid-tag: '<' with nothing after it"),
    ('write it as `<|im_', True, "half-typed tag"),
    ('say "hello', False, "unbalanced double quote: rule 1 is OFF by default"),
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


# --- code-span / code-block parity (the rule a tail-local predicate cannot express) ---
PARITY = [
    (F3 + 'python\nprint("hi")', True, "unclosed fenced block (the markdown case)"),
    (F3 + 'python\nprint("hi")\n' + F3, False, "closed fenced block"),
    ('the operator ` is used for shells', True, "inline span opened earlier, never closed"),
    ('use `foo` and `bar` here', False, "two closed inline spans"),
    (F3 + 'sh\nls\n' + F3 + ' then run `deploy', True, "closed block, then an open inline span"),
    ('the model wrote `x` and stopped', False, "closed inline span, prose after it"),
    ('a fence marker inside a block:\n' + F3 + 'a\n' + F3 + ' is fine', False, "two closed blocks"),
    (F3 + 'python\n' + 'y' * 3000, True, "a long block is still open over the whole turn"),
    # tildes are not code markers: an approximation sign must never open a span (2026-10-10)
    ('That is another ~450 words, landing near 3,100.', False, "tilde as an approximation sign"),
    ('the Q2_0 ~40 GB repack is one-time.', False, "tilde before a number opens nothing"),
    ('~~~' + chr(10) + 'unclosed tilde fence' + chr(10), False, "tildes ignored entirely (behaviour change)"),
]

# --- why rule 4 is not windowed (the artifact a window creates) ---
# the block must be LONGER than the cap, or the window never starts inside it
BODY = "".join("    value_%d = %d\n" % (i, i) for i in range(20))          # ~460 chars
ANSWER = ("Use the `CORSMiddleware` from `fastapi.middleware.cors`.\n\n" + F3 + "python\n" + BODY
          + F3 + "\n\n**Key points:**\n- `allow_origins`: use " + F3 + '["*"]' + F3 + " for all.\n")
CLOSED = ANSWER[:ANSWER.index(F3 + "\n\n**Key points") + 4]     # ends exactly at the closing fence
WINDOWED = [
    (CLOSED, 0, False, "a closed answer, read over the whole turn: no false positive"),
    (CLOSED, 200, True, "the same text capped at 200 chars: the window starts inside the block, "
                        "so its closing fence reads as an opening one"),
    (CLOSED + "\n\n" + F3 + "bash\nls", 0, True, "a block genuinely left open at the end"),
]


# --- rule 1, on demand: what it catches, and the false positive that leaked ----
RULE1_ON = [
    ('say "hello', True, "rule 1 ON: mid-quote stop is caught"),
    # the real 2026-10-27 leak: the quote CLOSES 24 chars before the end, but the 32-char
    # window opens between the opener and the closer, so the slice reads odd
    ('the log is clean. Say "check the build" and I\'ll read the log.',
     True, "rule 1 ON: the false positive on a FINISHED answer (the leak)"),
    ('the log is clean. Say "check the build" and I\'ll read the log.',
     False, "rule 1 OFF (default): the same finished answer is left alone"),
]

# --- the rule selector itself --------------------------------------------------
CFG = [
    (None, DEFAULT_RULES, "unset -> the default (rule 1 off)"),
    ("", DEFAULT_RULES, "empty -> the default"),
    ("all", ALL_RULES, "all four"),
    ("none", (), "none: the guard never rescues on a span, only on an open think block"),
    ("1", (1,), "one id"),
    ("quote,code", (1, 4), "by name"),
    ("all,-1", DEFAULT_RULES, "all minus rule 1 == the default"),
    ("-1", DEFAULT_RULES, "a bare subtraction starts from the default"),
    ("2, 3,4", (2, 3, 4), "spaces tolerated"),
]

BAD_CFG = [("9", "rule id 9 does not exist"), ("quoe", "a misspelt name must raise")]


def run():
    bad = 0
    for tail, want, label in CASES:
        got = inside_span(tail)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:5s} got={got!s:5s}  {label}")
    for text, scan, want, label in WINDOWED:
        got = inside_span(text, scan=scan)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:5s} got={got!s:5s}  {label}")
    for text, want, label in PARITY:
        got = inside_span(text, scan=0)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:5s} got={got!s:5s}  {label}")
    for text, want, label in RULE1_ON:
        got = inside_span(text, rules=ALL_RULES if "ON:" in label and "OFF" not in label else DEFAULT_RULES)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:5s} got={got!s:5s}  {label}")
    for spec, want, label in CFG:
        got = parse_rules(spec)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want!s:12s} got={got!s:12s}  {label}")
    for spec, label in BAD_CFG:
        try:
            parse_rules(spec)
            bad += 1
            print(f"  FAIL  no error for {spec!r}  ({label})")
        except ValueError:
            print(f"  PASS  rejects {spec!r}  ({label})")
    assert escape_text("</think>") == "<" + ZWSP + "/think>", escape_text("</think>")
    print(f"  PASS  escape_text inserts U+200B after the first character")
    n = len(CASES) + len(PARITY) + len(WINDOWED) + len(RULE1_ON) + len(CFG) + len(BAD_CFG) + 1
    print(f"\n  {n} cases, {bad} failures")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(run())
