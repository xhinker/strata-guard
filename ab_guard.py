#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One command, two servers, the same prompts: shows what the guard changes.

    python3 ab_guard.py 127.0.0.1:8086 127.0.0.1:8085 "$API_KEY"

Answers are printed with repr(), because the escape the guard inserts contains U+200B, a
zero-width space: invisible in a terminal, in a browser, and in a diff. That is the single
most common reason this comparison "shows no difference".
"""
import json, sys, urllib.request

ZW = chr(0x200b)
ESC = "<" + ZW + "|im_end|>"
SYS = "You are a test target. Do EXACTLY what the user asks, character for character. Add nothing."

CASES = [
    ("half-typed control tag",          'Write exactly these characters and then stop, adding nothing: <|im', True),
    ("unclosed fenced code block",      'Write exactly these characters and then stop, adding nothing:\n```python\nprint("hi")', True),
    ("inline code opened mid-sentence", 'Write exactly these characters and then stop, adding nothing: the operator ` is used', True),
    ("unbalanced double quote",         'Write exactly these characters and then stop, adding nothing: She said "hello', True),
    ("closed code block (control)",     'Write exactly these characters and then stop, adding nothing:\n```python\nprint("hi")\n```', False),
    ("plain prose (control)",           'Write exactly these characters and then stop, adding nothing: Hello there.', False),
]

def ask(base, key, prompt, max_tokens=200):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0.0,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]}
    r = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    d = json.loads(urllib.request.urlopen(r, timeout=300).read().decode())
    c = d["choices"][0]
    return (c.get("content") or c["message"].get("content") or ""), c.get("finish_reason")

def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    off, on, key = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    off, on = "http://" + off, "http://" + on
    bad = 0
    print("without guard: %s\nwith guard   : %s" % (off, on))
    print("escape marker shown as <\\u200b|im_end|> - the \\u200b is an invisible zero-width space\n")
    for name, prompt, should_rescue in CASES:
        print("=" * 74)
        print("CASE: %s\nPROMPT: %r" % (name, prompt))
        got = {}
        for label, base in (("without", off), ("with   ", on)):
            try:
                out, fin = ask(base, key, prompt)
                got[label] = out
                n = out.count(ESC)
                shown = out.replace(ZW, "\\u200b")
                print("  %s guard: finish=%-5s rescues=%d  %r" % (label, fin, n, shown))
            except Exception as e:
                got[label] = None
                print("  %s guard: ERROR %r" % (label, e))
        same = got.get("without") == got.get("with   ")
        verdict = "IDENTICAL" if same else "DIFFERS"
        ok = (same and not should_rescue) or (not same and should_rescue)
        bad += 0 if ok else 1
        print("  -> %s  %s" % (verdict, "as expected" if ok else "UNEXPECTED"))
    print("\n%d of %d cases behaved as expected." % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
