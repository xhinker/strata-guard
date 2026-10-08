"""probe_guard.py - does the control-span rescue guard fire?

The guard acts on exactly one condition: the generation ENDS (its last token is one of
the eos ids) while the tail looks like an open quote / code span / half-typed tag.
So every "positive" probe below has to end mid-span, and the negative controls must not.

    python3 probe_guard.py [PORT] [API_KEY]
"""
import json, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8084"
KEY  = sys.argv[2] if len(sys.argv) > 2 else ""
BASE = f"http://127.0.0.1:{PORT}"

LT, BT, QT, PIPE = "<", "`", '"', "|"          # built, never written as a tag literal

SYS = ("You are a test target. Do EXACTLY what the user asks, character for character. "
       "Never add explanations, never close anything the user asked you to leave open.")

PROBES = [
    ("POSITIVE 1 - half-typed tag at the end",
     f"End your reply with these exact characters and then stop: {LT}{PIPE}im  - do not complete the tag.",
     "tail ends on '<' with no '>' or space -> rule 3"),
    ("POSITIVE 2 - unclosed code span at the end (less reliable: the model may close it)",
     f"End your reply with exactly one {BT} character and type nothing after it.",
     "tail ends on an opening backtick -> rule 2"),
    ("POSITIVE 3 - unbalanced double quote at the end",
     f"Write one short sentence that ends with an opening double quote, exactly like: She said {QT}   - then stop.",
     "odd number of quotes in the window -> rule 1"),
    ("POSITIVE 4 - open pipe at the end",
     f"End your reply with the two characters {LT}{PIPE} and stop.",
     "tail ends on '|' -> rule 2"),
    ("NEGATIVE 1 - a finished tag (should NOT rescue)",
     f"Print these two markers and nothing else: {LT}think> and {LT}/think>",
     "tail ends on '>' -> predicate false"),
    ("NEGATIVE 2 - plain prose (should NOT rescue)",
     "Say hello in one short sentence.",
     "no span at all"),
]

def ask(user):
    body = {"model": "x", "max_tokens": 250, "temperature": 0.7,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    d = json.loads(urllib.request.urlopen(r, timeout=240).read().decode())
    c = d["choices"][0]
    return c.get("finish_reason"), c["message"].get("content") or "", d["usage"]

print(f"  probing {BASE}")
print("  HOW TO READ IT: a positive probe should log '[control-span] rescue N: the turn ended inside a")
print("  span ...' on the server's stderr, and its answer should contain the escaped spelling")
print("  (an invisible U+200B) instead of stopping at the dangling character.  Negatives must not log.\n")
for name, prompt, why in PROBES:
    try:
        fin, content, usage = ask(prompt)
    except Exception as exc:
        print(f"  {name}\n     ERROR {exc}\n"); continue
    print(f"  {name}")
    print(f"     why      : {why}")
    print(f"     finish   : {fin} | completion={usage.get('completion_tokens')}")
    print(f"     answer   : {content[:180]!r}")
    print()
