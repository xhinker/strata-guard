"""probe_think_split.py - trigger a quoted thinking-close and show what think_split logged.

Sends a prompt that makes the model quote the chat-template's closing marker inside
its own reasoning, then prints the [think-split] lines from the server's tmux pane.

  python3 probe_think_split.py --url http://127.0.0.1:8084/v1 --tmux strata

--dry prints the request instead of sending it.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, urllib.request

LT = "<"
CLOSE = LT + "/think>"
PROMPT = (
    "Think step by step about how a chat template closes its thinking block. "
    "About halfway through your reasoning, write that closing marker literally, character for character, "
    "as part of your reasoning text (do not put it in a code fence). Then continue reasoning for two or "
    "three more sentences, and only then give a one-sentence final answer."
)


def pane_lines(session: str, n: int = 400) -> list[str]:
    try:
        out = subprocess.run(["tmux", "capture-pane", "-p", "-t", session, "-S", f"-{n}"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [l for l in out.splitlines() if "[think-split]" in l]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8084/v1")
    ap.add_argument("--key", default="")
    ap.add_argument("--tmux", default="strata")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    body = {"model": "x", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 400, "temperature": 0.7,
            "chat_template_kwargs": {"preserve_thinking": False}}
    if a.dry:
        print(json.dumps(body, indent=2))
        return 0

    before = len(pane_lines(a.tmux))
    req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {a.key}"} if a.key else {})})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
    m = d["choices"][0]["message"]
    rc, c = m.get("reasoning_content") or "", m.get("content") or ""
    print(f"  responded in {time.time()-t0:.1f}s | finish={d['choices'][0].get('finish_reason')}")
    print(f"  reasoning {len(rc)} chars | content {len(c)} chars")
    print(f"  reasoning tail : {rc[-70:]!r}")
    print(f"  content   head : {c[:70]!r}")
    time.sleep(1.0)
    lines = pane_lines(a.tmux)
    new = lines[before:]
    print(f"\n  [think-split] lines from the last {len(new)} new:")
    for l in new[-6:]:
        print("   ", l.strip()[:170])
    if not lines:
        print("   (none - is STRATA_THINK_SPLIT=log in the server environment, and is --tmux right?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
