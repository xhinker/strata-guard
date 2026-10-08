"""stats_from_log.py - restore the Monitor's per-request figures.

The Monitor's "reused" and "tok/s" columns come from the service's request records, and the
record builder reads them off StrataEngine.last (`last["reused"]`, `last["prompt_ms"]`,
`last["decode_ms"]`, `last["generated"]` - see Service's history.append and request_timings()).
`last` is meant to be filled when the engine's DONE line is parsed.  On this server it stays
empty in the streaming solo path, so those two columns render "-".

The engine, however, always writes its own per-request summary to its log:

    strata serve: prompt 47677 tokens = 43231 reused + 4446 read in 3187 ms (1394.9 tok/s),
                  452 generated in 4116 ms (109.8 tok/s), drafts accepted 247 of 454, 6 checkpoints

so this module re-states that line into `engine.last` when a request finishes, using exactly the
field names the record builder expects.  It only fills keys that are missing, never overwrites.

Install with install(server_module, log_path); disable with STRATA_STATS_FROM_LOG=0.
"""
from __future__ import annotations
import os, re, sys

SUMMARY = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read_n>\d+)(?: of \d+)? read in "
    r"(?P<read_ms>[\d.]+) ms \((?P<pp>[\d.]+) tok/s\), (?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms "
    r"\((?P<tg>[\d.]+) tok/s\)")


def _note(msg, *a):
    print("[strata-stats] " + (msg % a if a else msg), file=sys.stderr, flush=True)


def read_last_summary(log_path: str, tail_bytes: int = 262144) -> dict | None:
    """The newest engine summary in the log, in the record builder's field names."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    m = None
    for m in SUMMARY.finditer(text):
        pass                                  # keep the last match
    if not m:
        return None
    return {"prompt_tokens": int(m["prompt"]), "reused": int(m["reused"]), "prompt_read": int(m["read_n"]),
            "prompt_ms": float(m["read_ms"]), "decode_ms": float(m["gen_ms"]), "generated": int(m["gen"]),
            "prompts_per_second": float(m["pp"]), "predicted_per_second": float(m["tg"])}


def install(server_module, log_path: str):
    if os.environ.get("STRATA_STATS_FROM_LOG", "1") in ("0", "off", "false", "no"):
        return None
    cls = server_module.StrataEngine
    original = cls.generate
    if getattr(original, "_stats_from_log", False):
        return None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            yield from original(self, ids, max_new, sampling, cancel, embeddings)
        finally:
            fresh = read_last_summary(log_path) if log_path else None
            if fresh:
                cur = dict(getattr(self, "last", None) or {})
                filled = [k for k, v in fresh.items() if cur.get(k) is None]
                for k in filled:
                    cur[k] = fresh[k]
                try:
                    self.last = cur
                except Exception:                                   # a read-only property: set the dict in place
                    tp = getattr(self, "last", None)
                    if isinstance(tp, dict):
                        tp.update({k: fresh[k] for k in filled})
                if filled:
                    _note("filled %s from the engine log (reused=%d, decode=%.1f tok/s)",
                          ",".join(filled), fresh["reused"], fresh["predicted_per_second"])

    generate._stats_from_log = True
    cls.generate = generate
    _note("ENABLED (log=%s)", log_path)
    return generate
