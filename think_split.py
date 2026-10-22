"""think_split.py - which id closed the thinking block?

Strata's OutputParser splits reasoning -> content on the TEXT of the closing
marker.  A model that QUOTES the marker while reasoning (illustrating the chat
template, say) looks identical from the text side, and the rest of its reasoning
is then delivered as the answer.

The text cannot tell the two apart; the ID can.  This module records, for every
reasoning->content transition, the id that completed the marker:

  STRATA_THINK_SPLIT=log     (default when installed) - log only, no behaviour change
  STRATA_THINK_SPLIT=off     - do nothing

The log line says whether the marker arrived as the pack's RESERVED id (the model
really closed its thinking) or was SPELLED out of ordinary tokens (it was writing
the characters - a quoted marker).  Strict handling can then be built against the
measured branch instead of a guess.
"""
from __future__ import annotations

import os
import sys
import threading

THINK_END = "<" + "/think>"          # built by concatenation: no raw tag in source

_tls = threading.local()


def _note(msg, *a):
    print("[think-split] " + (msg % a if a else msg), file=sys.stderr, flush=True)


def reserved_id(tok):
    """The one id whose text is the marker, or None when the tokenizer spells it."""
    try:
        ids = tok.encode(THINK_END, parse_special=True)
    except Exception:
        return None
    return ids[0] if len(ids) == 1 else None


def install(server_module, tok, mode: str | None = None):
    """Patch Detokenizer.push (records the id) and OutputParser.feed (logs the transition)."""
    mode = (mode or os.environ.get("STRATA_THINK_SPLIT", "off")).strip().lower()
    if mode not in ("log", "strict"):
        return None
    if mode == "strict":
        _note("strict is not implemented yet - running as log")
    detok_cls = server_module.Detokenizer
    parser_cls = server_module.OutputParser
    if getattr(detok_cls.push, "_think_split", False) or getattr(parser_cls.feed, "_think_split", False):
        return None
    rid = reserved_id(tok)
    _note("ENABLED (log; the pack's reserved marker id is %s)", rid)

    orig_push = detok_cls.push

    def push(self, t):
        _tls.last = t
        return orig_push(self, t)

    orig_feed = parser_cls.feed

    def feed(self, delta):
        before = getattr(self, "state", None)
        evs = orig_feed(self, delta)
        if before == "reasoning" and getattr(self, "state", None) == "content":
            t = getattr(_tls, "last", None)
            if rid is None:
                how = "unknown (the tokenizer spells the marker)"
            elif t == rid:
                how = "RESERVED id -> the model really closed its thinking"
            else:
                how = "SPELLED out of ordinary tokens -> a quoted marker"
            _note("reasoning -> content by id %s: %s", t, how)
        return evs

    push._think_split = True
    feed._think_split = True
    detok_cls.push = push
    parser_cls.feed = feed
    return push
