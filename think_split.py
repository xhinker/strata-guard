"""think_split.py - which id closed the thinking block, and what to do about it.

Strata's OutputParser splits reasoning -> content on the TEXT of the closing
marker.  A model that QUOTES the marker while reasoning (illustrating the chat
template, say) looks identical from the text side, and the rest of its reasoning
is then delivered as the answer.

The text cannot tell the two apart; the ID can.  This module records, for every
reasoning->content transition, the id that completed the marker:

  STRATA_THINK_SPLIT=log      - log only, no behaviour change
  STRATA_THINK_SPLIT=strict   - a split caused by a QUOTED marker is undone: what
                                follows it stays REASONING, where it belongs
  STRATA_THINK_SPLIT=off      - do nothing

The log line says whether the marker arrived as the pack's RESERVED id (the model
really closed its thinking) or was SPELLED out of ordinary tokens (it was writing
the characters - a quoted marker).

Strict deliberately acts only when the pack HAS a reserved id: without one every
close is spelled and a spelled close carries no information, so nothing is
rewritten.  It also leaves a turn alone when the split released tool calls, which
cannot be un-emitted.  The quoted marker goes back into the reasoning text in the
escaped spelling (U+200B after its first character, the same convention as
control_span.escape_text), so a downstream splitter that matches the raw tag text
does not read it as a real close.
"""
from __future__ import annotations

import os
import sys
import threading

THINK_END = "<" + "/think>"          # built by concatenation: no raw tag in source
ZWSP = "\u200b"

_tls = threading.local()


def _note(msg, *a):
    print("[think-split] " + (msg % a if a else msg), file=sys.stderr, flush=True)


def escaped(text: str) -> str:
    """The unambiguous spelling of a control tag: U+200B right after its first character."""
    return text[0] + ZWSP + text[1:] if text else text


def reserved_id(tok):
    """The one id whose text is the marker, or None when the tokenizer spells it."""
    try:
        ids = tok.encode(THINK_END, parse_special=True)
    except Exception:
        return None
    return ids[0] if len(ids) == 1 else None


def _event_cls(parser_cls):
    """The Event class this parser emits, found through its bases (tests subclass the parser)."""
    for k in getattr(parser_cls, "__mro__", [parser_cls]):
        mod = sys.modules.get(getattr(k, "__module__", ""))
        ev = getattr(mod, "Event", None) if mod is not None else None
        if ev is not None:
            return ev
    return None


def _keep_reasoning(parser, evs, event_cls):
    """Undo a split caused by a QUOTED marker: whatever followed it is still reasoning."""
    if any(getattr(e, "kind", "") == "tool_call" for e in evs):
        _note("strict: left alone - this split released tool calls, which cannot be un-emitted")
        return evs
    if event_cls is None:
        _note("strict: left alone - the Event class could not be resolved")
        return evs
    parser.state = "reasoning"
    marker = escaped(THINK_END)
    out, inserted = [], False
    for e in evs:
        if getattr(e, "kind", "") == "content":
            if not inserted:
                out.append(event_cls("reasoning", marker))
                inserted = True
            out.append(event_cls("reasoning", e.text))
        else:
            out.append(e)
    if not inserted:                    # the marker ended the delta: nothing followed it
        out.append(event_cls("reasoning", marker))
    _note("strict: the rest of this turn stays REASONING; the quoted marker is re-inserted as %r", marker)
    return out


def install(server_module, tok, mode: str | None = None):
    """Patch Detokenizer.push (records the id) and OutputParser.feed (logs the split; strict undoes it)."""
    mode = (mode or os.environ.get("STRATA_THINK_SPLIT", "off")).strip().lower()
    if mode not in ("log", "strict"):
        return None
    detok_cls = server_module.Detokenizer
    parser_cls = server_module.OutputParser
    if getattr(detok_cls.push, "_think_split", False) or getattr(parser_cls.feed, "_think_split", False):
        return None
    rid = reserved_id(tok)
    _note("ENABLED (%s; the pack's reserved marker id is %s)", mode, rid)

    orig_push = detok_cls.push

    def push(self, t):
        _tls.last = t
        return orig_push(self, t)

    orig_feed = parser_cls.feed
    event_cls = _event_cls(parser_cls)

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
            if mode == "strict" and rid is not None and t != rid:
                evs = _keep_reasoning(self, evs, event_cls)
        return evs

    push._think_split = True
    feed._think_split = True
    detok_cls.push = push
    parser_cls.feed = feed
    return push
