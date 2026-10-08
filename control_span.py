"""control_span.py - control-tag rescue guard for Strata's Python server layer.

WHY HERE
--------
The Strata *engine* works in token ids only (it streams `T <id>` lines and has no
detokenizer), and it decides the end of a turn with a post-sample check on its
eos ids (generate.cpp: eos_ids = {248044, 248046}).  A mask-style guard there
would need a precomputed per-id table and edits at three places inside a 10k-line
GPU file.  The Python server, by contrast, owns the tokenizer and sees the real
text - so the same semantics (llama.cpp's "substitution" arm) can be implemented
exactly, with no upstream edits at all.

WHAT IT DOES
------------
Qwen spells its own control tags with characters that ARE reserved vocabulary
entries (`<|im_end|>` is one id and is also the end-of-turn token).  When the
model is inside an open quote/code span and emits that id, it almost always meant
to PRINT the tag - and the turn dies mid-sentence.  When the engine stops and the
predicate below says "inside a span", this guard appends the ESCAPED spelling of
the stop token (U+200B inserted) to the conversation and asks the engine to keep
going, so the caller receives a complete answer instead of a truncated one.

Predicate ported 1:1 from the llama.cpp fork (common/control-span-predicate.h).
Enabled only when STRATA_CONTROL_SPAN_GUARD=on or --control-span-guard on.
"""

from __future__ import annotations

import os
import re
import sys
import types

DEFAULT_EOS = (248044, 248046)          # <|endoftext|>, <|im_end|>   (engine default)
ZWSP = "\u200b"

_WHITESPACE = " \t\r\n\f\v"
# rule 2: a backtick counts as OPENING when the character before it is one of these
_OPENERS = (" ", "\t", "\n", "(", "[", ":", ",", "-", "=", ";")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "on", "true", "yes", "enabled")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def inside_span(tail: str) -> bool:
    """True when the tail looks like the model is inside a quote/code span."""
    # (1) odd number of double quotes INSIDE the window only (global parity was
    #     "too sticky": one lone quote far back kept the guard blocking forever)
    if tail.count('"') % 2 == 1:
        return True

    # (2) the tail ends on '|' or on an opening backtick (trailing whitespace ignored;
    #     '>' is deliberately NOT an opener - "`<|im_end|>`" is a legitimate ending)
    n = len(tail)
    while n > 0 and tail[n - 1] in _WHITESPACE:
        n -= 1
    if n > 0:
        last = tail[n - 1]
        if last == "|":
            return True
        if last == "`":
            prev = tail[n - 2] if n >= 2 else " "
            if prev in _OPENERS:
                return True

    # (3) mid-tag: a '<' with no '>' and no whitespace after it
    lt = tail.rfind("<")
    if lt != -1:
        rest = tail[lt + 1:]
        if ">" not in rest and not any(c in _WHITESPACE for c in rest):
            return True

    return False


def escape_text(raw: str) -> str:
    """The unambiguous spelling: U+200B right after the first character."""
    if not raw:
        return raw
    return raw[0] + ZWSP + raw[1:]


class Guard:
    CONTROL_TYPE = 3          # token_type.json: 3 = CONTROL (<|im_start|>, <|im_end|>, <|endoftext|>)

    def __init__(self, tok, eos_ids, types=None, window=32, max_rescues=2, log=sys.stderr):
        self.tok, self.eos_ids, self.types = tok, set(eos_ids), types
        # the token to spell out when the stream itself carried no eos (the resident engine returns on
        # DONE without yielding it): the configured turn-end id, i.e. the last one listed.
        ordered = [int(x) for x in eos_ids]
        self.turn_id = ordered[-1] if ordered else 248046
        self.window, self.max_rescues, self.log = window, max_rescues, log

    def tail_text(self, ids, keep=64):
        """Decode the recent ids the way llama.cpp builds its tail: CONTROL tokens render as
        the empty piece (common_token_to_piece(..., special=false)).  Rendering them as text
        would let the stop token itself close the span the predicate looks for."""
        recent = ids[-keep:]
        if self.types is not None:
            recent = [t for t in recent
                      if t >= len(self.types) or self.types[t] != self.CONTROL_TYPE]
        return self.tok.decode(recent)

    def _note(self, msg, *a):
        print("[control-span] " + (msg % a if a else msg), file=self.log, flush=True)

    def escape_ids(self, token_id: int) -> list[int]:
        """ids that spell `token_id`'s text escaped so it cannot re-tokenise as one id."""
        raw = self.tok.decode([token_id])
        if not raw:
            return []
        out = self.tok.encode(escape_text(raw), parse_special=False)
        if out and out[0] == token_id:                       # the escape did not break it
            stronger = "".join(c + ZWSP for c in raw)
            out = self.tok.encode(stronger, parse_special=False)
        return [t for t in out if t != token_id]

    def guard(self, engine, ids, max_new, sampling, cancel, embeddings=None):
        """Wrap one generation: same ids out, plus a rescue when the stop is inside a span."""
        produced: list[int] = []
        rescues = 0
        while True:
            for t in engine(ids + produced, max(1, max_new - len(produced)), sampling, cancel, embeddings):
                produced.append(t)
                yield t
            if cancel.is_set() or not produced:
                return
            tail = self.tail_text(produced)[-self.window:]
            # How the turn ended matters.  The resident engine streams `T <id>` lines and a bare
            # `DONE`; it does NOT yield the eos token (server.py: ~1370), so an eos test alone never
            # fires in production.  The reliable signal is: the stream ended SHORT of the cap and the
            # request was not cancelled -> the engine stopped on an end-of-turn token.  The eos test is
            # kept as a bonus for callers that do yield it (the MockEngine, for one).
            ended_early = len(produced) + 1 < max_new and not cancel.is_set()
            ended_on_eos = bool(produced) and produced[-1] in self.eos_ids
            if not ((ended_early or ended_on_eos) and inside_span(tail)):
                return
            if rescues >= self.max_rescues:
                self._note("stop inside a span, but the rescue cap (%d) is reached - letting it end",
                           self.max_rescues)
                return
            target = produced[-1] if ended_on_eos else self.turn_id
            esc = self.escape_ids(target)
            if not esc:
                return
            rescues += 1
            self._note("rescue %d: the turn ended inside a span (%s, tail=%r) - continuing with the "
                       "escaped spelling of token %d as %s", rescues,
                       "eos token" if ended_on_eos else "short of the cap, so a turn-end token",
                       tail[-24:], target, esc)
            for t in esc:                                    # the caller sees the escaped text
                produced.append(t)
                yield t


def install(server_module, tok, types=None, eos_ids=None, force=False):
    """Monkey-patch StrataEngine.generate. Returns the guard object (or None if off)."""
    if not force and not _env_flag("STRATA_CONTROL_SPAN_GUARD"):
        return None
    eos = eos_ids or tuple(int(x) for x in
                           (os.environ.get("STRATA_CONTROL_SPAN_EOS") or
                            ",".join(str(e) for e in DEFAULT_EOS)).split(",") if x.strip())
    guard = Guard(tok, eos, types=types,
                  window=_env_int("STRATA_CONTROL_SPAN_WINDOW", 32),
                  max_rescues=_env_int("STRATA_CONTROL_SPAN_RESCUES", 2))
    cls = server_module.StrataEngine
    original = cls.generate

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            return guard.guard(lambda *a: original(self, *a), ids, max_new, sampling, cancel, embeddings)
        except Exception as exc:                             # never break generation
            guard._note("disabled for this request: %s", exc)
            return original(self, ids, max_new, sampling, cancel, embeddings)

    cls.generate = generate
    guard._note("ENABLED (eos ids %s, window %d, max rescues %d)",
                sorted(eos), guard.window, guard.max_rescues)
    return guard
