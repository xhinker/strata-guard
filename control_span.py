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


def _open_code_span(text: str) -> bool:
    """True when a code block or inline code span is still open at the end of `text`.

    Rule 2 below only catches a span that is open AT THE END of the reply (the tail ends on an
    opening backtick).  A model writing markdown usually stops somewhere else: inside a fenced
    block, or with an inline span opened earlier and never closed.  Both are truncations, and
    neither was covered - measured 2026-10-08, where an unclosed ``` block came back identical
    from a guarded and an unguarded server.

    A single backtick OPENS a span only where the ported predicate says it opens one: at the
    start of the text or after one of _OPENERS.  Anywhere else ("done`") it reads as a closing
    backtick, so it must not open anything - that case is a test on purpose.
    """
    i, n = 0, len(text)
    fence = None          # the fence character while inside a ``` / ~~~ block
    inline = False
    while i < n:
        c = text[i]
        if c not in ("`", "~"):
            i += 1
            continue
        j = i
        while j < n and text[j] == c:
            j += 1
        prev = text[i - 1] if i else " "
        if j - i >= 3:                                  # a fence: opens or closes a block
            if fence is None:
                fence = c
            elif c == fence:
                fence = None
        elif fence is None:                             # inline backticks, ignored inside a block
            if inline:
                inline = False                          # closing
            elif prev in _OPENERS:
                inline = True                           # opening (start of text counts as a space)
        i = j
    return fence is not None or inline


def inside_span(text: str, window: int = 32, scan: int | None = None) -> bool:
    """True when the model looks cut off inside a quote, a code span or a code block.

    `text` is the recent generated text (up to `scan` characters).  Rules 1-3 stay deliberately
    local - they look at the last `window` characters only, because global quote parity was
    "too sticky": one lone quote far back kept the guard blocking forever.  Rule 4 cannot be
    local: a fenced block is longer than any tail, so it reads the whole `scan` window.
    """
    tail = text[-window:]
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

    # (4) an OPEN code block or code span - the case a tail-local rule cannot see: the model
    #     stopped inside a fenced block, or after an inline span it opened earlier and left open.
    #
    # This one cannot be windowed.  A window that starts INSIDE a code block sees that block's
    # closing fence as an opening one, and reports a perfectly closed answer as an open span
    # (measured: the last 200 characters of a complete FastAPI answer return True).  Parity is
    # only meaningful over the whole turn, so `scan` defaults to 0 = no limit; a positive value
    # caps the work for very long outputs and accepts that artifact.
    if _open_code_span(text if not scan else text[-scan:]):
        return True

    return False


def escape_text(raw: str) -> str:
    """The unambiguous spelling: U+200B right after the first character."""
    if not raw:
        return raw
    return raw[0] + ZWSP + raw[1:]


class Guard:
    CONTROL_TYPE = 3          # token_type.json: 3 = CONTROL (<|im_start|>, <|im_end|>, <|endoftext|>)

    def __init__(self, tok, eos_ids, types=None, window=32, max_rescues=2, log=sys.stderr,
                 scan=0):
        self.tok, self.eos_ids, self.types = tok, set(eos_ids), types
        self.scan = max(0, scan)          # 0 = the whole turn (see inside_span rule 4)
        # the token to spell out when the stream itself carried no eos (the resident engine returns on
        # DONE without yielding it): the configured turn-end id, i.e. the last one listed.
        ordered = [int(x) for x in eos_ids]
        self.turn_id = ordered[-1] if ordered else 248046
        self.window, self.max_rescues, self.log = window, max_rescues, log

    def tail_text(self, ids, keep=64):
        """Decode the recent ids the way llama.cpp builds its tail: CONTROL tokens render as
        the empty piece (common_token_to_piece(..., special=false)).  Rendering them as text
        would let the stop token itself close the span the predicate looks for."""
        recent = [t for t in ids[-keep:] if isinstance(t, int)]   # None heartbeats are not ids
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
        """Wrap one generation: same ids out, plus a rescue when the stop is inside a span.

        A turn can end two ways and the check belongs in different places for each.

        (a) THE ENGINE YIELDS THE TURN-END ID.  This is what the current engine does, and it
            is the shape that made the guard dead: the consumer (serve/server.py:3297,
            `if t in self.stop_ids:  finish = "stop"; raw_ids.append(t); break`) stops pulling
            the moment it sees that id, which closes this generator at its `yield` - so any
            check written after the loop never runs.  The decision therefore has to be made
            BEFORE the id is handed over: withhold it, rescue if the tail says the model was
            quoting something, and only hand it over once the turn really is over (so the
            caller still gets finish_reason "stop").
        (b) THE ENGINE RETURNS WITHOUT YIELDING A TURN-END ID (older builds: it streams `T <id>`
            lines then a bare `DONE`).  Here the end of the stream IS the end of the turn, so
            the check belongs after the loop, and "ended short of the cap" is the signal.
        """
        produced: list[int] = []
        rescues = 0
        while True:
            budget = max(1, max_new - len(produced))
            gen = engine(ids + produced, budget, sampling, cancel, embeddings)
            stop_id = None
            for t in gen:
                if t is None:
                    # The engine emits None as a heartbeat while it is quiet (serve/server.py
                    # does the same: pass it on, never treat it as a generated token).
                    yield t
                    continue
                if t in self.eos_ids:
                    stop_id = t
                    break                    # withhold it: yielding it would end the consumer
                produced.append(t)
                yield t
            if stop_id is not None:
                try:
                    gen.close()              # the same abandonment the server's stop path does
                except Exception:
                    pass

            if cancel.is_set():
                if stop_id is not None:
                    yield stop_id
                return
            if not produced and stop_id is None:
                return

            recent = self.tail_text(produced)[-self.scan:]
            tail = recent[-self.window:]
            # (b): the stream ended by itself, so a stop short of the cap is an end-of-turn.
            ended_by_stream = stop_id is None and len(produced) + 1 < max_new
            stopping = stop_id is not None or ended_by_stream
            if not (stopping and inside_span(recent, window=self.window, scan=self.scan)):
                if stop_id is not None:
                    yield stop_id            # a genuine end of turn: let it end as "stop"
                return
            if rescues >= self.max_rescues:
                self._note("stop inside a span, but the rescue cap (%d) is reached - letting it end",
                           self.max_rescues)
                if stop_id is not None:
                    yield stop_id
                return

            target = stop_id if stop_id is not None else self.turn_id
            esc = self.escape_ids(target)
            if not esc:
                if stop_id is not None:
                    yield stop_id
                return
            rescues += 1
            self._note("rescue %d: the turn ended inside a span (%s, tail=%r) - continuing with the "
                       "escaped spelling of token %d as %s", rescues,
                       "the engine yielded the turn-end id" if stop_id is not None
                       else "the stream ended short of the cap, so a turn-end token",
                       tail[-24:], target, esc)
            for t in esc:                    # the caller sees the escaped text
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
                  max_rescues=_env_int("STRATA_CONTROL_SPAN_RESCUES", 2),
                  scan=_env_int("STRATA_CONTROL_SPAN_SCAN", 0))
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
