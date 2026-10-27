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
DEFAULT_THINK = (248068, 248069)        # think open/close (token_type.json: 4 = USER_DEFINED)
ZWSP = "\u200b"

_WHITESPACE = " \t\r\n\f\v"
# rule 2: a backtick counts as OPENING when the character before it is one of these
_OPENERS = (" ", "\t", "\n", "(", "[", ":", ",", "-", "=", ";")

# --------------------------------------------------------------------------------------
# The predicate is FOUR rules, and they are not equally trustworthy, so each one is
# addressable by id (and by name).  --control-span-rules / STRATA_CONTROL_SPAN_RULES picks
# which run; DEFAULT_RULES leaves rule 1 OFF (see _rule_quote for why).
# --------------------------------------------------------------------------------------
RULE_QUOTE, RULE_DANGLING, RULE_HALF_TAG, RULE_CODE = 1, 2, 3, 4
ALL_RULES = (RULE_QUOTE, RULE_DANGLING, RULE_HALF_TAG, RULE_CODE)
DEFAULT_RULES = (RULE_DANGLING, RULE_HALF_TAG, RULE_CODE)      # rule 1 off by default

RULE_NAMES = {RULE_QUOTE: "quote", RULE_DANGLING: "dangling", RULE_HALF_TAG: "half-tag",
              RULE_CODE: "code"}


def rule_name(rule_id):
    return RULE_NAMES.get(int(rule_id), str(rule_id))


def parse_rules(spec, default=DEFAULT_RULES):
    """Turn a config value into a sorted tuple of rule ids.

    Accepts ids ("2,3,4"), names ("dangling,half-tag,code"), the words "all" and "none", and
    a "-" prefix to subtract from the default ("all,-1", or just "-1").  None or "" means the
    default.  An unknown token raises, so a typo in a config file fails loudly at start
    instead of quietly running a different predicate.
    """
    if spec is None:
        return tuple(default)
    if not isinstance(spec, str):
        spec = ",".join(str(x) for x in spec)   # a tuple/list/set of ids, straight from code
    toks = [t.strip().lower() for t in str(spec).replace(" ", ",").split(",") if t.strip()]
    if not toks:
        return tuple(default)
    by_name = {v: k for k, v in RULE_NAMES.items()}
    out, subtract = set(), set()
    for t in toks:
        neg = t.startswith("-")
        t2 = t[1:].strip() if neg else t
        if t2 in ("all", "*"):
            base = set(ALL_RULES)
        elif t2 in ("none", "off"):
            base = set()
        elif t2.isdigit():
            rid = int(t2)
            if rid not in RULE_NAMES:
                raise ValueError("unknown control-span rule id %r (valid: 1-4, all, none)" % t2)
            base = {rid}
        elif t2 in by_name:
            base = {by_name[t2]}
        else:
            raise ValueError("unknown control-span rule %r (valid: 1-4, %s, all, none)"
                             % (t2, ", ".join(sorted(by_name))))
        (subtract if neg else out).update(base)
    if not out and subtract:
        out = set(default)          # a bare "-1" means "the default, minus rule 1"
    return tuple(sorted(out - subtract))


def _fmt_rules(rules):
    return "[" + ",".join("%d:%s" % (r, rule_name(r)) for r in sorted(rules)) + "]"


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
    fence = None          # the fence character while inside a ``` block (backticks only)
    inline = False
    while i < n:
        c = text[i]
        if c != "`":
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


def _rule_quote(tail):
    """Rule 1 - an odd number of double quotes in the tail.  OFF by default.

    This is the rule that leaked a control tag into a finished answer (2026-10-27), and the
    fault is structural, not a wrong window size.  Quote parity is a property of a WHOLE
    sequence: with s(i) = parity over text[0..i], the target is s(n) == 1, but a slice
    computes s(n) XOR s(n-window).  The two agree only when s(n-window) == 0 - when the
    window boundary happens to fall outside any quote - and the boundary state is exactly
    what the slice throws away.  Measured on the real case: opener 4645, closer 4661, window
    opened at 4653, so the window held the closer but not the opener: odd, so "inside a
    quote", while the turn (6 quotes) was closed.  Widen the window and the error just moves.

    It fails the other way too: a quote opened before the window plus two quotes inside it
    reads even, so a genuine mid-quote stop goes unrescued.

    Whole-turn parity is the correct STATE but the wrong INFERENCE - prose is not a
    well-formed depth-1 bracket language (quotes inside code spans, JSON, tables), and one
    lone quote 4,000 characters back keeps the guard blocking forever ("too sticky").  A
    truncation risk depends on the state AND the distance since the opener; neither rule as
    written carries the second term.  Re-enable with --control-span-rules all for the
    llama.cpp predicate 1:1.
    """
    return tail.count('"') % 2 == 1


def _rule_dangling(tail):
    """Rule 2 - the tail ends on '|' or on an opening backtick (trailing whitespace ignored).

    '>' is deliberately NOT an opener: a quoted, escaped turn-end tag is a legitimate ending.
    Genuinely local - the evidence is an unfinished token, so it sits in the last characters.
    """
    n = len(tail)
    while n > 0 and tail[n - 1] in _WHITESPACE:
        n -= 1
    if n == 0:
        return False
    last = tail[n - 1]
    if last == "|":
        return True
    if last == "`":
        prev = tail[n - 2] if n >= 2 else " "
        return prev in _OPENERS
    return False


def _rule_half_tag(tail):
    """Rule 3 - a '<' with no '>' and no whitespace after it: the model is mid-tag.

    Also genuinely local: a half-typed tag cannot be longer than the window.
    """
    lt = tail.rfind("<")
    if lt == -1:
        return False
    rest = tail[lt + 1:]
    return ">" not in rest and not any(c in _WHITESPACE for c in rest)


def _rule_set(rules):
    """Normalise ids, names or a config string to a set of rule ids."""
    return set(parse_rules(rules))


def inside_span(text: str, window: int = 32, scan: int | None = None, rules=None) -> bool:
    """True when the model looks cut off inside a quote, a code span or a code block.

    `text` is the recent generated text (up to `scan` characters).  Rules 2 and 3 stay
    deliberately local - they look at the last `window` characters only, because they detect
    unfinished TOKENS and that evidence cannot be far from the end.  Rule 4 cannot be local:
    a fenced block is longer than any tail, so it reads the whole `scan` window.  Rule 1 is
    neither: it is a nesting STATE, its evidence can be arbitrarily far back, and a slice of
    it is meaningless - which is why it is off by default (see _rule_quote).

    `rules` selects which run: ids/names, a config string, or None for DEFAULT_RULES.
    """
    rules = DEFAULT_RULES if rules is None else _rule_set(rules)
    tail = text[-window:]
    if RULE_QUOTE in rules and _rule_quote(tail):
        return True
    if RULE_DANGLING in rules and _rule_dangling(tail):
        return True
    if RULE_HALF_TAG in rules and _rule_half_tag(tail):
        return True
    # Rule 4: an OPEN code block or code span - the case a tail-local predicate cannot see:
    # the model stopped inside a fenced block, or after an inline span it opened earlier and
    # left open.  Parity is only meaningful over the whole turn, so `scan` defaults to 0 = no
    # limit; a positive value caps the work for very long outputs and accepts that artifact.
    if RULE_CODE in rules and _open_code_span(text if not scan else text[-scan:]):
        return True

    return False


def escape_text(raw: str) -> str:
    """The unambiguous spelling: U+200B right after the first character."""
    if not raw:
        return raw
    return raw[0] + ZWSP + raw[1:]


class Guard:
    CONTROL_TYPE = 3          # token_type.json: 3 = CONTROL (<|im_start|>, <|im_end|>, <|endoftext|>)

    def __init__(self, tok, eos_ids, types=None, window=32, max_rescues=5, log=sys.stderr,
                 scan=0, think_ids=None, rules=None):
        self.tok, self.eos_ids, self.types = tok, set(eos_ids), types
        self.scan = max(0, scan)          # 0 = the whole turn (see inside_span rule 4)
        # the token to spell out when the stream itself carried no eos (the resident engine returns on
        # DONE without yielding it): the configured turn-end id, i.e. the last one listed.
        ordered = [int(x) for x in eos_ids]
        self.turn_id = ordered[-1] if ordered else 248046
        self.window, self.max_rescues, self.log = window, max_rescues, log
        self.rules = parse_rules(rules)          # None -> DEFAULT_RULES (rule 1 off)
        # the thinking-block markers (USER_DEFINED tokens, so they appear in the stream as real ids):
        # a stop while the most recent of the two is the OPEN tag is a truncation, not an end of turn.
        t = tuple(int(x) for x in (think_ids or DEFAULT_THINK))
        self.think_open, self.think_close = t[0], t[1]

    def tail_text(self, ids, keep=64):
        """Decode the recent ids the way llama.cpp builds its tail: CONTROL tokens render as
        the empty piece (common_token_to_piece(..., special=false)).  Rendering them as text
        would let the stop token itself close the span the predicate looks for."""
        recent = [t for t in ids[-keep:] if isinstance(t, int)]   # None heartbeats are not ids
        if self.types is not None:
            recent = [t for t in recent
                      if t >= len(self.types) or self.types[t] != self.CONTROL_TYPE]
        return self.tok.decode(recent)

    def answer_text(self, produced):
        """Decode the ANSWER only - everything after the most recent think-CLOSE tag, with
        no id cap.  Two artifacts this avoids, both measured on 2026-10-10:

        (1) tail_text caps at its last `keep` ids (64).  A window that starts INSIDE a
            code block reads that block's CLOSING fence as an OPENING one, so a complete
            answer looks truncated: for the answer that triggered this, keep=64 said True
            and keep=128, keep=256 and the full length all said False.  inside_span's own
            rule 4 says parity is only meaningful over the whole turn, so the span check
            must not be handed a 64-id slice.
        (2) The reasoning block is prose ABOUT code - inline backticks, nested quoting,
            runs of three from quoting quoted text - and its backtick parity says nothing
            about whether the answer is complete.  A stop while thinking is still open is
            already handled by _thinking_open, which is id-based and stays on the full
            context."""
        k = -1
        for i, t in enumerate(produced):
            if t == self.think_close:
                k = i
        tail = produced[k + 1:]
        return self.tail_text(tail, keep=max(1, len(tail)))

    def _thinking_open(self, ids):
        """True when the most recent think tag in `ids` is the OPEN tag: the model is still
        inside the thinking block, so a stop here is a truncation, not an end of turn.

        `ids` must be the FULL context (prompt + generated), not just the generated ids:
        the chat template injects the think-OPEN tag into the PROMPT (the model does not
        generate it), so scanning only the generated ids is blind to it. Once the model has
        finished thinking it has emitted a CLOSE tag, which is more recent than the prompt's
        open tag, so this correctly returns False for a turn that is already answering.

        Id-based, not text-based: a model QUOTING the tag spells it as character pieces and
        never emits the single reserved id, so this cannot fire on quoted text.
        """
        for t in reversed(ids):
            if t == self.think_open or t == self.think_close:
                return t == self.think_open
        return False

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
        rescued_at = None       # len(produced) right after the last escape (one-shot per span)
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

            recent = self.answer_text(produced)[-self.scan:]
            tail = recent[-self.window:]
            # (b): the stream ended by itself, so a stop short of the cap is an end-of-turn.
            ended_by_stream = stop_id is None and len(produced) + 1 < max_new
            stopping = stop_id is not None or ended_by_stream
            span = inside_span(recent, window=self.window, scan=self.scan,
                                 rules=self.rules)
            thinking = self._thinking_open(ids + produced)
            if not (stopping and (span or thinking)):
                if stop_id is not None:
                    yield stop_id            # a genuine end of turn: let it end as "stop"
                return

            # ONE-SHOT PER SPAN: the escape just injected is a complete rendering of what
            # the model wanted to print.  A stop with nothing new in between is the same
            # intent expressed again - honour it instead of appending another copy (which
            # is how one quoted marker used to become five).
            if rescued_at is not None and len(produced) == rescued_at:
                self._note("stop %s, but nothing was written after rescue %d - letting it end",
                           "inside an open thinking block" if thinking else "inside a span",
                           rescues)
                if stop_id is not None:
                    yield stop_id
                return
            if rescues >= self.max_rescues:
                self._note("stop %s, but the rescue cap (%d) is reached - letting it end",
                           "inside an open thinking block" if thinking else "inside a span",
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
            self._note("rescue %d: the turn ended %s (%s, tail=%r) - continuing with the "
                       "escaped spelling of token %d as %s", rescues,
                       "inside an open thinking block" if thinking else "inside a span",
                       "the engine yielded the turn-end id" if stop_id is not None
                       else "the stream ended short of the cap, so a turn-end token",
                       tail[-24:], target, esc)
            for t in esc:                    # the caller sees the escaped text
                produced.append(t)
                yield t
            rescued_at = len(produced)      # snapshot AFTER the escape: 'nothing new' is measured from here

def install(server_module, tok, types=None, eos_ids=None, force=False):
    """Monkey-patch StrataEngine.generate. Returns the guard object (or None if off)."""
    if not force and not _env_flag("STRATA_CONTROL_SPAN_GUARD"):
        return None
    eos = eos_ids or tuple(int(x) for x in
                           (os.environ.get("STRATA_CONTROL_SPAN_EOS") or
                            ",".join(str(e) for e in DEFAULT_EOS)).split(",") if x.strip())
    think = tuple(int(x) for x in
                  (os.environ.get("STRATA_CONTROL_SPAN_THINK") or
                   ",".join(str(x) for x in DEFAULT_THINK)).split(",") if x.strip())
    guard = Guard(tok, eos, types=types,
                  window=_env_int("STRATA_CONTROL_SPAN_WINDOW", 32),
                  max_rescues=_env_int("STRATA_CONTROL_SPAN_RESCUES", 5),
                  scan=_env_int("STRATA_CONTROL_SPAN_SCAN", 0),
                  think_ids=think,
                  rules=parse_rules(os.environ.get("STRATA_CONTROL_SPAN_RULES")))
    cls = server_module.StrataEngine
    original = cls.generate

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        try:
            return guard.guard(lambda *a: original(self, *a), ids, max_new, sampling, cancel, embeddings)
        except Exception as exc:                             # never break generation
            guard._note("disabled for this request: %s", exc)
            return original(self, ids, max_new, sampling, cancel, embeddings)

    cls.generate = generate
    guard._note("ENABLED (eos ids %s, think ids %s, window %d, max rescues %d, rules %s)",
                sorted(eos), sorted(think), guard.window, guard.max_rescues,
                _fmt_rules(guard.rules))
    return guard
