#!/usr/bin/env python3
"""serve_with_guard.py - run Strata's server with the control-span rescue guard installed.

Usage (same arguments as serve/server.py, plus the guard's own):

    python3 Strata-guard/serve_with_guard.py --engine strata \
        --config strata-3090x3-iq3s.json --port 8081 --control-span-guard on

The guard is OFF unless STRATA_CONTROL_SPAN_GUARD=on or --control-span-guard on.
No upstream file is modified; the patch exists only for this process.

    --control-span-guard on|off     enable/disable (default: off, or the env var)
    --control-span-window N         characters inspected at the end of the text (32)
    --control-span-rescues N        at most N rescues per request (2)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

STRATA = os.environ.get("STRATA_ROOT", "/mnt/data_2t/github_repos/Strata")
sys.path.insert(0, STRATA)
sys.path.insert(0, str(Path(STRATA) / "tools"))   # strata_tokenizer lives here (tools/strata_tokenizer.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import control_span  # noqa: E402
import stats_from_log  # noqa: E402


def _take_flag(argv, name, cast=str, default=None):
    """Pull `--name value` (or `--name=value`) out of argv; return (argv, value)."""
    out, value, i = [], default, 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            if i + 1 < len(argv):
                value = cast(argv[i + 1]); i += 2; continue
            value = default; i += 1; continue
        if a.startswith(name + "="):
            value = cast(a.split("=", 1)[1]); i += 1; continue
        out.append(a); i += 1
    return out, value


def _tokenizer_from_pack(pack: Path):
    """Build the same tokenizer the server builds (server.py, main()); return (tok, types)."""
    import strata_tokenizer as ST
    vocab = json.loads((pack / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (pack / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((pack / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types), types


def _ensure_venv() -> None:
    """strata_tokenizer is a compiled module that lives in Strata's venv: re-exec there once."""
    try:
        import strata_tokenizer  # noqa: F401
        return
    except ImportError:
        pass
    for cand in (Path(STRATA) / ".venv" / "bin" / "python",
                 Path(STRATA) / ".venv" / "bin" / "python3"):
        if cand.exists() and Path(sys.executable).resolve() != cand.resolve():
            os.execv(str(cand), [str(cand), str(Path(__file__).resolve()), *sys.argv[1:]])
    print("[control-span] strata_tokenizer is not importable and no venv python was found",
          file=sys.stderr)


def _config_log(argv) -> str | None:
    """The engine log the config names (`log`), which is also what the start narrator reads."""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            p = Path(argv[i + 1])
            if not p.is_absolute():
                p = Path(STRATA) / p
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
            log = cfg.get("log")
            if log and not Path(log).is_absolute():
                log = str(Path(STRATA) / log)
            return log
    return None


def main() -> int:
    _ensure_venv()
    argv = list(sys.argv[1:])
    argv, guard_flag = _take_flag(argv, "--control-span-guard", str, None)
    argv, window = _take_flag(argv, "--control-span-window", int, None)
    argv, rescues = _take_flag(argv, "--control-span-rescues", int, None)
    if guard_flag is not None:
        os.environ["STRATA_CONTROL_SPAN_GUARD"] = guard_flag
    if window is not None:
        os.environ["STRATA_CONTROL_SPAN_WINDOW"] = str(window)
    if rescues is not None:
        os.environ["STRATA_CONTROL_SPAN_RESCUES"] = str(rescues)

    # the pack path: from --pack on the command line, else the model config's args
    pack = None
    for i, a in enumerate(argv):
        if a == "--pack" and i + 1 < len(argv):
            pack = Path(argv[i + 1])
    if pack is None:
        for i, a in enumerate(argv):
            if a == "--config" and i + 1 < len(argv):
                cfg_path = Path(argv[i + 1])
                if not cfg_path.is_absolute():
                    cfg_path = Path(STRATA) / cfg_path
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    args = cfg.get("args") or []
                    if "--pack" in args:
                        pack = Path(args[args.index("--pack") + 1])
                except Exception:
                    pack = None
    tok = None
    types = None
    if pack is not None:
        tdir = pack if (pack / "vocab.json").exists() else pack / "tokenizer"
        try:
            tok, types = _tokenizer_from_pack(tdir)
        except Exception as exc:
            print(f"[control-span] could not build the tokenizer from {tdir}: {exc}", file=sys.stderr)

    sys.argv = [sys.argv[0]] + argv
    from serve import server as S       # importing does not start the server
    if tok is None:
        print("[control-span] no pack/tokenizer - the guard stays off", file=sys.stderr)
    else:
        control_span.install(S, tok, types=types)
    # the Monitor's per-request figures: fill engine.last from the engine's own summary line.
    # Installed AFTER the guard, so its finally runs last (after any rescue has finished).
    log_path = os.environ.get("STRATA_STATS_LOG") or _config_log(argv)
    if log_path:
        stats_from_log.install(S, log_path)
    else:
        print("[strata-stats] no log path found - the Monitor columns stay empty", file=sys.stderr)
    return S.main()


if __name__ == "__main__":
    raise SystemExit(main())
