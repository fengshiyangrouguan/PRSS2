"""Non-invasive utility attribution for the external-agents telemetry (§7.6).

Split out of the former single-module ``telemetry.py`` (mechanical move; see
the package ``__init__`` for the full design contract).
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "attribute_utilities",
]

# ---------------------------------------------------------------------------
# Utility attribution — non-invasive, honestly bounded (plan §7.6)
# ---------------------------------------------------------------------------

#: Tokens that, on their own, would false-positive a helper named ``solve`` /
#: ``run`` (the canonical solver entry points the agent invokes for unrelated
#: reasons). Bash matching uses exact-word boundaries and python matching
#: requires a ``helpers/``-prefixed token, so these never spuriously match.
_GENERIC_HELPER_NAMES = frozenset({"solve", "run", "main", "test"})


def _bash_helper_file_used(name: str, command: str) -> bool:
    """Whether the advertised ``source ./helpers/<name>.sh`` file form is invoked.

    This is the ``helpers/``-anchored filename branch of :func:`_bash_token_used`,
    extracted so it can be applied to GENERIC helper names too: the ``helpers/``
    anchor makes the match generic-SAFE (a bare ``run`` command cannot
    false-positive on ``helpers/run.sh``), unlike the bare word-boundary rule in
    :func:`_bash_token_used` which is generic-UNsafe and so stays gated off the
    generic names. The ``source`` line itself registers here — the bare
    word-boundary rule rejects it because ``<name>`` is followed by ``.`` (the
    ``(?![\\w.])`` lookahead).
    """
    if not name or not command:
        return False
    esc = re.escape(name)
    return re.search(rf"helpers/{esc}\.sh", command) is not None


def _bash_token_used(name: str, command: str) -> bool:
    """Whether bash helper ``name`` is invoked as a command word in ``command``.

    Matches the helper *function/script name* on a word boundary so a helper
    called ``parse`` does not match ``parser`` and a generic ``run`` does not
    match ``run_tests`` (plan §7.6). The name must appear as its own token (the
    command word or a later argument), not as a substring of another identifier.

    The advertised bash form ``source ./helpers/<name>.sh`` (injection.py
    ``_HELPERS_NOTE_BASH``) is matched FIRST via :func:`_bash_helper_file_used`
    (the ``helpers/<name>.sh`` filename) so the ``source`` line itself registers;
    the word-boundary search remains the fallback for the post-source function
    call (``<name> args``). The filename form is ``helpers/``-anchored, so a
    generic name cannot false-positive on it.
    """
    if not name or not command:
        return False
    # Advertised ``source ./helpers/<name>.sh`` form (helpers/-anchored filename).
    if _bash_helper_file_used(name, command):
        return True
    esc = re.escape(name)
    return re.search(rf"(?<![\w.]){esc}(?![\w.])", command) is not None


def _py_helper_used(name: str, command: str) -> bool:
    """Whether python helper ``helpers/<name>.py`` is referenced in ``command``.

    Requires either the staged *filename* (``helpers/<name>.py`` or the
    bash-staged python-lib form ``helpers/_lib_<name>.py``) **or** an import
    token (``from helpers import <name>``, ``import helpers.<name>``,
    ``helpers.<name>(``). The import-token requirement is essential because
    ``from helpers import x`` never contains the filename — without it the helper
    would false-*negative*. The ``helpers/_lib_<name>.py`` form is the advertised
    ``python3 ./helpers/_lib_<name>.py <args>`` bash invocation (injection.py
    ``_HELPERS_NOTE_BASH`` + ``build_python_lib_file``); without it a python
    helper called through the bash entry point would false-*negative*. All
    matched forms carry the ``helpers``-prefix, so a generically named helper
    (``solve``/``run``) cannot false-positive on an unrelated bare
    ``solve``/``run`` command (plan §7.6).
    """
    if not name or not command:
        return False
    esc = re.escape(name)
    patterns = (
        rf"helpers/{esc}\.py",                       # staged filename
        rf"helpers/_lib_{esc}\.py",                  # bash-staged python-lib form
        rf"from\s+helpers\s+import\s+(?:[\w,\s]*\b){esc}\b",  # from helpers import name
        rf"import\s+helpers\.{esc}\b",               # import helpers.name
        rf"helpers\.{esc}\s*\(",                     # helpers.name(
    )
    return any(re.search(p, command) for p in patterns)


def attribute_utilities(
    command_history: Optional[list[str]],
    utilities_available: Optional[list[str]],
    attribution_available: bool,
) -> tuple[Optional[list[str]], dict[str, int]]:
    """Recover which injected utilities were used, non-invasively (plan §7.6).

    Computed purely from the executed ``command_history`` and the list of staged
    ``utilities_available`` — the staged helper source is **never** instrumented
    (that would fork behavior from the builtin path and pollute the archived
    ``solver_lib_*.py``). bash helpers match an exact command-word; python helpers
    match a ``helpers/<name>.py`` filename **or** an import token, which avoids
    both false negatives (``from helpers import x``) and false positives
    (``solve``/``run`` matching unrelated commands).

    The ``None`` vs ``[]`` distinction is load-bearing:

    * ``attribution_available is False`` → returns ``(None, {})`` — the backend
      cannot expose a command stream, so usage is *unmeasurable on this backend*,
      **not** zero. Analysis must treat ``None`` as unknown.
    * ``attribution_available is True`` → returns ``([...], {...})`` where ``[]``
      means "captured, none of the injected utilities were called".

    Args:
        command_history: Ordered commands the agent issued, or ``None``/``[]``.
        utilities_available: Staged helper names (bash function/script names and
            python ``helpers/<name>.py`` basenames). May be ``None``/empty.
        attribution_available: Whether the backend captured a command stream.

    Returns:
        ``(utilities_called, utilities_call_counts)`` where ``utilities_called``
        is a *sorted* list (or ``None`` when unmeasurable) and
        ``utilities_call_counts`` maps each *called* utility to its approximate
        invocation count.
    """
    if not attribution_available:
        # Unmeasurable on this backend — preserve the None-vs-[] distinction.
        return None, {}

    available = list(utilities_available or [])
    if not available:
        return [], {}

    commands = list(command_history or [])
    counts: dict[str, int] = {}
    for cmd in commands:
        if not cmd:
            continue
        for name in available:
            # Try the helpers/-anchored forms first — the python-helper forms
            # (``helpers/<name>.py`` / ``helpers/_lib_<name>.py`` / import token)
            # AND the bash file form (``helpers/<name>.sh``). Both are
            # ``helpers/``-anchored, so they are generic-SAFE: an advertised
            # ``source ./helpers/run.sh`` registers even for a generic name
            # (solve/run/main/test). Only if neither anchored form matched do we
            # fall back to the bare word-boundary bash rule, which is generic-
            # UNsafe and so stays gated off the generic names.
            hit = _py_helper_used(name, cmd) or _bash_helper_file_used(name, cmd)
            if not hit and name not in _GENERIC_HELPER_NAMES:
                hit = _bash_token_used(name, cmd)
            if hit:
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts.keys()), counts
