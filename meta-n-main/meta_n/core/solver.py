"""Layer 1 Solver — LLM-based task solver that produces scripts."""

from __future__ import annotations

import json
import logging
import re

from meta_n.core.llm_client import LLMClient
from meta_n.core.meta_layer import TaskDescription
from meta_n.core.prompts import (
    SOLVER_PROMPT,
    SOLVER_PROMPT_CLASSIFY,
    SOLVER_PROMPT_OPENEVOLVE,
    SOLVER_PROMPT_PYTHON,
)
from meta_n.core.solver_output import validate_solver_output

logger = logging.getLogger(__name__)


def extract_fenced_block(response: str, tags: tuple[str, ...]) -> str | None:
    """First fenced code block matching one of ``tags`` (in order), stripped;
    None if none match. The wildcard tag ``""`` matches any fence and must be
    LAST so labeled blocks win over interstitial prose (see _extract_* below).
    """
    for tag in tags:
        match = re.search(rf"```{tag}\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _balanced_json_object(response: str) -> str | None:
    """First string-aware brace-balanced span that ``json.loads`` to a dict
    containing at least one ``"case_<n>"`` key; None otherwise. Tracks JSON
    in-string / escape state so braces inside quoted label values do not
    miscount depth. O(n) per candidate start; bounded by response length.

    Used by ``extract_case_json`` (behind its ``balanced_fallback`` gate)
    to recover objects the flat ``[^{}]*`` regex cannot match — nested values
    or a ``{``/``}`` inside a label string. Spans that fail ``json.loads`` or
    lack a ``case_<n>`` key are skipped in favour of the next ``{`` start.
    """
    for candidate in re.finditer(r"\{", response):
        start = candidate.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(response)):
            ch = response[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    span = response[start : i + 1]
                    try:
                        obj = json.loads(span)
                    except json.JSONDecodeError:
                        break  # not JSON from this start; try the next '{'
                    if isinstance(obj, dict) and any(
                        re.fullmatch(r"case_\d+", key) for key in obj
                    ):
                        return span
                    break  # valid JSON but wrong shape; try the next '{'
        # Inner loop exhausted without a depth-0 close: unbalanced from this
        # start — fall through to the next candidate '{'.
    return None


def extract_case_json(response: str, *, balanced_fallback: bool = False) -> str | None:
    """Classify extraction chain: fenced -> flat regex -> (``balanced_fallback``
    ON only) balanced scan; None when nothing matches. Shared by
    ``Layer1Solver._extract_json`` and the text-classification live parse site
    (``_coerce_predictions``) so both honour the same F075 ordering contract.
    """
    result = extract_fenced_block(response, ("json", ""))
    if result is not None:
        return result
    match = re.search(r"\{[^{}]*(?:\"case_\d+\"[^{}]*)+\}", response, re.DOTALL)
    if match:
        return match.group(0)
    if balanced_fallback:
        return _balanced_json_object(response)
    return None


class Layer1Solver:
    """LLM-based task solver. Produces bash or Python scripts from task descriptions."""

    # Class-level default (not just an ``__init__`` assignment) so instances
    # built via ``Layer1Solver.__new__`` — a supported construction path in
    # extraction-only contexts — still resolve the flag to OFF.
    balanced_json_fallback: bool = False

    def __init__(
        self,
        llm_client: LLMClient,
        language: str = "bash",
        balanced_json_fallback: bool = False,
    ):
        """
        Args:
            llm_client: LLM client for completions
            language: Solution language — 'bash' or 'python'
            balanced_json_fallback: When True, ``_extract_json`` recovers a
                string-aware brace-balanced JSON object (nested values, braces
                inside label strings) after the fenced and flat-regex
                extractions both fail. Default False — the classify extraction
                path is byte-identical to the flag-less behaviour.
        """
        self.llm_client = llm_client
        self.language = language
        self.balanced_json_fallback = balanced_json_fallback

    async def solve(
        self, task: TaskDescription, additional_context: str = "",
        temperature: float | None = None,
        *,
        seed: int | None = None,
        _verdict: dict | None = None,
    ) -> tuple[str, str, int]:
        """
        Generate a script to solve the task.

        Args:
            task: Task description.
            additional_context: Optional context string injected by upper meta-layers.
            temperature: LLM temperature override (default 0.3 to match production).
            seed: optional per-request CRN seed (paired eval). Forwarded to
                ``llm_client.complete``; a no-op unless the backend honours a
                per-request seed. ``None`` (default) ⇒ existing behaviour
                unchanged.
            _verdict: Optional out-parameter filled with this call's validity
                verdict -- ``ok``, ``reason`` (see ``solver_output``),
                ``finish_reason`` and ``attempts``. The return type is unchanged
                because four call sites unpack the 3-tuple; callers that must
                act on an invalid program pass a dict and check ``ok``.

        Returns:
            Tuple of (script, reasoning, tokens_used)
        """
        context_section = ""
        if additional_context:
            context_section = f"\n## Additional Context\n{additional_context}\n"

        # Pick prompt based on language (task metadata can override)
        lang = task.metadata.get("solution_language", self.language)
        if lang == "classify":
            template = SOLVER_PROMPT_CLASSIFY
            prompt_name = "SOLVER_PROMPT_CLASSIFY"
            prompt = template.format(
                task_description=task.description,
                additional_context=context_section,
                cases_text=task.metadata.get("cases_text", ""),
            )
        elif lang == "python":
            template = SOLVER_PROMPT_PYTHON
            prompt_name = "SOLVER_PROMPT_PYTHON"
            prompt = template.format(
                task_description=task.description,
                additional_context=context_section,
            )
        elif lang == "openevolve":
            template = SOLVER_PROMPT_OPENEVOLVE
            prompt_name = "SOLVER_PROMPT_OPENEVOLVE"
            prompt = template.format(
                task_description=task.description,
                additional_context=context_section,
            )
        else:
            template = SOLVER_PROMPT
            prompt_name = "SOLVER_PROMPT"
            prompt = template.format(
                task_description=task.description,
                additional_context=context_section,
            )

        logger.debug(
            "Solving task=%s lang=%s prompt=%s context_len=%d",
            task.task_id, lang, prompt_name,
            len(additional_context) if additional_context else 0,
        )

        meta: dict = {}
        # complete() rather than complete_with_breakdown(): the wire call is the
        # same code, and the prompt/completion/reasoning split arrives through
        # ``meta`` (see LLMClient(_meta=...)), so the call site stays exactly the
        # one the seed-threading tests pin.
        response, tokens = await self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 if temperature is None else temperature,
            seed=seed,
            _meta=meta,
        )

        if lang == "classify":
            script = self._extract_json(response)
        elif lang in ("python", "openevolve"):
            script = self._extract_python_code(response)
        else:
            script = self._extract_bash_script(response)

        ok, reason = validate_solver_output(
            script, lang, meta.get("finish_reason"))

        if not script.strip():
            logger.warning(
                "Empty script extracted for task=%s (response_len=%d)",
                task.task_id, len(response),
            )
        elif not ok:
            # Not DEBUG: this is the difference between "the model produced a
            # weak solution" and "the model produced no solution", and the two
            # were previously indistinguishable here.
            logger.warning(
                "Invalid solver output for task=%s: %s "
                "(response_len=%d script_len=%d finish_reason=%s attempts=%s)",
                task.task_id, reason, len(response), len(script),
                meta.get("finish_reason"), meta.get("attempts"),
            )
        else:
            logger.debug(
                "Solved task=%s tokens=%d script_len=%d",
                task.task_id, tokens, len(script),
            )

        if _verdict is not None:
            _verdict.update({
                "ok": ok,
                "reason": reason,
                "finish_reason": meta.get("finish_reason"),
                # attempts == 1 means the call succeeded on its FIRST wire
                # attempt; anything higher means a transport retry rescued it.
                # Kept here so "raw first-attempt success" can be reported
                # separately from "succeeded after retry" instead of the retry
                # being invisible.
                "attempts": meta.get("attempts"),
                "escalated": meta.get("escalated"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "reasoning_tokens": meta.get("reasoning_tokens"),
                "reasoning_reported": meta.get("reasoning_reported"),
                "total_tokens": tokens,
                "response_len": len(response),
                "script_len": len(script),
            })
        return script, response, tokens

    def _extract_bash_script(self, response: str) -> str:
        """Extract bash script from fenced code block."""
        # Explicit None check (not ``or``): a matched-but-empty block must
        # return "" rather than fall through to the whole response.
        result = extract_fenced_block(response, ("bash", "sh", ""))
        return result if result is not None else response.strip()

    def _extract_json(self, response: str) -> str:
        """Extract JSON from fenced code block.

        Ordering contract: fenced -> flat regex -> (``balanced_json_fallback``
        ON only) balanced scan -> raw response. The flag can therefore only
        change outcomes where the flag-less code returns the raw prose.
        """
        result = extract_case_json(
            response, balanced_fallback=self.balanced_json_fallback
        )
        return result if result is not None else response.strip()

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from fenced code block."""
        result = extract_fenced_block(response, ("python", "py", ""))
        return result if result is not None else response.strip()
