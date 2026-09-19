"""Prompt templates for the Terminus 2-inspired agentic solver.

These templates define the multi-turn interaction between the agentic solver
and the LLM. The response format uses XML tags (more robust than JSON when
the response contains code with special characters).
"""

# ---------------------------------------------------------------------------
# System prompt — sent as the first user message
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM_PROMPT = """You are a code-writing agent. Each turn you OUTPUT CODE.
You do not describe code, summarize code, or explain code — you output executable code.

## Task
{task_description}

{language_instructions}

{context_section}

## OUTPUT FORMAT — exact, no deviation

Your response MUST begin with `<code lang="{language}">` as the FIRST characters.
Do NOT write any preamble, greeting, or explanation before the code block.

<code lang="{language}">
# COMPLETE solution here. Full code every turn — never a diff or patch.
# Self-contained and immediately executable.
</code>

<status>working</status>

That is the entire response. Two blocks: <code>...</code> then <status>...</status>.

Use `<status>complete</status>` only after your code has executed and scored >= 0.5.

## Hard rules
- The <code> block is the response. No code = invalid response = wasted turn.
- Begin with `<code lang="{language}">` — literally the first characters of your message.
- Do NOT write `<analysis>`, `<plan>`, descriptions, or summaries — just code + status.
- Each turn emits the FULL code from scratch (not a patch, not a diff).
- On turn 2+, read the previous execution result and emit a NEW complete code that fixes
  the issues observed. Output the corrected code directly — do not narrate.
"""

# ---------------------------------------------------------------------------
# R2 — behavioral preamble (orthogonal base-agent floor-raiser).
#
# Injected into AGENTIC_SYSTEM_PROMPT by AgenticSolver._build_system_message
# behind ``--agentic-preamble`` (default OFF). The block is inserted AFTER the
# role line and BEFORE "## Task" so "## OUTPUT FORMAT" / "## Hard rules" remain
# last and most-salient. The block is injected at RENDER time (the module
# constant above is left verbatim), so when the flag is OFF the rendered system
# prompt is byte-identical to HEAD. Trailing blank line keeps a clean break
# before "## Task" when inserted via ``replace("## Task\n", ...)``.
# ---------------------------------------------------------------------------

AGENTIC_PREAMBLE = """## How to work (brief)
- Efficiency: do the minimum that solves the task — no extra scaffolding, tests, or files.
- One solution in place: emit the FULL corrected solution each turn; never make v2/_fixed variants.
- On failure: read the actual error first, list 5-7 plausible causes, fix the most likely, verify.

"""

# ---------------------------------------------------------------------------
# Language-specific instructions (plugged into {language_instructions})
# ---------------------------------------------------------------------------

LANG_INSTRUCTIONS_BASH = """## Solution Language: Bash
- Output a complete bash script inside `<code lang="bash">` tags
- The script should be self-contained and executable
- Use `set -e` for error handling
- No external dependencies unless you install them in the script
- Keep it simple and direct"""

LANG_INSTRUCTIONS_PYTHON = """## Solution Language: Python
- Output the complete `solve()` function inside `<code lang="python">` tags
- The function must match the signature shown in the problem description
- Use only the Python standard library (do NOT use numpy, scipy, or other external packages)
- An `llm(prompt, *, temperature=0.3, max_tokens=512) -> str` helper is available at runtime
  for LLM reasoning when useful (e.g. classification, text understanding). For multiple calls,
  prefer `llm_batch(prompts: list[str], *, temperature=0.3, max_tokens=512) -> list[str]`
  which runs them concurrently. Both are optional — pure Python solutions are preferred for
  algorithmic tasks.
- Focus on producing a correct, efficient solution
- Do NOT include test code or if __name__ blocks"""

LANG_INSTRUCTIONS_CLASSIFY = """## Solution Language: Python (Classification)
- Output a `solve(cases, labels, few_shot) -> dict` function inside `<code lang="python">` tags
- Return a dict mapping case_id to predicted label
- Use EXACTLY the label names from the valid labels list
- Classify EVERY case — do not skip any
- For multi-label tasks, separate multiple labels with semicolons (e.g. "Label A;Label B")
- An `llm()` and `llm_batch()` helper is available at runtime for LLM-based classification"""

LANG_INSTRUCTIONS_OPENEVOLVE = """## Solution Language: Python (Scientific Computing)

REGARDLESS of how the task above is framed (math problem, research challenge, expert
mission, etc.), your response MUST be Python code in a <code lang="python"> block.
You are NOT being asked to explain the problem, write a research note, or describe an
approach — you are being asked to OUTPUT EXECUTABLE CODE.

- Begin your response with `<code lang="python">` as the first characters.
- The block must define the EXACT function name and signature shown in the task's
  reference implementation. Improve the algorithm; keep the interface identical.
- A naive but correct stub (e.g. random / greedy / simple heuristic) is valid output —
  emit code even if you are unsure. Do NOT respond with "I cannot" or "let me think".
- You may use numpy, scipy, jax, optax, sympy, and the Python standard library.
- Do NOT include test code, `if __name__ == "__main__"` blocks, or example invocations.
- The evaluator imports your module and calls the specified function automatically."""

LANG_INSTRUCTIONS_BASH_SWEBENCH = """## Solution Language: Bash (SWE-bench Verified)

You are fixing a real bug in an open-source repository inside a prebuilt Docker container.

**Environment**
- The repository is checked out at `/testbed`, which is your bash script's working
  directory by default (the image's WORKDIR). You can use relative paths to repo files
  or absolute `/testbed/...` paths.
- The repo's Python environment is the `testbed` conda env. To activate before running
  Python tools: `source /opt/miniconda3/bin/activate && conda activate testbed`.
- You are root inside the container; no sudo needed.

**Goal**
- Modify the repo so that a hidden set of FAIL_TO_PASS tests start passing,
  AND the existing PASS_TO_PASS tests still pass.
- Do NOT modify anything under the repo's test directories — the verifier resets
  test files before grading, but editing them wastes turns.

**How to make changes**
- Edit files in place (`sed -i`, `python -c`, `cat heredoc > file`) OR write a unified
  diff and apply it with `patch --fuzz=5 -p1 -i <diff>` (cwd is already `/testbed`).
- End your script with `git --no-pager diff` so the trace surfaces exactly what you
  changed (this is observed by the meta-loop, not the grader).

**Don'ts**
- Do not `pip install` new dependencies — the `testbed` env already has everything
  the repo's tests need. Adding packages risks version conflicts the grader doesn't expect.
- Do not modify files in `tests/` — those get reset by the verifier before grading.
- Avoid network calls (curl/wget/git fetch). The container has internet, but anything
  fetched at solve time isn't reproducible and may differ from what the grader sees."""

# ---------------------------------------------------------------------------
# Observation template — shown after each execution
# ---------------------------------------------------------------------------

OBSERVATION_TEMPLATE = """## Execution Result (Turn {turn}/{max_turns})

Score: {score:.4f}
Exit code: {exit_code}
Duration: {duration_s:.1f}s

### stdout
{stdout}

### stderr
{stderr}

{eval_feedback_section}

Analyze the results carefully. If the score is below 0.5 or there are errors/timeouts,
you MUST iterate — identify the root cause and fix it. Do NOT signal completion
unless the solution is genuinely working.
"""

# ---------------------------------------------------------------------------
# R1 — error-hint taxonomy (orthogonal base-agent floor-raiser).
#
# Rendered into the observation by AgenticSolver._build_observation behind
# ``--agentic-error-hints`` (default OFF). The KEYS MUST match the class
# strings returned by ``meta_n.core.meta_layer.classify_error``. Only the
# ACTIONABLE classes have an entry; the non-actionable classes
# (Unknown error / Runtime error / Turn starvation / Environment fault) are
# deliberately absent, so ``error_hint`` returns "" and NO hint is rendered for
# them. The hint is injected at RENDER time (the OBSERVATION_TEMPLATE constant
# above is left verbatim), so when the flag is OFF the rendered observation is
# byte-identical to HEAD.
# ---------------------------------------------------------------------------

ERROR_HINTS: dict[str, str] = {
    "Numeric instability": (
        "Guard against nan / divide-by-zero / overflow; clamp or special-case "
        "degenerate inputs before returning."
    ),
    "Dependency error": (
        "Use ONLY the Python standard library — do not import numpy/scipy/etc."
    ),
    "Timeout": (
        "Reduce algorithmic complexity or add an early exit; the run hit the "
        "time cap before finishing."
    ),
    "Constraint violation": (
        "Re-check feasibility/bounds (capacity, overlap, boundaries) before "
        "emitting the answer."
    ),
    "Indexing error": (
        "Check off-by-one, empty collections, and dict keys before indexing."
    ),
    "Syntax error": (
        "Emit COMPLETE, valid code — no truncation, balanced brackets/quotes."
    ),
    "Format/parse error": (
        "Begin the response with the code block and emit the full solution in "
        "one block."
    ),
}


def error_hint(error_class: str) -> str:
    """Return the actionable hint for ``error_class``, or "" if non-actionable.

    The keys mirror the class strings returned by
    ``meta_n.core.meta_layer.classify_error``. Non-actionable classes
    (Unknown error / Runtime error / Turn starvation / Environment fault) are
    absent from :data:`ERROR_HINTS` and therefore yield "" — no hint rendered.
    """
    return ERROR_HINTS.get(error_class, "")


# ---------------------------------------------------------------------------
# Completion confirmation — shown when agent first signals complete
# ---------------------------------------------------------------------------

COMPLETION_CONFIRMATION = """## Completion Check

You indicated the task is complete. Here is the execution state:

Score: {score:.4f}
{eval_feedback_section}

Review this carefully before confirming:
- If score < 0.5 or there are errors/timeouts, respond with `<status>working</status>`
  and include the complete fixed code in the same message.
- If score >= 0.5 and the solution is genuinely correct, respond with `<status>complete</status>`.

A score of 0.0 means the solution failed entirely — you should NOT confirm completion.
"""

# ---------------------------------------------------------------------------
# Parse error feedback — shown when response couldn't be parsed
# ---------------------------------------------------------------------------

PARSE_ERROR_FEEDBACK = """Your previous response could not be parsed.
Error: {error}

Respond with EXACTLY this format. Begin with `<code lang="{language}">` — first characters.
No preamble. No analysis. Just code + status.

<code lang="{language}">
# Your complete solution
</code>

<status>working</status>
"""

# ---------------------------------------------------------------------------
# Context summarization prompt
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """Summarize the following conversation between a solver agent and an
execution environment. Preserve:
1. The original task description (verbatim if short, summarized if long)
2. Key observations from each execution attempt (scores, errors, what worked)
3. What approaches were tried and their outcomes
4. The most recent code that was generated

Be concise but preserve all information needed to continue solving.

Conversation:
{conversation}
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_language_instructions(
    solver_language: str,
    task_metadata: dict | None = None,
) -> str:
    """Return the language-specific instruction block for the given solver language.

    When ``task_metadata`` carries a benchmark hint that has its own bash variant
    (currently only ``swe_bench_verified``), return the benchmark-specific block.
    Otherwise dispatch on ``solver_language`` alone.
    """
    if (
        solver_language == "bash"
        and task_metadata
        and task_metadata.get("benchmark") == "swe_bench_verified"
    ):
        return LANG_INSTRUCTIONS_BASH_SWEBENCH
    return {
        "bash": LANG_INSTRUCTIONS_BASH,
        "python": LANG_INSTRUCTIONS_PYTHON,
        "classify": LANG_INSTRUCTIONS_CLASSIFY,
        "openevolve": LANG_INSTRUCTIONS_OPENEVOLVE,
    }.get(solver_language, LANG_INSTRUCTIONS_PYTHON)
