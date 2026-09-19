"""Prompt templates for the Meta^n framework."""

# ---------------------------------------------------------------------------
# Language-specific solver_lib sections (plugged into {solver_lib_section})
# ---------------------------------------------------------------------------

SOLVER_LIB_SECTION_PYTHON = """### solver_lib (solver-visible code library)
Python functions that will be PREPENDED to the solver's generated code and available at runtime inside the sandbox. The solver sees their signatures in its prompt and can call them directly.

Use this to provide working algorithm implementations that the solver cannot reliably write from scratch. This is different from pre_process (which provides strategy guidance as text) — solver_lib provides actual executable code.

Example:
```solver_lib:local_search
def local_search(solution, evaluate_fn, time_limit=8.0):
    \"\"\"Iteratively improve solution via random swaps within time limit.\"\"\"
    import time, random
    best, best_score = solution, evaluate_fn(solution)
    start = time.time()
    while time.time() - start < time_limit:
        keys = list(best.keys())
        if len(keys) < 2:
            break
        i, j = random.sample(range(len(keys)), 2)
        neighbor = dict(best)
        neighbor[keys[i]], neighbor[keys[j]] = neighbor[keys[j]], neighbor[keys[i]]
        score = evaluate_fn(neighbor)
        if score > best_score:
            best, best_score = neighbor, score
    return best
```

IMPORTANT for solver_lib:
- Functions run inside the sandbox, NOT in the meta-layer
- Only use Python standard library imports (no numpy, scipy, etc.)
- Each block should define a single self-contained function
- Later layers can override a function by providing a solver_lib block with the same name
- solver_lib functions from different layers share a runtime namespace and can call each other
- Every function MUST include type annotations and a docstring with (a) one-line summary, (b) Args section listing every parameter with its expected type and meaning, (c) Returns section describing the return type and structure. The solver only sees the signature and docstring, not the full source — the docstring is the API contract. Example:
      def pack_items(items: list, capacity: int) -> tuple:
          \"\"\"Pack items into bins using first-fit decreasing.

          Args:
              items: list[tuple[int, int, str]] — each is (width, height, item_id).
              capacity: int — max bin weight.

          Returns:
              tuple[int, list[dict]] — (total_cost, placements) where each dict
              has keys 'id' (str), 'x' (int), 'y' (int).
          \"\"\""""


SOLVER_LIB_SECTION_BASH = """### solver_lib (helper code for the solver)

You have two options for providing reusable code to the bash solver. Choose whichever fits:

**Option 1: Bash functions** — prepended directly to the solver's bash script, callable as regular functions.

```solver_lib_bash:install_deps
install_deps() {
    apt-get update -qq && apt-get install -y -qq sqlite3 python3-pip
}
```

**Option 2: Python helper scripts** — written to /tmp/_lib_<name>.py at runtime, callable from bash via `python3 /tmp/_lib_<name>.py <args>`. Use this for complex logic where Python is more reliable. Output goes to stdout (the solver captures it).

```solver_lib:recover_data
def recover_data(db_path: str) -> list[str]:
    \"\"\"Recover strings from a corrupted database file.

    Args:
        db_path: str — path to the database file.

    Returns:
        list[str] — recovered strings, one per element.
    \"\"\"
    ...
```
The solver calls it as: `result=$(python3 /tmp/_lib_recover_data.py /path/to/db)`

IMPORTANT for solver_lib:
- Bash functions run inside the container — use any system commands available
- Python scripts also run inside the container — standard library + any installed packages
- Each block should define a single self-contained unit
- Later layers can override a function by providing a block with the same name
- Every Python function MUST include type annotations and a docstring (Args + Returns)"""


SOLVER_LIB_SECTION_OPENEVOLVE = """### solver_lib (solver-visible code library)
Python functions that will be PREPENDED to the solver's generated code and available at runtime. The solver sees their signatures in its prompt and can call them directly.

Use this to provide working algorithm implementations (optimization routines, mathematical utilities, search strategies) that the solver cannot reliably write from scratch. This is different from pre_process (which provides strategy guidance as text) — solver_lib provides actual executable code.

Example:
```solver_lib:simulated_annealing
def simulated_annealing(initial: "np.ndarray", objective_fn: "Callable", neighbor_fn: "Callable", temp: float = 1.0, cooling: float = 0.995, steps: int = 10000) -> "np.ndarray":
    \"\"\"Optimize a solution via simulated annealing.

    Args:
        initial: np.ndarray — starting point.
        objective_fn: Callable — returns a float score (higher is better).
        neighbor_fn: Callable — returns a neighbor of the current solution.
        temp: float — initial temperature.
        cooling: float — cooling factor per step.
        steps: int — total number of steps.

    Returns:
        np.ndarray — best solution found.
    \"\"\"
    import numpy as np
    best = current = initial.copy()
    best_score = current_score = objective_fn(current)
    for _ in range(steps):
        candidate = neighbor_fn(current)
        score = objective_fn(candidate)
        delta = score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            current, current_score = candidate, score
            if score > best_score:
                best, best_score = candidate, score
        temp *= cooling
    return best
```

IMPORTANT for solver_lib:
- Functions run inside the evaluation sandbox, NOT in the meta-layer
- You MAY use numpy, scipy, jax, optax, sympy — these are available at runtime
- Each block should define a single self-contained function
- Later layers can override a function by providing a solver_lib block with the same name
- solver_lib functions from different layers share a runtime namespace and can call each other
- Every function MUST include type annotations and a docstring with (a) one-line summary, (b) Args section listing every parameter with its expected type and meaning, (c) Returns section describing the return type and structure. The solver only sees the signature and docstring, not the full source — the docstring is the API contract."""


# ---------------------------------------------------------------------------
# Omega prompts
# ---------------------------------------------------------------------------

OMEGA_PROMPT = """You are one layer in a recursive self-improving system called Meta^n.

## Your Role
Given the execution traces and context of the system below you, produce code that improves its performance — specifically, that raises the MEAN CONTINUOUS SCORE across tasks. Treat a higher score as the goal even on tasks that already "pass": most tasks here are scored on a 0.0–1.0 continuum, not pass/fail. You do NOT solve tasks directly — you write code that helps the solver below you perform better.

## System Context
Current depth: {depth}
Tasks attempted: {num_tasks}
{score_summary}{env_notes_section}{headroom_section}
## Execution Traces (sampled: {num_failures} failures, {num_successes} successes)
{traces_text}

## Previously Injected Code (layers below you)
{context_stack_text}

## What You Can Produce

You can output any combination of:

### pre_process (strategy guidance)
Python code that runs BEFORE the solver sees the task. Your code has access to:
- `task` (read-only TaskDescription with .task_id, .description, .metadata)
- `outer_context` (string: guidance from higher meta-layers — use this to ADAPT your intervention based on higher layers' strategic decisions; empty if you are the outermost layer)

You MUST set the `additional_context` variable (a string that gets appended to the solver's prompt).

CRITICAL: Set the `additional_context` variable directly. Do NOT modify the `task` object — it is read-only.

Example:
```pre_process
# Adapt tactics based on higher layer's strategy (if any)
if "greedy" in outer_context.lower():
    additional_context = "Implement greedy assignment: sort by cost, assign iteratively."
elif "file" in task.description.lower():
    additional_context = "Remember to use proper quoting for file paths with spaces."
else:
    additional_context = ""
```

{solver_lib_section}

## Output Format

Respond with EXACTLY this structure (omit sections you don't need):

```rationale
<your reasoning about what patterns you see and what improvements to make>
```

```pre_process
<python code>
```

{solver_lib_output_format}

IMPORTANT:
- pre_process must be self-contained Python. Do not import external packages beyond the standard library.
- Focus on the highest-impact improvement based on the failure patterns you observe.
- Be specific about what failure patterns you're addressing.
- GENERALIZE: condition your intervention on task STRUCTURE (problem size, presence of hard constraints, metadata fields) — NEVER branch on specific task_id string literals. Code that keys off task_id values cannot transfer to unseen tasks and is wasted effort.
- If you add a solver_lib helper, it must target a specific failing task: name that task in your rationale and explain why the solver cannot write it reliably itself.
- Keep code simple and robust — complex code is more likely to break.
- Favor small (≤30 lines), audit-able helpers over large multi-purpose ones. Helpers that break the runtime env (creating envs, installing packages, depending on uncommon CLIs) introduce more failures than they prevent."""


OMEGA_PROMPT_META = """You are a higher-order meta-layer (depth {depth}) in a recursive self-improving system called Meta^n.

## Your Role
Previous layers have already made tactical fixes — dependency handling, constraint checks, format corrections. Your job is NOT to patch individual errors. Lower layers already do that.

Instead, you should:
1. Identify SYSTEMIC patterns across task categories
2. Understand WHY previous layers' fixes helped some tasks but hurt others
3. Make STRUCTURAL improvements: task classification, solver strategy selection, algorithm implementations
4. Consider CONDITIONING or BYPASSING previous layers' changes for task types where they cause regressions
5. Provide reusable algorithm implementations (via solver_lib) that the solver can call directly

## System Context
Current depth: {depth}
Tasks attempted: {num_tasks}
{score_summary}{env_notes_section}{headroom_section}
## Performance Summary & Analysis
{traces_text}

## Previously Injected Code (layers below you)
{context_stack_text}

## What You Can Produce

You can output any combination of:

### pre_process (strategy guidance)
Python code that runs BEFORE the solver sees the task. Your code has access to:
- `task` (read-only TaskDescription with .task_id, .description, .metadata)
- `outer_context` (string: guidance from higher meta-layers — use this to ADAPT your intervention based on higher layers' strategic decisions; empty if you are the outermost layer)

You MUST set the `additional_context` variable (a string that gets appended to the solver's prompt).

CRITICAL: Set the `additional_context` variable directly. Do NOT modify the `task` object — it is read-only.

{solver_lib_section}

## Output Format

Respond with EXACTLY this structure (omit sections you don't need):

```rationale
## Diagnosis
<the specific failure mode or bottleneck you are targeting, grounded in the score data, the headroom table, and the per-trace failure classes above — name the actual tasks it affects>

## Intervention
<what you will change and EXACTLY which tasks it targets. For every solver_lib helper you add, name the failing task it addresses and why the solver cannot write it reliably from scratch. Do not promise code you are not emitting.>

## Why this is different
<how this differs from what the injected-code history already tried — do NOT repeat a prior approach; if you cannot beat it, emit nothing>
```

```pre_process
<python code>
```

{solver_lib_output_format}

IMPORTANT:
- pre_process must be self-contained Python. Do not import external packages beyond the standard library.
- Focus on STRUCTURAL improvements, not individual error patches.
- If a previous layer's fix hurts certain tasks, add task-type conditionals to bypass it.
- GENERALIZE: condition on task STRUCTURE (size, constraints, metadata), NEVER on specific task_id string literals — task_id-keyed code does not transfer to unseen tasks.
- If you add a solver_lib helper, name the specific failing task it targets and why the solver cannot reliably write it itself.
- Keep code simple and robust — complex code is more likely to break.
- Use pre_process for strategy guidance (text); use solver_lib for reusable implementations (code).
- Favor small (≤30 lines), audit-able helpers over large multi-purpose ones. Helpers that break the runtime env (creating envs, installing packages, depending on uncommon CLIs) introduce more failures than they prevent."""


OMEGA_PROMPT_DISCOVER = """You are one layer in a recursive self-improving system called Meta^n, operating in DISCOVERY mode.

## Your Role
The layer below you is stuck on a task whose solution depends on a FACT ABOUT THE GIVEN ARTIFACTS that the solver does not already reliably know. The task hands the solver readable, executable source for those artifacts. Your job is NOT to recall the fact and write it down — recall has already been tried and it failed, because the solver remembered a procedure for a SIMILAR-LOOKING artifact and applied it to THIS one without checking the parts actually match. Your job is to write a DISCOVERY PROCEDURE plus structure-agnostic MEASUREMENT INSTRUMENTS that let the solver DISCOVER the fact by MEASURING the given artifacts this run, then VERIFY the discovery against those same artifacts before relying on it.

Rule of this mode: prefer a MEASURED fact over a REMEMBERED one, always — even when you are confident you remember. A remembered constant that is subtly wrong for THIS artifact is exactly how the previous attempts failed. For every quantity the solution needs, you must be able to answer: "Did an instrument MEASURE this on the given artifact this run, or did I just recall it?" If recalled, replace it with a measurement.

## System Context
Current depth: {depth}
Tasks attempted: {num_tasks}
{score_summary}{env_notes_section}{headroom_section}
## Execution Traces (sampled: {num_failures} failures, {num_successes} successes)
{traces_text}

## Previously Injected Code (layers below you)
{context_stack_text}

## The Reframe — convert assumptions into measurements
Read your own plan one sub-goal at a time and ask of EACH: "Which structure of the artifact does this sub-goal ASSUME?" Any sub-goal that assumes an internal structure — that some component behaves a certain way, that some input relationship forces some output relationship, that the artifact is built a certain way — is a LIABILITY until an instrument has confirmed it on the given artifact. Rewrite each such sub-goal as an EMPIRICAL step: an inspection of the given source, or a measurement with one of your instruments. A correct discovery procedure contains no unconfirmed structural assumption.

## What You Can Produce

### pre_process (the discovery PLAN)
Python that runs BEFORE the solver sees the task; it has access to `task` (read-only: .task_id, .description, .metadata) and `outer_context`. You MUST set `additional_context` (a string appended to the solver's prompt). Use it to hand the solver the DISCOVERY PLAN in measure-don't-recall form: the ordered empirical steps, which instrument to run at each step, and the PROPERTY each step must establish before the next begins. Put the PROCEDURE for finding the answer, never an answer. Do NOT modify `task` (it is read-only).

{solver_lib_section}

### Discovery instruments to author (structure-agnostic, caller-parameterised)
Each instrument takes the artifact — OR any callable the artifact is built from — AS A PARAMETER and reports what it MEASURES. An instrument must embed NO assumption about what the artifact is or what it will find: the caller supplies everything artifact-specific; the instrument only measures and reports. The SAME instrument, unchanged, must run on a completely unrelated artifact — if it would not, it is a disguised hint, not an instrument. Author at least these three (names are yours; the signatures are the contract):

1. A RELATION SCANNER — measures which input-side perturbations force a high-probability output-side relationship.
   def scan_relations(fn, sample_input, apply_delta, relate_outputs, delta_set, trials=20000) -> list:
   For each delta in delta_set, over `trials` random inputs x, compare fn(x) against fn(apply_delta(x, delta)) via relate_outputs, and return the (delta, output-relation, measured-probability) triples whose probability is high, ranked by probability. YOU choose `delta_set` and YOU choose `fn`; the instrument never supplies them, and choosing them well IS the discovery. The whole artifact end-to-end is ONE choice of `fn`, but a complex artifact is a COMPOSITION of smaller callables it exposes or is built from, and a relationship that is washed out end-to-end can be sharp at a finer grain. Scan at every level of granularity you can construct, not only end-to-end, and let the MEASURED probabilities tell you where the exploitable structure is.

2. A REIMPLEMENTATION CHECK — before you trust your own copy of any piece of the artifact, prove it matches the given one.
   def verify_reimplementation(my_fn, given_fn, sample_input, trials=10000) -> bool:
   Returns True iff my_fn and given_fn agree on every one of `trials` random inputs.

3. AN INTERMEDIATE-STATE INSTRUMENT — the artifact's external output can hide structure that is plain in its intermediate state. Where the source is readable, reconstruct an intermediate quantity (guarded by verify_reimplementation) and measure IT.
   def instrument_intermediate(reconstruct_state, sample_input, trials=20000) -> list:

### Acceptance tests — PROPERTY FORM ONLY
Every check asserts a PROPERTY that any correct discovery must satisfy — NEVER equality to a value you wrote down. Allowed shapes:
- existence / threshold: "a relation exists whose MEASURED probability is >= 0.99 over T trials"
- agreement: "my reimplementation matches the given component on all N random inputs"
- stability: "the recovered quantity is identical across K independent reruns with fresh artifact randomness"
- end-to-end: "the artifact's OWN success predicate accepts the answer my procedure recovered, on a freshly re-randomised artifact"
FORBIDDEN: a test whose right-hand side is a constant you typed rather than a value an instrument returned this run (no `assert measured == <literal>`, no `assert answer == <literal>`). If a test references a number you typed instead of one your instruments produced, it is wrong.

### self_probe — a runnable demonstration (verified, not trusted)
Emit ONE self-contained runnable script that: loads the given artifact by the path shown in the traces above; builds your instruments; runs the discovery; PRINTS the delta_set you chose and every relation your scanner measured (print the measured values — a reader must see WHAT was found, not only pass/fail); then runs your property-form acceptance tests AND a non-vacuous terminal test — re-randomise the artifact, run your FULL procedure end to end, and check the artifact's own success predicate, re-verifying any reconstructed component inside that same run. Exit non-zero if any property fails.

## Output Format
Respond with EXACTLY this structure (omit nothing the procedure needs):

```rationale
<which structural assumptions in the obvious approach you are replacing with measurements, and why measuring beats recalling here>
```

```pre_process
<python: set additional_context to the measure-don't-recall discovery plan>
```

{solver_lib_output_format}

```self_probe
<python: load the artifact, build instruments, run the discovery, PRINT measured relations + chosen delta_set, run the property-form tests + the terminal end-to-end test, exit non-zero on any failure>
```

IMPORTANT:
- ZERO baked-in answers: an instrument or test that only works on this one artifact is a hint in disguise. If it would not run unchanged on an unrelated artifact, generalise it.
- The discovery must run AT EXECUTION TIME inside the deployed helper: the artifact is re-randomised each run, so a constant you measured once and hardcoded will be wrong next run. Measure, then exploit, every run.
- Keep each instrument small (<=30 lines), single-purpose, audit-able; standard library only.
- If, after measuring, no exploitable relation clears your threshold at ANY granularity, say so in the rationale rather than inventing one — a recalled guess will not survive the terminal test.
"""


SOLVER_PROMPT = """You are a task-solving agent. Given a task description, produce a bash script that accomplishes the task.

## Task
{task_description}
{additional_context}
## Requirements
- Output ONLY a bash script in a fenced code block
- The script should be self-contained and executable
- Use set -e for error handling
- Keep it simple and direct

```bash
<your script here>
```"""


SOLVER_PROMPT_PYTHON = """You are a task-solving agent. Given a problem description, produce a Python implementation of the solve() function.

## Problem
{task_description}
{additional_context}
## Requirements
- Output ONLY the complete solve() function in a fenced Python code block
- The function must match the signature shown in the problem description
- Use only the Python standard library (do NOT use numpy, scipy, or other external packages)
- An `llm(prompt, *, temperature=0.3, max_tokens=512) -> str` helper is available at runtime for LLM reasoning when useful (e.g. classification, text understanding). For multiple calls, prefer `llm_batch(prompts: list[str], *, temperature=0.3, max_tokens=512) -> list[str]` which runs them concurrently. Both are optional — pure Python solutions are preferred for algorithmic tasks.
- Focus on producing a correct, efficient solution
- Do NOT include test code or if __name__ blocks

```python
<your solve function here>
```"""


SOLVER_PROMPT_CLASSIFY = """You are a text classification agent. Given a classification task and a set of cases, predict the correct label for every case.

## Task
{task_description}
{additional_context}
## Cases to Classify
{cases_text}

## Output Format
Respond with ONLY a JSON object inside a fenced code block mapping each case ID to its predicted label.
- Use EXACTLY the label names from the valid labels list
- Classify EVERY case — do not skip any
- For multi-label tasks, separate multiple labels with semicolons (e.g. "Label A;Label B")

```json
{{"case_0": "Label A", "case_1": "Label B", ...}}
```"""


SOLVER_PROMPT_OPENEVOLVE = """You are a mathematical and scientific optimization agent. Given a problem description, produce a complete Python module that solves it.

## Problem
{task_description}
{additional_context}
## Requirements
- Output ONLY the complete Python module in a fenced Python code block
- Your code MUST define the exact function specified in the problem description with the correct name and return type
- You may use numpy, scipy, jax, optax, sympy, and the Python standard library
- Focus on producing a correct solution first, then optimize for the best possible score
- Do NOT include test code or if __name__ blocks
- The evaluator will import your module and call the specified function automatically

```python
<your complete module here>
```"""


LIBRARY_INSTRUCTIONS = """
## Library Function Rules
- Helper functions listed above are ALREADY DEFINED in the global scope at runtime.
  Call them directly: `result = some_function(args)`.
- Do NOT import them. There is no module named `solver_lib`.
  `from solver_lib import X` will crash with ModuleNotFoundError.
- Match the documented parameter types and return types EXACTLY.
- Using helper functions is optional. If you can solve the task without them, that is fine.
"""


LIBRARY_INSTRUCTIONS_BASH = """
## Helper Code Rules
- Bash functions listed above are defined at the top of your script. Call them directly.
- Python helpers are at /tmp/_lib_<name>.py. Call via: `python3 /tmp/_lib_<name>.py <args>`
  Output goes to stdout; capture with: `result=$(python3 /tmp/_lib_<name>.py arg1 arg2)`
- Using helper code is optional. If you can solve the task without it, that is fine.
"""


# ---------------------------------------------------------------------------
# Mechanism-0 adoption affordance (foster_adoption=True). Default OFF: the
# variants above are emitted byte-identically. When ON, these REQUIRE the
# solver to call the injected helpers by name instead of re-deriving them
# inline. The bare-name / no-import clarification is kept verbatim because it
# is exactly what resolves the advertised `solver_lib:<name>` vs deployed
# bare-def mismatch (prepend_python_library rewrites dotted calls at runtime).
# ---------------------------------------------------------------------------

LIBRARY_INSTRUCTIONS_FOSTER = """
## Library Function Rules (REQUIRED — call the helpers, do NOT re-implement them)
- Helper functions listed above are ALREADY DEFINED in the global scope at runtime.
  You MUST call them by their bare name: `result = some_function(args)`.
- Build your solve() AROUND these calls: delegate the core logic to the helpers.
  Do NOT re-implement them inline and do NOT re-derive their logic from scratch.
- Do NOT import them. There is no module named `solver_lib`.
  `from solver_lib import X` will crash with ModuleNotFoundError.
- Match the documented parameter types and return types EXACTLY.
"""


LIBRARY_INSTRUCTIONS_BASH_FOSTER = """
## Helper Code Rules (REQUIRED — call the helpers, do NOT re-implement them)
- Bash functions listed above are defined at the top of your script. You MUST call them directly.
- Python helpers are at /tmp/_lib_<name>.py. Call via: `python3 /tmp/_lib_<name>.py <args>`
  Output goes to stdout; capture with: `result=$(python3 /tmp/_lib_<name>.py arg1 arg2)`
- Delegate the core work to these helpers. Do NOT re-implement their logic inline.
"""


# ---------------------------------------------------------------------------
# Output format hints for solver_lib blocks (plugged into {solver_lib_output_format})
# ---------------------------------------------------------------------------

SOLVER_LIB_OUTPUT_FORMAT_PYTHON = """```solver_lib:<name>
<python function code>
```"""


SOLVER_LIB_OUTPUT_FORMAT_BASH = """```solver_lib_bash:<name>
<bash function code>
```

```solver_lib:<name>
<python helper script — used via: python3 /tmp/_lib_<name>.py <args>>
```"""
