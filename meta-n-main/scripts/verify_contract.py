"""Minimal check of the exposed execution contract. ONE model call.

Steps (user protocol):
  1. the FINAL task description must contain BOTH "10-second" and the blocked
     imports list;
  2. spend exactly ONE DeepSeek call asking Omega to produce a helper for this
     task, given that description;
  3. feed the produced helper straight to the EXISTING
     `validate_library_function` -- no Meta^n run, no evaluator;
  4. it must avoid `sys` and every other BLOCKED_IMPORTS entry and PASS.

If it passes here, a real max-iterations=2/3 run is worth paying for.
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def stage1_description():
    from meta_n.integrations.co_bench import COBenchAdapter
    a = COBenchAdapter(data_dir="./data/co_bench",
                       task_names=["Bin packing - one-dimensional"],
                       instance_workers=1)
    task = a.load_tasks()[0]
    d = task.description
    print("=== STAGE 1: the description the model actually sees ===")
    print("  chars                :", len(d))
    print("  mentions 10-second   :", "10-second" in d)
    print("  mentions sandbox     :", "validation" in d)
    print("  lists blocked imports:", "sys" in d and "subprocess" in d)
    ok = ("10-second" in d) and ("sys" in d) and ("subprocess" in d)
    print("  %s both contracts present" % ("OK " if ok else "** "))
    return d, ok


def omega_prompt(description: str) -> str:
    return (
        "You are one layer in a recursive self-improving system called Meta^n.\n"
        "Given the execution traces of the layer below, you write improvement "
        "code that is injected into the layer beneath you.\n\n"
        "## Task\n" + description + "\n\n"
        "## Your output\n"
        "Reply with EXACTLY these fenced blocks and nothing else:\n\n"
        + FENCE + "rationale\n...\n" + FENCE + "\n\n"
        + FENCE + "pre_process\n# python that sets `additional_context`\n"
        + FENCE + "\n\n"
        + FENCE + "solver_lib:<function_name>\ndef <function_name>(...):\n"
        "    ...\n" + FENCE + "\n\n"
        "The library helper must be self-contained and must pass the helper "
        "code restrictions stated in the task.\n"
    )


def one_call(prompt: str) -> str:
    key = None
    for line in (Path(".env").read_text()).splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not found in .env")
    body = json.dumps({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "thinking": {"type": "disabled"},
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    u = d.get("usage", {})
    print("  call: ct=%s pt=%s" % (u.get("completion_tokens"),
                                   u.get("prompt_tokens")))
    return d["choices"][0]["message"].get("content") or ""


def stage2_call(description: str):
    print()
    print("=== STAGE 2: ONE Omega call (deepseek-flash, thinking disabled) ===")
    txt = one_call(omega_prompt(description))
    libs = {}
    i = 0
    while True:
        k = txt.find(FENCE + "solver_lib:", i)
        if k < 0:
            break
        nl = txt.find("\n", k)
        name = txt[k + len(FENCE) + len("solver_lib:"):nl].strip()
        e = txt.find(FENCE, nl)
        libs[name] = txt[nl + 1:e].strip()
        i = e + len(FENCE)
    print("  helpers produced:", list(libs))
    return libs, txt


def stage3_validate(libs):
    print()
    print("=== STAGE 3: feed to the EXISTING validate_library_function ===")
    from meta_n.core.code_library import validate_library_function
    from meta_n.utils.safety import BLOCKED_IMPORTS
    ok = True
    if not libs:
        print("  ** no helper block produced")
        return False, libs
    for name, body in libs.items():
        imports = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", body,
                             re.M | re.I)
        roots = {i.split(".")[0] for i in imports}
        offenders = sorted(roots & set(BLOCKED_IMPORTS))
        passed = validate_library_function(name, body, executor=None)
        print("  helper '%s': %d chars | imports=%s" % (name, len(body),
                                                        sorted(roots)))
        print("    blocked imports used : %s" % (offenders or "NONE"))
        print("    validate_library_function -> %s" % passed)
        ok &= passed and not offenders
    return ok, libs


def main() -> int:
    d, ok1 = stage1_description()
    libs, _ = stage2_call(d)
    ok3, libs = stage3_validate(libs)
    print()
    verdict = ok1 and ok3
    print("VERDICT:", "CONTRACT WORKS -- model avoids blocked imports and the "
          "helper validates" if verdict else "NOT YET (see above)")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
