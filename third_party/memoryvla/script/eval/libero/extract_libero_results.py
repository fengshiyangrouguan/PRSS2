import os
import re
from collections import defaultdict

# ===== user inputs =====
ckpt_paths = [
    # Add your log root directories here
    # such as: ./log/libero/memvla_libero_spatial--image_aug
]

# ===== regex =====
re_step = re.compile(r"step-(\d+)")
re_episode = re.compile(r"# episodes completed so far:\s*(\d+)")
re_total = re.compile(r"Current total success rate:\s*([\d\.]+)")
re_trials = re.compile(r"(\d+)trials", re.IGNORECASE)


def is_complete(fname: str, episodes: int) -> bool:
    fn = fname.lower()
    if any(k in fn for k in ["spatial", "object", "goal", "libero_10"]):
        return episodes == 500
    if "libero_90" in fn:
        m = re_trials.search(fn)
        if m:
            trials = int(m.group(1))
            return episodes >= 90 * trials
        return episodes >= 450
    return False


def get_category(fname: str):
    fn = fname.lower()

    # --- special handling for libero_90 with trials count ---
    if "libero_90" in fn:
        m = re.search(r"libero_90-(\d+)trials", fn)
        if m:
            trials = m.group(1)
            return f"90-{trials}trials"
        return "90"  # fallback if no explicit trials count

    # --- normal cases ---
    for key in ["spatial", "object", "goal", "libero_10"]:
        if key in fn:
            return key.replace("libero_", "")
    return "unknown"


def norm_basename(path: str) -> str:
    """Normalize base name to dash style for matching run dirs like 'memvla-libero-*.pt'."""
    b = os.path.basename(path)
    return b.replace("_", "-")


def find_eval_root(base: str) -> str | None:
    """Return the eval_root to use. Prefer base/eval_libero; fallback to parent/eval_libero."""
    direct = os.path.join(base, "eval_libero")
    if os.path.isdir(direct):
        return direct
    parent = os.path.join(os.path.dirname(base), "eval_libero")
    if os.path.isdir(parent):
        return parent
    return None


def candidate_run_dirs(eval_root: str, base: str) -> list[tuple[str, str]]:
    """
    Return [(name, run_dir_path), ...].
    - If eval_root is inside base: include all subdirs.
    - If eval_root is parent-shared: include only subdirs whose name starts with normalized base name.
    """
    inside = eval_root.startswith(os.path.abspath(base) + os.sep)
    all_subdirs = [
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d))
    ]
    if inside:
        return [(d, os.path.join(eval_root, d)) for d in sorted(all_subdirs)]
    # parent-shared: filter
    key = norm_basename(base)
    filtered = [d for d in all_subdirs if d.startswith(key)]
    return [(d, os.path.join(eval_root, d)) for d in sorted(filtered)]


def collect_txt_paths(run_dir: str) -> list[str]:
    """
    Collect *.txt files under run_dir (depth 1) and also under one nested level (depth 2).
    Handles structures like:
      run_dir/*.txt
      run_dir/<sub>/*.txt
    """
    txts = []
    # depth 1
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p) and p.endswith(".txt"):
            txts.append(p)
    # depth 2
    for name in os.listdir(run_dir):
        pdir = os.path.join(run_dir, name)
        if os.path.isdir(pdir):
            for sub in os.listdir(pdir):
                p = os.path.join(pdir, sub)
                if os.path.isfile(p) and p.endswith(".txt"):
                    txts.append(p)
    return sorted(txts)


def name_from_run_dir(run_dir_name: str) -> str:
    """Prefer numeric step → '20k'; else use run_dir_name as-is."""
    m = re_step.search(run_dir_name)
    if m:
        step_num = int(m.group(1))
        return f"{step_num // 1000}k"
    return run_dir_name


# ===== main =====
for base in ckpt_paths:
    eval_root = find_eval_root(base)
    if not eval_root:
        print(f"\n❌ {base} has no eval_libero (neither local nor parent), skipped")
        continue

    runs = candidate_run_dirs(eval_root, base)
    if not runs:
        print(f"\n❌ {base} no matching runs under {eval_root}, skipped")
        continue

    print(f"\n📂 {os.path.basename(base)}")

    # name -> { category -> max_rate }
    results = defaultdict(lambda: defaultdict(float))

    for run_dir_name, run_dir in runs:
        name = name_from_run_dir(run_dir_name)
        cat_to_rates = defaultdict(list)

        for txt_path in collect_txt_paths(run_dir):
            fname = os.path.basename(txt_path)
            category = get_category(fname)

            try:
                text = open(txt_path, "r", encoding="utf-8", errors="ignore").read()
            except:
                continue

            ep_match = None
            total_match = None
            for m in re_episode.finditer(text):
                ep_match = m
            for m in re_total.finditer(text):
                total_match = m
            if not ep_match or not total_match:
                continue

            episodes = int(ep_match.group(1))
            total_rate = float(total_match.group(1))

            if is_complete(fname, episodes):
                cat_to_rates[category].append(total_rate)

        for cat, vals in cat_to_rates.items():
            results[name][cat] = max(vals)

    if not results:
        print("(no complete results)")
        continue

    all_cats = sorted({c for run in results.values() for c in run.keys()})
    print("\n| Name | " + " | ".join(all_cats) + " |")
    print("|" + "|".join("---" for _ in range(len(all_cats) + 1)) + "|")

    def sort_key(n: str):
        return (0, int(n[:-1])) if n.endswith("k") and n[:-1].isdigit() else (1, n)

    for name in sorted(results.keys(), key=sort_key):
        vals = [f"{results[name][c]:.3f}" if c in results[name] else "-" for c in all_cats]
        print("| " + " | ".join([name] + vals) + " |")

    cat_best = defaultdict(lambda: ("", 0.0))
    for name, cats in results.items():
        for c, v in cats.items():
            if v > cat_best[c][1]:
                cat_best[c] = (name, v)

    print()
    for c, (name, v) in cat_best.items():
        print(f"⭐ best({c}): {name} ({v:.3f})")
