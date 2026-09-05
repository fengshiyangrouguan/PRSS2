import os
import re

def extract_success_rates(log_path: str):
    """
    Extract all success rates from a log file.
    Each line containing 'Average success: x.xxx' will be parsed and converted to a percentage (0–100).
    Returns:
        List[float]: list of success rates in percentage.
    """
    pattern = re.compile(r"Average success:?\s+([0-9]*\.?[0-9]+)")
    rates = []
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rates.append(float(match.group(1)) * 100)
    return rates


def process_root(root: str, dir_flag: str):
    """
    Traverse all evaluation folders under root/dir_flag, extract success rates,
    print per-task summaries, and generate a Markdown summary table.
    """
    file_configs = {
        "coke_can_vm.txt": ("Coke Can", 12),
        "move_near_vm.txt": ("Move Near", 4),
        "drawer_vm.txt": ("O/C Drawer", 216),
        "put_in_drawer_vm.txt": ("Put In Drawer", 12),
        "coke_can_va.txt": ("Coke Can", 33),
        "move_near_va.txt": ("Move Near", 10),
        "drawer_va.txt": ("O/C Drawer", 42),
        "put_in_drawer_va.txt": ("Put In Drawer", 7),
    }

    tasks = ["Coke Can", "Move Near", "O/C Drawer", "Put In Drawer"]
    eval_dir = os.path.join(root, dir_flag)
    if not os.path.isdir(eval_dir):
        print(f"[Warning] Eval directory not found: {eval_dir}")
        return

    data = {}
    notes = []

    subfolders = sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )

    for name in subfolders:
        name_label = name  # directly use the folder name
        entry = {"VM": {}, "VA": {}}
        step_path = os.path.join(eval_dir, name)

        for fname, (short, expected_count) in file_configs.items():
            full_path = os.path.join(step_path, fname)
            rates = extract_success_rates(full_path) if os.path.isfile(full_path) else []
            actual_count = len(rates)
            avg_rate = sum(rates) / actual_count if actual_count else None

            mode = "VM" if "vm" in fname else "VA"
            if actual_count == expected_count and avg_rate is not None:
                entry[mode][short] = avg_rate
            else:
                entry[mode][short] = None
                if actual_count != expected_count:
                    notes.append((name_label, fname, expected_count, actual_count, rates))

        data[name_label] = entry

    # Markdown summary table
    headers = ["Name", "AVG(VM)"] + tasks + [""] + ["AVG(VA)"] + tasks
    print(f"\n### Results for `{root}` ({dir_flag})\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")

    for name_label in sorted(data):
        vm_map, va_map = data[name_label]["VM"], data[name_label]["VA"]

        def make_cells(task_map):
            """Format task results and compute average across valid entries."""
            vals = [task_map[t] for t in tasks]
            valid = [v for v in vals if v is not None]
            avg_cell = f"{sum(valid) / len(valid):5.1f}" if valid else "-"
            task_cells = [f"{v:5.1f}" if v is not None else "-" for v in vals]
            return [avg_cell] + task_cells

        vm_cells = make_cells(vm_map)
        va_cells = make_cells(va_map)
        print("| " + " | ".join([name_label] + vm_cells + [""] + va_cells) + " |")

    # Notes for mismatched file counts
    if notes:
        print("\n**Mismatched counts:**")
        for name_label, fname, expected, actual, rates in notes:
            short = file_configs[fname][0]
            avg_str = f"{sum(rates)/actual:5.1f}" if actual else "-"
            print(f"- `{name_label}` → `{fname}` ({short}) expected {expected}, found {actual}, avg={avg_str}")
    print("\n" + "=" * 80 + "\n")


def process_multiple_roots(root_paths, dir_flag='eval_simpler'):
    """
    Process multiple root directories and print aggregated tables for each.
    """
    for root in root_paths:
        process_root(root, dir_flag)


if __name__ == "__main__":
    dir_flag = 'eval_simpler'
    root_paths = [
        # Add your log root directories here
        # such as: ./log/fractal/memvla_fractal--image_aug
    ]
    process_multiple_roots(root_paths, dir_flag)
