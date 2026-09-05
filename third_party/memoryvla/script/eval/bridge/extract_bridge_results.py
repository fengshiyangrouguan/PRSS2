import os
import re
import glob

def extract_and_print_success(root_paths, dir_flag='eval', print_style='text'):
    pattern = re.compile(r"Average success:?[ ]*([0-9]*\.?[0-9]+)")
    task_map = {
        "Carrot": "Carrot",
        "Eggplant": "Eggplant",
        "Spoon": "Spoon",
        "Cube": "Cube",
    }

    headers = ["name", "AVG"] + list(task_map.values())
    header_fmt = "{:>24s}  {:>6s}  {:>6s}  {:>8s}  {:>6s}  {:>6s}"
    row_fmt    = "{name:>24s}  {AVG:>6s}  {Carrot:>6s}  {Eggplant:>8s}  {Spoon:>6s}  {Cube:>6s}"

    for root in root_paths:
        eval_dir = os.path.join(root, dir_flag)
        records = []

        # Parse every subfolder under eval_dir as one evaluation run
        if os.path.isdir(eval_dir):
            subdirs = sorted(
                d for d in os.listdir(eval_dir)
                if os.path.isdir(os.path.join(eval_dir, d))
            )
            for sub in subdirs:
                run_name = sub  # e.g., "memvla-bridge.pt" or "step-020000-epoch-03-..."
                run_path = os.path.join(eval_dir, sub)

                tmp = {}
                for txt in glob.glob(os.path.join(run_path, "*.txt")):
                    raw_name = os.path.basename(txt)[:-4]
                    for long_name in task_map:
                        if raw_name.startswith(long_name):
                            with open(txt, "r") as f:
                                text = f.read()
                            m = pattern.search(text)
                            if m:
                                tmp[long_name] = float(m.group(1))
                            break

                if len(tmp) == len(task_map):
                    avg = sum(tmp.values()) / len(task_map) * 100
                    avg_str = f"{avg:5.1f}"
                else:
                    avg = None
                    avg_str = "-"

                rec = {"name": run_name, "AVG": avg_str, "avg_val": avg}
                for long_name, short in task_map.items():
                    val = tmp.get(long_name)
                    rec[short] = f"{val * 100:5.1f}" if val is not None else "-"
                records.append(rec)

        # Sort lexicographically by name (since “iter” is no longer meaningful)
        records.sort(key=lambda r: r["name"])

        title = root + f" ({dir_flag})"
        if not records:
            print(f"No records found for {title}\n")
            continue

        # Output
        if print_style == 'text':
            print("=" * 60)
            print(title)
            print(header_fmt.format(*headers))
            for rec in records:
                print(row_fmt.format(**rec))
            print()
        elif print_style == 'md':
            print(f"**{title}**\n")
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join("---" for _ in headers) + " |")
            for rec in records:
                row = [rec["name"], rec["AVG"]] + [rec[c] for c in task_map.values()]
                print("| " + " | ".join(row) + " |")
            print()

        # Best AVG
        valid = [r for r in records if r.get("avg_val") is not None]
        if valid:
            max_avg = max(r["avg_val"] for r in valid)
            best = [r for r in valid if abs(r["avg_val"] - max_avg) < 1e-6]
            print(f"--> Highest AVG for {root}: {max_avg:.1f}%")
            for br in best:
                print(f"    - name {br['name']}: {br['AVG']}%")
        else:
            print(f"--> No average values found for {root}.")
        print()


if __name__ == "__main__":
    dir_flag = 'eval_simpler'
    roots = [
        # Add your log root directories here
        # such as: ./log/bridge/memvla_bridge--image_aug
    ]
    extract_and_print_success(roots, dir_flag=dir_flag, print_style='md')
