#!/bin/bash
set -e

ckpt_paths=(
/PATH/TO/YOUR/CHECKPOINT_1
/PATH/TO/YOUR/CHECKPOINT_2
)

gpu_id=0

scripts=(
"script/eval/fractal/task_scripts/coke_can_vm.sh"
"script/eval/fractal/task_scripts/move_near_vm.sh"
"script/eval/fractal/task_scripts/put_in_drawer_vm.sh"
"script/eval/fractal/task_scripts/move_near_va.sh"
"script/eval/fractal/task_scripts/coke_can_va.sh"
"script/eval/fractal/task_scripts/drawer_va.sh"
"script/eval/fractal/task_scripts/put_in_drawer_va.sh"
"script/eval/fractal/task_scripts/drawer_vm.sh"
)

for ckpt_path in "${ckpt_paths[@]}"; do
    eval_dir=$(dirname "$(dirname "${ckpt_path}")")/eval_simpler/$(basename "${ckpt_path}")
    mkdir -p "${eval_dir}/logs"

    for s in "${scripts[@]}"; do
        base_name="$(basename "${s}" .sh)"
        log_file="${eval_dir}/logs/${base_name}.log"
        echo "=== START：${s} （GPU ID=${gpu_id}）===" | tee -a "${log_file}"
        bash "${s}" -g "${gpu_id}" -k "${ckpt_path}" 2>&1 | tee -a "${log_file}" || true
        echo "=== DONE：${s} ===" | tee -a "${log_file}"
        echo
    done
done

echo "✅ All done."
