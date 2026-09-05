#!/bin/bash

gpu_id=0
ckpt_paths=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -g) gpu_id="$2"; shift 2 ;;
    -k) ckpt_paths+=("$2"); shift 2 ;;
    *) echo "Error: Unknown arg: $1"; exit 1 ;;
  esac
done

if [ "${#ckpt_paths[@]}" -eq 0 ]; then
  echo "Error: At least one ckpt path must be specified via -k"
  exit 1
fi

echo "GPU ID: $gpu_id"
echo "CKPTs: ${ckpt_paths[*]}"


env_name=MoveNearGoogleBakedTexInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

urdf_version_arr=(
None
"recolor_tabletop_visual_matching_1"
"recolor_tabletop_visual_matching_2"
"recolor_cabinet_visual_matching_1"
)

EvalOverlay() {
  ckpt_path=$1
  env_name=$2
  scene_name=$3
  extra_args=$4
  rgb_overlay_path=$5

  eval_dir=$(dirname "$(dirname "${ckpt_path}")")/eval_simpler/$(basename "${ckpt_path}")
  mkdir -p "${eval_dir}"
  log_file="${eval_dir}/move_near_vm.txt"

  {
    echo "+###############################+"
    date
    for i in "$@"; do
      echo "| $i"
    done
    echo "+###############################+"

    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path "${ckpt_path}" \
      --robot google_robot_static \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 80 \
      --env-name "${env_name}" \
      --scene-name "${scene_name}" \
      --rgb-overlay-path "${rgb_overlay_path}" \
      --robot-init-x 0.35 0.35 1 \
      --robot-init-y 0.21 0.21 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      ${extra_args}

    echo "+###############################+"
    date
    echo "Done!"
    echo "+###############################+"
  } 2>&1 | tee -a "${log_file}"
}

for ckpt_path in "${ckpt_paths[@]}"; do
  for urdf_version in "${urdf_version_arr[@]}"; do
    EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "--additional-env-build-kwargs urdf_version=${urdf_version} --additional-env-save-tags baked_except_bpb_orange" "$rgb_overlay_path"
  done
done
