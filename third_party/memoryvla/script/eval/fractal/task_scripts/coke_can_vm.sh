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

# coke_can_options
coke_can_options_arr=(
  "lr_switch=True"
  "upright=True"
  "laid_vertically=True"
)

urdf_version_arr=(
  None
  "recolor_tabletop_visual_matching_1"
  "recolor_tabletop_visual_matching_2"
  "recolor_cabinet_visual_matching_1"
)

env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./third_libs/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

EvalOverlay() {
  ckpt_path=$1
  env_name=$2
  scene_name=$3
  rgb_overlay_path=$4
  coke_can_option=$5
  urdf_version=$6
  
  eval_dir=$(dirname "$(dirname "${ckpt_path}")")/eval_simpler/$(basename "${ckpt_path}")
  mkdir -p "${eval_dir}"
  log_file="${eval_dir}/coke_can_vm.txt"

  {
    echo "+###############################+"
    date
    for i in "$@"; do
      echo "| $i"
    done
    echo "+###############################+"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path "${ckpt_path}" \
      --robot google_robot_static \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 80 \
      --env-name "${env_name}" \
      --scene-name "${scene_name}" \
      --rgb-overlay-path "${rgb_overlay_path}" \
      --robot-init-x 0.35 0.35 1 \
      --robot-init-y 0.20 0.20 1 \
      --obj-init-x -0.35 -0.12 5 \
      --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version}
    
    echo "+###############################+"
    date
    echo "Done!"
    echo "+###############################+"
  } 2>&1 | tee -a "${log_file}"
}

for ckpt_path in "${ckpt_paths[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    for urdf_version in "${urdf_version_arr[@]}"; do
      EvalOverlay "$ckpt_path" "$env_name" "$scene_name" "$rgb_overlay_path" "$coke_can_option" "$urdf_version"
    done
  done
done
