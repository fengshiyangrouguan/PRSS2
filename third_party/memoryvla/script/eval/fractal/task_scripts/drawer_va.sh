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

env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0
OpenBottomDrawerCustomInScene-v0
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0
CloseBottomDrawerCustomInScene-v0
)

EvalSim() {
  ckpt_path=$1
  env_name=$2
  scene_name=$3
  EXTRA_ARGS=$4

  eval_dir=$(dirname $(dirname ${ckpt_path}))/eval_simpler/$(basename ${ckpt_path})
  mkdir -p ${eval_dir}
  log_file=${eval_dir}/drawer_va.txt

  {
    echo "+###############################+"
    date
    for i in "$@"; do
      echo "| $i"
    done
    echo "+###############################+"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      ${EXTRA_ARGS}

    echo "+###############################+"
    date
    echo "Done!"
    echo "+###############################+"
  } 2>&1 | tee -a "${log_file}"
}

# base setup
scene_name=frl_apartment_stage_simple
EXTRA_ARGS="--enable-raytracing"
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
  done
done

# backgrounds
declare -a scene_names=(
"modern_bedroom_no_roof"
"modern_office_no_roof"
)
EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    for scene_name in "${scene_names[@]}"; do
      EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
      done
  done
done

# lightings
scene_name=frl_apartment_stage_simple
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
    EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
  done
done

# new cabinets
scene_name=frl_apartment_stage_simple
for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    EvalSim "${ckpt_path}" "${env_name}" "${scene_name}" "${EXTRA_ARGS}"
  done
done
