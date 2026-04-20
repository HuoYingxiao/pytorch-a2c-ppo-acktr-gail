#!/usr/bin/env bash
set -euo pipefail

# Grid search launcher for KAFE hyperparameters.
# It reuses train.sh and varies:
#   - KAFE_DAMPING
#   - KAFE_MAX_STEP_SIZE
#   - KAFE_TARGET_KL
#   - KAFE_KL_CLIP
#
# Supports multi-GPU parallel execution with simple slot scheduling.

ALGO="kafe_shared"

DAMPINGS=("1e-1")
MAX_STEP_SIZES=("0.1" "0.05" "0.01" "0.005" "0.001")
TARGET_KLS=("0.01" "0.005" "0.001" "0.05" "0.1")
KL_CLIPS=("0.01" "0.005" "0.001" "0.05" "0.1")

GPU_IDS=("0 1")
JOBS_PER_GPU=3
LOG_DIR="grid_logs"
POLL_INTERVAL=5

WANDB_GROUP="kafe-grid"
RUN_PREFIX="kafe-grid"
DRY_RUN=0
START_INDEX=1
END_INDEX=0

TRAIN_ARGS=()

declare -a SLOT_GPUS=()
declare -a SLOT_PIDS=()
declare -a SLOT_RUN_NAMES=()

print_help() {
  cat <<EOF
Usage: ./kafe_grid_search.sh [options] [-- extra_train_sh_args]

Options:
  --dampings <csv>         Comma-separated KAFE_DAMPING values
  --max-step-sizes <csv>   Comma-separated KAFE_MAX_STEP_SIZE values
  --target-kls <csv>       Comma-separated KAFE_TARGET_KL values
  --kl-clips <csv>         Comma-separated KAFE_KL_CLIP values
  --algo <name>            train.sh algo to run (default: kafe_shared)
  --gpu-ids <csv>          Comma-separated physical GPU ids to use (default: 0)
  --jobs-per-gpu <int>     Concurrent runs per GPU (default: 1)
  --log-dir <path>         Directory for per-run logs (default: grid_logs)
  --poll-interval <sec>    Scheduler polling interval in seconds (default: 5)
  --wandb-group <name>     wandb group for all runs (default: kafe-grid)
  --run-prefix <name>      Prefix used in wandb run names (default: kafe-grid)
  --start-index <int>      Start from this 1-based run index
  --end-index <int>        Stop at this 1-based run index (0 means no limit)
  --dry-run                Print commands without launching
  --help, -h               Show this help

Examples:
  ./kafe_grid_search.sh
  ./kafe_grid_search.sh --gpu-ids 0,1 --jobs-per-gpu 1 -- --num-env-steps 2000000
  ./kafe_grid_search.sh --dampings 1e-1,3e-2 --target-kls 0.01,0.003 --gpu-ids 0,1 --jobs-per-gpu 2
  ./kafe_grid_search.sh --start-index 10 --end-index 18 -- --gpu-ids 1
EOF
}

parse_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

slot_count() {
  local gpu_count=${#GPU_IDS[@]}
  echo $((gpu_count * JOBS_PER_GPU))
}

init_slots() {
  local index=0
  local gpu
  local replica
  for gpu in "${GPU_IDS[@]}"; do
    for ((replica = 0; replica < JOBS_PER_GPU; replica++)); do
      SLOT_GPUS[index]="$gpu"
      SLOT_PIDS[index]=""
      SLOT_RUN_NAMES[index]=""
      index=$((index + 1))
    done
  done
}

refresh_slots() {
  local i
  for ((i = 0; i < ${#SLOT_GPUS[@]}; i++)); do
    if [[ -n "${SLOT_PIDS[i]}" ]] && ! kill -0 "${SLOT_PIDS[i]}" 2>/dev/null; then
      wait "${SLOT_PIDS[i]}" || true
      echo "[kafe_grid_search] Slot $i on GPU ${SLOT_GPUS[i]} finished: ${SLOT_RUN_NAMES[i]}"
      SLOT_PIDS[i]=""
      SLOT_RUN_NAMES[i]=""
    fi
  done
}

wait_for_free_slot() {
  while true; do
    refresh_slots
    local i
    for ((i = 0; i < ${#SLOT_GPUS[@]}; i++)); do
      if [[ -z "${SLOT_PIDS[i]}" ]]; then
        echo "$i"
        return 0
      fi
    done
    sleep "$POLL_INTERVAL"
  done
}

wait_for_all_slots() {
  while true; do
    refresh_slots
    local busy=0
    local i
    for ((i = 0; i < ${#SLOT_GPUS[@]}; i++)); do
      if [[ -n "${SLOT_PIDS[i]}" ]]; then
        busy=1
        break
      fi
    done
    if [[ $busy -eq 0 ]]; then
      return 0
    fi
    sleep "$POLL_INTERVAL"
  done
}

sanitize_name() {
  echo "$1" | tr '/ ' '__' | tr -cd '[:alnum:]._=-'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dampings) parse_csv "$2" DAMPINGS; shift 2 ;;
    --max-step-sizes) parse_csv "$2" MAX_STEP_SIZES; shift 2 ;;
    --target-kls) parse_csv "$2" TARGET_KLS; shift 2 ;;
    --kl-clips) parse_csv "$2" KL_CLIPS; shift 2 ;;
    --algo) ALGO="$2"; shift 2 ;;
    --gpu-ids) parse_csv "$2" GPU_IDS; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --end-index) END_INDEX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    --) shift; TRAIN_ARGS+=("$@"); break ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

if [[ ${#DAMPINGS[@]} -eq 0 || ${#MAX_STEP_SIZES[@]} -eq 0 || \
      ${#TARGET_KLS[@]} -eq 0 || ${#KL_CLIPS[@]} -eq 0 || \
      ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "All hyperparameter grids and gpu ids must be non-empty."
  exit 1
fi

if [[ "$JOBS_PER_GPU" -lt 1 ]]; then
  echo "--jobs-per-gpu must be >= 1"
  exit 1
fi

mkdir -p "$LOG_DIR"
init_slots

run_index=0
selected_runs=0
total_runs=$((${#DAMPINGS[@]} * ${#MAX_STEP_SIZES[@]} * ${#TARGET_KLS[@]} * ${#KL_CLIPS[@]}))

echo "[kafe_grid_search] Total combinations: $total_runs"
echo "[kafe_grid_search] DAMPINGS=${DAMPINGS[*]}"
echo "[kafe_grid_search] MAX_STEP_SIZES=${MAX_STEP_SIZES[*]}"
echo "[kafe_grid_search] TARGET_KLS=${TARGET_KLS[*]}"
echo "[kafe_grid_search] KL_CLIPS=${KL_CLIPS[*]}"
echo "[kafe_grid_search] GPU_IDS=${GPU_IDS[*]}"
echo "[kafe_grid_search] JOBS_PER_GPU=$JOBS_PER_GPU"
echo "[kafe_grid_search] LOG_DIR=$LOG_DIR"
echo "[kafe_grid_search] TOTAL_SLOTS=$(slot_count)"

for damping in "${DAMPINGS[@]}"; do
  for max_step_size in "${MAX_STEP_SIZES[@]}"; do
    for target_kl in "${TARGET_KLS[@]}"; do
      for kl_clip in "${KL_CLIPS[@]}"; do
        run_index=$((run_index + 1))

        if [[ $run_index -lt $START_INDEX ]]; then
          continue
        fi
        if [[ $END_INDEX -gt 0 && $run_index -gt $END_INDEX ]]; then
          break 4
        fi

        run_name="${RUN_PREFIX}-$(printf "%03d" "$run_index")-d${damping}-s${max_step_size}-t${target_kl}-k${kl_clip}"
        safe_run_name=$(sanitize_name "$run_name")
        log_path="${LOG_DIR}/${safe_run_name}.log"

        slot_index=0
        assigned_gpu="${GPU_IDS[0]}"
        if [[ $DRY_RUN -eq 0 ]]; then
          slot_index=$(wait_for_free_slot)
          assigned_gpu="${SLOT_GPUS[slot_index]}"
        fi

        cmd=(
          bash train.sh
          --algo "$ALGO"
          --gpu-ids "$assigned_gpu"
          --use-wandb
          --wandb-group "$WANDB_GROUP"
          --wandb-name "$run_name"
          --kafe-damping "$damping"
          --kafe-max-step-size "$max_step_size"
          --kafe-target-kl "$target_kl"
          --kafe-kl-clip "$kl_clip"
        )

        if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
          cmd+=("${TRAIN_ARGS[@]}")
        fi

        selected_runs=$((selected_runs + 1))
        echo
        echo "[kafe_grid_search] Run $run_index/$total_runs"
        echo "[kafe_grid_search] GPU=$assigned_gpu slot=$slot_index log=$log_path"
        printf '[kafe_grid_search] Command:'
        printf ' %q' "${cmd[@]}"
        echo

        if [[ $DRY_RUN -eq 0 ]]; then
          (
            stdbuf -oL -eL "${cmd[@]}"
          ) >"$log_path" 2>&1 &
          SLOT_PIDS[slot_index]="$!"
          SLOT_RUN_NAMES[slot_index]="$run_name"
          echo "[kafe_grid_search] Started pid=${SLOT_PIDS[slot_index]}"
        fi
      done
    done
  done
done

if [[ $DRY_RUN -eq 0 ]]; then
  echo
  echo "[kafe_grid_search] Waiting for all running jobs to finish..."
  wait_for_all_slots
fi

echo
echo "[kafe_grid_search] Finished. Selected runs: $selected_runs"
