#!/usr/bin/env bash
set -euo pipefail

# Grid search launcher for Procgen + KAFE shared.
# This script launches procgen_train.sh with environment-variable overrides so
# you can keep the single-run defaults in procgen_train.sh and sweep only the
# parameters you care about here.

LAUNCHER="./procgen_train.sh"

ENV_NAME="procgen-coinrun"
PROCGEN_DISTRIBUTION_MODE="hard"
PROCGEN_NUM_LEVELS="500"
PROCGEN_START_LEVEL="0"
PROCGEN_EVAL_NUM_LEVELS=""
PROCGEN_EVAL_START_LEVEL=""

SEEDS=("1")
DAMPINGS=("1e-1")
MAX_STEP_SIZES=("0.5" "0.1" "0.01" "0.05")
TARGET_KLS=("0.01" "0.005" "0.001" "0.1")
KL_CLIPS=("none")
FISHER_CLIPS=("0.1" "0.05" "0.01")
SEARCH_MODE="full"

GPU_IDS=("0")
JOBS_PER_GPU=1
LOG_DIR="grid_logs/procgen_kafe_shared"
POLL_INTERVAL=5

WANDB_GROUP="procgen-kafe-shared-grid"
RUN_PREFIX="procgen-kafe-shared"
DRY_RUN=0
START_INDEX=1
END_INDEX=0

EXTRA_ENV_VARS=()
EXTRA_ENV_VARS+=("NUM_ENV_STEPS=10000000")

declare -a SLOT_GPUS=()
declare -a SLOT_PIDS=()
declare -a SLOT_RUN_NAMES=()

print_help() {
  cat <<EOF
Usage: ./procgen_kafe_shared_grid_search.sh [options]

Options:
  --launcher <path>               Launcher script to execute (default: ./procgen_train.sh)
  --env-name <name>               Procgen env id (default: procgen-coinrun)
  --procgen-distribution-mode <m> Procgen distribution mode (default: hard)
  --procgen-num-levels <int>      Procgen train num_levels (default: 500)
  --procgen-start-level <int>     Procgen train start_level (default: 0)
  --procgen-eval-num-levels <int> Procgen eval num_levels (default: disabled)
  --procgen-eval-start-level <i>  Procgen eval start_level (default: disabled)
  --seeds <csv>                   Comma-separated seeds (default: 1)
  --dampings <csv>                Comma-separated KAFE_DAMPING values
  --max-step-sizes <list>         KAFE_MAX_STEP_SIZE values; comma or space separated
  --kafe-max-step-size <list>     Alias of --max-step-sizes
  --target-kls <list>             KAFE_TARGET_KL values; comma or space separated
  --kl-clips <list>               KAFE_KL_CLIP values; use 'none' to disable (default: none)
  --kafe-fisher-clip <list>       KAFE_FISHER_CLIP values; use 'none' to disable
  --search-mode <mode>            'focused' (default) or 'full'
  --gpu-ids <csv>                 Comma-separated physical GPU ids (default: 0)
  --jobs-per-gpu <int>            Concurrent runs per GPU (default: 1)
  --log-dir <path>                Directory for per-run logs
  --poll-interval <sec>           Scheduler polling interval (default: 5)
  --wandb-group <name>            wandb group shared by all runs
  --run-prefix <name>             Prefix used in wandb run names
  --start-index <int>             Start from this 1-based run index
  --end-index <int>               Stop at this 1-based run index (0 means no limit)
  --set <KEY=VALUE>               Extra environment override passed to procgen_train.sh
  --dry-run                       Print commands without launching
  --help, -h                      Show this help

Examples:
  ./procgen_kafe_shared_grid_search.sh
  ./procgen_kafe_shared_grid_search.sh --gpu-ids 0,1 --jobs-per-gpu 1
  ./procgen_kafe_shared_grid_search.sh --search-mode full
  ./procgen_kafe_shared_grid_search.sh --seeds 1,2,3 --target-kls 0.01,0.005 --kl-clips none,0.01
  ./procgen_kafe_shared_grid_search.sh --set NUM_ENV_STEPS=10000000 --set NUM_PROCESSES=16
EOF
}

parse_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "$value"
}

parse_list_flexible() {
  local -n out_ref="$1"
  shift
  local raw=""
  local token
  for token in "$@"; do
    if [[ -n "$raw" ]]; then
      raw+=","
    fi
    raw+="$token"
  done

  out_ref=()
  local part
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part=$(trim_spaces "$part")
    if [[ -n "$part" ]]; then
      out_ref+=("$part")
    fi
  done
}

consume_list_args() {
  local target_name="$1"
  shift
  local -n target_ref="$target_name"
  local consumed_ref_name="$1"
  shift
  local -n consumed_ref="$consumed_ref_name"

  local values=()
  local index=1
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --* ]]; then
      break
    fi
    values+=("$1")
    shift
    index=$((index + 1))
  done

  if [[ ${#values[@]} -eq 0 ]]; then
    echo "Missing list value for option."
    exit 1
  fi

  parse_list_flexible target_ref "${values[@]}"
  consumed_ref=$((index - 1))
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
      echo "[procgen_kafe_shared_grid] Slot $i on GPU ${SLOT_GPUS[i]} finished: ${SLOT_RUN_NAMES[i]}" >&2
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

should_run_combo() {
  local max_step_size="$1"
  local target_kl="$2"
  local kl_clip="$3"
  local fisher_clip="$4"

  if [[ "$SEARCH_MODE" == "full" ]]; then
    return 0
  fi

  if [[ "$SEARCH_MODE" != "focused" ]]; then
    echo "Unknown search mode: $SEARCH_MODE" >&2
    exit 1
  fi

  # Focused mode keeps only trust-region settings where the outer step cap
  # is strictly larger than the KL target.
  if awk "BEGIN { exit !($max_step_size > $target_kl) }"; then
    return 0
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --procgen-distribution-mode) PROCGEN_DISTRIBUTION_MODE="$2"; shift 2 ;;
    --procgen-num-levels) PROCGEN_NUM_LEVELS="$2"; shift 2 ;;
    --procgen-start-level) PROCGEN_START_LEVEL="$2"; shift 2 ;;
    --procgen-eval-num-levels) PROCGEN_EVAL_NUM_LEVELS="$2"; shift 2 ;;
    --procgen-eval-start-level) PROCGEN_EVAL_START_LEVEL="$2"; shift 2 ;;
    --seeds)
      consumed=0
      consume_list_args SEEDS consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --dampings)
      consumed=0
      consume_list_args DAMPINGS consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --max-step-sizes|--kafe-max-step-size)
      consumed=0
      consume_list_args MAX_STEP_SIZES consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --target-kls)
      consumed=0
      consume_list_args TARGET_KLS consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --kl-clips)
      consumed=0
      consume_list_args KL_CLIPS consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --kafe-fisher-clip)
      consumed=0
      consume_list_args FISHER_CLIPS consumed "${@:2}"
      shift $((consumed + 1))
      ;;
    --search-mode) SEARCH_MODE="$2"; shift 2 ;;
    --gpu-ids) parse_csv "$2" GPU_IDS; shift 2 ;;
    --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --end-index) END_INDEX="$2"; shift 2 ;;
    --set) EXTRA_ENV_VARS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) print_help; exit 0 ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

if [[ ! -f "$LAUNCHER" ]]; then
  echo "Launcher not found: $LAUNCHER"
  exit 1
fi

if [[ ${#SEEDS[@]} -eq 0 || ${#DAMPINGS[@]} -eq 0 || \
      ${#MAX_STEP_SIZES[@]} -eq 0 || ${#TARGET_KLS[@]} -eq 0 || \
      ${#KL_CLIPS[@]} -eq 0 || ${#FISHER_CLIPS[@]} -eq 0 || \
      ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "All hyperparameter grids and gpu ids must be non-empty."
  exit 1
fi

if [[ "$JOBS_PER_GPU" -lt 1 ]]; then
  echo "--jobs-per-gpu must be >= 1"
  exit 1
fi

if [[ "$SEARCH_MODE" != "focused" && "$SEARCH_MODE" != "full" ]]; then
  echo "--search-mode must be one of: focused, full"
  exit 1
fi

mkdir -p "$LOG_DIR"
init_slots

run_index=0
selected_runs=0
total_runs=0
for seed in "${SEEDS[@]}"; do
  for damping in "${DAMPINGS[@]}"; do
    for max_step_size in "${MAX_STEP_SIZES[@]}"; do
      for target_kl in "${TARGET_KLS[@]}"; do
        for kl_clip in "${KL_CLIPS[@]}"; do
          for fisher_clip in "${FISHER_CLIPS[@]}"; do
            if should_run_combo "$max_step_size" "$target_kl" "$kl_clip" "$fisher_clip"; then
              total_runs=$((total_runs + 1))
            fi
          done
        done
      done
    done
  done
done

echo "[procgen_kafe_shared_grid] Total combinations: $total_runs"
echo "[procgen_kafe_shared_grid] SEARCH_MODE=$SEARCH_MODE"
echo "[procgen_kafe_shared_grid] SEEDS=${SEEDS[*]}"
echo "[procgen_kafe_shared_grid] DAMPINGS=${DAMPINGS[*]}"
echo "[procgen_kafe_shared_grid] MAX_STEP_SIZES=${MAX_STEP_SIZES[*]}"
echo "[procgen_kafe_shared_grid] TARGET_KLS=${TARGET_KLS[*]}"
echo "[procgen_kafe_shared_grid] KL_CLIPS=${KL_CLIPS[*]}"
echo "[procgen_kafe_shared_grid] FISHER_CLIPS=${FISHER_CLIPS[*]}"
echo "[procgen_kafe_shared_grid] GPU_IDS=${GPU_IDS[*]}"
echo "[procgen_kafe_shared_grid] JOBS_PER_GPU=$JOBS_PER_GPU"
echo "[procgen_kafe_shared_grid] LOG_DIR=$LOG_DIR"
echo "[procgen_kafe_shared_grid] TOTAL_SLOTS=$(slot_count)"

for seed in "${SEEDS[@]}"; do
  for damping in "${DAMPINGS[@]}"; do
    for max_step_size in "${MAX_STEP_SIZES[@]}"; do
      for target_kl in "${TARGET_KLS[@]}"; do
        for kl_clip in "${KL_CLIPS[@]}"; do
          for fisher_clip in "${FISHER_CLIPS[@]}"; do
            if ! should_run_combo "$max_step_size" "$target_kl" "$kl_clip" "$fisher_clip"; then
              continue
            fi

            run_index=$((run_index + 1))

            if [[ $run_index -lt $START_INDEX ]]; then
              continue
            fi
            if [[ $END_INDEX -gt 0 && $run_index -gt $END_INDEX ]]; then
              break 6
            fi

            kl_label="$kl_clip"
            if [[ "$kl_label" == "none" ]]; then
              kl_label="off"
            fi

            fisher_label="$fisher_clip"
            if [[ "$fisher_label" == "none" ]]; then
              fisher_label="off"
            fi

            run_name="${RUN_PREFIX}-$(printf "%03d" "$run_index")-${ENV_NAME#procgen-}-seed${seed}-d${damping}-s${max_step_size}-t${target_kl}-k${kl_label}-f${fisher_label}"
            safe_run_name=$(sanitize_name "$run_name")
            log_path="${LOG_DIR}/${safe_run_name}.log"

            slot_index=0
            assigned_gpu="${GPU_IDS[0]}"
            if [[ $DRY_RUN -eq 0 ]]; then
              slot_index=$(wait_for_free_slot)
              assigned_gpu="${SLOT_GPUS[slot_index]}"
            fi

            env_cmd=(
              env
              ALGO="kafe_shared"
              ENV_NAME="$ENV_NAME"
              PROCGEN_DISTRIBUTION_MODE="$PROCGEN_DISTRIBUTION_MODE"
              PROCGEN_NUM_LEVELS="$PROCGEN_NUM_LEVELS"
              PROCGEN_START_LEVEL="$PROCGEN_START_LEVEL"
              SEED="$seed"
              GPU_IDS="$assigned_gpu"
              WANDB_GROUP="$WANDB_GROUP"
              WANDB_NAME="$run_name"
              KAFE_DAMPING="$damping"
              KAFE_MAX_STEP_SIZE="$max_step_size"
              KAFE_TARGET_KL="$target_kl"
            )

            if [[ -n "$PROCGEN_EVAL_NUM_LEVELS" ]]; then
              env_cmd+=(PROCGEN_EVAL_NUM_LEVELS="$PROCGEN_EVAL_NUM_LEVELS")
            fi
            if [[ -n "$PROCGEN_EVAL_START_LEVEL" ]]; then
              env_cmd+=(PROCGEN_EVAL_START_LEVEL="$PROCGEN_EVAL_START_LEVEL")
            fi
            if [[ "$kl_clip" == "none" ]]; then
              env_cmd+=(KAFE_KL_CLIP="")
            else
              env_cmd+=(KAFE_KL_CLIP="$kl_clip")
            fi
            if [[ "$fisher_clip" == "none" ]]; then
              env_cmd+=(KAFE_FISHER_CLIP="")
            else
              env_cmd+=(KAFE_FISHER_CLIP="$fisher_clip")
            fi
            if [[ ${#EXTRA_ENV_VARS[@]} -gt 0 ]]; then
              env_cmd+=("${EXTRA_ENV_VARS[@]}")
            fi

            cmd=("${env_cmd[@]}" bash "$LAUNCHER")

            selected_runs=$((selected_runs + 1))
            echo
            echo "[procgen_kafe_shared_grid] Run $run_index/$total_runs"
            echo "[procgen_kafe_shared_grid] GPU=$assigned_gpu slot=$slot_index log=$log_path"
            printf '[procgen_kafe_shared_grid] Command:'
            printf ' %q' "${cmd[@]}"
            echo

            if [[ $DRY_RUN -eq 0 ]]; then
              (
                stdbuf -oL -eL "${cmd[@]}"
              ) >"$log_path" 2>&1 &
              SLOT_PIDS[slot_index]="$!"
              SLOT_RUN_NAMES[slot_index]="$run_name"
              echo "[procgen_kafe_shared_grid] Started pid=${SLOT_PIDS[slot_index]}"
            fi
          done
        done
      done
    done
  done
done

if [[ $DRY_RUN -eq 0 ]]; then
  echo
  echo "[procgen_kafe_shared_grid] Waiting for all running jobs to finish..."
  wait_for_all_slots
fi

echo "[procgen_kafe_shared_grid] Finished. Selected runs: $selected_runs"
