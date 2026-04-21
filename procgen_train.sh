#!/usr/bin/env bash
set -euo pipefail

# Procgen-only training launcher.
# Edit the values in this section directly for quick experiments.

set_default() {
  local var_name="$1"
  local default_value="$2"
  if [[ -z "${!var_name+x}" ]]; then
    printf -v "$var_name" '%s' "$default_value"
  fi
}

set_default ALGO "kafe_shared"
set_default ENV_NAME "procgen-coinrun"

set_default PROCGEN_DISTRIBUTION_MODE "hard"
set_default PROCGEN_NUM_LEVELS "500"
set_default PROCGEN_START_LEVEL "0"
set_default PROCGEN_EVAL_NUM_LEVELS ""
set_default PROCGEN_EVAL_START_LEVEL ""

set_default SEED "1"
set_default NUM_PROCESSES "32"
set_default NUM_STEPS "256"
set_default NUM_MINI_BATCH "8"
set_default PPO_EPOCH "4"
set_default NUM_ENV_STEPS "25000000"

set_default LR "5e-4"
set_default EPS "1e-5"
set_default CLIP_PARAM "0.01"
set_default VALUE_LOSS_COEF "0.5"
set_default ENTROPY_COEF "0.0"
set_default MAX_GRAD_NORM "5.0"
set_default GAMMA "0.99"
set_default GAE_LAMBDA "0.95"

set_default LOG_INTERVAL "1"
set_default EVAL_INTERVAL "10"
set_default SAVE_INTERVAL "100"
set_default LOG_DIR "/tmp/gym/"
set_default SAVE_DIR "./trained_models/"

set_default USE_GAE "1"
set_default USE_LINEAR_LR_DECAY "1"
set_default USE_PROPER_TIME_LIMITS "0"
set_default RECURRENT_POLICY "0"
set_default CUDA_DETERMINISTIC "0"

set_default USE_CUDA "1"
set_default GPU_IDS "0"

set_default USE_WANDB "1"
set_default WANDB_PROJECT "a2c-ppo-acktr-gail-procgen1"
set_default WANDB_ENTITY ""
set_default WANDB_NAME ""
set_default WANDB_GROUP ""
set_default WANDB_TAGS ""

# KAFE-only options. Ignored unless ALGO is kafe or kafe_shared.
set_default KAFE_DAMPING "1e-1"
set_default KAFE_MAX_STEP_SIZE "0.1"
set_default KAFE_TARGET_KL "0.01"
set_default KAFE_KL_CLIP ""
set_default KAFE_FISHER_CLIP ""
set_default KAFE_KERNEL_NUM_ANCHORS "16"
set_default KAFE_KERNEL_SIGMA "0.1"
set_default KAFE_STATISTIC "logp"
set_default KAFE_CRITIC_LR ""

# Additional raw main.py args can be appended here.
EXTRA_ARGS=()

if [[ "$ALGO" != "ppo" && "$ALGO" != "kafe" && "$ALGO" != "kafe_shared" ]]; then
  echo "procgen_train.sh supports ALGO in {ppo, kafe, kafe_shared}"
  exit 1
fi

if [[ "$ENV_NAME" != procgen-* && "$ENV_NAME" != procgen:* ]]; then
  echo "ENV_NAME must use procgen-<name> or procgen:<name>"
  exit 1
fi

if [[ "$USE_CUDA" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  echo "[procgen_train.sh] CUDA enabled. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  echo "[procgen_train.sh] CPU mode enabled."
fi

CMD=(
  python main.py
  --algo "$ALGO"
  --env-name "$ENV_NAME"
  --seed "$SEED"
  --procgen-distribution-mode "$PROCGEN_DISTRIBUTION_MODE"
  --procgen-num-levels "$PROCGEN_NUM_LEVELS"
  --procgen-start-level "$PROCGEN_START_LEVEL"
  --num-processes "$NUM_PROCESSES"
  --num-steps "$NUM_STEPS"
  --num-mini-batch "$NUM_MINI_BATCH"
  --ppo-epoch "$PPO_EPOCH"
  --num-env-steps "$NUM_ENV_STEPS"
  --lr "$LR"
  --eps "$EPS"
  --clip-param "$CLIP_PARAM"
  --value-loss-coef "$VALUE_LOSS_COEF"
  --entropy-coef "$ENTROPY_COEF"
  --max-grad-norm "$MAX_GRAD_NORM"
  --gamma "$GAMMA"
  --gae-lambda "$GAE_LAMBDA"
  --log-interval "$LOG_INTERVAL"
  --save-interval "$SAVE_INTERVAL"
  --log-dir "$LOG_DIR"
  --save-dir "$SAVE_DIR"
)

if [[ -n "$PROCGEN_EVAL_NUM_LEVELS" ]]; then
  CMD+=(--procgen-eval-num-levels "$PROCGEN_EVAL_NUM_LEVELS")
fi
if [[ -n "$PROCGEN_EVAL_START_LEVEL" ]]; then
  CMD+=(--procgen-eval-start-level "$PROCGEN_EVAL_START_LEVEL")
fi

if [[ "$USE_CUDA" -eq 0 ]]; then
  CMD+=(--no-cuda)
fi
if [[ "$CUDA_DETERMINISTIC" -eq 1 ]]; then
  CMD+=(--cuda-deterministic)
fi
if [[ "$USE_GAE" -eq 1 ]]; then
  CMD+=(--use-gae)
fi
if [[ "$USE_LINEAR_LR_DECAY" -eq 1 ]]; then
  CMD+=(--use-linear-lr-decay)
fi
if [[ "$USE_PROPER_TIME_LIMITS" -eq 1 ]]; then
  CMD+=(--use-proper-time-limits)
fi
if [[ "$RECURRENT_POLICY" -eq 1 ]]; then
  CMD+=(--recurrent-policy)
fi
if [[ -n "$EVAL_INTERVAL" ]]; then
  CMD+=(--eval-interval "$EVAL_INTERVAL")
fi

if [[ "$ALGO" == "kafe" || "$ALGO" == "kafe_shared" ]]; then
  CMD+=(
    --kafe-damping "$KAFE_DAMPING"
    --kafe-max-step-size "$KAFE_MAX_STEP_SIZE"
    --kafe-target-kl "$KAFE_TARGET_KL"
    --kafe-kernel-num-anchors "$KAFE_KERNEL_NUM_ANCHORS"
    --kafe-kernel-sigma "$KAFE_KERNEL_SIGMA"
    --kafe-statistic "$KAFE_STATISTIC"
  )
  if [[ -n "$KAFE_KL_CLIP" ]]; then
    CMD+=(--kafe-kl-clip "$KAFE_KL_CLIP")
  fi
  if [[ -n "$KAFE_FISHER_CLIP" ]]; then
    CMD+=(--kafe-fisher-clip "$KAFE_FISHER_CLIP")
  fi
  if [[ -n "$KAFE_CRITIC_LR" ]]; then
    CMD+=(--kafe-critic-lr "$KAFE_CRITIC_LR")
  fi
fi

if [[ "$USE_WANDB" -eq 1 ]]; then
  CMD+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_NAME" ]]; then
    CMD+=(--wandb-name "$WANDB_NAME")
  fi
  if [[ -n "$WANDB_GROUP" ]]; then
    CMD+=(--wandb-group "$WANDB_GROUP")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    CMD+=(--wandb-tags "$WANDB_TAGS")
  fi
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[procgen_train.sh] Launch command:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
