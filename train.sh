#!/usr/bin/env bash
set -euo pipefail

# Unified training launcher for this repository.
# Supports algorithm selection, GPU visibility, env parallelism and wandb settings.
conda activate pytorch-a2c-ppo-acktr-gail

ALGO="kafe_shared"
ENV_NAME="PongNoFrameskip-v4"
SEED=1
NUM_PROCESSES=8
NUM_STEPS=128
NUM_MINI_BATCH=4
PPO_EPOCH=1
NUM_ENV_STEPS=10000000
LOG_INTERVAL=1
EVAL_INTERVAL=""
SAVE_INTERVAL=100
USE_GAE=1
USE_LINEAR_LR_DECAY=1
USE_PROPER_TIME_LIMITS=0

LR="2.5e-4"
CLIP_PARAM="0.1"
VALUE_LOSS_COEF="0.5"
ENTROPY_COEF="0.01"
MAX_GRAD_NORM="0.5"
GAMMA="0.99"
GAE_LAMBDA="0.95"

# KAFE preset group
KAFE_DAMPING="1e-1"
KAFE_MAX_STEP_SIZE="0.1"
KAFE_TARGET_KL="0.01"
KAFE_KL_CLIP="0.1"
KAFE_KERNEL_NUM_ANCHORS="16"
KAFE_KERNEL_SIGMA="0.1"
KAFE_STATISTIC="logp"
KAFE_CRITIC_LR=""

# ACKTR preset group
ACKTR_LR="7e-4"
ACKTR_EPS="1e-5"
ACKTR_ALPHA="0.99"
ACKTR_NUM_STEPS="20"
ACKTR_NUM_PROCESSES="32"
ACKTR_USE_GAE=0

USE_CUDA=1
GPU_IDS="1"

USE_WANDB=1
WANDB_PROJECT="a2c-ppo-acktr-gail"
WANDB_ENTITY=""
WANDB_NAME=""
WANDB_GROUP=""
WANDB_TAGS=""

EXTRA_ARGS=()

print_help() {
  cat <<EOF
Usage: ./train.sh [options] [-- extra_main_args]

Core options:
  --algo <a2c|ppo|acktr|kafe|kafe_shared>  Algorithm (default: ppo)
  --env-name <name>                Gym environment id (default: PongNoFrameskip-v4)
  --seed <int>                     Random seed (default: 1)

Parallel and GPU options:
  --num-processes <int>            Number of env workers (default: 8)
  --num-steps <int>                Rollout length per worker (default: 128)
  --gpu-ids <ids>                  CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1 (default: 0)
  --cpu                            Force CPU mode

Training hyperparameters:
  --lr <float>                     Learning rate (default: 2.5e-4)
  --clip-param <float>             PPO/KAFE clip param (default: 0.1)
  --value-loss-coef <float>        Value loss coefficient (default: 0.5)
  --entropy-coef <float>           Entropy coefficient (default: 0.01)
  --max-grad-norm <float>          Max grad norm (default: 0.5)
  --gamma <float>                  Discount factor (default: 0.99)
  --gae-lambda <float>             GAE lambda (default: 0.95)
  --ppo-epoch <int>                PPO/KAFE update epochs (default: 4)
  --num-mini-batch <int>           PPO/KAFE mini-batches (default: 4)
  --num-env-steps <int>            Total env steps (default: 10000000)

Algorithm parameter groups:
  KAFE group (used when --algo kafe):
    --kafe-damping <float>         (default: 1e-2)
    --kafe-max-step-size <float>   (default: 0.05)
    --kafe-target-kl <float>       (default: 0.01)
    --kafe-kl-clip <float>         (default: use --kafe-target-kl)
    --kafe-kernel-num-anchors <n>  (default: 16)
    --kafe-kernel-sigma <float>    (default: 1.0)
    --kafe-statistic <name>        logits|probs|logp|score (default: logp)
    --kafe-critic-lr <float>       (default: use --lr)

  ACKTR group (used when --algo acktr):
    --acktr-lr <float>             (default: 7e-4)
    --acktr-eps <float>            (default: 1e-5)
    --acktr-alpha <float>          (default: 0.99)
    --acktr-num-steps <int>        (default: 20)
    --acktr-num-processes <int>    (default: 32)
    --acktr-use-gae                Enable GAE for ACKTR preset (default: off)

Logging/eval/checkpoint:
  --log-interval <int>             Print interval in updates (default: 1)
  --eval-interval <int>            Eval interval in updates (optional)
  --save-interval <int>            Save interval in updates (default: 100)

Feature toggles:
  --no-gae                         Disable GAE
  --no-linear-lr-decay             Disable linear LR decay
  --use-proper-time-limits         Enable proper time limit handling

wandb options:
  --use-wandb                      Enable wandb logging
  --wandb-project <name>           wandb project
  --wandb-entity <name>            wandb entity/team
  --wandb-name <name>              wandb run name
  --wandb-group <name>             wandb group
  --wandb-tags <csv>               wandb tags, comma-separated

Examples:
  ./train.sh --algo ppo --env-name PongNoFrameskip-v4 --gpu-ids 0 --num-processes 8 --use-wandb --wandb-project rl-debug --wandb-name pong-ppo
  ./train.sh --algo a2c --env-name BreakoutNoFrameskip-v4 --cpu --num-processes 16
  ./train.sh --algo kafe_shared --env-name PongNoFrameskip-v4 --gpu-ids 1 -- --kafe-target-kl 0.02
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo) ALGO="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;

    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --num-steps) NUM_STEPS="$2"; shift 2 ;;
    --num-mini-batch) NUM_MINI_BATCH="$2"; shift 2 ;;
    --ppo-epoch) PPO_EPOCH="$2"; shift 2 ;;
    --num-env-steps) NUM_ENV_STEPS="$2"; shift 2 ;;

    --lr) LR="$2"; shift 2 ;;
    --clip-param) CLIP_PARAM="$2"; shift 2 ;;
    --value-loss-coef) VALUE_LOSS_COEF="$2"; shift 2 ;;
    --entropy-coef) ENTROPY_COEF="$2"; shift 2 ;;
    --max-grad-norm) MAX_GRAD_NORM="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --gae-lambda) GAE_LAMBDA="$2"; shift 2 ;;

    --kafe-damping) KAFE_DAMPING="$2"; shift 2 ;;
    --kafe-max-step-size) KAFE_MAX_STEP_SIZE="$2"; shift 2 ;;
    --kafe-target-kl) KAFE_TARGET_KL="$2"; shift 2 ;;
    --kafe-kl-clip) KAFE_KL_CLIP="$2"; shift 2 ;;
    --kafe-kernel-num-anchors) KAFE_KERNEL_NUM_ANCHORS="$2"; shift 2 ;;
    --kafe-kernel-sigma) KAFE_KERNEL_SIGMA="$2"; shift 2 ;;
    --kafe-statistic) KAFE_STATISTIC="$2"; shift 2 ;;
    --kafe-critic-lr) KAFE_CRITIC_LR="$2"; shift 2 ;;

    --acktr-lr) ACKTR_LR="$2"; shift 2 ;;
    --acktr-eps) ACKTR_EPS="$2"; shift 2 ;;
    --acktr-alpha) ACKTR_ALPHA="$2"; shift 2 ;;
    --acktr-num-steps) ACKTR_NUM_STEPS="$2"; shift 2 ;;
    --acktr-num-processes) ACKTR_NUM_PROCESSES="$2"; shift 2 ;;
    --acktr-use-gae) ACKTR_USE_GAE=1; shift ;;

    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;

    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --cpu) USE_CUDA=0; shift ;;

    --no-gae) USE_GAE=0; shift ;;
    --no-linear-lr-decay) USE_LINEAR_LR_DECAY=0; shift ;;
    --use-proper-time-limits) USE_PROPER_TIME_LIMITS=1; shift ;;

    --use-wandb) USE_WANDB=1; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-name) WANDB_NAME="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --wandb-tags) WANDB_TAGS="$2"; shift 2 ;;

    --help|-h) print_help; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

case "$ALGO" in
  a2c|ppo|acktr|kafe|kafe_shared) ;;
  *)
    echo "Unsupported algo: $ALGO"
    echo "Use one of: a2c, ppo, acktr, kafe, kafe_shared"
    exit 1
    ;;
esac

# Apply algorithm-specific preset groups.
if [[ "$ALGO" == "acktr" ]]; then
  LR="$ACKTR_LR"
  NUM_STEPS="$ACKTR_NUM_STEPS"
  NUM_PROCESSES="$ACKTR_NUM_PROCESSES"
  USE_GAE="$ACKTR_USE_GAE"
  echo "[train.sh] Using ACKTR preset group."
fi

if [[ "$USE_CUDA" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  echo "[train.sh] CUDA enabled. CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  echo "[train.sh] CPU mode enabled."
fi

CMD=(
  python main.py
  --algo "$ALGO"
  --env-name "$ENV_NAME"
  --seed "$SEED"
  --num-processes "$NUM_PROCESSES"
  --num-steps "$NUM_STEPS"
  --num-mini-batch "$NUM_MINI_BATCH"
  --ppo-epoch "$PPO_EPOCH"
  --num-env-steps "$NUM_ENV_STEPS"
  --lr "$LR"
  --clip-param "$CLIP_PARAM"
  --value-loss-coef "$VALUE_LOSS_COEF"
  --entropy-coef "$ENTROPY_COEF"
  --max-grad-norm "$MAX_GRAD_NORM"
  --gamma "$GAMMA"
  --gae-lambda "$GAE_LAMBDA"
  --log-interval "$LOG_INTERVAL"
  --save-interval "$SAVE_INTERVAL"
)

if [[ "$USE_CUDA" -eq 0 ]]; then
  CMD+=(--no-cuda)
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
if [[ -n "$EVAL_INTERVAL" ]]; then
  CMD+=(--eval-interval "$EVAL_INTERVAL")
fi

if [[ "$ALGO" == "kafe" || "$ALGO" == "kafe_shared" ]]; then
  echo "[train.sh] Using KAFE preset group."
  CMD+=(
    --kafe-damping "$KAFE_DAMPING"
    --kafe-max-step-size "$KAFE_MAX_STEP_SIZE"
    --kafe-target-kl "$KAFE_TARGET_KL"
    --kafe-kl-clip "$KAFE_KL_CLIP"
    --kafe-kernel-num-anchors "$KAFE_KERNEL_NUM_ANCHORS"
    --kafe-kernel-sigma "$KAFE_KERNEL_SIGMA"
    --kafe-statistic "$KAFE_STATISTIC"
  )
  if [[ -n "$KAFE_CRITIC_LR" ]]; then
    CMD+=(--kafe-critic-lr "$KAFE_CRITIC_LR")
  fi
fi

if [[ "$ALGO" == "acktr" ]]; then
  CMD+=(
    --eps "$ACKTR_EPS"
    --alpha "$ACKTR_ALPHA"
  )
fi

if [[ "$USE_WANDB" -eq 1 ]]; then
  CMD+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then CMD+=(--wandb-entity "$WANDB_ENTITY"); fi
  if [[ -n "$WANDB_NAME" ]]; then CMD+=(--wandb-name "$WANDB_NAME"); fi
  if [[ -n "$WANDB_GROUP" ]]; then CMD+=(--wandb-group "$WANDB_GROUP"); fi
  if [[ -n "$WANDB_TAGS" ]]; then CMD+=(--wandb-tags "$WANDB_TAGS"); fi
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[train.sh] Launch command:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
