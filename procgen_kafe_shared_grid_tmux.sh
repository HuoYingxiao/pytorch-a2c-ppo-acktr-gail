#!/usr/bin/env bash
set -euo pipefail

GRID_SCRIPT="./procgen_kafe_shared_grid_search.sh"
SESSION_NAME=""
TMUX_LOG_DIR="tmux_logs"

print_help() {
  cat <<EOF
Usage: ./procgen_kafe_shared_grid_tmux.sh [options] -- [grid_search_args...]

Options:
  --session-name <name>   tmux session name; default auto-generated
  --grid-script <path>    Grid search script path (default: ./procgen_kafe_shared_grid_search.sh)
  --tmux-log-dir <path>   Directory for tmux wrapper logs (default: tmux_logs)
  --help, -h              Show this help

Examples:
  ./procgen_kafe_shared_grid_tmux.sh -- --gpu-ids 0,1 --jobs-per-gpu 2
  ./procgen_kafe_shared_grid_tmux.sh --session-name procgen-grid-a -- --seeds 1 --target-kls 0.01,0.005
EOF
}

GRID_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --grid-script) GRID_SCRIPT="$2"; shift 2 ;;
    --tmux-log-dir) TMUX_LOG_DIR="$2"; shift 2 ;;
    --help|-h) print_help; exit 0 ;;
    --) shift; GRID_ARGS=("$@"); break ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  exit 1
fi

if [[ ! -f "$GRID_SCRIPT" ]]; then
  echo "Grid script not found: $GRID_SCRIPT"
  exit 1
fi

mkdir -p "$TMUX_LOG_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")
if [[ -z "$SESSION_NAME" ]]; then
  SESSION_NAME="procgen-grid-${timestamp}"
fi

tmux_log="${TMUX_LOG_DIR}/${SESSION_NAME}.log"

cmd=(bash "$GRID_SCRIPT")
if [[ ${#GRID_ARGS[@]} -gt 0 ]]; then
  cmd+=("${GRID_ARGS[@]}")
fi

quoted_cmd=$(printf '%q ' "${cmd[@]}")
tmux_shell_cmd="cd $(printf '%q' "$PWD") && stdbuf -oL -eL ${quoted_cmd}> >(tee -a $(printf '%q' "$tmux_log")) 2>&1"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$tmux_shell_cmd"

echo "[procgen_kafe_shared_grid_tmux] Started detached tmux session: $SESSION_NAME"
echo "[procgen_kafe_shared_grid_tmux] Attach: tmux attach -t $SESSION_NAME"
echo "[procgen_kafe_shared_grid_tmux] Log: $tmux_log"
printf '[procgen_kafe_shared_grid_tmux] Command:'
printf ' %q' "${cmd[@]}"
echo
