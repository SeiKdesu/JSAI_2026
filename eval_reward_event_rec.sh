#!/usr/bin/env bash
#
# eval_reward_event_rec.sh
# -----------------------------------------------------------------------------
# reward_event_rec の学習済み checkpoint を評価する。
#
# 何をするか:
#   - 指定した学習ラン (proposed_400k / baseline_400k など) の最新 checkpoint を自動検出
#   - embodied の eval_only スクリプト (mode='eval') で N episodes 実行
#   - 探索ノイズなし・replay 追加なし・gradient update なし
#   - episode/score を metrics.jsonl / scores.jsonl に記録
#
# 使い方:
#   ./eval_reward_event_rec.sh proposed_400k            # seed=0, task=atari_breakout
#   SEED=1 ./eval_reward_event_rec.sh baseline_400k
#   TASK=atari_pong SEED=0 ./eval_reward_event_rec.sh proposed_400k
#
#   # checkpoint を直接指定したい場合:
#   CKPT_PATH=/path/to/ckpt/20260603-120000/ ./eval_reward_event_rec.sh proposed_400k
#
# 結果の確認:
#   python3 analyze_eval.py <eval_logdir>
# -----------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON=python
fi

RUNMODE="${1:-proposed_400k}"
[ $# -gt 0 ] && shift

# ---- 環境変数で上書き可能なパラメータ ----------------------------------------
TASK="${TASK:-atari_breakout}"
SIZE="${SIZE:-size50m}"
SEED="${SEED:-0}"
LOGROOT="${LOGROOT:-$HOME/logdir/reward_event_rec}"
PLATFORM="${PLATFORM:-cuda}"

# eval 設定
# 100 episodes 相当のステップ数を確保する。
# Breakout の平均 episode 長 ~500 agent steps → 100 episodes × 3000 = 300k agent steps で十分。
EVAL_STEPS="${EVAL_STEPS:-300000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

timestamp() { date +%Y%m%d-%H%M%S; }

# ---- 学習ラン logdir から最新 checkpoint を解決する --------------------------
resolve_checkpoint() {
  local run_name="$1"
  # CKPT_PATH が環境変数で直接指定されている場合はそちらを優先
  if [ -n "${CKPT_PATH:-}" ]; then
    echo "${CKPT_PATH}"
    return
  fi

  # 最新の学習 timestamp ディレクトリを探す
  local seed_dir="${LOGROOT}/${TASK}/${run_name}/seed${SEED}"
  if [ ! -d "${seed_dir}" ]; then
    echo "ERROR: Training logdir not found: ${seed_dir}" >&2
    echo "  Run './run_reward_event_rec.sh ${run_name}' first." >&2
    exit 1
  fi

  # 最新の timestamp ディレクトリ (辞書順最後 = 最新)
  local latest_run
  latest_run=$(ls -1 "${seed_dir}" | sort | tail -1)
  if [ -z "${latest_run}" ]; then
    echo "ERROR: No run directories found under ${seed_dir}" >&2
    exit 1
  fi

  local ckpt_dir="${seed_dir}/${latest_run}/ckpt"
  if [ ! -f "${ckpt_dir}/latest" ]; then
    echo "ERROR: No checkpoint found at ${ckpt_dir}/latest" >&2
    echo "  The training run may not have saved a checkpoint yet." >&2
    exit 1
  fi

  local ckpt_name
  ckpt_name=$(cat "${ckpt_dir}/latest")
  local ckpt_path="${ckpt_dir}/${ckpt_name}"

  if [ ! -f "${ckpt_path}/done" ]; then
    echo "ERROR: Checkpoint incomplete (no 'done' file): ${ckpt_path}" >&2
    exit 1
  fi

  echo "${ckpt_path}"
}

# ---- eval 実行 ---------------------------------------------------------------
run_eval() {
  local run_name="$1"
  local ckpt_path
  ckpt_path=$(resolve_checkpoint "${run_name}")

  local eval_logdir="${LOGROOT}/${TASK}/${run_name}_eval/seed${SEED}/$(timestamp)"
  mkdir -p "${eval_logdir}"

  # checkpoint から step 数を推定 (フォルダ名に含まれる場合)
  local ckpt_step
  ckpt_step=$(basename "${ckpt_path}" | grep -oE '[0-9]{12}$' || echo "unknown")

  echo "=============================================================="
  echo " EVAL RUN   : ${run_name}"
  echo " TASK       : ${TASK}    SIZE: ${SIZE}    SEED: ${SEED}"
  echo " CHECKPOINT : ${ckpt_path}"
  echo " CKPT STEP  : ${ckpt_step} (agent steps, ×4 = raw frames)"
  echo " EVAL STEPS : ${EVAL_STEPS} agent steps (>${EVAL_EPISODES} episodes)"
  echo " EVAL LOGDIR: ${eval_logdir}"
  echo "=============================================================="

  set -x
  "${PYTHON}" dreamerv3/main.py \
    --logdir "${eval_logdir}" \
    --configs atari "${SIZE}" \
    --task "${TASK}" \
    --seed "${SEED}" \
    --script eval_only \
    --run.steps "${EVAL_STEPS}" \
    --run.envs 1 \
    --run.from_checkpoint "${ckpt_path}" \
    --jax.platform "${PLATFORM}" \
    --logger.filter "episode/score|episode/length|fps/policy" \
    "$@"
  set +x

  echo ""
  echo ">>> Evaluation complete. Analyzing results..."
  "${PYTHON}" analyze_eval.py "${eval_logdir}" --episodes "${EVAL_EPISODES}" --run "${run_name}"
}

case "${RUNMODE}" in
  proposed_400k)
    run_eval "proposed_400k" "$@"
    ;;

  baseline_400k)
    run_eval "baseline_400k" "$@"
    ;;

  proposed)
    run_eval "proposed" "$@"
    ;;

  baseline)
    run_eval "baseline" "$@"
    ;;

  *)
    # 任意の run_name を受け付ける
    run_eval "${RUNMODE}" "$@"
    ;;
esac
