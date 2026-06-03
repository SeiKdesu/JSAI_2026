#!/usr/bin/env bash
#
# run_reward_event_rec.sh
# -----------------------------------------------------------------------------
# Reward-Event Contrastive Reconstruction (提案手法) を起動するスクリプト。
#
# 何をする手法か:
#   DreamerV3 の reconstruction loss は全画素一様で、Atari のボール/弾のような
#   小さい報酬関連物体を埋もれさせる。本手法は「報酬イベント前後で、非報酬時と
#   比べ特異的に変化する領域」を非学習で抽出 (reward_event_prior) し、その重みで
#   reconstruction loss を空間的に再配分する。
#
# 実装形態 (確定版 = mode: mult):
#   標準 image rec loss を「その場で」重み付けに置換する (A2):
#       loss_image = Σ_hwc  (x_hat - x)^2 · W_eff(h,w)
#   W_eff は空間平均=1 に正規化した重み (A3) なので rec の総量は保たれ、
#   「rec を強めただけ」という交絡なしに容量を報酬物体へ寄せる。
#       W~      = clip(1 + alpha·prior_n, clip_min, clip_max) / mean(...)
#       W_eff   = 1 + gate · blend · (W~ - 1)        # gate: 報酬イベント不足なら 1(=uniform)
#   enable=False で完全に純 DreamerV3 と一致。mode=aux で旧 additive 版に切替可。
#
#   検証(容量制限AEのdose-response, 3seeds): 報酬物体の再構成MSE −29% (normalize=percentile)。
#
# 使い方:
#   ./run_reward_event_rec.sh smoke       # CPU で素早い疎通確認 (debug config, enable=True)
#   ./run_reward_event_rec.sh proposed    # 提案手法 (mode=mult) で本学習
#   ./run_reward_event_rec.sh baseline    # 純 DreamerV3 (enable=False)
#   ./run_reward_event_rec.sh ablation    # 同一 seed で baseline -> proposed を続けて起動
#
# 主要パラメータは環境変数で上書き可能 (例):
#   TASK=atari_pong SEED=1 ./run_reward_event_rec.sh proposed
#   ALPHA=3.0 BLEND=1.0 NORMALIZE=percentile ./run_reward_event_rec.sh proposed
#   REC_MODE=aux SCALE=0.3 ./run_reward_event_rec.sh proposed   # 旧 additive 版を試す
# -----------------------------------------------------------------------------
set -euo pipefail

# ---- リポジトリルートへ移動 -------------------------------------------------
cd "$(dirname "$0")"

# ---- Python 実行コマンドの解決 ----------------------------------------------
PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON=python
fi

# ---- 起動モード (smoke/proposed/baseline/ablation) --------------------------
# 第1引数を起動モードとして取り出し、残りは main.py への追加フラグとして渡す。
RUNMODE="${1:-proposed}"
[ $# -gt 0 ] && shift

# ---- 共通設定 (環境変数で上書き可) ------------------------------------------
# Breakout は小さなボール/ブロックという報酬関連物体が多く、効果検証に好適。
TASK="${TASK:-atari_breakout}"
SIZE="${SIZE:-size50m}"
SEED="${SEED:-0}"
STEPS="${STEPS:-5.1e7}"
LOGROOT="${LOGROOT:-$HOME/logdir/reward_event_rec}"
PLATFORM="${PLATFORM:-cuda}"   # cpu / cuda / tpu

# ---- reward_event_rec ハイパーパラメータ (環境変数で上書き可) ----------------
REC_MODE="${REC_MODE:-mult}"           # mult (確定版, rec本体を再配分) / aux (旧 additive)
BLEND="${BLEND:-1.0}"                  # mult: 再配分強度 W_eff = 1 + blend·(W~-1)
SCALE="${SCALE:-0.05}"                 # aux のみ: 加算 event_rec の loss scale
ALPHA="${ALPHA:-2.0}"
BETA="${BETA:-1.0}"
WINDOW="${WINDOW:-5}"
MIN_EVENT_COUNT="${MIN_EVENT_COUNT:-20}"
NORMALIZE="${NORMALIZE:-percentile}"   # percentile (検証で最良/安定) / mean
PERCENTILE="${PERCENTILE:-95}"
HUD_PENALTY="${HUD_PENALTY:-0.1}"
HUD_HEIGHT_RATIO="${HUD_HEIGHT_RATIO:-0.15}"
CLIP_MIN="${CLIP_MIN:-1.0}"
CLIP_MAX="${CLIP_MAX:-5.0}"
LOG_MAPS="${LOG_MAPS:-True}"

# reward_event 系メトリクスもログに出すフィルタ (既存キー + reward_event)。
FILTER="${FILTER:-score|length|fps|ratio|train/loss/|train/rand/|reward_event}"

timestamp() { date +%Y%m%d-%H%M%S; }

# reward_event_rec の共通フラグを組み立てる。第1引数: enable (True/False)
rer_flags() {
  local enable="$1"
  printf '%s ' \
    "--agent.reward_event_rec.enable" "${enable}" \
    "--agent.reward_event_rec.mode" "${REC_MODE}" \
    "--agent.reward_event_rec.blend" "${BLEND}" \
    "--agent.reward_event_rec.scale" "${SCALE}" \
    "--agent.reward_event_rec.alpha" "${ALPHA}" \
    "--agent.reward_event_rec.beta" "${BETA}" \
    "--agent.reward_event_rec.window" "${WINDOW}" \
    "--agent.reward_event_rec.min_event_count" "${MIN_EVENT_COUNT}" \
    "--agent.reward_event_rec.normalize" "${NORMALIZE}" \
    "--agent.reward_event_rec.percentile" "${PERCENTILE}" \
    "--agent.reward_event_rec.hud_penalty" "${HUD_PENALTY}" \
    "--agent.reward_event_rec.hud_height_ratio" "${HUD_HEIGHT_RATIO}" \
    "--agent.reward_event_rec.clip_min" "${CLIP_MIN}" \
    "--agent.reward_event_rec.clip_max" "${CLIP_MAX}" \
    "--agent.reward_event_rec.log_maps" "${LOG_MAPS}"
}

# 1 ラン起動する。引数: <run_name> <enable True/False> [追加フラグ...]
launch() {
  local name="$1"; shift
  local enable="$1"; shift
  local logdir="${LOGROOT}/${TASK}/${name}/seed${SEED}/$(timestamp)"
  mkdir -p "${logdir}"

  echo "=============================================================="
  echo " RUN        : ${name}"
  echo " TASK       : ${TASK}    SIZE: ${SIZE}    SEED: ${SEED}"
  echo " enable     : ${enable}"
  if [ "${enable}" = "True" ]; then
    echo " rec_mode   : ${REC_MODE} (blend=${BLEND}, scale=${SCALE}[aux only])"
    echo " rer params : alpha=${ALPHA} beta=${BETA} window=${WINDOW}"
    echo "              min_event_count=${MIN_EVENT_COUNT} normalize=${NORMALIZE} pct=${PERCENTILE}"
    echo "              hud_penalty=${HUD_PENALTY} hud_height_ratio=${HUD_HEIGHT_RATIO}"
    echo "              clip=[${CLIP_MIN}, ${CLIP_MAX}]"
  fi
  echo " logdir     : ${logdir}"
  echo "=============================================================="

  set -x
  "${PYTHON}" dreamerv3/main.py \
    --logdir "${logdir}" \
    --configs atari "${SIZE}" \
    --task "${TASK}" \
    --seed "${SEED}" \
    --run.steps "${STEPS}" \
    --jax.platform "${PLATFORM}" \
    --logger.filter "${FILTER}" \
    $(rer_flags "${enable}") \
    "$@"
  set +x
}

case "${RUNMODE}" in
  smoke)
    # 既存コードを壊していないか + 提案手法の shape を CPU で素早く確認する。
    # debug config で小さいネット・短い学習にし、enable=True (mode=mult) で起動。
    smoke_log="${LOGROOT}/smoke/$(timestamp)"
    mkdir -p "${smoke_log}"
    echo ">>> SMOKE TEST (debug config, CPU, enable=True, mode=${REC_MODE})"
    set -x
    "${PYTHON}" dreamerv3/main.py \
      --logdir "${smoke_log}" \
      --configs atari debug \
      --task "${TASK}" \
      --jax.platform cpu \
      --run.steps 1500 \
      --logger.filter "${FILTER}" \
      $(rer_flags True) \
      --agent.reward_event_rec.min_event_count 1
    set +x
    echo ">>> SMOKE 完了: shape error 無く学習でき、reward_event/* が出れば OK"
    ;;

  proposed)
    launch "proposed" True "$@"
    ;;

  baseline)
    # enable=False。提案手法ブロックは丸ごとスキップされ純 DreamerV3 と完全一致。
    launch "baseline" False "$@"
    ;;

  ablation)
    # 同一 seed で baseline と proposed を順に学習し、スコア/再構成で比較する。
    launch "baseline" False "$@"
    launch "proposed" True "$@"
    ;;

  proposed_400k)
    # DyMoDreamer / Atari 100k 比較用: 400k raw frames = 100k agent steps
    # ログ上の step=400000 に対応 (multiplier=4 でログ出力されるため)。
    # save_every=120 (秒) で最終 checkpoint を 400k 直前に保存する。
    STEPS=1e5
    launch "proposed_400k" True --run.save_every 120 "$@"
    ;;

  baseline_400k)
    # DyMoDreamer / Atari 100k 比較用 baseline: 提案手法を無効化した純 DreamerV3。
    # proposed_400k と同一 seed・同一環境設定で実行すること。
    STEPS=1e5
    launch "baseline_400k" False --run.save_every 120 "$@"
    ;;

  *)
    echo "Unknown run mode: ${RUNMODE}" >&2
    echo "Usage: $0 {smoke|proposed|baseline|ablation|proposed_400k|baseline_400k}" >&2
    exit 1
    ;;
esac
