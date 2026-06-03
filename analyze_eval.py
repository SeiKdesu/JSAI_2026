#!/usr/bin/env python3
"""
analyze_eval.py
---------------
eval_reward_event_rec.sh で生成した評価ログを解析し、
DyMoDreamer / Atari 100k と比較できる形式で統計を出力する。

使い方:
  python3 analyze_eval.py <eval_logdir>
  python3 analyze_eval.py <eval_logdir> --episodes 100 --run proposed_400k

出力:
  - mean raw score, std, stderr
  - human-normalized score (HNS)
  - 比較表 (Markdown 形式)

Breakout 標準スコア (DQN論文 / Mnih et al. 2015):
  random  =   1.7
  human   =  30.5
"""

import argparse
import json
import math
import pathlib
import sys


# Atari ゲームごとの random / human スコア (Wang et al. 2016 Dueling DQN 参照)
ATARI_SCORES = {
    'atari_breakout':    {'random':  1.7,  'human':  30.5},
    'atari_pong':        {'random': -20.7, 'human':  9.7},
    'atari_seaquest':    {'random':  68.4, 'human': 42054.7},
    'atari_space_invaders': {'random': 148.0, 'human': 1652.3},
    'atari_qbert':       {'random': 163.9, 'human': 13455.0},
    'atari_beam_rider':  {'random': 363.9, 'human': 7456.0},
}


def load_scores(logdir: pathlib.Path, max_episodes: int) -> list[float]:
    """scores.jsonl から episode/score を max_episodes 件読み込む。"""
    scores_file = logdir / 'scores.jsonl'
    metrics_file = logdir / 'metrics.jsonl'

    scores = []

    # scores.jsonl を優先 (episode/score のみが記録されている)
    target = scores_file if scores_file.exists() else metrics_file
    if not target.exists():
        print(f"ERROR: No metrics file found in {logdir}", file=sys.stderr)
        sys.exit(1)

    with open(target) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'episode/score' in d:
                scores.append(float(d['episode/score']))
                if len(scores) >= max_episodes:
                    break

    return scores


def human_normalized_score(score: float, random: float, human: float) -> float:
    """τ = (score - random) / (human - random)"""
    return (score - random) / (human - random)


def main():
    parser = argparse.ArgumentParser(description='Analyze eval episode scores.')
    parser.add_argument('logdir', type=pathlib.Path,
                        help='eval logdir (contains scores.jsonl or metrics.jsonl)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--run', type=str, default='',
                        help='Run name label for output table')
    parser.add_argument('--task', type=str, default='atari_breakout',
                        help='Task name for HNS lookup')
    parser.add_argument('--checkpoint-step', type=int, default=400000,
                        help='Checkpoint raw frame step (default: 400000)')
    args = parser.parse_args()

    logdir = args.logdir
    if not logdir.exists():
        print(f"ERROR: Logdir not found: {logdir}", file=sys.stderr)
        sys.exit(1)

    scores = load_scores(logdir, args.episodes)
    n = len(scores)

    if n == 0:
        print(f"ERROR: No episode scores found in {logdir}", file=sys.stderr)
        sys.exit(1)

    if n < args.episodes:
        print(f"WARNING: Only {n} episodes found (requested {args.episodes}).",
              file=sys.stderr)

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n)

    # Human-normalized score
    task_key = args.task if args.task in ATARI_SCORES else 'atari_breakout'
    ref = ATARI_SCORES[task_key]
    hns = human_normalized_score(mean, ref['random'], ref['human'])

    run_label = args.run or logdir.parent.parent.name  # 親ディレクトリ名をフォールバック

    # ---- 出力 ----------------------------------------------------------------
    print()
    print(f"Task        : {task_key}")
    print(f"Run         : {run_label}")
    print(f"Episodes    : {n}")
    print(f"Mean score  : {mean:.2f}")
    print(f"Std         : {std:.2f}")
    print(f"Stderr      : {stderr:.2f}")
    print(f"HNS         : {hns:.4f}  (random={ref['random']}, human={ref['human']})")
    print()

    # Markdown テーブル (DyMoDreamer 比較用)
    print("| run | checkpoint step (raw frames) | eval episodes | mean raw score | std | stderr | HNS |")
    print("|-----|-----------------------------:|-------------:|---------------:|----:|-------:|----:|")
    print(f"| {run_label:<20} | {args.checkpoint_step:>28,} | {n:>12} | {mean:>14.2f} | {std:>3.2f} | {stderr:>6.2f} | {hns:.4f} |")
    print()

    # JSON でも保存 (eval_logdir/result.json)
    result = {
        'run': run_label,
        'task': task_key,
        'episodes': n,
        'checkpoint_step_raw_frames': args.checkpoint_step,
        'mean_raw_score': round(mean, 4),
        'std': round(std, 4),
        'stderr': round(stderr, 4),
        'hns': round(hns, 6),
        'random_score': ref['random'],
        'human_score': ref['human'],
        'all_scores': scores,
    }
    result_path = logdir / 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {result_path}")


if __name__ == '__main__':
    main()
