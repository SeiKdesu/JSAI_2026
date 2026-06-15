"""
rer_ab.py — baseline(DreamerV3) vs 提案手法 の A/B 比較（full実験の前の妥当性検証）

同一 seed・同一ネットで以下3条件を回し、report 時の診断メトリクスを比較する:
  COND=baseline  reward_event_rec.enable=False（純DreamerV3）
  COND=pct       enable=True, normalize=percentile（configデフォルト）
  COND=mean      enable=True, normalize=mean（HUD再増幅に頑健か検証）

注目メトリクス（agent.py に実装済み）:
  reward_event/recon_mse_event_region   報酬領域の再構成MSE（小さいほど良い）
  reward_event/recon_mse_bg_region      背景の再構成MSE
  reward_event/recon_mse_region_over_bg 報酬領域/背景の比（提案手法で下がってほしい）
  reward_event/scale_ratio_to_rec       補助lossの相対強度
  episode/score                          スコア（toyなので参考値）

使い方:
  COND=baseline SEED=0 python rer_ab.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from rer_experiment import BreakoutLike, patched_make_env  # noqa: F401

COND = os.environ.get('COND', 'baseline')
SEED = os.environ.get('SEED', '0')
STEPS = os.environ.get('STEPS', '20000')


def run():
  from dreamerv3 import main as dmain
  dmain.make_env = patched_make_env

  logdir = os.path.abspath(f'./logdir_ab/{COND}_seed{SEED}')
  enable = 'False' if COND == 'baseline' else 'True'
  normalize = 'mean' if COND == 'mean' else 'percentile'

  argv = [
      '--logdir', logdir,
      '--configs', 'atari', 'debug',
      '--task', 'breakout_like',
      '--jax.platform', 'cpu',
      '--seed', SEED,
      '--run.steps', STEPS,
      '--run.log_every', '8',
      '--run.report_every', '400',
      '--run.train_ratio', '32',
      '--batch_size', '8',
      '--batch_length', '16',
      '--report_length', '16',
      # debug より少し大きめの「再構成が意味を持つ」ネット + 安いimagination
      '--agent.imag_length', '5',
      '--agent.dyn.rssm.deter', '128',
      '--agent.dyn.rssm.hidden', '96',
      '--agent.dyn.rssm.stoch', '8',
      '--agent.dyn.rssm.classes', '8',
      '--agent.dyn.rssm.blocks', '4',
      '--agent.enc.simple.depth', '16',
      '--agent.dec.simple.depth', '16',
      '--agent.enc.simple.units', '128',
      '--agent.dec.simple.units', '128',
      '--agent.enc.simple.layers', '2',
      '--agent.dec.simple.layers', '2',
      # reward_event_rec
      '--agent.reward_event_rec.enable', enable,
      '--agent.reward_event_rec.min_event_count', '1',
      '--agent.reward_event_rec.normalize', normalize,
      '--agent.reward_event_rec.window', '5',
      '--agent.reward_event_rec.alpha', '2.0',
      '--agent.reward_event_rec.beta', '1.0',
      '--agent.reward_event_rec.scale', '0.05',
      '--agent.reward_event_rec.hud_penalty', '0.1',
      '--logger.filter', 'reward_event|episode/score|episode/length|loss/image',
  ]
  print(f'COND={COND} enable={enable} normalize={normalize} '
        f'seed={SEED} steps={STEPS}')
  dmain.main(argv)


if __name__ == '__main__':
  run()
