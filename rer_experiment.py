"""
rer_experiment.py — 実験0: Reward-Event Contrastive Reconstruction の loss 発火確認

ALE (実Breakout) は py3.13/mac で wheel が無いため、Breakout を模した合成環境
`BreakoutLike` を使って DreamerV3 の本物の学習ループ (embodied.run.train) を回し、
agent.loss() 内の reward_event_prior が実際に発火するかを確認する。

BreakoutLike の作り:
  - 上部に HUD 帯 (毎ステップ点滅 = 報酬非相関の大域変化、HUD penalty の検証用)
  - 動くボール (2x2)、下部パドル (行動で左右)
  - ブロック群。ボールがブロックに当たると消えて reward=1（報酬相関の局所変化）
これにより
  base_map: HUD とボール軌跡で高い
  event_map: それ + ブロック消失領域
  prior = ReLU(event - beta*base): HUD/軌跡は相殺、ブロック消失だけ残る
が期待される。

使い方:
  python rer_experiment.py            # 既定: min_event_count=1, debug config, CPU
"""
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import elements
import embodied


class BreakoutLike(embodied.Env):

  def __init__(self, task, size=(64, 64), length=300, seed=0):
    del task
    self.H, self.W = int(size[0]), int(size[1])
    self.length = length
    self.rng = np.random.default_rng(seed)
    self.hud_h = max(2, int(0.15 * self.H))  # 上部 HUD 帯
    self.count = 0
    self.done = True
    self._reset_state()

  def _reset_state(self):
    H, W = self.H, self.W
    self.count = 0
    self.done = False
    # ボール
    self.by = H // 2
    self.bx = W // 2
    self.vy = -1
    self.vx = 1 if self.rng.random() < 0.5 else -1
    # パドル
    self.px = W // 2
    # ブロック: HUD 直下に数行。True=存在
    self.brick_rows = list(range(self.hud_h + 2, self.hud_h + 8))
    self.bricks = np.ones((len(self.brick_rows), W), bool)
    # まばらに（4列に1個）ブロックを置く
    self.bricks[:, ::4] = True
    self.bricks[:, [i for i in range(W) if i % 4 != 0]] = False

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (self.H, self.W, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.int32, (), 0, 3),  # 0 左,1 停,2 右
    }

  def step(self, action):
    if action['reset'] or self.done:
      self._reset_state()
      return self._obs(0.0, is_first=True)

    self.count += 1
    # パドル
    a = int(action['action'])
    self.px = int(np.clip(self.px + (a - 1) * 2, 2, self.W - 3))

    # ボール移動
    self.by += self.vy
    self.bx += self.vx
    # 壁反射
    if self.bx <= 1 or self.bx >= self.W - 2:
      self.vx *= -1
      self.bx = int(np.clip(self.bx, 1, self.W - 2))
    if self.by <= self.hud_h:
      self.vy *= -1
      self.by = self.hud_h + 1

    reward = 0.0
    # ブロック衝突
    if self.by in self.brick_rows:
      ri = self.brick_rows.index(self.by)
      if self.bricks[ri, self.bx]:
        self.bricks[ri, self.bx] = False
        reward = 1.0
        self.vy *= -1  # 跳ね返る
      if not self.bricks.any():  # 全消し→補充
        self.bricks[:, ::4] = True
        self.bricks[:, [i for i in range(self.W) if i % 4 != 0]] = False

    # 下端
    if self.by >= self.H - 2:
      # パドルで拾えれば反射、外せば終了
      if abs(self.bx - self.px) <= 3:
        self.vy *= -1
        self.by = self.H - 3
      else:
        self.done = True

    if self.count >= self.length:
      self.done = True

    return self._obs(
        reward, is_last=self.done, is_terminal=self.done and
        (self.by >= self.H - 2))

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    H, W = self.H, self.W
    img = np.zeros((H, W, 3), np.uint8)
    # HUD: 毎ステップ点滅する大域変化（報酬非相関の distractor）
    hud_val = (self.count * 53) % 200 + 30
    img[:self.hud_h, :, :] = hud_val
    # ブロック（緑）
    for ri, row in enumerate(self.brick_rows):
      cols = np.where(self.bricks[ri])[0]
      img[row, cols, 1] = 200
    # ボール（白 2x2）
    by, bx = int(self.by), int(self.bx)
    img[by:by + 2, bx:bx + 2, :] = 255
    # パドル（赤）
    img[H - 2:H, self.px - 3:self.px + 3, 0] = 230
    return dict(
        image=img,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )


def patched_make_env(config, index, **overrides):
  return BreakoutLike('breakout', size=(64, 64), length=300, seed=index)


def run():
  from dreamerv3 import main as dmain
  dmain.make_env = patched_make_env  # 環境を差し替え

  logdir = os.path.abspath('./logdir_rer_exp0')
  argv = [
      '--logdir', logdir,
      '--configs', 'atari', 'debug',
      '--task', 'breakout_like',
      '--jax.platform', 'cpu',
      '--run.steps', '4000',
      '--run.log_every', '4',
      '--run.train_ratio', '32',
      '--batch_size', '8',
      '--batch_length', '16',
      # reward_event_rec を有効化（デバッグ用に min_event_count=1）
      '--agent.reward_event_rec.enable', 'True',
      '--agent.reward_event_rec.min_event_count', '1',
      '--agent.reward_event_rec.window', '5',
      '--agent.reward_event_rec.alpha', '2.0',
      '--agent.reward_event_rec.beta', '1.0',
      '--agent.reward_event_rec.scale', '0.05',
      '--agent.reward_event_rec.hud_penalty', '0.1',
      '--agent.reward_event_rec.normalize', 'percentile',
      # reward_event 系メトリクスをロガーに通す（既定の jsonl 出力を使用）
      '--logger.filter', 'reward_event|loss/event_rec',
  ]
  print('Running BreakoutLike training with reward_event_rec enabled...')
  dmain.main(argv)


if __name__ == '__main__':
  run()
