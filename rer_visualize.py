"""
rer_visualize.py — map visualization の確認

BreakoutLike を転がして (B,T,H,W,C) のバッチを作り、agent.py の本物の関数
(reward_event_prior, _gray_to_rgb, _weight_to_rgb, _overlay) をそのまま呼んで
各種マップを PNG 保存する。学習済み decoder は不要（overlay は実フレームを使う）。

出力 (./rer_maps/):
  diff_map.png          raw frame difference D_t
  motion_weight.png     naive motion-only weight (HUD/背景も拾う)
  prior_map.png         ReLU(event_mean - beta*base_mean) + HUD penalty
  weight_map.png        reward-event contrastive weight (実際に loss で使う)
  overlay.png           weight を実フレームに赤ヒートで重畳
  panel.png             diff | motion | prior | weight | overlay 横並び
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax.numpy as jnp
from PIL import Image

import elements
from rer_experiment import BreakoutLike
from dreamerv3 import agent as A


def collect_batch(B=8, T=64, seed=0):
  imgs = np.zeros((B, T, 64, 64, 3), np.uint8)
  rews = np.zeros((B, T), np.float32)
  for b in range(B):
    env = BreakoutLike('x', length=10_000, seed=seed + b)
    o = env.step({'reset': True})
    rng = np.random.default_rng(seed + b)
    for t in range(T):
      imgs[b, t] = o['image']
      rews[b, t] = o['reward']
      a = {'reset': bool(o['is_last']), 'action': np.int32(rng.integers(3))}
      o = env.step(a)
  return imgs, rews


def save(arr, path):
  arr = np.asarray(arr).astype(np.uint8)
  Image.fromarray(arr).resize(
      (arr.shape[1] * 4, arr.shape[0] * 4), Image.NEAREST).save(path)


class Cfg:
  enable = True; scale = 0.05; alpha = 2.0; beta = 1.0; window = 5
  min_event_count = 1; normalize = 'percentile'; percentile = 95
  hud_penalty = 0.1; hud_height_ratio = 0.15; clip_min = 1.0; clip_max = 5.0
  log_maps = True


def main():
  os.makedirs('rer_maps', exist_ok=True)
  imgs, rews = collect_batch()
  print(f'batch images {imgs.shape}, reward events = {(rews != 0).sum()}')

  image = jnp.asarray(imgs, jnp.float32) / 255
  reward = jnp.asarray(rews)
  cfg = Cfg()

  (weight, prior, motion_weight, diff_map,
   ec, bc, gate) = A.reward_event_prior(image, reward, cfg)
  print(f'event_count={float(ec):.0f} base_count={float(bc):.0f} '
        f'gate={float(gate):.0f}')
  print(f'weight        min/mean/max = {float(weight.min()):.2f}/'
        f'{float(weight.mean()):.2f}/{float(weight.max()):.2f}')
  print(f'motion_weight min/mean/max = {float(motion_weight.min()):.2f}/'
        f'{float(motion_weight.mean()):.2f}/{float(motion_weight.max()):.2f}')

  lo, hi = cfg.clip_min, cfg.clip_max
  ev = (jnp.abs(reward) > 0).reshape(-1)
  idx = int(jnp.argmax(ev))
  T = image.shape[1]
  sample01 = image[idx // T, idx % T]

  diff_img = A._gray_to_rgb(diff_map)
  motion_img = A._weight_to_rgb(motion_weight, lo, hi)
  prior_img = A._gray_to_rgb(prior)
  weight_img = A._weight_to_rgb(weight, lo, hi)
  overlay_img = A._overlay(sample01, weight)
  panel = jnp.concatenate(
      [diff_img, motion_img, prior_img, weight_img, overlay_img], 1)

  save(diff_img, 'rer_maps/diff_map.png')
  save(motion_img, 'rer_maps/motion_weight.png')
  save(prior_img, 'rer_maps/prior_map.png')
  save(weight_img, 'rer_maps/weight_map.png')
  save(overlay_img, 'rer_maps/overlay.png')
  save(panel, 'rer_maps/panel.png')
  save((np.asarray(sample01) * 255), 'rer_maps/sample_frame.png')

  # 定量チェック: raw map レベルで HUD がどれだけ落ちるか（正規化前）
  hud_h = int(round(cfg.hud_height_ratio * 64))
  dm = np.asarray(diff_map); pr = np.asarray(prior)
  print(f'\n[raw map, 正規化前] HUD strip rows[0,{hud_h}) の平均値:')
  print(f'  diff_map (motion)   HUD={dm[:hud_h].mean():.4f}  '
        f'非HUD={dm[hud_h:].mean():.4f}  '
        f'-> motion は HUD が支配的')
  print(f'  prior  (contrastive) HUD={pr[:hud_h].mean():.4f}  '
        f'非HUD={pr[hud_h:].mean():.4f}  '
        f'peak={pr.max():.4f}')
  ratio = pr[:hud_h].mean() / max(pr.max(), 1e-8)
  print(f'  -> prior の HUD はピークの {ratio*100:.1f}% まで抑制されている')
  print('\n[注意] weight は percentile-95 正規化のため、スパースな prior では '
        '分母が極小になり HUD 残差が再増幅されることがある。')
  print('       weight_map(タイル4)で HUD が灰色に戻るのはこのため。')
  print('       より頑健にしたい場合は normalize=mean または percentile を下げる。')
  print('\nSaved PNGs to ./rer_maps/  (panel.png = diff|motion|prior|weight|overlay)')


if __name__ == '__main__':
  main()
