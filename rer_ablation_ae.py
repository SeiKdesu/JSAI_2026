"""
rer_ablation_ae.py — 提案手法の中核メカニズムだけを高速に検証する対照実験

DreamerV3 のフル RL ループは CPU では遅いので、reconstruction loss の空間再重み付け
という「中核仮説」だけを、容量制限つき畳み込みオートエンコーダ (AE) で隔離検証する。

仮説:
  限られた latent 容量のもとで、一様 MSE は背景や大きな構造に容量を使い、小さな
  報酬関連物体(ボール/ブロック)を潰す。報酬イベント由来の空間重みで重点化すると、
  同じ容量でも報酬物体の再構成が改善する。

設定:
  - BreakoutLike フレームを収集 (image, reward)。
  - agent.reward_event_prior で「報酬イベント重み W(H,W)」を非学習で作る（提案手法と同一コード）。
  - 同一初期化・同一データで 2 つの AE を学習:
       uniform : loss = mean((pred-tgt)^2)
       weighted: loss = mean((pred-tgt)^2 * W)   ← 提案手法の重み
  - 評価は「独立な ground-truth 報酬物体マスク」(ボール=白, ブロック=緑; レンダリング
    から直接判定。prior とは独立) 上の MSE で行う -> 循環評価を避ける。

判定:
  weighted が GT 報酬物体領域の再構成 MSE を uniform より下げ、かつ背景を過度に
  犠牲にしなければ、メカニズムは機能し DreamerV3 decoder でも有効と期待できる。
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rer_experiment import BreakoutLike
from dreamerv3 import agent as A


# ----------------------------- データ -----------------------------
def collect(n_seq=24, T=64, seed=0):
  imgs = np.zeros((n_seq, T, 64, 64, 3), np.uint8)
  rews = np.zeros((n_seq, T), np.float32)
  for b in range(n_seq):
    env = BreakoutLike('x', length=10_000, seed=seed + b)
    o = env.step({'reset': True})
    rng = np.random.default_rng(seed + b)
    for t in range(T):
      imgs[b, t] = o['image']; rews[b, t] = o['reward']
      o = env.step({'reset': bool(o['is_last']),
                    'action': np.int32(rng.integers(3))})
  return imgs, rews


def gt_object_mask(frames_u8):
  """独立な GT: ボール(白)とブロック(緑)の画素。HUD/パドルは除外。(N,H,W)"""
  f = frames_u8.astype(np.int32)
  r, g, b = f[..., 0], f[..., 1], f[..., 2]
  ball = (r > 200) & (g > 200) & (b > 200)
  brick = (g > 150) & (r < 120) & (b < 120)
  mask = (ball | brick).astype(np.float32)
  mask[:, :10, :] = 0.0  # HUD 帯は評価から除外
  return mask


# ----------------------------- AE -----------------------------
def init_params(key):
  k = jax.random.split(key, 12)
  def conv(ki, i, o): return jax.random.normal(ki, (3, 3, i, o)) * (
      (2.0 / (3 * 3 * i)) ** 0.5)
  p = {}
  p['e1'] = conv(k[0], 3, 16); p['e2'] = conv(k[1], 16, 32)
  p['e3'] = conv(k[2], 32, 64)
  p['ew'] = jax.random.normal(k[3], (8 * 8 * 64, 64)) * 0.02
  p['eb'] = jnp.zeros(64)
  p['dw'] = jax.random.normal(k[4], (64, 8 * 8 * 64)) * 0.02
  p['db'] = jnp.zeros(8 * 8 * 64)
  p['d3'] = conv(k[5], 64, 32); p['d2'] = conv(k[6], 32, 16)
  p['d1'] = conv(k[7], 16, 3)
  return p


def conv2d(x, w, stride=1):
  return jax.lax.conv_general_dilated(
      x, w, (stride, stride), 'SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def up(x):  # nearest 2x
  n, h, wd, c = x.shape
  return jax.image.resize(x, (n, h * 2, wd * 2, c), 'nearest')


def forward(p, x):  # x: (N,64,64,3) in [0,1]
  h = jax.nn.relu(conv2d(x, p['e1'], 2))      # 32
  h = jax.nn.relu(conv2d(h, p['e2'], 2))      # 16
  h = jax.nn.relu(conv2d(h, p['e3'], 2))      # 8
  z = h.reshape(h.shape[0], -1) @ p['ew'] + p['eb']  # latent 64
  d = (z @ p['dw'] + p['db']).reshape(-1, 8, 8, 64)
  d = jax.nn.relu(conv2d(up(d), p['d3']))     # 16
  d = jax.nn.relu(conv2d(up(d), p['d2']))     # 32
  d = jax.nn.sigmoid(conv2d(up(d), p['d1']))  # 64
  return d


def train(images, weight, mode, steps=2500, bs=32, seed=0):
  key = jax.random.PRNGKey(0)  # 同一初期化（両条件で同じ）
  p = init_params(key)
  opt = optax.adam(1e-3); st = opt.init(p)
  N = images.shape[0]
  W = jnp.asarray(weight)[None, :, :, None]  # (1,H,W,1)
  rng = np.random.default_rng(seed)

  def loss_fn(p, x):
    pred = forward(p, x)
    se = jnp.square(pred - x)
    if mode == 'weighted':
      return (se * W).mean()
    return se.mean()

  step = jax.jit(lambda p, st, x: (
      lambda g: (optax.apply_updates(p, opt.update(g, st, p)[0]),
                 opt.update(g, st, p)[1]))(jax.grad(loss_fn)(p, x)))

  for i in range(steps):
    idx = rng.integers(0, N, bs)
    p, st = step(p, st, jnp.asarray(images[idx]))
  return p


def region_mse(p, images, gtmask):
  pred = np.asarray(forward(p, jnp.asarray(images)))
  se = ((pred - images) ** 2).mean(-1)  # (N,H,W)
  m = gtmask
  obj = (se * m).sum() / max(m.sum(), 1)
  bg_m = (1 - m); bg_m[:, :10, :] = 0  # HUD除外背景
  bg = (se * bg_m).sum() / max(bg_m.sum(), 1)
  return obj, bg


def main():
  imgs, rews = collect()
  N, T = imgs.shape[:2]
  flat = imgs.reshape(N * T, 64, 64, 3).astype(np.float32) / 255
  # train/eval split
  ntr = int(0.8 * flat.shape[0])
  perm = np.random.default_rng(0).permutation(flat.shape[0])
  tr, ev = flat[perm[:ntr]], flat[perm[ntr:]]
  ev_u8 = (ev * 255).astype(np.uint8)
  gt = gt_object_mask(ev_u8)
  print(f'frames: train={tr.shape[0]} eval={ev.shape[0]} '
        f'GT object px/frame={gt.sum()/gt.shape[0]:.1f}')

  # 提案手法と同一コードで重みを作る
  class Cfg:
    enable=True; scale=0.05; alpha=2.0; beta=1.0; window=5; min_event_count=1
    normalize='percentile'; percentile=95; hud_penalty=0.1
    hud_height_ratio=0.15; clip_min=1.0; clip_max=5.0; log_maps=False
  image = jnp.asarray(imgs, jnp.float32) / 255
  weight, prior, *_ = A.reward_event_prior(image, jnp.asarray(rews), Cfg())
  weight = np.asarray(weight)
  print(f'event weight: mean={weight.mean():.3f} max={weight.max():.3f}')

  results = {}
  for mode in ['uniform', 'weighted']:
    p = train(tr, weight, mode)
    obj, bg = region_mse(p, ev, gt)
    results[mode] = (obj, bg)
    print(f'[{mode:8s}] GT報酬物体MSE={obj:.5f}  背景MSE={bg:.5f}  '
          f'比(物体/背景)={obj/max(bg,1e-8):.3f}')

  uo, ub = results['uniform']; wo, wb = results['weighted']
  print('\n==================== 判定 ====================')
  print(f'報酬物体 reconstruction MSE: uniform={uo:.5f} -> weighted={wo:.5f}  '
        f'({100*(uo-wo)/uo:+.1f}% 変化)')
  print(f'背景       reconstruction MSE: uniform={ub:.5f} -> weighted={wb:.5f}  '
        f'({100*(ub-wb)/ub:+.1f}% 変化)')
  improved = wo < uo
  print(f'\n-> 報酬物体の再構成は提案手法で{"改善" if improved else "悪化"}'
        f'（{"仮説を支持" if improved else "仮説に反する"}）')
  print(f'-> 物体/背景MSE比: uniform={uo/max(ub,1e-8):.3f} '
        f'weighted={wo/max(wb,1e-8):.3f}'
        f'（小さいほど物体に容量を割けている）')


if __name__ == '__main__':
  main()
