"""
rer_ablation_sweep.py — 「DreamerV3 に勝つには何が必要か」を特定する dose-response

rer_ablation_ae.py と同じ容量制限 AE・同一GT評価を使い、loss の formulation/scale を
振って、報酬物体の再構成 MSE がどれだけ下がるかを複数 seed で測る。

比較する loss:
  uniform        : mean(se)                          ← 純DreamerV3相当
  add@0.05       : mean(se) + 0.05*mean(se*W)        ← 現状configの加算補助(scale 0.05)
  add@0.3        : mean(se) + 0.30*mean(se*W)
  add@1.0        : mean(se) + 1.00*mean(se*W)
  mult           : mean(se*W)                        ← rec loss自体を重み付け(原式 (1+aW)se)
  W は agent.reward_event_prior（提案手法と同一コード）。
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax, jax.numpy as jnp, optax
from dreamerv3 import agent as A
import rer_ablation_ae as AE  # reuse collect / model / gt_object_mask


def loss_for(mode):
  def f(p, x, W):
    se = jnp.square(AE.forward(p, x) - x)
    u = se.mean()
    w = (se * W).mean()
    return {
        'uniform': u,
        'add@0.05': u + 0.05 * w,
        'add@0.3': u + 0.30 * w,
        'add@1.0': u + 1.00 * w,
        'mult': w,
    }[mode]
  return f


def train(images, W, mode, steps=1800, bs=32, seed=0):
  p = AE.init_params(jax.random.PRNGKey(0))  # 同一初期化
  opt = optax.adam(1e-3); st = opt.init(p)
  Wj = jnp.asarray(W)[None, :, :, None]
  lf = loss_for(mode)
  @jax.jit
  def step(p, st, x):
    g = jax.grad(lambda p: lf(p, x, Wj))(p)
    u, st2 = opt.update(g, st, p)
    return optax.apply_updates(p, u), st2
  rng = np.random.default_rng(seed)
  N = images.shape[0]
  for i in range(steps):
    p, st = step(p, st, jnp.asarray(images[rng.integers(0, N, bs)]))
  return p


def main():
  class Cfg:
    enable=True; scale=0.05; alpha=2.0; beta=1.0; window=5; min_event_count=1
    normalize='percentile'; percentile=95; hud_penalty=0.1
    hud_height_ratio=0.15; clip_min=1.0; clip_max=5.0; log_maps=False

  modes = ['uniform', 'add@0.05', 'add@0.3', 'add@1.0', 'mult']
  seeds = [0, 1, 2]
  agg = {m: [] for m in modes}
  bg_agg = {m: [] for m in modes}

  for sd in seeds:
    imgs, rews = AE.collect(n_seq=24, seed=sd)
    N, T = imgs.shape[:2]
    flat = imgs.reshape(N * T, 64, 64, 3).astype(np.float32) / 255
    perm = np.random.default_rng(sd).permutation(flat.shape[0])
    ntr = int(0.8 * flat.shape[0])
    tr, ev = flat[perm[:ntr]], flat[perm[ntr:]]
    gt = AE.gt_object_mask((ev * 255).astype(np.uint8))
    W = np.asarray(A.reward_event_prior(
        jnp.asarray(imgs, jnp.float32) / 255, jnp.asarray(rews), Cfg())[0])
    for m in modes:
      p = train(tr, W, m, seed=sd)
      obj, bg = AE.region_mse(p, ev, gt)
      agg[m].append(obj); bg_agg[m].append(bg)
    print(f'seed {sd} done: ' +
          ' '.join(f'{m}={np.mean(agg[m][-1]):.4f}' for m in modes))

  print('\n==================== 集計 (seeds=%s) ====================' % seeds)
  base = np.mean(agg['uniform'])
  print(f'{"mode":10s} {"obj_MSE(mean±std)":22s} {"vs uniform":>12s} '
        f'{"bg_MSE":>9s}')
  for m in modes:
    o = np.array(agg[m]); b = np.array(bg_agg[m])
    rel = 100 * (base - o.mean()) / base
    tag = '' if m == 'uniform' else f'{rel:+.1f}%'
    print(f'{m:10s} {o.mean():.4f} ± {o.std():.4f}        {tag:>12s} '
          f'{b.mean():9.4f}')
  print('\n(obj_MSE が小さい/負の%(改善)ほど、報酬物体の再構成が良い)')


if __name__ == '__main__':
  main()
