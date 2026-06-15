"""
rer_check_mult_mean1.py — 実装した mult(mean-1) 式そのものを AE で裏取り

agent.py の mult モードと同一の重み:
    wtilde = weight / mean(weight)         # mean-1 正規化 (A3)
    loss   = mean(se * wtilde)             # rec 本体を重み付け (A2)
を uniform(=純DreamerV3相当) と比較し、報酬物体MSEが下がるか(目標 ~-22%)を確認する。
weight は agent.reward_event_prior（提案手法と同一コード）。normalize は新既定の 'mean'。
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax, jax.numpy as jnp, optax
from dreamerv3 import agent as A
import rer_ablation_ae as AE


def loss_for(mode):
  def f(p, x, Wt):
    se = jnp.square(AE.forward(p, x) - x)
    if mode == 'mult_mean1':
      return (se * Wt).mean()      # Wt = mean-1 正規化重み (実装と同一)
    return se.mean()               # uniform
  return f


def train(images, Wt, mode, steps=1800, bs=32, seed=0):
  p = AE.init_params(jax.random.PRNGKey(0))     # 同一初期化
  opt = optax.adam(1e-3); st = opt.init(p)
  Wtj = jnp.asarray(Wt)[None, :, :, None]
  lf = loss_for(mode)
  @jax.jit
  def step(p, st, x):
    g = jax.grad(lambda p: lf(p, x, Wtj))(p)
    u, st2 = opt.update(g, st, p)
    return optax.apply_updates(p, u), st2
  rng = np.random.default_rng(seed); N = images.shape[0]
  for _ in range(steps):
    p, st = step(p, st, jnp.asarray(images[rng.integers(0, N, bs)]))
  return p


def main():
  class Cfg:
    enable=True; mode='mult'; blend=1.0; scale=0.05; alpha=2.0; beta=1.0
    window=5; min_event_count=1; normalize=os.environ.get('NORM','mean'); percentile=95
    hud_penalty=0.1; hud_height_ratio=0.15; clip_min=1.0; clip_max=5.0
    log_maps=False

  seeds = [0, 1, 2]
  res = {'uniform': [], 'mult_mean1': []}
  for sd in seeds:
    imgs, rews = AE.collect(n_seq=24, seed=sd)
    N, T = imgs.shape[:2]
    flat = imgs.reshape(N * T, 64, 64, 3).astype(np.float32) / 255
    perm = np.random.default_rng(sd).permutation(flat.shape[0])
    ntr = int(0.8 * flat.shape[0])
    tr, ev = flat[perm[:ntr]], flat[perm[ntr:]]
    gt = AE.gt_object_mask((ev * 255).astype(np.uint8))
    weight = np.asarray(A.reward_event_prior(
        jnp.asarray(imgs, jnp.float32) / 255, jnp.asarray(rews), Cfg())[0])
    Wt = weight / max(weight.mean(), 1e-8)       # mean-1 正規化（実装と同一）
    assert abs(Wt.mean() - 1.0) < 1e-5, Wt.mean()
    for m in res:
      p = train(tr, Wt, m, seed=sd)
      obj, bg = AE.region_mse(p, ev, gt)
      res[m].append((obj, bg))
    uo = res['uniform'][-1][0]; wo = res['mult_mean1'][-1][0]
    print(f'seed {sd}: uniform={uo:.4f}  mult_mean1={wo:.4f}  '
          f'({100*(uo-wo)/uo:+.1f}%)  [Wt.mean=1.000 Wt.max={Wt.max():.2f}]')

  uo = np.array([o for o, _ in res['uniform']])
  wo = np.array([o for o, _ in res['mult_mean1']])
  ub = np.array([b for _, b in res['uniform']])
  wb = np.array([b for _, b in res['mult_mean1']])
  print('\n==================== 集計 (seeds=%s) ====================' % seeds)
  print(f'報酬物体MSE  uniform={uo.mean():.4f}±{uo.std():.4f}  '
        f'mult_mean1={wo.mean():.4f}±{wo.std():.4f}  '
        f'改善={100*(uo.mean()-wo.mean())/uo.mean():+.1f}%')
  print(f'背景MSE      uniform={ub.mean():.4f}  mult_mean1={wb.mean():.4f}')
  ok = wo.mean() < uo.mean()
  print(f'\n-> 実装した mult(mean-1) 式は報酬物体の再構成を'
        f'{"改善（手順1は有効）" if ok else "改善せず"}。'
        f'背景MSEは{"ほぼ不変" if abs(wb.mean()-ub.mean())<0.001 else "変化あり"}'
        f'（mean-1正規化で総量保存）')


if __name__ == '__main__':
  main()
