"""
rer_ema_compare.py
-----------------------------------------------------------------------------
"per-batch prior (現状)" vs "EMA prior (update_rate を使う提案修正)" の
安定性・局在精度を numpy で比較する自己完結ハーネス。

背景:
  agent.py:reward_event_prior() は prior を毎バッチ「その場の数個の報酬イベント
  だけ」から生計算する (event_mean = M を報酬時刻で平均)。Breakout/size25m 規模では
  1 バッチあたり報酬イベントが ~3-12 個しかなく (実測 median≈9.5 @batch16,
  ~5 @batch8)、min_event_count=20 ではゲートがほぼ開かない。閾値を下げて開かせると
  prior が数イベント由来で非常にノイジーになる、というジレンマを検証する。

  config には update_rate:0.01 が宣言されているがコード未使用 (デッドコード)。
  本ハーネスは「event_mean/base_mean をバッチ横断で EMA 蓄積する」版を実装し、
  少数イベントでも prior が安定するか・真の報酬領域に局在するかを数値で示す。

  ground truth: 報酬領域 (reward_region) を既知位置に埋め込むので、
  prior が「正しい場所」に重みを集中できているかを IoU/mass-in-region で測れる。
"""

import numpy as np

f32 = np.float32
relu = lambda x: np.maximum(x, 0.0)


class Cfg:
  alpha = 2.0
  beta = 1.0
  window = 5
  normalize = 'percentile'
  percentile = 95
  hud_penalty = 0.1
  hud_height_ratio = 0.15
  clip_min = 1.0
  clip_max = 5.0
  update_rate = 0.05  # EMA 版のみ使用 (configs.yaml の宣言値 0.01 はやや遅いので
                      #  60 バッチの短い smoke では 0.05 で立ち上がりを見る)


def _motion(image, window):
  """agent.py と同一: 窓内最大のフレーム差分 M (B,T,H,W)。"""
  diff = np.abs(image[:, 1:] - image[:, :-1])
  diff = np.concatenate([np.zeros_like(image[:, :1]), diff], 1)
  D = diff.mean(-1)
  M = D.copy()
  for k in range(1, max(1, int(window))):
    shifted = np.pad(D, [(0, 0), (k, 0), (0, 0), (0, 0)])[:, :M.shape[1]]
    M = np.maximum(M, shifted)
  return M


def _prior_from_means(event_mean, base_mean, cfg, H, W):
  """event_mean/base_mean (H,W) から agent.py と同じ式で weight/prior を作る。"""
  prior = relu(event_mean - cfg.beta * base_mean)
  hud_h = int(round(cfg.hud_height_ratio * H))
  if hud_h > 0:
    hud = np.ones((H, W), f32)
    hud[:hud_h, :] = cfg.hud_penalty
    prior = prior * hud
  denom = prior.mean() if cfg.normalize == 'mean' else np.percentile(prior, cfg.percentile)
  prior_n = prior / max(denom, 1e-8)
  weight = np.clip(1.0 + cfg.alpha * prior_n, cfg.clip_min, cfg.clip_max)
  return weight, prior


def per_batch_means(image, reward, cfg):
  """現状: このバッチの報酬/非報酬時刻だけで event_mean/base_mean を計算。"""
  M = _motion(image, cfg.window)
  event = (np.abs(reward) > 0).astype(f32)
  base = 1.0 - event
  ec, bc = event.sum(), base.sum()
  ew, bw = event[:, :, None, None], base[:, :, None, None]
  event_mean = (M * ew).sum((0, 1)) / max(ec, 1.0)
  base_mean = (M * bw).sum((0, 1)) / max(bc, 1.0)
  return event_mean, base_mean, ec


class EMAState:
  """提案修正: event_mean/base_mean をバッチ横断で EMA 蓄積する。
  報酬イベントがあるバッチだけ event_mean を更新 (空バッチで 0 を注入しない)。"""
  def __init__(self):
    self.event_mean = None
    self.base_mean = None

  def update(self, image, reward, cfg):
    bm_e, bm_b, ec = per_batch_means(image, reward, cfg)
    r = cfg.update_rate
    # base は毎バッチ豊富にあるので常時更新。
    self.base_mean = bm_b if self.base_mean is None else (1 - r) * self.base_mean + r * bm_b
    # event はイベントがある時だけ更新。
    if ec >= 1:
      self.event_mean = bm_e if self.event_mean is None else (1 - r) * self.event_mean + r * bm_e
    return ec


def make_batch(B, T, H, W, C, lam_events, reward_region, noise, rng):
  """Breakout 風: 報酬領域が報酬直前に動く + 全体に背景ノイズ。
  lam_events: このバッチの報酬イベント期待数 (Poisson)。"""
  image = np.full((B, T, H, W, C), 0.3, f32)
  # 背景ノイズ (per-batch 推定を撹乱する大域変化)。
  image += noise * rng.standard_normal((B, T, H, W, C)).astype(f32)
  reward = np.zeros((B, T), f32)
  (r0, r1, c0, c1) = reward_region
  n_ev = int(rng.poisson(lam_events))
  cands = [(b, t) for b in range(B) for t in range(3, T)]
  rng.shuffle(cands)
  for (b, t) in cands[:n_ev]:
    image[b, t - 2:t, r0:r1, c0:c1] += 0.5  # 報酬対象の動き
    reward[b, t] = 1.0
  return image, reward


def mass_in_region(weight, region):
  """weight の総量のうち真の報酬領域に入る割合 (局在精度, 高いほど良い)。
  weight-1 (uniform からの上乗せ分) で測る。"""
  (r0, r1, c0, c1) = region
  excess = np.maximum(weight - 1.0, 0.0)
  total = excess.sum()
  if total < 1e-8:
    return 0.0
  return float(excess[r0:r1, c0:c1].sum() / total)


def main():
  B, T, H, W, C = 8, 64, 32, 32, 3  # batch8 (B*T=512) を模す
  region = (22, 26, 14, 18)         # HUD 外の小領域 = 真の報酬物体
  region_frac = (26 - 22) * (18 - 14) / (H * W)
  cfg = Cfg()
  rng = np.random.default_rng(0)

  N_BATCH = 60
  LAM = 5.0     # 1 バッチ平均 ~5 イベント (batch8 の実測レンジ)
  NOISE = 0.04  # 背景ノイズ

  ema = EMAState()
  pb_mass, ema_mass = [], []
  pb_w_series, ema_w_series = [], []
  ecs = []

  for i in range(N_BATCH):
    img, rew = make_batch(B, T, H, W, C, LAM, region, NOISE, rng)
    # --- 現状: per-batch prior ---
    em, bm, ec = per_batch_means(img, rew, cfg)
    w_pb, _ = _prior_from_means(em, bm, cfg, H, W)
    # --- 提案修正: EMA prior ---
    ema.update(img, rew, cfg)
    w_ema, _ = _prior_from_means(ema.event_mean, ema.base_mean, cfg, H, W)

    ecs.append(ec)
    pb_mass.append(mass_in_region(w_pb, region))
    ema_mass.append(mass_in_region(w_ema, region))
    pb_w_series.append(w_pb)
    ema_w_series.append(w_ema)

  pb_w_series = np.stack(pb_w_series)    # (N,H,W)
  ema_w_series = np.stack(ema_w_series)
  # バッチ間の重みマップのブレ = 各画素の時系列 std を平均 (低いほど安定)。
  pb_jitter = pb_w_series.std(0).mean()
  ema_jitter = ema_w_series.std(0).mean()
  # 立ち上がり後 (後半30バッチ) の局在精度。
  pb_loc = np.mean(pb_mass[30:])
  ema_loc = np.mean(ema_mass[30:])

  print('=' * 68)
  print(' per-batch prior (現状)  vs  EMA prior (update_rate 提案修正)')
  print('=' * 68)
  print(f' batch: B={B} T={T} (B*T={B*T})   events/batch mean={np.mean(ecs):.1f} '
        f'(min={int(min(ecs))} max={int(max(ecs))})')
  print(f' 真の報酬領域が画面に占める面積比: {region_frac*100:.1f}%')
  print(f' (mass-in-region のランダム期待値 ≈ {region_frac*100:.1f}%)')
  print('-' * 68)
  print(f' {"指標":<34}{"per-batch":>14}{"EMA":>14}')
  print(f' {"バッチ間ブレ weight.std (低いほど安定)":<30}{pb_jitter:>14.4f}{ema_jitter:>14.4f}')
  print(f' {"局在精度 mass-in-region 後半30 (高いほど良)":<28}{pb_loc*100:>12.1f}%{ema_loc*100:>12.1f}%')
  print('-' * 68)
  improve_jit = (pb_jitter - ema_jitter) / max(pb_jitter, 1e-8) * 100
  improve_loc = (ema_loc - pb_loc) / max(pb_loc, 1e-8) * 100
  print(f' EMA はブレを {improve_jit:+.0f}% / 局在精度を {improve_loc:+.0f}% 変化させた')
  print('=' * 68)
  # 判定: EMA が「より安定 かつ より局在」なら提案修正に意味あり。
  verdict = (ema_jitter < pb_jitter) and (ema_loc > pb_loc)
  print(' 判定:', 'EMA 修正は有効 (安定かつ局在改善)' if verdict
        else 'この設定では EMA の優位は限定的')


if __name__ == '__main__':
  main()
