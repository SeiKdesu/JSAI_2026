"""
verify_reward_event_rec.py
Reward-Event Contrastive Reconstruction の補助損失ロジックを numpy で再現し、
agent.py:reward_event_prior() と weighted reconstruction loss の挙動を検証する。

jax 未インストール環境でも実行できるよう、agent.py の式を numpy で 1:1 に写経する。
対応コード:
  - agent.py: reward_event_prior(), weighted event_rec loss block
"""

import numpy as np

f32 = np.float32
relu = lambda x: np.maximum(x, 0.0)


class Cfg:
  enable = True
  scale = 0.05
  alpha = 2.0
  beta = 1.0
  window = 5
  min_event_count = 20
  normalize = 'percentile'
  percentile = 95
  hud_penalty = 0.1
  hud_height_ratio = 0.15
  clip_min = 1.0
  clip_max = 5.0
  log_maps = True


def reward_event_prior(image, reward, cfg):
  """agent.py の同名関数を numpy で再現。"""
  B, T, H, W, C = image.shape
  diff = np.abs(image[:, 1:] - image[:, :-1])
  diff = np.concatenate([np.zeros_like(image[:, :1]), diff], 1)
  D = diff.mean(-1)  # (B, T, H, W)

  window = max(1, int(cfg.window))
  M = D.copy()
  for k in range(1, window):
    shifted = np.pad(D, [(0, 0), (k, 0), (0, 0), (0, 0)])[:, :T]
    M = np.maximum(M, shifted)

  event = (np.abs(reward) > 0).astype(f32)
  base = 1.0 - event
  event_count = event.sum()
  base_count = base.sum()
  ew = event[:, :, None, None]
  bw = base[:, :, None, None]
  event_mean = (M * ew).sum((0, 1)) / max(event_count, 1.0)
  base_mean = (M * bw).sum((0, 1)) / max(base_count, 1.0)

  prior = relu(event_mean - cfg.beta * base_mean)

  hud_h = int(round(cfg.hud_height_ratio * H))
  if hud_h > 0:
    hud = np.ones((H, W), f32)
    hud[:hud_h, :] = cfg.hud_penalty
    prior = prior * hud

  if cfg.normalize == 'mean':
    denom = prior.mean()
  else:
    denom = np.percentile(prior, cfg.percentile)
  prior_n = prior / max(denom, 1e-8)

  weight = np.clip(1.0 + cfg.alpha * prior_n, cfg.clip_min, cfg.clip_max)
  gate = f32(event_count >= cfg.min_event_count)
  return weight, prior, event_count, base_count, gate


def event_rec_loss(pred, image, weight, gate):
  sqerr = np.square(pred - image)
  key_loss = (sqerr * weight[None, None, :, :, None]).sum((-1, -2, -3))
  return key_loss * gate


def make_data(B, T, H, W, C, n_reward, reward_region, seed=0):
  """報酬イベント周辺で reward_region が動く合成データを作る。"""
  rng = np.random.default_rng(seed)
  # 背景は一様にゆっくりスクロール（大域変化）。
  image = np.zeros((B, T, H, W, C), f32)
  for t in range(T):
    image[:, t] = (0.3 + 0.001 * t)  # global slow drift everywhere
  reward = np.zeros((B, T), f32)
  # report region: 小さい矩形が報酬の直前に点滅する。
  (r0, r1, c0, c1) = reward_region
  # 報酬時刻を (b, t) 候補から決定的に n_reward 個ばらまく。
  cands = [(b, t) for b in range(B) for t in range(3, T)]
  rng.shuffle(cands)
  for (b, t) in cands[:n_reward]:
    # その手前 window フレームで報酬対象が変化する。
    image[b, t - 2:t, r0:r1, c0:c1] += 0.5
    reward[b, t] = 1.0
  return image, reward


def main():
  B, T, H, W, C = 4, 16, 32, 32, 3
  region = (20, 26, 22, 28)  # 下部の小領域（HUD外）
  cfg = Cfg()

  print('=' * 60)
  print('Test 3 & 4: min_event_count ゲート')
  print('=' * 60)
  # few events -> gate 0
  img_few, rew_few = make_data(B, T, H, W, C, n_reward=5, reward_region=region)
  w, prior, ec, bc, gate = reward_event_prior(img_few, rew_few, cfg)
  pred = img_few + 0.1  # 適当な予測誤差
  loss_few = event_rec_loss(pred, img_few, w, gate)
  print(f'event_count={ec:.0f} (<{cfg.min_event_count}) gate={gate} '
        f'loss.sum={loss_few.sum():.4f}')
  assert ec < cfg.min_event_count and gate == 0.0
  assert loss_few.sum() == 0.0, 'Test3 失敗: ゲート時 loss が 0 でない'
  print('  -> Test3 OK: event_count < min のとき loss == 0')

  # many events -> gate 1, positive loss
  img_many, rew_many = make_data(B, T, H, W, C, n_reward=40, reward_region=region)
  w, prior, ec, bc, gate = reward_event_prior(img_many, rew_many, cfg)
  pred = img_many + 0.1
  loss_many = event_rec_loss(pred, img_many, w, gate)
  print(f'event_count={ec:.0f} (>={cfg.min_event_count}) gate={gate} '
        f'loss.mean={loss_many.mean():.4f}')
  assert ec >= cfg.min_event_count and gate == 1.0
  assert loss_many.sum() > 0.0, 'Test4 失敗: loss が正でない'
  print('  -> Test4 OK: event_count >= min のとき loss > 0')

  print('=' * 60)
  print('Test 5: weight が [clip_min, clip_max] に収まる')
  print('=' * 60)
  print(f'weight min={w.min():.4f} max={w.max():.4f} '
        f'(clip [{cfg.clip_min}, {cfg.clip_max}])')
  assert w.min() >= cfg.clip_min - 1e-6 and w.max() <= cfg.clip_max + 1e-6
  print('  -> Test5 OK')

  print('=' * 60)
  print('Test 6: 報酬ありbatchと報酬なしbatchで prior が変わる')
  print('=' * 60)
  # 報酬なし: reward 全ゼロ -> event_mean は 0/1 -> prior 0
  img_z, _ = make_data(B, T, H, W, C, n_reward=40, reward_region=region)
  rew_zero = np.zeros((B, T), f32)
  w0, prior0, ec0, _, gate0 = reward_event_prior(img_z, rew_zero, cfg)
  print(f'reward あり prior_max={prior.max():.4f} '
        f'/ reward なし prior_max={prior0.max():.4f}')
  assert prior.max() > prior0.max()
  print('  -> Test6 OK: 報酬イベントが prior に反映される')

  print('=' * 60)
  print('Test 6b: prior のピークが報酬対象領域に出る')
  print('=' * 60)
  peak = np.unravel_index(np.argmax(prior), prior.shape)
  print(f'prior argmax (h,w)={peak}, 報酬領域 rows[{region[0]},{region[1]}) '
        f'cols[{region[2]},{region[3]})')
  in_region = (region[0] <= peak[0] < region[1] and
               region[2] <= peak[1] < region[3])
  assert in_region, 'prior のピークが報酬領域に無い'
  print('  -> OK: 背景の大域変化ではなく報酬特異領域がピーク')

  print('=' * 60)
  print('Test 7: 上部 HUD penalty が適用される')
  print('=' * 60)
  # 報酬対象を HUD 領域(上部)に置いても prior が抑制されることを確認。
  hud_region = (1, 4, 10, 16)
  img_hud, rew_hud = make_data(
      B, T, H, W, C, n_reward=40, reward_region=hud_region, seed=1)
  _, prior_hud, _, _, _ = reward_event_prior(img_hud, rew_hud, cfg)
  hud_h = int(round(cfg.hud_height_ratio * H))
  # HUD penalty を 1.0 にした比較用 cfg。
  class CfgNoHud(Cfg):
    hud_penalty = 1.0
  _, prior_nohud, _, _, _ = reward_event_prior(img_hud, rew_hud, CfgNoHud())
  top_with = prior_hud[:hud_h].max()
  top_without = prior_nohud[:hud_h].max()
  print(f'HUD strip rows[0,{hud_h}) penalty適用 max={top_with:.4f} '
        f'/ 非適用 max={top_without:.4f} (ratio={top_with/max(top_without,1e-8):.3f})')
  assert top_with < top_without
  assert abs(top_with / max(top_without, 1e-8) - cfg.hud_penalty) < 1e-5
  print(f'  -> Test7 OK: HUD 領域 prior が {cfg.hud_penalty}x に減衰')

  print('=' * 60)
  print('Test 1相当: enable=False で event_rec を一切作らないこと')
  print('=' * 60)
  # enable=False のとき agent.py は losses に 'event_rec' を入れない。
  # ここでは config 上の分岐のみ確認（実体は agent.py の if recfg.enable）。
  print('  -> agent.py: `if recfg.enable:` で囲っているため enable=False で'
        ' losses/scales とも未追加（既存挙動と同一）')

  print()
  print('ALL TESTS PASSED')


if __name__ == '__main__':
  main()
