import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


def _normalize_map(m, cfg):
  if cfg.normalize == 'mean':
    denom = m.mean()
  else:  # percentile
    denom = jnp.percentile(m, cfg.percentile)
  return m / jnp.maximum(denom, 1e-8)


def _event_base_means(image, reward, cfg):
  """Per-batch raw motion statistics split by reward / non-reward steps.

  Args:
    image: float32 array (B, T, H, W, C) scaled to [0, 1].
    reward: float32 array (B, T).
    cfg: reward_event_rec config block.

  Returns:
    event_mean, base_mean: (H, W) mean windowed motion at reward / non-reward
      steps (raw, before HUD penalty and normalization).
    motion_map, diff_map: (H, W) naive aggregate maps for visualization.
    event_count, base_count: scalar counts of reward / non-reward steps.
  """
  B, T, H, W, C = image.shape

  # D_t = |x_t - x_{t-1}| reduced over channels; D_0 = 0.
  diff = jnp.abs(image[:, 1:] - image[:, :-1])
  diff = jnp.concatenate([jnp.zeros_like(image[:, :1]), diff], 1)
  D = diff.mean(-1)  # (B, T, H, W)

  # M_t = max over the past `window` frames (tau in [-K, 0]).
  window = max(1, int(cfg.window))
  M = D
  for k in range(1, window):
    shifted = jnp.pad(D, [(0, 0), (k, 0), (0, 0), (0, 0)])[:, :T]
    M = jnp.maximum(M, shifted)

  event = (jnp.abs(reward) > 0).astype(f32)  # (B, T)
  base = 1.0 - event
  event_count = event.sum()
  base_count = base.sum()
  ew = event[:, :, None, None]
  bw = base[:, :, None, None]
  event_mean = (M * ew).sum((0, 1)) / jnp.maximum(event_count, 1.0)  # (H, W)
  base_mean = (M * bw).sum((0, 1)) / jnp.maximum(base_count, 1.0)  # (H, W)

  diff_map = D.mean((0, 1))  # (H, W) raw frame difference
  motion_map = M.mean((0, 1))  # (H, W) windowed motion saliency (all frames)
  return event_mean, base_mean, motion_map, diff_map, event_count, base_count


def _means_to_weight(event_mean, base_mean, cfg, H, W):
  """Turn (event_mean, base_mean) into the reconstruction weight + raw prior.

  Identical math whether the means are a single batch (per-batch mode) or the
  cross-batch EMA accumulation (prior_ema mode); keeping it in one place means
  both paths normalize and clip exactly the same way.
  """
  prior = jax.nn.relu(event_mean - cfg.beta * base_mean)  # (H, W)
  # Down-weight the top HUD strip used by Atari score/lives displays.
  hud_h = int(round(cfg.hud_height_ratio * H))
  if hud_h > 0:
    hud = jnp.ones((H, W), f32).at[:hud_h, :].set(cfg.hud_penalty)
    prior = prior * hud
  prior_n = _normalize_map(prior, cfg)
  weight = jnp.clip(1.0 + cfg.alpha * prior_n, cfg.clip_min, cfg.clip_max)
  return jax.lax.stop_gradient(weight), prior


def reward_event_prior(image, reward, cfg):
  """Per-batch Reward-Event Contrastive prior over the image plane.

  Builds a non-learned spatial map that highlights regions which change
  specifically around reward events, relative to non-reward times. This is the
  stateless per-batch estimate; the EMA accumulation (prior_ema mode) lives in
  RerPriorEMA and reuses _event_base_means / _means_to_weight.

  Returns:
    weight: (H, W) stop-gradient reconstruction weight in [clip_min, clip_max].
    prior: (H, W) raw ReLU(event_mean - beta * base_mean) before normalize.
    motion_weight: (H, W) naive motion-only weight (no contrast, no HUD penalty)
      shown for comparison; this is what a pure frame-difference mask produces.
    diff_map: (H, W) raw mean frame difference D_t averaged over batch/time.
    event_count, base_count: scalar counts of reward / non-reward steps.
    gate: scalar 1.0 if event_count >= min_event_count else 0.0.
  """
  B, T, H, W, C = image.shape
  event_mean, base_mean, motion_map, diff_map, ecount, bcount = (
      _event_base_means(image, reward, cfg))
  weight, prior = _means_to_weight(event_mean, base_mean, cfg, H, W)

  # Naive motion-only weight (DyMoDreamer-like): no event/base contrast and no
  # HUD penalty, so it also lights up the HUD and global background motion.
  motion_n = _normalize_map(motion_map, cfg)
  motion_weight = jnp.clip(
      1.0 + cfg.alpha * motion_n, cfg.clip_min, cfg.clip_max)
  motion_weight = jax.lax.stop_gradient(motion_weight)

  gate = (ecount >= cfg.min_event_count).astype(f32)
  return (weight, prior, motion_weight, diff_map, ecount, bcount, gate)


class RerPriorEMA(nj.Module):
  """Cross-batch EMA of the reward-event motion statistics.

  The per-batch prior averages M over only the handful of reward steps in one
  batch (~3-12 for Breakout/size25m), so it is very noisy and the min_event
  gate almost never opens. This accumulates event_mean / base_mean across train
  steps (the previously-unused `update_rate`), giving a stable prior even when
  each batch contributes few events, and gates on the cumulative count seen.
  """

  rate: float = 0.01

  def __init__(self, shape):
    self.emean = nj.Variable(jnp.zeros, shape, f32, name='emean')
    self.bmean = nj.Variable(jnp.zeros, shape, f32, name='bmean')
    self.ecorr = nj.Variable(jnp.zeros, (), f32, name='ecorr')  # debias
    self.seen = nj.Variable(jnp.zeros, (), f32, name='seen')  # cumulative events

  def update(self, event_mean, base_mean, ecount):
    r = self.rate
    g = (ecount >= 1).astype(f32)
    # Base steps are plentiful every batch -> always update.
    self.bmean.write((1 - r) * self.bmean.read() + r * sg(base_mean))
    # Event steps may be absent -> only fold in batches that saw an event, so
    # empty batches do not drag the estimate toward zero.
    new_e = (1 - r) * self.emean.read() + r * sg(event_mean)
    self.emean.write(jnp.where(g > 0, new_e, self.emean.read()))
    new_c = (1 - r) * self.ecorr.read() + r
    self.ecorr.write(jnp.where(g > 0, new_c, self.ecorr.read()))
    self.seen.write(self.seen.read() + sg(f32(ecount)))

  def stats(self):
    # Debias the event mean (zero init) like embodied.jax.Normalize does.
    corr = jnp.maximum(self.rate, self.ecorr.read())
    return sg(self.emean.read() / corr), sg(self.bmean.read())

  def count(self):
    return self.seen.read()


def _gray_to_rgb(m):
  """Min-max normalize a (H, W) map to a (H, W, 3) uint8 grayscale image."""
  lo, hi = m.min(), m.max()
  n = (m - lo) / jnp.maximum(hi - lo, 1e-8)
  u = (jnp.clip(n, 0, 1) * 255).astype(jnp.uint8)
  return jnp.repeat(u[..., None], 3, -1)


def _weight_to_rgb(w, lo, hi):
  """Map a weight (H, W) in [lo, hi] to a (H, W, 3) uint8 grayscale image."""
  span = max(float(hi) - float(lo), 1e-8)
  n = jnp.clip((w - lo) / span, 0, 1)
  u = (n * 255).astype(jnp.uint8)
  return jnp.repeat(u[..., None], 3, -1)


def _overlay(sample01, w):
  """Overlay weight (H, W) as a red heat on a grayscale frame (H, W, C)->RGB."""
  gray = sample01.mean(-1)  # (H, W), robust to any channel count
  base = gray * 0.6
  lo, hi = w.min(), w.max()
  hn = (w - lo) / jnp.maximum(hi - lo, 1e-8)
  red = jnp.maximum(base, hn)
  rgb = jnp.stack([red, base, base], -1)
  return (jnp.clip(rgb, 0, 1) * 255).astype(jnp.uint8)


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    rercfg = self.config.reward_event_rec
    if rercfg.enable and rercfg.mode == 'aux':
      scales['event_rec'] = rercfg.scale
    self.scales = scales

    # Cross-batch EMA accumulators for the reward-event prior (one per image
    # key). Only built when the method and EMA mode are both enabled, so the
    # baseline (enable=False) creates no extra state.
    self.rer_ema = {}
    if rercfg.enable and rercfg.prior_ema:
      for key, space in obs_space.items():
        if key in dec_space and isimage(space):
          H, W = space.shape[0], space.shape[1]
          self.rer_ema[key] = RerPriorEMA(
              (H, W), rate=rercfg.update_rate, name=f'rer_ema_{key}')

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    # Reward-Event Contrastive Reconstruction (auxiliary, global prior version).
    report_outs = {}
    recfg = self.config.reward_event_rec
    if recfg.enable:
      mult = (recfg.mode == 'mult')
      event_rec = jnp.zeros((B, T), f32)
      imgkeys = [k for k in self.dec.imgkeys if k in recons]
      primary = imgkeys[0] if imgkeys else None
      for key in imgkeys:
        image = f32(obs[key]) / 255  # (B, T, H, W, C) in [0, 1]
        pred = recons[key].pred()  # raw decoder mean, same [0, 1] scale
        H, W = image.shape[2], image.shape[3]
        # Per-batch raw statistics (always computed for metrics / fallback).
        em, bm, motion_map, diff_map, ecount, bcount = (
            _event_base_means(image, obs['reward'], recfg))
        ema = self.rer_ema.get(key)
        if ema is not None:
          # EMA mode: accumulate across train steps, weight from the stable
          # estimate, and gate on the cumulative number of events seen.
          if training:
            ema.update(em, bm, ecount)
          em_use, bm_use = ema.stats()
          gate = (ema.count() >= recfg.min_event_count).astype(f32)
        else:
          # Per-batch mode: estimate and gate from this batch alone.
          em_use, bm_use = em, bm
          gate = (ecount >= recfg.min_event_count).astype(f32)
        weight, prior = _means_to_weight(em_use, bm_use, recfg, H, W)
        motion_n = _normalize_map(motion_map, recfg)
        motion_weight = sg(jnp.clip(
            1.0 + recfg.alpha * motion_n, recfg.clip_min, recfg.clip_max))
        sqerr = jnp.square(pred - sg(image))
        uniform_key_loss = sqerr.sum((-1, -2, -3))  # standard rec (B, T)
        # A3: mean-1 normalized weight -> pure spatial redistribution, so the
        # total reconstruction magnitude is preserved (no confound with simply
        # scaling up the rec loss). gate/blend fall back to uniform safely.
        wtilde = weight / jnp.maximum(weight.mean(), 1e-8)  # (H, W), mean 1
        w_eff = 1.0 + gate * recfg.blend * (wtilde - 1.0)
        w_eff = sg(w_eff)
        weighted_key_loss = (
            sqerr * w_eff[None, None, :, :, None]).sum((-1, -2, -3))
        if mult:
          # A2: reweight the standard image rec loss in place (no extra term).
          losses[key] = weighted_key_loss
        else:
          # Legacy additive auxiliary term at `scale` (uses raw weight, gated).
          event_rec = event_rec + (
              sqerr * weight[None, None, :, :, None]).sum((-1, -2, -3)) * gate
        if key == primary:
          applied = w_eff if mult else (weight * gate + (1.0 - gate))
          metrics['reward_event/event_count'] = ecount
          metrics['reward_event/base_count'] = bcount
          metrics['reward_event/prior_mean'] = prior.mean()
          metrics['reward_event/prior_max'] = prior.max()
          metrics['reward_event/weight_mean'] = applied.mean()
          metrics['reward_event/weight_max'] = applied.max()
          metrics['reward_event/motion_weight_mean'] = motion_weight.mean()
          metrics['reward_event/enabled'] = gate
          # Redistribution effect: >1 means the reward region is currently
          # reconstructed worse than average (the term the method targets).
          metrics['reward_event/weighted_over_uniform'] = (
              weighted_key_loss.mean() / jnp.maximum(
                  uniform_key_loss.mean(), 1e-8))
          metrics['reward_event/loss'] = (
              weighted_key_loss if mult else event_rec).mean()
          # Maps are only consumed by report(); skip the work during training.
          if recfg.log_maps and not training:
            # Pick a real frame at the first reward event for the overlay.
            ev = (jnp.abs(obs['reward']) > 0).reshape(-1)
            idx = jnp.argmax(ev)
            sample01 = image[idx // T, idx % T]  # (H, W, C) in [0, 1]
            diff_img = _gray_to_rgb(diff_map)
            motion_img = _gray_to_rgb(motion_weight)
            prior_img = _gray_to_rgb(prior)
            weight_img = _gray_to_rgb(applied)
            overlay_img = _overlay(sample01, applied)
            report_outs['reward_event/diff_map'] = diff_img
            report_outs['reward_event/motion_weight_map'] = motion_img
            report_outs['reward_event/prior_map'] = prior_img
            report_outs['reward_event/weight_map'] = weight_img
            report_outs['reward_event/overlay'] = overlay_img
            # Side-by-side panel: diff | motion | prior | weight | overlay.
            report_outs['reward_event/panel'] = jnp.concatenate(
                [diff_img, motion_img, prior_img, weight_img, overlay_img], 1)
      if not mult:
        losses['event_rec'] = event_rec

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    # Relative magnitude of the auxiliary event reconstruction loss.
    if self.config.reward_event_rec.enable and 'event_rec' in losses:
      scaled_event_rec = losses['event_rec'].mean() * self.scales['event_rec']
      imgkeys = [k for k in self.dec.imgkeys if k in losses]
      scaled_image_rec = sum(
          losses[k].mean() * self.scales[k] for k in imgkeys)
      total_before = loss - scaled_event_rec
      metrics['reward_event/scaled_loss'] = scaled_event_rec
      metrics['reward_event/scale_ratio_to_rec'] = (
          scaled_event_rec / jnp.maximum(scaled_image_rec, 1e-8))
      metrics['reward_event/scale_ratio_to_total'] = (
          scaled_event_rec / jnp.maximum(total_before, 1e-8))

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    outs.update(report_outs)
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)
    for k, v in outs.items():
      if k.startswith('reward_event/'):
        metrics[k] = v

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Region-decomposed reconstruction probe (independent of enable so the
    # baseline and the proposed method are measured the same way). Measures how
    # well the observed-half reconstruction matches the true frames inside the
    # reward-event region versus the background.
    recfg = self.config.reward_event_rec
    for key in self.dec.imgkeys:
      image = f32(obs[key]) / 255  # (B, T, H, W, C)
      ema = self.rer_ema.get(key)
      if ema is not None:
        # Use the accumulated EMA prior (read-only) so the region matches what
        # the loss actually weighted.
        em_use, bm_use = ema.stats()
        _, prior = _means_to_weight(
            em_use, bm_use, recfg, image.shape[2], image.shape[3])
      else:
        _, prior, _, _, _, _, _ = reward_event_prior(
            image, obs['reward'], recfg)
      thresh = jnp.percentile(prior, recfg.percentile)
      region = (prior > jnp.maximum(thresh, 1e-6)).astype(f32)  # (H, W)
      true = image[:RB, :T // 2]
      pred = obsrecons[key].pred()
      err = jnp.square(pred - true).mean(-1)  # (RB, T//2, H, W)
      reg = region[None, None]
      ev = (err * reg).sum((-1, -2)) / jnp.maximum(region.sum(), 1.0)
      bg = (err * (1 - reg)).sum((-1, -2)) / jnp.maximum((1 - region).sum(), 1.0)
      metrics['reward_event/recon_mse_event_region'] = ev.mean()
      metrics['reward_event/recon_mse_bg_region'] = bg.mean()
      metrics['reward_event/recon_mse_region_over_bg'] = (
          ev.mean() / jnp.maximum(bg.mean(), 1e-8))

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
  losses = {}
  metrics = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
