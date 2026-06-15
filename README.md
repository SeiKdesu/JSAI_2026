# Mastering Diverse Domains through World Models + RER

This repository contains a reimplementation of [DreamerV3][paper] enhanced with **Reward-Event Contrastive Reconstruction (RER)**. DreamerV3 is a scalable reinforcement learning algorithm that masters diverse applications with fixed hyperparameters. RER is a proposed extension that improves learning efficiency by spatially reallocating reconstruction loss to reward-relevant regions.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

## Reward-Event Contrastive Reconstruction (RER)

Standard DreamerV3 applies uniform reconstruction loss across all pixels. In many environments (like Atari Breakout), reward-relevant objects like balls or bullets are tiny and their reconstruction loss is easily overwhelmed by the background.

**RER** addresses this by:
1.  **Extracting Reward-Event Priors**: Automatically identifying regions that change specifically during reward events compared to non-rewarding transitions.
2.  **Spatial Loss Reallocation**: Weighting the reconstruction loss so the model focuses its capacity on these critical regions.
3.  **Total Loss Preservation**: Normalizing weights to ensure the total reconstruction loss remains consistent with the baseline, avoiding confounding effects from simply increasing the loss scale.

Experiments on Breakout show that RER can reduce reconstruction MSE for reward-relevant objects by ~29% and leads to faster score improvements as the "gate" for RER opens (typically after ~20 reward events).

## Instructions

The code requires Python 3.11+ and has been tested on Linux and macOS.

### Setup

Install dependencies:

```sh
pip install -U -r requirements.txt
```

### Running Experiments

Use the provided helper script to run RER experiments or baselines:

```sh
# Smoke test (CPU, debug config, verify shape/logging)
./run_reward_event_rec.sh smoke

# Run Proposed Method (RER enabled)
./run_reward_event_rec.sh proposed

# Run Baseline (Standard DreamerV3)
./run_reward_event_rec.sh baseline

# Run Ablation (Baseline followed by Proposed on same seed)
./run_reward_event_rec.sh ablation
```

You can override parameters via environment variables:
```sh
TASK=atari_pong SEED=1 ./run_reward_event_rec.sh proposed
```

### Hardware Constraints & Optimization

If running on older GPUs like **RTX 1080 Ti** (Pascal architecture), use these flags to avoid OOM and compatibility issues:
- `--jax.compute_dtype float32` (Pascal doesn't support bfloat16 hardware acceleration)
- `--jax.prealloc False`
- `--batch_size 8` (or lower)

These are automatically handled if you use the default settings in `run_reward_event_rec.sh` or can be added to your manual command.

### Visualization & Analysis

- **Scope**: View scalar metrics and images.
  ```sh
  pip install -U scope
  python -m scope.viewer --basedir ~/logdir/reward_event_rec --port 8000
  ```
- **RER Maps**: If `log_maps=True`, RER-specific heatmaps (`base_map`, `event_map`, `prior`) are logged to help verify spatial focusing.
- **Verification Scripts**:
  - `rer_visualize.py`: Visualize the RER heatmaps and loss distributions.
  - `verify_reward_event_rec.py`: Unit tests for the RER logic.

## Repository Structure

- `dreamerv3/`: Core algorithm code.
  - `agent.py`: Contains the RER implementation (search for `reward_event_rec`).
  - `configs.yaml`: Default hyperparameters.
- `embodied/`: Environment wrappers and infrastructure.
- `rer_*.py`: Scripts for RER-specific experiments, ablations, and visualization.
- `verify_*.py`: Verification and debugging scripts.
- `run_reward_event_rec.sh`: Main entry point for experiments.

## Citation

If you find the DreamerV3 implementation useful, please cite the original paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## Disclaimer

This repository is a reimplementation and extension of DreamerV3 based on the open-source DreamerV2 codebase. It is unrelated to Google or DeepMind.

[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
