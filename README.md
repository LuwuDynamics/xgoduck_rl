# XgoDuck RL

Reinforcement-learning environments for training XgoDuck, a small biped whose geometry and inertia differ from MicroDuck. Policies are trained with PPO on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) and exported to ONNX.

## Acknowledgments

This repository is a downstream project based on [microduck_rl](https://github.com/pollen-robotics/microduck_rl) by [Pollen Robotics](https://github.com/pollen-robotics). Training runs on [mjlab](https://github.com/mujocolab/mjlab). Joint actuation uses [BAM](https://github.com/Rhoban/bam) (Better Actuator Models) from Rhoban.

Thank you to the authors of those projects.

## What is different

The task code follows the MicroDuck recipes, but the robot does not:

- The MJCF and meshes use the XgoDuck model (`src/mjlab_microduck/robot/xgoduck/`).
- Actuators are the HLS1910 BAM model in `src/mjlab_microduck/robot/xgoduck/params/1910_m6.json`, not the MicroDuck XL330.

## Install

You need a CUDA GPU, Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
cd xgoduck_rl
uv sync
```

On ARM machines (DGX Spark / GB10, Jetson), the first `uv sync` downloads a large CUDA wheel and uv's default 30 s HTTP timeout can abort it. Set `UV_HTTP_TIMEOUT=600` for that first sync.

## Train

From the repository root:

```bash
uv run train Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096
uv run train Mjlab-Velocity-Swizzle-XgoDuck --env.scene.num-envs 4096
```

`uv run list-envs` prints every registered task. XgoDuck checkpoints and TensorBoard logs go under `logs/rsl_rl/<experiment>/`.

| Task | Terrain | Experiment directory |
|---|---|---|
| `Mjlab-Velocity-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_velocity/` |
| `Mjlab-Velocity-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_velocity/` |
| `Mjlab-StandUp-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_standup/` |
| `Mjlab-StandUp-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_standup/` |
| `Mjlab-GroundPick-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ground_pick/` |
| `Mjlab-GroundPick-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_ground_pick/` |
| `Mjlab-SitStand-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_sitstand/` |
| `Mjlab-SitStand-Rough-XgoDuck` | rough | `logs/rsl_rl/xgoduck_sitstand/` |
| `Mjlab-BallKick-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ball_kick_right/` |
| `Mjlab-BallKick-Left-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_ball_kick_left/` |
| `Mjlab-Roulade-Flat-XgoDuck` | flat | `logs/rsl_rl/xgoduck_roulade/` |
| `Mjlab-Velocity-Swizzle-XgoDuck` | flat | `logs/rsl_rl/xgoduck_velocity_swizzle/` |

## Play

`play` with no checkpoint flag loads the newest `model_*.pt` in that task's experiment directory:

```bash
uv run play Mjlab-Velocity-Flat-XgoDuck
uv run play Mjlab-Velocity-Swizzle-XgoDuck
```

Pass `--checkpoint-file path/to/model_XXXX.pt` to choose a specific checkpoint.

## ONNX

Export bakes observation normalization into the graph. Run it from the repository root and point it at a trained checkpoint:

```bash
uv run python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
    --checkpoint-file logs/rsl_rl/xgoduck_velocity/<run>/model_XXXX.pt \
    --onnx-file logs/rsl_rl/xgoduck_velocity/<run>/<run>.onnx
```

If you omit `--onnx-file`, the file is written to `output.onnx` in the current directory. `*.onnx` is gitignored. Keep the export next to the run that produced it, for example `logs/rsl_rl/xgoduck_velocity/<run>/<run>.onnx`.

## License

The code is licensed under Apache 2.0. See [LICENSE](LICENSE).
