"""Recover selected hip output angles after the initial XgoDuck kick."""
import math

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .mdp import joint_pos_rel_backlash


class HipHomeAfterKick:
    """Deadband L1 cost on selected hips, ramped in after the initial kick window.

    The clock starts at a physical environment reset, independently of the
    episode_length_buf that RSL-RL randomizes when training begins. Output
    angles include the series backlash hinge; encoder calibration bias is
    excluded from the physical-pose target. Use a negative reward weight.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        start = cfg.params["return_start_s"]
        full = cfg.params["return_full_s"]
        if not 0.0 <= start < full:
            raise ValueError("Require 0 <= return_start_s < return_full_s")
        if cfg.params["deadband_rad"] < 0.0:
            raise ValueError("deadband_rad must be nonnegative")
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.return_gate = torch.zeros_like(self.elapsed)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        return_start_s: float,
        return_full_s: float,
        deadband_rad: float,
        metric_prefix: str = "kick_hip_roll",
    ) -> torch.Tensor:
        self.elapsed.add_(env.step_dt)
        progress = ((self.elapsed - return_start_s) / (return_full_s - return_start_s)).clamp(0.0, 1.0)
        self.return_gate = progress.square() * (3.0 - 2.0 * progress)
        error = joint_pos_rel_backlash(env, biased=False, asset_cfg=asset_cfg)
        error = torch.nan_to_num(error, nan=0.0, posinf=math.pi, neginf=-math.pi)
        cost = (error.abs() - deadband_rad).clamp_min(0.0).sum(dim=-1)
        env.extras["log"][f"Metrics/{metric_prefix}_error_deg"] = torch.rad2deg(error.abs()).mean()
        env.extras["log"][f"Metrics/{metric_prefix}_return_gate"] = self.return_gate.mean()
        return cost * self.return_gate

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.elapsed.zero_()
            self.return_gate.zero_()
        else:
            self.elapsed[env_ids] = 0.0
            self.return_gate[env_ids] = 0.0


# Keep existing callers compatible with the original roll-only reward.
HipRollHomeAfterKick = HipHomeAfterKick
