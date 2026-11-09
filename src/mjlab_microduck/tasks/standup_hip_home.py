"""Hip HOME refinement near a recovered stance, without taxing floor recovery."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from .mdp import joint_pos_rel_backlash

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def standup_joint_home_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    height_low: float,
    height_high: float,
    tilt_full_deg: float = 20.,
    tilt_zero_deg: float = 45.,
    freeze_s: float = 2.,
) -> torch.Tensor:
    """Positive summed joint error in radians; pair with a negative weight.

    Read the same encoder/output view as the actor, including gear play and
    encoder bias. The standing gate matches the existing arrival damper's
    height/tilt region. Frozen controls and low/prone poses pay no extra cost.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height = ((z-height_low)/max(height_high-height_low, 1e-6)).clamp(0., 1.)
    height = height.square() * (3.-2.*height)
    tilt = torch.acos((-asset.data.projected_gravity_b[:, 2]).clamp(-1., 1.))
    tilt_gate = ((math.radians(tilt_zero_deg)-tilt)
                 / max(math.radians(tilt_zero_deg-tilt_full_deg), 1e-6)).clamp(0., 1.)
    tilt_gate = tilt_gate.square() * (3.-2.*tilt_gate)
    active = (env.episode_length_buf * env.step_dt >= freeze_s).float()
    error = joint_pos_rel_backlash(env, biased=True, asset_cfg=asset_cfg)
    return error.abs().sum(dim=-1) * height * tilt_gate * active
