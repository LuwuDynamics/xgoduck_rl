"""Reward shaping specific to XgoDuck velocity tasks."""
from __future__ import annotations

import torch


def air_time_center_bonus(
    air_time: torch.Tensor,
    threshold_min: float = 0.125,
    threshold_max: float = 0.30,
    center_bonus: float = 0.5,
) -> torch.Tensor:
    """Per-foot bonus, zero at/outside the old strict interval endpoints.

    The original binary reward remains the base. A parabolic bump adds
    center_bonus at the midpoint and tends to zero at both endpoints.
    """
    if threshold_max <= threshold_min:
        raise ValueError("threshold_max must exceed threshold_min")
    if center_bonus < 0:
        raise ValueError("center_bonus must be nonnegative")
    midpoint = (threshold_min + threshold_max) * 0.5
    half_width = (threshold_max - threshold_min) * 0.5
    in_range = (air_time > threshold_min) & (air_time < threshold_max)
    bump = (1.0 - ((air_time - midpoint) / half_width).square()).clamp(min=0.0)
    return bump * in_range * center_bonus


def feet_air_time_centered(
    env,
    sensor_name: str,
    threshold_min: float = 0.125,
    threshold_max: float = 0.30,
    command_name: str | None = None,
    command_threshold: float = 0.5,
    center_bonus: float = 0.5,
) -> torch.Tensor:
    """Keep the upstream base, command gate and air-time logging; add a bump."""
    from mjlab.tasks.velocity.mdp.rewards import feet_air_time

    base = feet_air_time(
        env,
        sensor_name=sensor_name,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    current_air_time = env.scene[sensor_name].data.current_air_time
    assert current_air_time is not None
    bonus = air_time_center_bonus(
        current_air_time, threshold_min, threshold_max, center_bonus
    ).sum(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            active = (
                torch.linalg.vector_norm(command[:, :2], dim=1)
                + command[:, 2].abs()
            ) > command_threshold
            bonus = bonus * active
    return base + bonus

def joint_pos_soft_limit_count(env, asset_cfg) -> torch.Tensor:
    """Return one per selected joint strictly outside its soft position limits.

    Penalty magnitude is independent of excess distance. The reward term uses
    weight -1, so two violating servo joints contribute -2 before dt scaling.
    """
    asset = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits
    assert limits is not None
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = limits[:, asset_cfg.joint_ids, 0]
    upper = limits[:, asset_cfg.joint_ids, 1]
    outside = (position < lower) | (position > upper)
    return outside.to(position.dtype).sum(dim=-1)
