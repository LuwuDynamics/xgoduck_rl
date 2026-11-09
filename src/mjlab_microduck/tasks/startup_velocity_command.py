"""Seven direct velocity buckets plus stand-then-small-command practice."""
from dataclasses import dataclass
import math

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from .mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg


class StartupVelocityCommand(VelocityCommandCommandOnly):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isclose(7 * cfg.independent_command_fraction + cfg.startup_fraction, 1.):
            raise ValueError("Seven direct buckets plus startup_fraction must sum to one")
        for duration in (cfg.startup_stand_time_s, cfg.startup_move_time_s):
            if not 0 < duration[0] <= duration[1]:
                raise ValueError("Startup phase durations must be positive and ordered")
        self.scenario_bucket = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.startup_target = torch.zeros_like(self.vel_command_b)
        self.startup_move_duration = torch.zeros_like(self.time_left)

    def _resample_command(self, env_ids):
        # Retain the upstream command initialization, bypassing the legacy
        # seven-direction/standing sampler in VelocityCommandCommandOnly.
        UniformVelocityCommand._resample_command(self, env_ids)
        p = self.cfg.independent_command_fraction
        bucket = (torch.rand(len(env_ids), device=self.device) / p).long().clamp_max(7)
        self.scenario_bucket[env_ids] = bucket
        ranges = (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)
        for axis, bounds in enumerate(ranges):
            self.vel_command_b[env_ids, axis] = torch.empty(len(env_ids), device=self.device).uniform_(*bounds)
        # 0..5 are the six single-axis directions; 6 retains a full mixed draw.
        self.vel_command_b[env_ids[bucket < 6]] = 0.
        for kind in range(6):
            ids = env_ids[bucket == kind]
            axis, sign = kind // 2, 1. if kind % 2 == 0 else -1.
            maximum = ranges[axis][1] if sign > 0 else -ranges[axis][0]
            self.vel_command_b[ids, axis] = sign * torch.empty(len(ids), device=self.device).uniform_(0., maximum)

        self.startup_target[env_ids] = 0.
        self.startup_move_duration[env_ids] = 0.
        startup_ids = env_ids[bucket == 7]
        direction = torch.randint(6, (len(startup_ids),), device=self.device)
        small_ranges = (self.cfg.startup_vx, self.cfg.startup_vy, self.cfg.startup_yaw)
        for kind in range(6):
            ids = startup_ids[direction == kind]
            axis, sign = kind // 2, 1. if kind % 2 == 0 else -1.
            self.startup_target[ids, axis] = sign * torch.empty(len(ids), device=self.device).uniform_(*small_ranges[axis])
        standing = torch.empty(len(startup_ids), device=self.device).uniform_(*self.cfg.startup_stand_time_s)
        moving = torch.empty_like(standing).uniform_(*self.cfg.startup_move_time_s)
        self.startup_move_duration[startup_ids] = moving
        # Reserve the whole sequence so the ordinary 3-8 s resampler cannot
        # discard a startup command before its movement phase is exercised.
        self.time_left[startup_ids] = standing + moving
        self.vel_command_b[startup_ids] = 0.
        self.is_standing_env[env_ids] = bucket == 7
        for flag in (self.is_heading_env, self.is_world_env, self.is_forward_env):
            flag[env_ids] = False
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self):
        super()._update_command()
        ids = (self.scenario_bucket == 7).nonzero().flatten()
        moving = self.time_left[ids] <= self.startup_move_duration[ids]
        self.is_standing_env[ids] = ~moving
        self.vel_command_b[ids] = torch.where(moving.unsqueeze(-1), self.startup_target[ids], 0.)
        self.vel_command_w[ids] = self.vel_command_b[ids]


@dataclass(kw_only=True)
class StartupVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    # Per-draw probabilities, not wall-clock occupancy percentages.
    independent_command_fraction: float = .1
    startup_fraction: float = .3
    startup_stand_time_s: tuple[float, float] = (2., 4.)
    startup_move_time_s: tuple[float, float] = (3., 5.)
    startup_vx: tuple[float, float] = (.05, .15)
    startup_vy: tuple[float, float] = (.04, .12)
    startup_yaw: tuple[float, float] = (.15, .4)

    def build(self, env):
        return StartupVelocityCommand(self, env)
