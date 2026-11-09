"""XgoDuck Swizzle with signed push commands and grounded passive skates.

Head commands remain at HOME. Heading and push-disturbance curricula introduce
turning and robustness gradually; servo dynamics and mass DR match XgoDuck Walk.
"""

from copy import deepcopy
from dataclasses import replace
from math import radians

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.xgoduck_roller_constants import (
    XGODUCK_ROLLER_ROBOT_CFG,
    XGODUCK_ROLLER_STAND_Z,
    XGODUCK_WHEEL_RADIUS,
)

from . import mdp
from .microduck_velocity_swizzle_env_cfg import (
    MicroduckSwizzleRlCfg,
    make_microduck_velocity_swizzle_env_cfg,
)
from .xgoduck_physics import configure_xgoduck_physics


WHEEL_BEARING_FRICTION_RANGE = (0.0008, 0.0012)  # N m per wheel.
WHEEL_BEARING_FRICTION_NOMINAL = 0.001
TIRE_SLIDING_FRICTION_RANGE = (0.4, 2.0)
TIRE_SLIDING_FRICTION_NOMINAL = 1.0


def swizzle_standing_curriculum(
    env, env_ids, command_name: str, start_fraction: float,
    end_fraction: float, end_step: int,
) -> torch.Tensor:
    """Reduce standing resamples while sharing the remainder across four modes."""
    del env_ids
    progress = min(1.0, max(0.0, float(env.common_step_counter) / end_step))
    fraction = start_fraction + (end_fraction - start_fraction) * progress
    env.command_manager.get_term(command_name).cfg.rel_standing_envs = fraction
    return torch.tensor([fraction], device=env.device)


class XgoduckSwizzleCommand(mdp.RelativeHeadingVelocityCommand):
    """Sample stand, straight forward/backward, and turning forward/backward."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        if count == 0:
            return
        standing = torch.rand(count, device=self.device) < self.cfg.rel_standing_envs
        # Conditional on moving, these four disjoint modes have equal probability:
        # 0 forward, 1 backward, 2 backward + turn, 3 forward + turn.
        mode = torch.randint(4, (count,), device=self.device)
        backward = ~standing & ((mode == 1) | (mode == 2))
        turning = ~standing & ((mode == 2) | (mode == 3))
        forward = ~standing & ~backward
        x_lo, x_hi = self.cfg.ranges.lin_vel_x
        turn_lo, turn_hi = self.cfg.ranges.ang_vel_z
        if not (x_lo < 0.0 < x_hi and turn_lo < 0.0 < turn_hi):
            raise ValueError("Swizzle command modes require signed push and heading ranges")

        # Keep the existing uniform magnitude ranges, excluding only exact zero;
        # an all-zero command belongs exclusively to the standing mode.
        epsilon = torch.finfo(self.vel_command_b.dtype).eps
        push = torch.where(
            backward,
            -torch.empty(count, device=self.device).uniform_(epsilon, -x_lo),
            torch.empty(count, device=self.device).uniform_(epsilon, x_hi),
        )
        turn_left = torch.rand(count, device=self.device) < 0.5
        heading = torch.where(
            turn_left,
            torch.empty(count, device=self.device).uniform_(epsilon, turn_hi),
            -torch.empty(count, device=self.device).uniform_(epsilon, -turn_lo),
        )
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = torch.where(standing, 0.0, push)
        self.vel_command_b[env_ids, 2] = torch.where(turning, heading, 0.0)
        self.is_standing_env[env_ids] = standing
        self.is_heading_env[env_ids] = turning
        self.is_forward_env[env_ids] = forward & ~turning
        self.is_world_env[env_ids] = False

        # The third input remains a relative heading error. Turning modes sample
        # a left/right goal relative to the current heading instead of a random
        # world heading that would saturate most samples at the clip limit.
        quat = self.robot.data.root_link_quat_w[env_ids]
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self._target_heading_w[env_ids] = yaw + self.vel_command_b[env_ids, 2]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self) -> None:
        super()._update_command()
        # The relative-heading parent recomputes slot 2 for every environment.
        # Enforce the sampled mode after it so stand/straight stay exactly zero.
        self.vel_command_b[~self.is_heading_env, 2] = 0.0
        self.vel_command_b[self.is_standing_env] = 0.0
        self.vel_command_w[:] = self.vel_command_b

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=True)
            push = server.gui.add_slider(
                "Push strength", min=-0.6, max=0.6, step=0.05, initial_value=0.0,
            )
            lateral = server.gui.add_slider(
                "Lateral (fixed zero)", min=-0.1, max=0.1, step=0.05,
                initial_value=0.0, disabled=True,
            )
            heading = server.gui.add_slider(
                "Heading error (rad)", min=-0.5, max=0.5,
                step=0.05, initial_value=0.0,
            )
            zero = server.gui.add_button("Zero")

            @zero.on_click
            def _(_event):
                push.value = 0.0
                heading.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_sliders = [push, lateral, heading]
        self._joystick_get_env_idx = get_env_idx


class XgoduckSwizzleCommandCfg(mdp.RelativeHeadingVelocityCommandCfg):
    def build(self, env):
        return XgoduckSwizzleCommand(self, env)


def make_xgoduck_velocity_swizzle_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = deepcopy(make_microduck_velocity_swizzle_env_cfg(play=play))
    cfg.scene.entities = {"robot": deepcopy(XGODUCK_ROLLER_ROBOT_CFG)}
    cfg.actions["joint_pos"].joint_names = (r"^(?!passive_).*",)
    cfg.commands["twist"] = XgoduckSwizzleCommandCfg(**vars(cfg.commands["twist"]))
    cfg.commands["twist"].ranges.lin_vel_x = (-0.6, 0.6)
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].init_velocity_prob = 0.0
    cfg.commands["twist"].rel_standing_envs = 0.3
    cfg.commands["twist"].rel_forward_envs = 0.0
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=swizzle_standing_curriculum,
        params={"command_name": "twist", "start_fraction": 0.3,
                "end_fraction": 0.1, "end_step": 2500 * 24},
    )

    # Robot-specific sensing and propulsion shaping.
    cfg.observations["actor"].terms["joint_pos"].noise.n_min = -radians(1.0)
    cfg.observations["actor"].terms["joint_pos"].noise.n_max = radians(1.0)
    cfg.rewards["wheel_speed"].weight = 10.0
    cfg.rewards["wheel_speed"].params["wheel_radius"] = XGODUCK_WHEEL_RADIUS
    cfg.rewards["body_speed"] = RewardTermCfg(
        func=mdp.skating_body_speed_reward,
        weight=5.0,
        params={"command_name": "twist", "vel_scale": 0.3, "bidirectional": True},
    )
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["com_height_target"].params.update(
        target_height_min=XGODUCK_ROLLER_STAND_Z - 0.015,
        target_height_max=XGODUCK_ROLLER_STAND_Z + 0.005,
    )

    for sensor in cfg.scene.sensors:
        if sensor.name == "feet_ground_contact":
            sensor.primary.pattern = r"^(ankle_left|ankle_right)$"
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    cfg.events["randomize_head_com"].params["asset_cfg"].body_names = (
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
    )
    cfg.terminations["nan_state"].params["sensor_names"] = ("feet_ground_contact",)

    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["head_pose"].zero_command_prob = 1.0
    cfg.curriculum.pop("head_pose_range", None)
    cfg.curriculum.pop("head_pose_tracking_weight", None)
    cfg.rewards["head_pose_tracking"].weight = 1.0
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=mdp.head_pose_bias_penalty,
        weight=3.0,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    bearing = cfg.events["randomize_wheel_friction"]
    bearing.params["ranges"] = (
        (WHEEL_BEARING_FRICTION_NOMINAL,) * 2 if play
        else WHEEL_BEARING_FRICTION_RANGE
    )
    bearing.params["shared_random"] = False
    cfg.curriculum.pop("wheel_friction", None)
    cfg.events["randomize_tire_sliding_friction"] = EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=(r"^(left|right)_(front|rear)_tire_collision$",),
            ),
            "ranges": (
                (TIRE_SLIDING_FRICTION_NOMINAL,) * 2 if play
                else TIRE_SLIDING_FRICTION_RANGE
            ),
            "operation": "abs",
            "axes": [0],
            "shared_random": True,
        },
    )
    cfg.rewards["leg_symmetry"].weight = 6.0
    cfg.rewards["com_height_target"].weight = 1.2
    cfg.rewards["joint_torques_l2"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -3.0
    cfg.rewards["neck_action_rate_l2"].weight = -2.0
    cfg.curriculum.pop("action_rate_weight", None)
    cfg.rewards["heading_tracking"].weight *= 0.5
    for stage in cfg.curriculum["heading_tracking_weight"].params["weight_stages"]:
        stage["weight"] *= 0.5

    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="wheel_ground_contact",
        primary=ContactMatch(
            mode="body", entity="robot",
            pattern=("left_front_tire", "left_rear_tire",
                     "right_front_tire", "right_rear_tire"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="netforce", num_slots=1,
    ))
    cfg.rewards["four_wheels_grounded"] = RewardTermCfg(
        func=mdp.four_wheels_grounded_reward, weight=0.5,
        params={"sensor_name": "wheel_ground_contact"},
    )
    angular_noise = cfg.observations["actor"].terms["base_ang_vel"].noise
    angular_noise.n_min, angular_noise.n_max = -radians(10.0), radians(10.0)
    gravity_noise = cfg.observations["actor"].terms["projected_gravity"].noise
    gravity_noise.n_min, gravity_noise.n_max = -0.03, 0.03
    max_push = {
        "x": (-0.35, 0.35), "y": (-0.35, 0.35),
        "roll": (-radians(20.0), radians(20.0)),
        "pitch": (-radians(20.0), radians(20.0)),
        "yaw": (-radians(20.0), radians(20.0)),
    }
    cfg.events["push_robot"].params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in max_push
    }
    cfg.curriculum["push_velocity_scale"] = CurriculumTermCfg(
        func=mdp.push_velocity_linear_curriculum,
        params={"event_name": "push_robot", "max_velocity_range": max_push,
                "end_step": 2000 * 24},
    )
    if play:
        cfg.commands["twist"].rel_standing_envs = 0.1
        cfg.rewards["heading_hold"].weight = 0.0
        cfg.rewards["heading_tracking"].weight = 1.5
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
    return configure_xgoduck_physics(cfg)


XgoduckSwizzleRlCfg = replace(
    MicroduckSwizzleRlCfg,
    experiment_name="xgoduck_velocity_swizzle",
    num_steps_per_env=24,
    run_name="xgoduck_swizzle",
    logger="tensorboard",
    upload_model=False,
)
