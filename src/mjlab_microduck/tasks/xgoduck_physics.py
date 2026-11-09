"""Shared HLS1910 dynamics for all XgoDuck task families."""
from copy import deepcopy
import math
import re

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.robot.xgoduck_constants import XGODUCK_WALK_ROBOT_CFG
from .backlash import make_backlash_variant

HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
SERVO_JOINTS = (r"^(?!passive_).*",)
FOOT_FRICTION_RANGE = (0.2, 0.8)
SERVO_FRICTION_SCALE_RANGE = (0.9, 1.3)


def _exclude_backlash(value):
    if isinstance(value, SceneEntityCfg):
        if value.name == "robot" and value.joint_names:
            patterns = (value.joint_names,) if isinstance(value.joint_names, str) else value.joint_names
            value.joint_names = tuple(
                p if "_backlash" in p or re.fullmatch(r"[A-Za-z0-9_]+", p) else r"^(?!passive_.*_backlash)" + p.lstrip("^")
                for p in patterns
            )
    elif isinstance(value, dict):
        for child in value.values():
            _exclude_backlash(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _exclude_backlash(child)


def configure_xgoduck_physics(cfg, *, randomize_mass_inertia: bool = True):
    """Keep task geometry/rewards; use Walk's servo model, timing and mass DR."""
    # Several upstream factories share nested defaults. Never mutate those.
    cfg = deepcopy(cfg)
    robot = cfg.scene.entities["robot"]
    robot.articulation.actuators = deepcopy(XGODUCK_WALK_ROBOT_CFG.articulation.actuators)
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    # Ordinary sole friction shares one range across XgoDuck tasks.
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params.update(
            ranges=FOOT_FRICTION_RANGE, distribution="uniform",
        )
    if "randomize_joint_friction" in cfg.events:
        cfg.events["randomize_joint_friction"].params["scale_range"] = (
            SERVO_FRICTION_SCALE_RANGE
        )
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"].body_names = HEAD_BODIES
    if randomize_mass_inertia:
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(.95) / 2., math.log(1.05) / 2.),
            },
        )
    else:
        cfg.events.pop("randomize_mass_inertia", None)

    reset = cfg.events.get("reset_robot_joints")
    if reset is not None:
        reset.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
        passive_reset = deepcopy(reset)
        passive_reset.params.update(
            asset_cfg=SceneEntityCfg("robot", joint_names=(r"^passive_.*",)),
            position_range=(0., 0.), velocity_range=(0., 0.),
        )
        # Zero passive hinges BEFORE task-specific ground/ball placement events.
        events = {}
        for name, event in cfg.events.items():
            if name in ("reset_backlash_joints", "reset_passive_joints"):
                continue
            events[name] = event
            if name == "reset_robot_joints":
                events["reset_passive_joints"] = passive_reset
        cfg.events = events
    for name in ("encoder_bias", "randomize_armature"):
        if name in cfg.events:
            cfg.events[name].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
    for group in ("actor", "critic"):
        for name in ("joint_pos", "joint_vel"):
            term = cfg.observations[group].terms.get(name)
            if term is not None:
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
    cfg = make_backlash_variant(cfg, robot)
    if "dof_pos_limits" in cfg.rewards:
        cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
    for reward in cfg.rewards.values():
        if reward is not None:
            _exclude_backlash(reward.params)
    return cfg
