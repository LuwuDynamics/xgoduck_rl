"""XgoDuck kick and forward-roll recipes using the HLS1910 full-contact model."""

from copy import deepcopy
from dataclasses import replace
import math
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_STANDUP_ROBOT_CFG,
    XGODUCK_ROULADE_ROBOT_CFG,
    XGODUCK_STAND_Z,
)
from . import mdp
from .xgoduck_physics import configure_xgoduck_physics
from .kick_hip_roll import HipHomeAfterKick
from .kick_ground_contact import non_foot_ground_contact, failure_event_cost
from .microduck_ball_kick_env_cfg import (
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)
from .microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    STAND_Z as MICRODUCK_STAND_Z,
    make_microduck_roulade_env_cfg,
)

# Ball placement retains clearance with the transferred 22-degree HOME pose.
# At the nearest noisy placement, ball rear x=0.110-0.015-0.035=0.060 m:
# ~15 mm clearance from the nominal toe, with room for joint/tilt noise.
BALL_OFFSET_X = 0.110
BALL_OFFSET_ABS_Y = 0.04133
# World up in jaw_soft's local frame at XgoDuck HOME (MicroDuck differs).
HEAD_TOP_AXIS = (0.0, 1.0, 0.0)


def _configure_robot(cfg):
    # Replace only the robot: BallKick must retain its separate ball entity.
    cfg.scene.entities["robot"] = deepcopy(XGODUCK_STANDUP_ROBOT_CFG)
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        XGODUCK_STAND_Z - 0.005, XGODUCK_STAND_Z + 0.005,
    )
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.terminations["nan_state"].params["sensor_names"] = ("feet_ground_contact",)
    for group in cfg.observations.values():
        for term in group.terms.values():
            if term.func.__name__ == "foot_contact_forces":
                term.func = mdp.foot_contact_forces_safe
    return cfg


def _configure_ballkick_collision_masks(cfg):
    # Terrain uses bit 1. Feet use bit 2; the ball uses bit 4.
    # Keep every collision body supported by terrain, with no robot self-pairs.
    robot = cfg.scene.entities["robot"]
    for collision in robot.collisions:
        collision.contype = {r"^(left|right)_foot_collision$": 2, r".*_collision": 0}
        collision.conaffinity = {r"^(left|right)_foot_collision$": 5, r".*_collision": 1}
    # The base task shares its ball asset; isolate these masks to XgoDuck.
    ball = deepcopy(cfg.scene.entities["ball"])
    ball.collisions = (CollisionCfg(
        geom_names_expr=("ball_geom",), contype=4, conaffinity=3,
    ),)
    cfg.scene.entities["ball"] = ball
    cfg.rewards.pop("self_collisions", None)
    cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if sensor.name != "self_collision")
    return cfg


def make_xgoduck_ball_kick_env_cfg(play=False, kick_foot="right"):
    cfg = _configure_robot(make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot))
    cfg = _configure_ballkick_collision_masks(cfg)
    # Only the two foot collision geoms may touch terrain. Contact clocks latch
    # brief touches across all four physics substeps without a force threshold.
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=r".*_collision", entity="robot",
            exclude=(r"^(left|right)_foot_collision$",),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1, track_air_time=True,
    ))
    cfg.terminations["non_foot_ground_contact"] = TerminationTermCfg(
        func=non_foot_ground_contact,
    )
    cfg.rewards["failure"] = RewardTermCfg(func=failure_event_cost, weight=-20.0)
    cfg.rewards["ball_forward_velocity"].weight = 8.0
    cfg.rewards["pose_stand_legs"].params["std"] = 0.4
    # Increase only the XgoDuck kick angular-velocity cost; other tricks retain theirs.
    cfg.rewards["body_ang_vel"].weight *= 2.0
    cfg.rewards["height_stand"].params["target_height"] = XGODUCK_STAND_Z
    spawn = cfg.events["set_ground_state"].params
    spawn["standing_z_min"] = XGODUCK_STAND_Z - 0.005
    spawn["standing_z_max"] = XGODUCK_STAND_Z + 0.005
    cfg.events["reset_ball"].params["offset"] = (
        BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y,
    )
    # After kicking, allow each hip roll/yaw within 5 degrees of HOME.
    for axis in ("roll", "yaw"):
        cfg.rewards[f"hip_{axis}_home_after_kick"] = RewardTermCfg(
            func=HipHomeAfterKick,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(f"left_hip_{axis}", f"right_hip_{axis}"),
                ),
                "return_start_s": 0.6,
                "return_full_s": 1.1,
                "deadband_rad": math.radians(5.0),
                "metric_prefix": f"kick_hip_{axis}",
            },
        )
    return configure_xgoduck_physics(cfg)


def make_xgoduck_roulade_env_cfg(play=False):
    cfg = _configure_robot(make_microduck_roulade_env_cfg(play=play))
    cfg.scene.entities["robot"] = deepcopy(XGODUCK_ROULADE_ROBOT_CFG)
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    # Root reset takes offsets; the final pose is grounded after all pose events.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    cfg.events.pop("base_com", None)  # Empty legacy selector; explicit CoM events follow.
    dz = XGODUCK_STAND_Z - MICRODUCK_STAND_Z
    for name in ("roulade_landing_composite", "roulade_height_after_roll",
                 "roulade_landing_sharp", "roulade_stand_tax"):
        cfg.rewards[name].params["target_height"] = XGODUCK_STAND_Z
    cfg.rewards["roulade_rise_velocity"].params["max_height"] = XGODUCK_STAND_Z + 0.01
    for key in ("height_low", "height_high"):
        cfg.rewards["arrival_damping"].params[key] += dz
    spawn = cfg.events["set_roulade_state"].params
    spawn["standing_z_min"] = XGODUCK_STAND_Z - 0.005
    spawn["standing_z_max"] = XGODUCK_STAND_Z + 0.005
    spawn["midroll_z_min"] += dz
    spawn["midroll_z_max"] += dz
    spawn["head_top_axis"] = HEAD_TOP_AXIS
    cfg.events["ground_roulade_reset"] = EventTermCfg(
        func=mdp.reset_root_to_ground, mode="reset", params={"clearance": 0.0005},
    )
    cfg.events["randomize_head_com"].params["asset_cfg"].body_names = (
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
    )
    # Playback starts standing; training keeps the reverse curriculum.
    if play:
        spawn.update(standing_prob=1.0, midroll_prob=0.0)
        cfg.curriculum.pop("roulade_spawn_mix", None)
    return configure_xgoduck_physics(cfg)


XgoduckBallKickRlCfg = replace(
    MicroduckBallKickRlCfg, logger="tensorboard", experiment_name="xgoduck_ball_kick_right",
    run_name="xgoduck", upload_model=False,
)
XgoduckBallKickLeftRlCfg = replace(
    XgoduckBallKickRlCfg, experiment_name="xgoduck_ball_kick_left",
)
XgoduckRouladeRlCfg = replace(
    MicroduckRouladeRlCfg, logger="tensorboard", experiment_name="xgoduck_roulade",
    run_name="xgoduck", upload_model=False,
    # Keep mirroring off until XgoDuck's action-axis mapping is validated.
    algorithm=replace(MicroduckRouladeRlCfg.algorithm, symmetry_cfg=None),
)
