"""XgoDuck kick and forward-roll recipes using the HLS1910 full-contact model."""

from copy import deepcopy
from dataclasses import replace
from math import radians

from mjlab.managers import EventTermCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_STANDUP_ROBOT_CFG,
    XGODUCK_STAND_Z,
    XGODUCK_ROULADE_ROBOT_CFG,
)
from . import mdp
from .xgoduck_physics import configure_xgoduck_physics
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
    return configure_xgoduck_physics(cfg)


def make_xgoduck_ball_kick_env_cfg(play=False, kick_foot="right"):
    cfg = _configure_robot(make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot))
    cfg.rewards["height_stand"].params["target_height"] = XGODUCK_STAND_Z
    spawn = cfg.events["set_ground_state"].params
    spawn["standing_z_min"] = XGODUCK_STAND_Z - 0.005
    spawn["standing_z_max"] = XGODUCK_STAND_Z + 0.005
    cfg.events["reset_ball"].params["offset"] = (
        BALL_OFFSET_X, -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y,
    )
    return cfg


def make_xgoduck_roulade_env_cfg(play=False, *, angle_observation=False):
    cfg = _configure_robot(make_microduck_roulade_env_cfg(play=play))
    cfg.scene.entities["robot"] = deepcopy(XGODUCK_ROULADE_ROBOT_CFG)
    # Pooled contact storage; the standing reset avoids a second full-world
    # collision pass and its large convex-collision scratch allocations.
    cfg.sim.nconmax = 64
    cfg.events.pop("base_com", None)  # Inherited empty selector; use the explicit CoM events.
    # Keep simulating after landing so the same actor learns to remain standing.
    cfg.episode_length_s = 7.0
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(cfg.commands["twist"].ranges, name, (0.0, 0.0))
    dz = XGODUCK_STAND_Z - MICRODUCK_STAND_Z
    for name in ("roulade_landing_composite", "roulade_height_after_roll",
                 "roulade_landing_sharp", "roulade_stand_tax"):
        cfg.rewards[name].params["target_height"] = XGODUCK_STAND_Z
    cfg.rewards["roulade_rise_velocity"].params["max_height"] = XGODUCK_STAND_Z + 0.01
    for name in ("roulade_landing_composite", "roulade_landing_sharp"):
        cfg.rewards[name].params["require_feet"] = True
    for key in ("height_low", "height_high"):
        cfg.rewards["arrival_damping"].params[key] += dz
    spawn = cfg.events["set_roulade_state"].params
    spawn["standing_z_min"] = XGODUCK_STAND_Z - 0.005
    spawn["standing_z_max"] = XGODUCK_STAND_Z + 0.005
    spawn["midroll_z_min"] += dz
    spawn["midroll_z_max"] += dz
    spawn["head_top_axis"] = HEAD_TOP_AXIS
    # Pick the final configuration first; ground it geometrically afterwards.
    for key in ("standing_z_min", "standing_z_max", "midroll_z_min", "midroll_z_max"):
        spawn[key] = XGODUCK_STAND_Z
    cfg.events["ground_roulade_reset"] = EventTermCfg(
        func=mdp.reset_root_to_ground,
        mode="reset",
        params={"clearance": 0.0005},
    )
    cfg.events["randomize_head_com"].params["asset_cfg"].body_names = (
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
    )
    # Both ablations start a complete roll at zero; intermediate-roll spawns
    # would make the angle at landing depend on the sampled starting posture.
    spawn.update(standing_prob=1.0, midroll_prob=0.0)
    cfg.curriculum.pop("roulade_spawn_mix", None)
    if angle_observation:
        for group in ("actor", "critic"):
            terms = cfg.observations[group].terms
            terms.pop("body_command")
            terms["roulade_angle"] = ObservationTermCfg(
                func=mdp.roulade_angle_observation,
                noise=UniformNoiseCfg(n_min=-radians(30), n_max=radians(30))
                if group == "actor" else None,
            )
            terms["body_command"] = ObservationTermCfg(
                func=mdp.zero_command_padding, params={"dim": 5},
            )
    # The baseline keeps the original actor/critic observation layout. The
    # angle variant exposes only signed cumulative radians, without a latch,
    # maximum-angle feature, or additional angle-tracking reward.
    if play:
        spawn.update(standing_prob=1.0, midroll_prob=0.0)
        cfg.curriculum.pop("roulade_spawn_mix", None)
        cfg.observations["actor"].enable_corruption = False
        for name in (
            "encoder_bias", "randomize_com", "randomize_head_com",
            "randomize_armature", "randomize_joint_friction",
        ):
            cfg.events.pop(name, None)
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = False
        for name in ("base_ang_vel", "projected_gravity", "joint_vel"):
            term = cfg.observations["actor"].terms[name]
            term.delay_min_lag = term.delay_max_lag = 0
            if "max_angle_deg" in term.params:
                term.params["max_angle_deg"] = 0.0
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
        articulation = cfg.scene.entities["robot"].articulation
        articulation.actuators = tuple(replace(
            servo,
            vin_range=(sum(servo.vin_range) / 2.0,) * 2,
            vin_drop_gain_range=(sum(servo.vin_drop_gain_range) / 2.0,) * 2,
        ) for servo in articulation.actuators)
        cfg.curriculum = {}
    return cfg


def make_xgoduck_roulade_angle_env_cfg(play=False):
    return make_xgoduck_roulade_env_cfg(play=play, angle_observation=True)


XgoduckBallKickRlCfg = replace(
    MicroduckBallKickRlCfg, logger="tensorboard", experiment_name="xgoduck_ball_kick_right",
    run_name="xgoduck", upload_model=False,
)
XgoduckBallKickLeftRlCfg = replace(
    XgoduckBallKickRlCfg, experiment_name="xgoduck_ball_kick_left",
)
XgoduckRouladeRlCfg = replace(
    MicroduckRouladeRlCfg, logger="tensorboard", experiment_name="xgoduck_roulade",
    run_name="no_angle", upload_model=False,
    # Keep mirroring off until XgoDuck's action-axis mapping is validated.
    algorithm=replace(MicroduckRouladeRlCfg.algorithm, symmetry_cfg=None),
)
XgoduckRouladeAngleRlCfg = replace(
    XgoduckRouladeRlCfg,
    experiment_name="xgoduck_roulade_angle",
    run_name="angle_noise30deg",
)
