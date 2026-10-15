"""XgoDuck kick and forward-roll recipes using the HLS1910 full-contact model."""

from copy import deepcopy
from dataclasses import replace

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_STANDUP_ROBOT_CFG,
    XGODUCK_STAND_Z,
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


def make_xgoduck_roulade_env_cfg(play=False):
    cfg = _configure_robot(make_microduck_roulade_env_cfg(play=play))
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
    # Playback starts standing; training keeps the reverse curriculum.
    if play:
        spawn.update(standing_prob=1.0, midroll_prob=0.0)
        cfg.curriculum.pop("roulade_spawn_mix", None)
    return cfg


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
