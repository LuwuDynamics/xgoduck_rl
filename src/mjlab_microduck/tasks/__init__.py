from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    DESCENT_END as _GP_DESCENT_END,
    HOLD_END as _GP_HOLD_END,
    RISE_END as _GP_RISE_END,
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# XgoDuck velocity: walk recipe with the XgoDuck XML and spawn height.
from dataclasses import replace

import math

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_SIT_Z,
    XGODUCK_STAND_Z,
    XGODUCK_STANDUP_ROBOT_CFG,
    XGODUCK_WALK_ROBOT_CFG,
)
from . import mdp as microduck_mdp
from .microduck_standup_env_cfg import STAND_Z as _MICRODUCK_STANDUP_STAND_Z

# Drop-from-air standup: 10 cm above measured stand height, 2 s frozen, then rise.
_XGODUCK_STANDUP_DROP_DZ = 0.10
_XGODUCK_STANDUP_FREEZE_S = 2.0
# 2 s freeze + MicroDuck's 6 s rise window.
_XGODUCK_STANDUP_EPISODE_S = 8.0

# view.py pick_conf (deg): legs + neck/head so the mouth can reach the floor.
_XGODUCK_PICK_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": math.radians(-5),
    "left_hip_pitch": math.radians(-80),
    "left_knee": 0.0,
    "left_ankle": math.radians(50),
    "neck_pitch": math.radians(-120),
    "head_pitch": math.radians(-90),
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": math.radians(5),
    "right_hip_pitch": math.radians(80),
    "right_knee": 0.0,
    "right_ankle": math.radians(-50),
}

# TensorBoard logs under logs/rsl_rl/xgoduck_velocity/.
XgoduckRlCfg = replace(
    MicroduckRlCfg,
    logger="tensorboard",
    experiment_name="xgoduck_velocity",
    run_name="xgoduck",
    upload_model=False,
)


from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from .xgoduck_physics import configure_xgoduck_physics


def _xgoduck_velocity_cfg(**kwargs):
    cfg = make_microduck_velocity_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": XGODUCK_WALK_ROBOT_CFG}
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        XGODUCK_STAND_Z - 0.005,
        XGODUCK_STAND_Z + 0.005,
    )
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
        }
    actor = cfg.observations["actor"].terms
    actor["base_ang_vel"].noise.n_min = -0.06
    actor["base_ang_vel"].noise.n_max = 0.06
    actor["projected_gravity"].noise.n_min = -0.03
    actor["projected_gravity"].noise.n_max = 0.03
    actor["joint_pos"].noise.n_min = -0.01
    actor["joint_pos"].noise.n_max = 0.01
    # Keep the forward command cap explicit in the XgoDuck walk recipe.
    cfg.commands["twist"].ranges.lin_vel_x = (-0.4, 0.4)
    cfg.rewards["track_linear_velocity"].weight = 2.2
    cfg.rewards["track_linear_velocity"].params["std"] = 0.30

    # Shorter swings and lower foot lift for a more conservative walking gait.
    cfg.rewards["air_time"].params["threshold_min"] = 0.100
    cfg.rewards["air_time"].params["threshold_max"] = 0.250
    cfg.rewards["foot_clearance"].params["target_height"] = 0.015
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.015
    cfg.commands["twist"].rel_backward_envs = 0.15
    cfg.commands["twist"].backward_speed_range = (-0.2, -0.1)

    # Hold all four head command deltas at zero until iteration 1500, then
    # restore the existing normal ranges in stages, reaching 100% at 3000.
    head_curriculum = cfg.curriculum["head_pose_range"].params
    normal_head_ranges = head_curriculum["range_stages"][-1]["ranges"]
    head_curriculum["range_stages"] = [
        {
            "step": iteration * XgoduckRlCfg.num_steps_per_env,
            "ranges": tuple((lo * scale, hi * scale) for lo, hi in normal_head_ranges),
        }
        for iteration, scale in (
            (0, 0.0), (1500, 0.05), (2000, 0.35), (2500, 0.65), (3000, 1.0)
        )
    ]
    cfg.commands["head_pose"].ranges = head_curriculum["range_stages"][0]["ranges"]

    # Price sustained neck compensation, not normal stride-frequency motion.
    # Positive weight: the function returns a negative excess bias in radians.
    cfg.rewards["neck_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.neck_pose_bias_penalty,
        weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0,
                "deadband_rad": math.radians(3.0)},
    )
    cfg.curriculum["neck_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "neck_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * XgoduckRlCfg.num_steps_per_env, "weight": 0.25},
                {"step": 1000 * XgoduckRlCfg.num_steps_per_env, "weight": 0.5},
            ],
        },
    )
    cfg.rewards["pose"].weight = 1.3
    # Hip yaw inward stop is 0.22 rad.
    _std_s = dict(cfg.rewards["pose"].params["std_standing"])
    _std_w = dict(cfg.rewards["pose"].params["std_walking"])
    _std_r = dict(cfg.rewards["pose"].params["std_running"])
    _std_s[r".*hip_yaw.*"] = 0.05
    _std_w[r".*hip_yaw.*"] = 0.15
    _std_r[r".*hip_yaw.*"] = 0.25
    cfg.rewards["pose"].params["std_standing"] = _std_s
    cfg.rewards["pose"].params["std_walking"] = _std_w
    cfg.rewards["pose"].params["std_running"] = _std_r
    return configure_xgoduck_physics(cfg)


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-XgoDuck",
    env_cfg=_xgoduck_velocity_cfg(),
    play_env_cfg=_xgoduck_velocity_cfg(play=True),
    rl_cfg=XgoduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-XgoDuck",
    env_cfg=_xgoduck_velocity_cfg(rough=True),
    play_env_cfg=_xgoduck_velocity_cfg(play=True, rough=True),
    rl_cfg=XgoduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# XgoDuck StandUp / GroundPick / SitStand use the full-contact model.
# Heights measured on the upstream trunk are shifted to XGODUCK_STAND_Z.
XgoduckStandUpRlCfg = replace(
    MicroduckStandUpRlCfg,
    logger="tensorboard",
    experiment_name="xgoduck_standup",
    run_name="xgoduck",
    upload_model=False,
)
XgoduckGroundPickRlCfg = replace(
    MicroduckGroundPickRlCfg,
    logger="tensorboard",
    experiment_name="xgoduck_ground_pick",
    run_name="xgoduck",
    upload_model=False,
)
XgoduckSitStandRlCfg = replace(
    MicroduckSitStandRlCfg,
    logger="tensorboard",
    experiment_name="xgoduck_sitstand",
    run_name="xgoduck",
    upload_model=False,
)


def _retarget_xgoduck_standup(cfg):
    """Keep MicroDuck standup rewards; only shift heights and use drop+freeze."""
    stand = XGODUCK_STAND_Z
    sit = XGODUCK_SIT_Z
    dz = stand - _MICRODUCK_STANDUP_STAND_Z
    r = cfg.rewards
    for name in (
        "height_stand",
        "height_stand_sharp",
        "height_stand_l1",
        "standing_composite",
    ):
        r[name].params["target_height"] = stand
    r["com_upward_velocity"].params["max_height"] = stand + 0.010
    r["upright_sharp"].params["height_low"] = sit
    r["upright_sharp"].params["height_high"] = stand
    r["arrival_damping"].params["height_low"] += dz
    r["arrival_damping"].params["height_high"] += dz
    if "head_pose_bias" in r:
        r["head_pose_bias"].params["gate_height_low"] += dz
        r["head_pose_bias"].params["gate_height_high"] += dz
    if "body_pose_tracking" in r:
        r["body_pose_tracking"].params["nominal_height"] = stand

    drop_z = stand + _XGODUCK_STANDUP_DROP_DZ
    cfg.events["set_ground_state"].func = microduck_mdp.set_random_drop_state
    cfg.events["set_ground_state"].params = {
        "drop_z_min": drop_z - 0.005,
        "drop_z_max": drop_z + 0.005,
        "euler_abs_max": math.pi,
    }
    cfg.curriculum.pop("ground_state_mix", None)
    cfg.episode_length_s = _XGODUCK_STANDUP_EPISODE_S

    joint = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = microduck_mdp.FreezeThenJointPositionActionCfg(
        entity_name=joint.entity_name,
        actuator_names=joint.actuator_names,
        scale=joint.scale,
        offset=joint.offset,
        preserve_order=joint.preserve_order,
        use_default_offset=joint.use_default_offset,
        clip=joint.clip,
        freeze_s=_XGODUCK_STANDUP_FREEZE_S,
    )
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = (
            microduck_mdp.push_by_setting_velocity_after_delay
        )
        cfg.events["push_robot"].params["delay_s"] = _XGODUCK_STANDUP_FREEZE_S

    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    return configure_xgoduck_physics(cfg)


def _xgoduck_standup_cfg(**kwargs):
    cfg = make_microduck_standup_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": XGODUCK_STANDUP_ROBOT_CFG}
    return _retarget_xgoduck_standup(cfg)


def _xgoduck_ground_pick_cfg(**kwargs):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cfg = make_microduck_ground_pick_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": XGODUCK_STANDUP_ROBOT_CFG}
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        XGODUCK_STAND_Z - 0.005,
        XGODUCK_STAND_Z + 0.005,
    )
    # Hip yaw/roll stay at the default pose during the crouch.
    cfg.rewards["hip_yaw_roll_hold"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-7.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*hip_yaw.*", r".*hip_roll.*")
            ),
        },
    )
    # Measured send_conf crouch balances at gx≈0.54 (~33° forward), gy≈0.
    cfg.rewards["descent_gravity_x"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_gravity_x_phased,
        weight=1.0,
        params={
            "command_name": "twist",
            "max_forward": 0.65,
            "descent_end": _GP_DESCENT_END,
            "hold_end": _GP_HOLD_END,
            "rise_end": _GP_RISE_END,
        },
    )
    cfg.rewards["descent_gravity_y_abs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_gravity_y_abs_phased,
        weight=-1.0,
        params={
            "command_name": "twist",
            "descent_end": _GP_DESCENT_END,
            "hold_end": _GP_HOLD_END,
            "rise_end": _GP_RISE_END,
        },
    )
    cfg.rewards["mouth_ground_proximity"].params["std"] = 0.04
    cfg.rewards["neck_vel_descent"].weight = -0.01
    # Light STAND↔pick interpolation (view.py pick_conf). Down-gate via blend.
    cfg.rewards["descent_pick_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=1.0,
        params={
            "command_name": "twist",
            "std": 0.4,
            "target_pose": _XGODUCK_PICK_POSE,
            "descent_end": _GP_DESCENT_END,
            "hold_end": _GP_HOLD_END,
            "rise_end": _GP_RISE_END,
        },
    )
    return configure_xgoduck_physics(cfg)


def _retarget_xgoduck_sitstand(cfg):
    """Shift MicroDuck sit/stand heights onto XGODUCK kinematic trunk z."""
    stand = XGODUCK_STAND_Z
    sit = XGODUCK_SIT_Z
    r = cfg.rewards
    for name in (
        "posture_height",
        "posture_height_sharp",
        "posture_height_l1",
        "posture_stillness",
        "posture_composite",
    ):
        r[name].params["sit_z"] = sit
        r[name].params["stand_z"] = stand
    r["rise_bootstrap"].params["max_height"] = stand + 0.010
    # MicroDuck: SIT_UPRIGHT_Z = SIT_Z + 0.015, STAND_UPRIGHT_Z = STAND_Z - 0.015
    r["upright_while_tall"].params["height_low"] = sit + 0.015
    r["upright_while_tall"].params["height_high"] = stand - 0.015
    cfg.commands["twist"].sit_z = sit
    cfg.commands["twist"].stand_z = stand
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        stand - 0.005,
        stand + 0.005,
    )
    gs = cfg.events["set_ground_state"].params
    gs["sitting_z_min"] = sit
    gs["sitting_z_max"] = sit + 0.015
    gs["standing_z_min"] = stand - 0.005
    gs["standing_z_max"] = stand + 0.005
    return configure_xgoduck_physics(cfg)


def _xgoduck_sitstand_cfg(**kwargs):
    cfg = make_microduck_sitstand_env_cfg(**kwargs)
    cfg.scene.entities = {"robot": XGODUCK_STANDUP_ROBOT_CFG}
    return _retarget_xgoduck_sitstand(cfg)


register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-XgoDuck",
    env_cfg=_xgoduck_standup_cfg(),
    play_env_cfg=_xgoduck_standup_cfg(play=True),
    rl_cfg=XgoduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-XgoDuck",
    env_cfg=_xgoduck_standup_cfg(rough=True),
    play_env_cfg=_xgoduck_standup_cfg(play=True, rough=True),
    rl_cfg=XgoduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-XgoDuck",
    env_cfg=_xgoduck_ground_pick_cfg(),
    play_env_cfg=_xgoduck_ground_pick_cfg(play=True),
    rl_cfg=XgoduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-XgoDuck",
    env_cfg=_xgoduck_ground_pick_cfg(rough=True),
    play_env_cfg=_xgoduck_ground_pick_cfg(play=True, rough=True),
    rl_cfg=XgoduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-XgoDuck",
    env_cfg=_xgoduck_sitstand_cfg(),
    play_env_cfg=_xgoduck_sitstand_cfg(play=True),
    rl_cfg=XgoduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-XgoDuck",
    env_cfg=_xgoduck_sitstand_cfg(rough=True),
    play_env_cfg=_xgoduck_sitstand_cfg(play=True, rough=True),
    rl_cfg=XgoduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# XgoDuck dynamic tricks (flat terrain); default BallKick is right-footed.
from .xgoduck_tricks_env_cfg import (
    make_xgoduck_ball_kick_env_cfg, make_xgoduck_roulade_env_cfg,
    XgoduckBallKickRlCfg, XgoduckBallKickLeftRlCfg, XgoduckRouladeRlCfg,
)

for _task_id, _factory, _kwargs, _runner_cfg in (
    ("Mjlab-BallKick-Flat-XgoDuck", make_xgoduck_ball_kick_env_cfg,
     {"kick_foot": "right"}, XgoduckBallKickRlCfg),
    ("Mjlab-BallKick-Left-Flat-XgoDuck", make_xgoduck_ball_kick_env_cfg,
     {"kick_foot": "left"}, XgoduckBallKickLeftRlCfg),
    ("Mjlab-Roulade-Flat-XgoDuck", make_xgoduck_roulade_env_cfg, {}, XgoduckRouladeRlCfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_factory(**_kwargs),
        play_env_cfg=_factory(play=True, **_kwargs),
        rl_cfg=_runner_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )


from .xgoduck_velocity_swizzle_env_cfg import (
    XgoduckSwizzleRlCfg,
    make_xgoduck_velocity_swizzle_env_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-XgoDuck",
    env_cfg=make_xgoduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_xgoduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=XgoduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


from mjlab_microduck.play_hook import maybe_inject_latest_checkpoint

# After all task ids are registered: `play TASK` with no checkpoint flag
# picks the latest local model_*.pt. Explicit wandb/file flags still win.
maybe_inject_latest_checkpoint()
