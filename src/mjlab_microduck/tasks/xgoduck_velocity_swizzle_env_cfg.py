"""XgoDuck adaptation of the MicroDuck grounded, symmetric Swizzle recipe.

The four head command deltas remain zero: neck_pitch, head_pitch, head_yaw
and head_roll are encouraged to hold their existing HOME positions. The
61-dimensional actor interface and 14 servo actions remain unchanged.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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


# Match the final stage of the inherited MicroDuck bearing-drag curriculum.
WHEEL_BEARING_FRICTION_NOMINAL = 0.0015  # N m per wheel.
TIRE_SLIDING_FRICTION_NOMINAL = 1.0
TIRE_SLIDING_FRICTION_RANGE = (0.2, 1.0)
HEAD_POSITION_WEIGHT = -1.0
HEAD_BIAS_WEIGHT = 3.0  # head_pose_bias_penalty already returns a negative cost.


def make_xgoduck_velocity_swizzle_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = deepcopy(make_microduck_velocity_swizzle_env_cfg(play=play))
    cfg.scene.entities = {"robot": deepcopy(XGODUCK_ROLLER_ROBOT_CFG)}
    cfg.actions["joint_pos"].joint_names = (r"^(?!passive_).*",)
    for sensor in cfg.scene.sensors:
        if sensor.name == "feet_ground_contact":
            sensor.primary.pattern = r"^(ankle_left|ankle_right)$"
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        XGODUCK_ROLLER_STAND_Z - 0.005,
        XGODUCK_ROLLER_STAND_Z + 0.005,
    )
    cfg.rewards["com_height_target"].params.update(
        target_height_min=XGODUCK_ROLLER_STAND_Z - 0.015,
        target_height_max=XGODUCK_ROLLER_STAND_Z + 0.005,
    )
    cfg.rewards["wheel_speed"].params["wheel_radius"] = XGODUCK_WHEEL_RADIUS

    # bearing_roll is a right-hip link on XgoDuck, not a head component.
    cfg.events["randomize_head_com"].params["asset_cfg"].body_names = (
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
    )
    cfg.terminations["nan_state"].params["sensor_names"] = ("feet_ground_contact",)

    # No head-motion curriculum: the default posture is the skating target.
    # Zero is a delta from HOME, not a command to set the physical joints to 0.
    cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    cfg.commands["head_pose"].zero_command_prob = 1.0
    cfg.curriculum.pop("head_pose_range", None)
    cfg.curriculum.pop("head_pose_tracking_weight", None)
    cfg.rewards.pop("head_pose_tracking", None)
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=mdp.neck_joint_pos_l2,
        weight=HEAD_POSITION_WEIGHT,
    )
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=mdp.head_pose_bias_penalty,
        weight=HEAD_BIAS_WEIGHT,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # Absolute reset sampling does not accumulate. Keep tire grip independent
    # of wheel-bearing drag; condim=3 does not add contact rolling resistance.
    bearing = cfg.events["randomize_wheel_friction"]
    if play:
        bearing.params["ranges"] = (WHEEL_BEARING_FRICTION_NOMINAL,) * 2
    bearing.params["shared_random"] = False
    cfg.events["randomize_tire_sliding_friction"] = EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", geom_names=(r"^(left|right)_(front|rear)_tire_collision$",)
            ),
            "ranges": (
                (TIRE_SLIDING_FRICTION_NOMINAL,) * 2
                if play else TIRE_SLIDING_FRICTION_RANGE
            ),
            "operation": "abs",
            "axes": [0],
            "shared_random": True,
        },
    )
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
    return configure_xgoduck_physics(cfg)


XgoduckSwizzleRlCfg = replace(
    MicroduckSwizzleRlCfg,
    experiment_name="xgoduck_velocity_swizzle",
    run_name="xgoduck_swizzle",
    logger="tensorboard",
    upload_model=False,
)
