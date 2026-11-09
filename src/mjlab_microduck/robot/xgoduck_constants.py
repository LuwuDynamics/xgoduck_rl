"""XgoDuck robot configs for the shared task stack.

Same joint, body, site, and foot-collision names as the upstream tasks.
Actuator is the HLS1910 BAM model. Speed comes from voltage and back-EMF
(max_velocity=100 disables the firmware goal-slew).

Walk uses a ±22° HOME pose and ±1.5° serial gear play.
All XgoDuck task models use the same ±1.5° servo play and HLS1910 drive.
Trunk heights keep the soles clear of the floor.
"""

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.actuator import BacklashEncoderBamActuatorCfg, FrictionDRBamActuatorCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "xgoduck"
_HLS1910_JSON: Path = _ROBOT_DIR / "params" / "1910_m6.json"

XGODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
XGODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
XGODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
XGODUCK_SITSTAND_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"

assert XGODUCK_WALK_XML.exists(), f"XML not found: {XGODUCK_WALK_XML}"
assert XGODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {XGODUCK_WALK_BACKLASH_XML}"
assert XGODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {XGODUCK_ALLCOLLISIONS_XML}"
assert _HLS1910_JSON.exists(), f"HLS1910 BAM params not found: {_HLS1910_JSON}"

# Trunk z at XGODUCK STAND (hip_pitch/ankle ±22°) so soles sit 1 mm above the floor.
# CAD-zero (all joints 0) is 0.129394 m; the HOME leg bend needs ~10 mm more.
XGODUCK_STAND_Z = 0.139158
# Same kinematic sole-clearance height at the sitstand SIT keyframe
# (knee ±1.35, hip_pitch ±0.4079, ankle/hip_roll 0; neck stays HOME).
XGODUCK_SIT_Z = 0.085868

_DEG22 = math.radians(22)
XGODUCK_HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, XGODUCK_STAND_Z),
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -_DEG22,
        r".*right_hip_pitch.*": _DEG22,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": _DEG22,
        r".*right_ankle.*": -_DEG22,
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Keep Walk HOME independent of tricks and roller configuration.
# At 22 degrees, nominal soles clear the floor by 1 mm.
XGODUCK_WALK_STAND_Z = XGODUCK_STAND_Z
XGODUCK_WALK_HOME_FRAME = deepcopy(XGODUCK_HOME_FRAME)
XGODUCK_WALK_HOME_FRAME.pos = (0.0, 0.0, XGODUCK_WALK_STAND_Z)
# Passive hinges must not match the broad servo HOME patterns.
XGODUCK_WALK_HOME_FRAME.joint_pos = {
    r"^passive_.*_backlash$": 0.0,
    **{r"^(?!passive_)" + pattern.lstrip("^"): value
       for pattern, value in XGODUCK_HOME_FRAME.joint_pos.items()},
}


# Named `*_collision` geoms. Bit 0 = world/floor, bit 1 = head↔base.
# Walk: feet only. Standup: every link has a mesh and hits the floor.
# Feet and other links: contype=1 conaffinity=0 → floor only, no robot-robot.
# trunk_collision (base) and head_collision (head_roll / jaw_soft):
#   contype=3 conaffinity=2 → floor and each other, not the other links.
XGODUCK_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    contype={
        r"^(trunk_collision|head_collision)$": 3,
        ".*_collision": 1,
    },
    conaffinity={
        r"^(trunk_collision|head_collision)$": 2,
        ".*_collision": 0,
    },
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)


# Get-up tasks: preserve every link's ground contact, and allow exactly
# head-trunk plus the head/left-foot/right-foot clique. Bit 0 is floor,
# bit 1 is head-trunk, bit 2 is shared by the head and the two soles.
XGODUCK_GETUP_COLLISION = replace(
    XGODUCK_COLLISION,
    contype={
        r"^head_collision$": 7,
        r"^trunk_collision$": 3,
        r"^(left|right)_foot_collision$": 5,
        ".*_collision": 1,
    },
    conaffinity={
        r"^head_collision$": 6,
        r"^trunk_collision$": 2,
        r"^(left|right)_foot_collision$": 4,
        ".*_collision": 0,
    },
)

# HLS1910 identification (sts3215 firmware). max_velocity=100 disables the
# firmware goal-slew so speed comes from voltage + back-EMF.
# kp_fw=5 matches the real Feetech P. BAM converts XML <position> to motor;
# XML kp is only a viewer companion.
actuators = BacklashEncoderBamActuatorCfg(
    json_path=str(_HLS1910_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=5.0,
    vin_range=(7.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=7.0,  # sag floor below sampled vin (7.4–8.0) so load can still drop V
    # mjlab applies delay through this buffer, not BAM JSON command_delay.
    # Every XgoDuck task uses 0–6 physics steps at 5 ms = 0–30 ms.
    delay_min_lag=0,
    delay_max_lag=6,
)


def _disable_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """CollisionCfg looks up geoms by name, so unnamed collision meshes leak."""
    for geom in spec.geoms:
        if not geom.name:
            geom.contype = 0
            geom.conaffinity = 0
    return spec


def get_walk_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_WALK_BACKLASH_XML)))


def get_standup_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_SITSTAND_BACKLASH_XML)))


# Shared backlash-servo defaults: 0–6 physics steps for every XgoDuck task.
_walk_actuators = BacklashEncoderBamActuatorCfg(
    json_path=str(_HLS1910_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=5.0,
    vin_range=(7.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=7.0,
    delay_min_lag=0,
    delay_max_lag=6,
)

XGODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=XGODUCK_WALK_HOME_FRAME,
    collisions=(XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        # Walk: 0–6 physics steps at 5 ms = 0–30 ms.
        actuators=(replace(_walk_actuators, delay_min_lag=0, delay_max_lag=6),),
        soft_joint_pos_limit_factor=0.9,
    ),
)

XGODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=deepcopy(XGODUCK_WALK_HOME_FRAME),
    collisions=(XGODUCK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


def get_sitstand_spec() -> mujoco.MjSpec:
    return _disable_unnamed_geoms(mujoco.MjSpec.from_file(str(XGODUCK_SITSTAND_BACKLASH_XML)))


# SitStand needs full-body contacts with the same ±1.5° gear play as Walk.
_sitstand_home = deepcopy(XGODUCK_HOME_FRAME)
_sitstand_home.joint_pos = {
    r"^passive_.*_backlash$": 0.0,
    **{r"^(?!passive_)" + pattern.lstrip("^"): value
       for pattern, value in XGODUCK_HOME_FRAME.joint_pos.items()},
}
XGODUCK_SITSTAND_ROBOT_CFG = deepcopy(XGODUCK_STANDUP_ROBOT_CFG)
XGODUCK_SITSTAND_ROBOT_CFG.spec_fn = get_sitstand_spec
XGODUCK_SITSTAND_ROBOT_CFG.init_state = _sitstand_home
XGODUCK_SITSTAND_ROBOT_CFG.articulation.actuators = (deepcopy(_walk_actuators),)


def get_roulade_spec() -> mujoco.MjSpec:
    """Keep the legacy Roulade trunk CoM independent of the Walk model tuning."""
    spec = get_standup_spec()
    trunk = spec.body("trunk_base")
    ipos = list(trunk.ipos)
    ipos[0] = -0.01327575578150381
    trunk.ipos = ipos
    return spec


XGODUCK_ROULADE_ROBOT_CFG = deepcopy(XGODUCK_STANDUP_ROBOT_CFG)
XGODUCK_ROULADE_ROBOT_CFG.spec_fn = get_roulade_spec
XGODUCK_ROULADE_ROBOT_CFG.articulation.actuators = (deepcopy(_walk_actuators),)
