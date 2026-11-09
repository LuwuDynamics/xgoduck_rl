"""Latest Swizzle CAD/HOME and servo fit, with get-up contact support."""
from copy import deepcopy
from dataclasses import replace
from math import radians
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from .xgoduck_constants import XGODUCK_HOME_FRAME, XGODUCK_GETUP_COLLISION, actuators

_ASSET_DIR=Path(__file__).parent/'xgoduck_roller'
ROLLER_STANDUP_XML=_ASSET_DIR/'robot_standup_swizzle.xml'
ROLLER_STAND_Z=0.1535429905506386
WHEEL_JOINTS=(r'^passive_.*wheel$',)
TIRE_GEOMS=(r'^(left|right)_(front|rear)_tire_collision$',)
_SKATE_GEOMS=r'^(left|right)_(ankle|(front|rear)_tire)_collision$'
_HIP_ROLL_GEOMS=r'^(left|right)_hip_roll_collision$'

ROLLER_HOME=deepcopy(XGODUCK_HOME_FRAME)
ROLLER_HOME.pos=(0.,0.,ROLLER_STAND_Z)
ROLLER_HOME.joint_pos={
    'left_hip_yaw':0.,'left_hip_roll':radians(2.),
    'left_hip_pitch':radians(-8.5),'left_knee':radians(8.5),'left_ankle':radians(17.),
    'neck_pitch':radians(25.),'head_pitch':radians(25.),'head_yaw':0.,'head_roll':0.,
    'right_hip_yaw':0.,'right_hip_roll':radians(-2.),
    'right_hip_pitch':radians(8.5),'right_knee':radians(-8.5),'right_ankle':radians(-17.),
    WHEEL_JOINTS[0]:0.,
}

# Ground support on all links; head-trunk and head/skates may also touch,
# matching the wheel-less recovery task's head/feet contact allowance.
# Bit 3 is emitted by hip-roll links and accepted by all skate geoms only.
ROLLER_GETUP_COLLISION=replace(
    XGODUCK_GETUP_COLLISION,
    contype={r'^head_collision$':7,r'^trunk_collision$':3,_SKATE_GEOMS:5,_HIP_ROLL_GEOMS:9,'.*_collision':1},
    conaffinity={r'^head_collision$':6,r'^trunk_collision$':2,_SKATE_GEOMS:12,'.*_collision':0},
    condim={TIRE_GEOMS[0]:3,'.*_collision':1},
    priority={TIRE_GEOMS[0]:1},friction={TIRE_GEOMS[0]:(1.,)},
)


def get_roller_standup_spec():
    return mujoco.MjSpec.from_file(str(ROLLER_STANDUP_XML))


_SERVO=replace(actuators,json_path=str(_ASSET_DIR/'1910_m6_standup_swizzle.json'),
               kp_fw=5.,delay_min_lag=3,delay_max_lag=6)
XGODUCK_ROLLER_STANDUP_ROBOT_CFG=EntityCfg(
    spec_fn=get_roller_standup_spec,init_state=ROLLER_HOME,
    collisions=(ROLLER_GETUP_COLLISION,),
    articulation=EntityArticulationInfoCfg(actuators=(_SERVO,),soft_joint_pos_limit_factor=.9),
)
