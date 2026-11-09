"""Independent XgoDuck skates: CAD inertia, XgoDuck HLS1910 servos, four passive wheels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from math import radians
import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab_microduck.robot.xgoduck_constants import XGODUCK_HOME_FRAME, actuators

XGODUCK_ROLLER_XML = Path(__file__).parent / 'xgoduck_roller/robot_backlash.xml'
# Lowest tire vertex at the roller-specific HOME pose touches the floor.
XGODUCK_ROLLER_STAND_Z = 0.15481704568257137
XGODUCK_WHEEL_RADIUS = 0.015  # 30 mm CAD tires; upstream MicroDuck uses 35 mm tires.
XGODUCK_ROLLER_HOME_FRAME = deepcopy(XGODUCK_HOME_FRAME)
XGODUCK_ROLLER_HOME_FRAME.pos = (0.0, 0.0, XGODUCK_ROLLER_STAND_Z)
# Roller HOME is independent of the wheel-less Walk posture.
XGODUCK_ROLLER_HOME_FRAME.joint_pos = {
    'left_hip_yaw': radians(0),
    'left_hip_roll': radians(0),
    'left_hip_pitch': radians(-10),
    'left_knee': radians(10),
    'left_ankle': radians(20),
    'neck_pitch': radians(25),
    'head_pitch': radians(25),
    'head_yaw': radians(0),
    'head_roll': radians(0),
    'right_hip_yaw': radians(0),
    'right_hip_roll': radians(0),
    'right_hip_pitch': radians(10),
    'right_knee': radians(-10),
    'right_ankle': radians(-20),
    r'^passive_.*wheel$': 0.0,
    r'^passive_.*_backlash$': 0.0,
}

def get_roller_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(XGODUCK_ROLLER_XML))

# Use the same HLS1910 fit, P and delay as the wheel-less XgoDuck.
_roller_actuators = deepcopy(actuators)

XGODUCK_ROLLER_ROBOT_CFG = EntityCfg(
    spec_fn=get_roller_spec,
    init_state=XGODUCK_ROLLER_HOME_FRAME,
    collisions=(),  # Named wheel/non-wheel masks are explicitly defined in the XML.
    articulation=EntityArticulationInfoCfg(
        actuators=(_roller_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
