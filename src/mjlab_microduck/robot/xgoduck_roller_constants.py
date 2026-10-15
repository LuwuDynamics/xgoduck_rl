"""XgoDuck roller geometry with the shared HLS1910 actuator model."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .xgoduck_constants import XGODUCK_HOME_FRAME, actuators


XGODUCK_ROLLER_XML = Path(__file__).parent / "xgoduck_roller" / "robot.xml"
XGODUCK_ROLLER_STAND_Z = 0.1638355460
XGODUCK_WHEEL_RADIUS = 0.015

# Preserve the walk model's 22-degree leg pose and four head HOME angles.
# The extra height leaves 1 mm clearance beneath the CAD tires.
XGODUCK_ROLLER_HOME_FRAME = deepcopy(XGODUCK_HOME_FRAME)
XGODUCK_ROLLER_HOME_FRAME.pos = (0.0, 0.0, XGODUCK_ROLLER_STAND_Z)
XGODUCK_ROLLER_HOME_FRAME.joint_pos[r"^passive_.*wheel$"] = 0.0


def get_roller_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(XGODUCK_ROLLER_XML))


XGODUCK_ROLLER_ROBOT_CFG = EntityCfg(
    spec_fn=get_roller_spec,
    init_state=XGODUCK_ROLLER_HOME_FRAME,
    # The MJCF defines separate tire, floor-contact and self-contact masks.
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(deepcopy(actuators),),
        soft_joint_pos_limit_factor=0.9,
    ),
)
