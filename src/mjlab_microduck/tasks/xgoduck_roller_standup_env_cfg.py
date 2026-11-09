"""Current XgoDuck StandUp recipe on the latest four-passive-wheel skates.

Keep recovery rewards, command/observation slots, 2-second HOME hold, and
8-second episodes. Only model/contact/height and roller dynamics are retargeted.
The actor still controls 14 servos, never the four passive wheels.
"""
from copy import deepcopy

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.robot.xgoduck_constants import XGODUCK_STAND_Z
from mjlab_microduck.robot.xgoduck_roller_standup_constants import (
    XGODUCK_ROLLER_STANDUP_ROBOT_CFG,ROLLER_STAND_Z,WHEEL_JOINTS,TIRE_GEOMS,
)


def make_xgoduck_roller_standup_env_cfg(play:bool=False):
    # Deferred import: this factory is registered after the original StandUp.
    from . import _xgoduck_standup_cfg
    cfg=deepcopy(_xgoduck_standup_cfg(play=play))
    # Keep the roller recovery recipe independent of wheel-less refinements.
    from .standup_training_rewards import configure_pose_tilt_gate
    cfg=configure_pose_tilt_gate(cfg,minimum_gate=0.)
    for axis in ('yaw','roll'):
        cfg.rewards[f'hip_{axis}_home_l1'].weight=-2.
    cfg.scene.entities={'robot':deepcopy(XGODUCK_ROLLER_STANDUP_ROBOT_CFG)}
    # Match the latest Swizzle control period and physical actuator delay.
    cfg.sim.mujoco.timestep=.004
    cfg.decimation=5
    # CAD inertias are explicit in the roller XML. Retain position-only CoM DR.
    cfg.events.pop('randomize_mass_inertia',None)

    dz=ROLLER_STAND_Z-XGODUCK_STAND_Z
    for name in ('height_stand','height_stand_sharp','height_stand_l1','standing_composite'):
        cfg.rewards[name].params['target_height']=ROLLER_STAND_Z
    cfg.rewards['com_upward_velocity'].params['max_height']=ROLLER_STAND_Z+.010
    for name in ('upright_sharp','arrival_damping','hip_yaw_home_l1','hip_roll_home_l1'):
        for key in ('height_low','height_high'):
            cfg.rewards[name].params[key]+=dz
    for key in ('gate_height_low','gate_height_high'):
        cfg.rewards['head_pose_bias'].params[key]+=dz
    cfg.rewards['body_pose_tracking'].params['nominal_height']=ROLLER_STAND_Z
    drop=cfg.events['set_ground_state'].params
    drop.update(drop_z_min=ROLLER_STAND_Z+.095,drop_z_max=ROLLER_STAND_Z+.105)

    # Preserve two aggregate foot entries in the critic; each skate includes
    # its ankle plate and both child tires, rather than missing sole geoms.
    sensor=next(s for s in cfg.scene.sensors if s.name=='feet_ground_contact')
    sensor.primary.mode='subtree'
    sensor.primary.pattern=r'^(ankle_left|ankle_right)$'
    # Tire traction has its own range, distinct from the wheel-less sole DR.
    friction=cfg.events.pop('foot_friction')
    friction.params.update(asset_cfg=SceneEntityCfg('robot',geom_names=TIRE_GEOMS),
        ranges=(1.,1.) if play else (.4,2.),operation='abs',axes=[0],shared_random=True)
    cfg.events['randomize_tire_sliding_friction']=friction
    cfg.events['randomize_wheel_friction']=EventTermCfg(
        func=dr.dof_frictionloss,mode='reset',params={
            'asset_cfg':SceneEntityCfg('robot',joint_names=WHEEL_JOINTS),
            'operation':'abs','ranges':(.001,.001) if play else (.0008,.0012),
            'shared_random':False,
        },
    )
    # Physics configuration inherited from StandUp already resets passive
    # joints separately and excludes every passive_* joint from servo slots.
    cfg.events['reset_passive_joints'].params['asset_cfg']=SceneEntityCfg('robot',joint_names=WHEEL_JOINTS)
    return cfg
