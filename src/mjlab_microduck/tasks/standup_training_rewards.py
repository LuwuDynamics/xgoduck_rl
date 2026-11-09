"""Shared tilt gating for XgoDuck StandUp and its roller variant only."""
from __future__ import annotations

import inspect
import math

import torch
from mjlab.managers import CurriculumTermCfg
from . import mdp
from .standup_hip_home import standup_joint_home_l1


def upright_linear_angle_gate(env, asset_name: str = 'robot') -> torch.Tensor:
    """Linear in angle to world +Z: 0deg->1, 45deg->.5, >=90deg->0.

    Read the current free-joint quaternion. Derived xquat/projected gravity
    can lag one physics substep when rewards run, before the final forward.
    This is an angle ramp, not a cosine ramp or the randomized actor IMU.
    """
    data=env.scene[asset_name].data
    q=data.data.qpos[:,data.indexing.free_joint_q_adr[3:7]]
    cosine=1.-2.*(q[:,1].square()+q[:,2].square())
    cosine=torch.nan_to_num(cosine,nan=-1.).clamp(-1.,1.)
    angle=torch.acos(cosine)
    return (1.-angle/(math.pi/2.)).clamp(0.,1.)


def _with_tilt_gate(base_func, public_name, *, minimum_gate: float = 0.):
    """Keep flat parameters/signatures so entity selectors resolve normally.

    Public module aliases also make these wrappers pickle/YAML serializable.
    Existing term-specific height, freeze, and arrival gates remain intact.
    """
    if not 0. <= minimum_gate <= 1.:raise ValueError('minimum_gate must be in [0, 1]')
    def gated(env, **params):
        value=base_func(env,**params)
        asset=params.get('asset_cfg')
        gate=upright_linear_angle_gate(env,asset.name if asset is not None else 'robot')
        gate=minimum_gate+(1.-minimum_gate)*gate
        return torch.where(gate>0.,value*gate,torch.zeros_like(value))
    gated.__name__=public_name
    gated.__qualname__=public_name
    gated.__module__=__name__
    gated.__signature__=inspect.signature(base_func)
    gated.__doc__='Tilt-gated '+(base_func.__doc__ or base_func.__name__)
    gated.ungated_func=base_func
    gated.tilt_gate_minimum=minimum_gate
    return gated


pose_target_match_tilt90=_with_tilt_gate(mdp.pose_target_match,'pose_target_match_tilt90')
pose_l1_penalty_tilt90=_with_tilt_gate(mdp.pose_l1_penalty,'pose_l1_penalty_tilt90')
head_pose_tracking_tilt90=_with_tilt_gate(mdp.head_pose_tracking,'head_pose_tracking_tilt90')
head_pose_bias_tilt90=_with_tilt_gate(mdp.head_pose_bias_penalty,'head_pose_bias_tilt90')
body_pose_tracking_tilt90=_with_tilt_gate(mdp.body_pose_tracking_locomotion,'body_pose_tracking_tilt90')
standing_composite_tilt90=_with_tilt_gate(mdp.standing_composite_score,'standing_composite_tilt90')
hip_home_l1_tilt90=_with_tilt_gate(standup_joint_home_l1,'hip_home_l1_tilt90')
upright_linear_tilt90=_with_tilt_gate(mdp.body_upright_linear,'upright_linear_tilt90')
upright_sharp_tilt90=_with_tilt_gate(mdp.upright_gaussian_at_height,'upright_sharp_tilt90')

POSE_TILT_REWARDS={
    'pose_stand_legs':pose_target_match_tilt90,
    'pose_stand_l1':pose_l1_penalty_tilt90,
    'head_pose_tracking':head_pose_tracking_tilt90,
    'head_pose_bias':head_pose_bias_tilt90,
    'body_pose_tracking':body_pose_tracking_tilt90,
    'standing_composite':standing_composite_tilt90,
    'hip_yaw_home_l1':hip_home_l1_tilt90,
    'hip_roll_home_l1':hip_home_l1_tilt90,
    'upright_linear':upright_linear_tilt90,
    'upright_sharp':upright_sharp_tilt90,
}



# The wheel-less task keeps half of each pose term even beyond 90 degrees.
# Separate public aliases preserve flat selectors and pickle/YAML signatures.
pose_target_match_tilt90_half=_with_tilt_gate(mdp.pose_target_match,'pose_target_match_tilt90_half',minimum_gate=.5)
pose_l1_penalty_tilt90_half=_with_tilt_gate(mdp.pose_l1_penalty,'pose_l1_penalty_tilt90_half',minimum_gate=.5)
head_pose_tracking_tilt90_half=_with_tilt_gate(mdp.head_pose_tracking,'head_pose_tracking_tilt90_half',minimum_gate=.5)
head_pose_bias_tilt90_half=_with_tilt_gate(mdp.head_pose_bias_penalty,'head_pose_bias_tilt90_half',minimum_gate=.5)
body_pose_tracking_tilt90_half=_with_tilt_gate(mdp.body_pose_tracking_locomotion,'body_pose_tracking_tilt90_half',minimum_gate=.5)
standing_composite_tilt90_half=_with_tilt_gate(mdp.standing_composite_score,'standing_composite_tilt90_half',minimum_gate=.5)
hip_home_l1_tilt90_half=_with_tilt_gate(standup_joint_home_l1,'hip_home_l1_tilt90_half',minimum_gate=.5)
upright_linear_tilt90_half=_with_tilt_gate(mdp.body_upright_linear,'upright_linear_tilt90_half',minimum_gate=.5)
upright_sharp_tilt90_half=_with_tilt_gate(mdp.upright_gaussian_at_height,'upright_sharp_tilt90_half',minimum_gate=.5)

POSE_TILT_HALF_REWARDS={
    'pose_stand_legs':pose_target_match_tilt90_half,
    'pose_stand_l1':pose_l1_penalty_tilt90_half,
    'head_pose_tracking':head_pose_tracking_tilt90_half,
    'head_pose_bias':head_pose_bias_tilt90_half,
    'body_pose_tracking':body_pose_tracking_tilt90_half,
    'standing_composite':standing_composite_tilt90_half,
    'hip_yaw_home_l1':hip_home_l1_tilt90_half,
    'hip_roll_home_l1':hip_home_l1_tilt90_half,
    'upright_linear':upright_linear_tilt90_half,
    'upright_sharp':upright_sharp_tilt90_half,
}

def configure_pose_tilt_gate(cfg, *, minimum_gate: float = 0.):
    """Select one pose-gate profile without stacking wrappers."""
    profiles={0.:POSE_TILT_REWARDS,.5:POSE_TILT_HALF_REWARDS}
    if minimum_gate not in profiles:raise ValueError('Unsupported pose tilt profile')
    for name,func in profiles[minimum_gate].items():
        term=cfg.rewards.get(name)
        if term is not None:
            assert getattr(term.func,'ungated_func',term.func) is func.ungated_func,name
            term.func=func
    return cfg


def linear_penalty_weight(env,env_ids,reward_name:str,start_weight:float,
                          final_weight:float,ramp_steps:int=5000*24):
    """Interpolate a live RewardManager weight over 5000 PPO updates (24 steps).

    Baselines are explicit arguments; no absent/zero penalty is silently
    interpreted as a different physical quantity.
    """
    del env_ids
    if ramp_steps<=0:raise ValueError('ramp_steps must be positive')
    fraction=max(0.,min(1.,float(env.common_step_counter)/ramp_steps))
    value=start_weight+(final_weight-start_weight)*fraction
    env.reward_manager.get_term_cfg(reward_name).weight=value
    return torch.tensor([value])


def configure_standup_penalty_course(cfg):
    """Only action-rate cost changes: -.1 -> -1.5 over 5000*24 steps."""
    cfg.rewards['action_rate_l2'].weight=-.1
    cfg.curriculum['action_rate_weight']=CurriculumTermCfg(
        func=linear_penalty_weight,params={
            'reward_name':'action_rate_l2','start_weight':-.1,
            'final_weight':-1.5,'ramp_steps':5000*24,
        },
    )
    return cfg
