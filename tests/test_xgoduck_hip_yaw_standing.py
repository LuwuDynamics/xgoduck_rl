"""Regression tests for head/hip name confusion and low-speed hip yaw drift."""
from types import SimpleNamespace
import re

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.tasks import _xgoduck_velocity_cfg, mdp


@pytest.mark.parametrize('kwargs', [{}, {'play': True}, {'rough': True},
                                  {'rough': True, 'play': True}])
def test_head_com_selection_resolves_only_head_bodies(kwargs):
    cfg = _xgoduck_velocity_cfg(**kwargs)
    names = cfg.events['randomize_head_com'].params['asset_cfg'].body_names
    model = cfg.scene.entities['robot'].spec_fn().compile()
    selected = {model.body(i).name for i in range(model.nbody)
                if any(re.fullmatch(pattern, model.body(i).name) for pattern in names)}
    assert selected == {'neck', 'neck_pitch', 'yaw_roll_motion', 'jaw_soft'}
    assert 'bearing_roll' not in selected
    term = cfg.rewards['hip_yaw_home']
    assert term.params['asset_cfg'].joint_names == ('left_hip_yaw', 'right_hip_yaw')
    assert term.weight < 0


@pytest.mark.parametrize('command,enabled', [
    ((0.0, 0.0, 0.0), True),
    ((0.029, 0.0, 0.0), True),
    ((-0.029, 0.0, 0.0), True),
    ((0.0, 0.029, 0.0), True),
    ((0.021, 0.021, 0.0), True),
    ((0.03, 0.0, 0.0), False),
    ((0.022, 0.022, 0.0), False),
    ((0.2, 0.0, 0.0), False),
    ((0.0, 0.0, 0.5), False),
    ((0.0, 0.0, -0.5), False),
    ((0.02, 0.0, 0.2), False),
])
def test_hip_cost_uses_command_speed_and_preserves_turns(command, enabled):
    # Two hips are deliberately separated by a head joint in this fixture.
    data = SimpleNamespace(
        joint_pos=torch.tensor([[0.2, 2.0, 0.6]]),
        default_joint_pos=torch.tensor([[-0.1, 0.0, 0.1]]),
        joint_vel=torch.zeros(1, 3),
    )
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.tensor([command], dtype=torch.float32)),
    )
    asset = SceneEntityCfg('robot', joint_ids=[0, 2])
    value = mdp.joint_pos_l1_when_standing(env, asset_cfg=asset)
    assert value.item() == pytest.approx(0.8 if enabled else 0.0)
    # A static, biased hip is still penalized; head yaw alone is excluded.
    data.joint_pos[:, [0, 2]] = data.default_joint_pos[:, [0, 2]]
    assert mdp.joint_pos_l1_when_standing(env, asset_cfg=asset).item() == 0.0


def test_backward_bucket_stays_straight_and_preserves_standing(monkeypatch):
    from dataclasses import replace

    cfg = _xgoduck_velocity_cfg().commands['twist']
    assert cfg.rel_backward_envs == 0.15
    assert cfg.backward_speed_range == (-0.2, -0.1)
    term = mdp.VelocityCommandCommandOnly.__new__(mdp.VelocityCommandCommandOnly)
    term.cfg = replace(cfg, rel_turn_in_place_envs=0.0)
    count, standing_count = 4096, 1024
    term._env = SimpleNamespace(device='cpu', num_envs=count)
    term.vel_command_b = torch.full((count, 3), 0.3)
    term.vel_command_b[:standing_count] = 0.0
    term.vel_command_w = term.vel_command_b.clone()
    term.is_standing_env = torch.arange(count) < standing_count
    term.is_heading_env = torch.ones(count, dtype=torch.bool)
    term.is_world_env = torch.ones(count, dtype=torch.bool)
    term.is_forward_env = torch.ones(count, dtype=torch.bool)
    monkeypatch.setattr(mdp.UniformVelocityCommand, '_resample_command',
                        lambda self, env_ids: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        term._resample_command(torch.arange(count))
    assert torch.count_nonzero(term.vel_command_b[:standing_count]) == 0
    backward = term.vel_command_b[:, 0] < 0
    fraction = backward.sum().item() / (count - standing_count)
    assert 0.12 < fraction < 0.18
    selected = term.vel_command_b[backward]
    assert torch.all((selected[:, 0] >= -0.2) & (selected[:, 0] <= -0.1))
    assert torch.count_nonzero(selected[:, 1:]) == 0
    assert not term.is_heading_env[backward].any()
    assert not term.is_world_env[backward].any()
    assert not term.is_forward_env[backward].any()
    torch.testing.assert_close(term.vel_command_w[backward], selected)
