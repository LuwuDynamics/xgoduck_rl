"""Ground-contact failure and a fixed terminal cost for XgoDuck kicks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def non_foot_ground_contact(
    env: ManagerBasedRlEnv, sensor_name: str = "non_foot_ground_contact",
) -> torch.Tensor:
    """Detect any ground touch since reset, including a released substep touch.

    ContactSensor updates its contact clocks every physics substep. A positive
    last_contact_time preserves a brief touch that ends before the policy step;
    both clocks are cleared by the sensor's normal per-environment reset.
    No force threshold: even a light contact counts.
    """
    data = env.scene.sensors[sensor_name].data
    touched = ((data.found > 0) | (data.current_contact_time > 0)
               | (data.last_contact_time > 0))
    return touched.flatten(start_dim=1).any(dim=1)


def failure_event_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One failure event, cancelling RewardManager's default dt scaling.

    The configured negative weight is charged on the terminal step, once even
    if multiple failure conditions coincide. Pure timeouts cost nothing.
    Include the existing tilt failure so it cannot bypass the contact cost.
    """
    return env.termination_manager.terminated.float() / env.step_dt
