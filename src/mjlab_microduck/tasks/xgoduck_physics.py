"""Shared physics timing and mass-randomization policy for XgoDuck tasks."""


def configure_xgoduck_physics(cfg, *, randomize_mass_inertia: bool = False):
    """Use 4 ms physics at 50 Hz and optionally retain inherited mass/inertia DR.

    Command delay is selected by each robot's actuator buffer, independently
    of the identification metadata stored in the BAM JSON.
    """
    if not randomize_mass_inertia:
        cfg.events.pop("randomize_mass_inertia", None)
    cfg.sim.mujoco.timestep = 0.004
    cfg.decimation = 5
    return cfg
