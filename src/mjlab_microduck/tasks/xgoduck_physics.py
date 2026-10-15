"""Shared CAD inertials and servo timing for XgoDuck tasks."""


def configure_xgoduck_physics(cfg):
    """Keep CAD mass/inertia fixed and apply a 4 ms actuator delay at 50 Hz.

    BAM's JSON delay is identification metadata in mjlab. The actuator uses
    delay_min_lag=delay_max_lag=1, so a 4 ms physics step makes the effective
    command delay match 1910_m6.json without interpolation or extra filtering.
    Five physics steps retain the existing 20 ms policy/control period.
    """
    cfg.events.pop("randomize_mass_inertia", None)
    cfg.sim.mujoco.timestep = 0.004
    cfg.decimation = 5
    return cfg
