"""Articulatory kinematic simulator: text -> jaw motion -> dual-IMU frames."""

from .articulation import PHONE_SPECS, Trajectory, synthesize
from .imu import (
    ScriptItem,
    SessionParams,
    SimSession,
    calibration_script,
    random_session_params,
    render_session,
    simulate_session,
    utterance_item,
)

__all__ = [
    "PHONE_SPECS",
    "ScriptItem",
    "SessionParams",
    "SimSession",
    "Trajectory",
    "calibration_script",
    "random_session_params",
    "render_session",
    "simulate_session",
    "synthesize",
    "utterance_item",
]
