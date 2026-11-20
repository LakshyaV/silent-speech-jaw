"""Signal processing: calibration, streaming front-end, activity segmentation."""

from .activity import Segment, Segmenter, SegmenterConfig, segment_activity, session_health, trim_to_activity
from .calibration import Calibration, calibrate, calibrate_session
from .frontend import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    N_FEATURES,
    Frontend,
    FrontendConfig,
    FrontendOutput,
    frames_for_interval,
    process_session,
)
from .resample import DualResampler, StreamResampler, TimestampSmoother

__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "N_FEATURES",
    "Calibration",
    "DualResampler",
    "Frontend",
    "FrontendConfig",
    "FrontendOutput",
    "Segment",
    "Segmenter",
    "SegmenterConfig",
    "StreamResampler",
    "TimestampSmoother",
    "calibrate",
    "calibrate_session",
    "frames_for_interval",
    "process_session",
    "segment_activity",
    "session_health",
    "trim_to_activity",
]
