"""The immutable 36-dimensional classifier feature contract."""

FREQUENCY_FEATURES = (
    "alignment_gain_mean",
    "alignment_gain_max",
    "peak_frequency_mean_hz",
    "peak_frequency_std_hz",
    "peak_frequency_cv",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "harmonic_score_mean",
    "harmonic_score_max",
    "spatial_phase_concentration_mean",
    "spatial_phase_concentration_max",
)

MORPHOLOGY_FEATURES = (
    "elongation_mean",
    "elongation_std",
    "elongation_cv_mean",
    "elongation_cv_max",
    "box_area_cv_mean",
    "box_area_cv_max",
    "box_aspect_cv_mean",
    "box_aspect_cv_max",
)

MOTION_FEATURES = (
    "path_length_px",
    "endpoint_displacement_px",
    "straightness",
    "speed_mean_px_per_s",
    "speed_max_px_per_s",
    "local_speed_mean_window_mean",
    "local_speed_std_window_mean",
    "local_acceleration_mean_window_mean",
    "local_acceleration_std_window_mean",
    "local_jerk_mean_window_mean",
    "local_curvature_mean_window_mean",
)

QUALITY_FEATURES = (
    "duration_ms",
    "roi_event_count_mean",
    "box_area_mean_px2",
    "box_area_std_px2",
    "box_aspect_mean",
    "box_aspect_std",
)

FEATURE_NAMES = (
    *FREQUENCY_FEATURES,
    *MORPHOLOGY_FEATURES,
    *MOTION_FEATURES,
    *QUALITY_FEATURES,
)

if len(FEATURE_NAMES) != 36 or len(set(FEATURE_NAMES)) != 36:
    raise RuntimeError("TADE-EF requires exactly 36 unique features")

