from tade_ef.schema import FEATURE_NAMES


def test_paper_schema_has_exactly_36_unique_features() -> None:
    assert len(FEATURE_NAMES) == 36
    assert len(set(FEATURE_NAMES)) == 36


def test_paper_schema_excludes_legacy_metadata_and_substitutions() -> None:
    excluded = {
        "focal_length",
        "distance",
        "observation_count",
        "acceleration_max_px_per_s2_max",
    }
    assert excluded.isdisjoint(FEATURE_NAMES)
    assert "local_speed_mean_window_mean" in FEATURE_NAMES
    assert "local_acceleration_std_window_mean" in FEATURE_NAMES

