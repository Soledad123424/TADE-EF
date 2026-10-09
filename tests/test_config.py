from pathlib import Path

from tade_ef.config import load_config


def test_paper_configuration_loads() -> None:
    config = load_config(Path("configs/paper.yaml"))
    assert config.window_ms == 20.0
    assert config.evidence.drone_threshold == 0.42
    assert config.evidence.classification_threshold == 0.5
    assert config.evidence.non_drone_threshold == 0.1
    assert config.evidence.non_drone_lock_ms == 500.0
    assert config.evidence.drone_threshold_by_fold == {"fold_1": 0.42, "fold_2": 0.45, "fold_3": 0.36}
    assert config.segmentation.min_duration_ms == 300.0
    assert config.segmentation.update_interval_ms == 100.0
    assert config.features.alignment_history_ms == 1000.0
    assert config.features.spectral_epsilon == 1.0e-12


def test_fold_specific_uav_thresholds() -> None:
    for fold, threshold in [(1, 0.42), (2, 0.45), (3, 0.36)]:
        config = load_config(Path("configs/paper.yaml"), fold=fold)
        assert config.evidence.drone_threshold == threshold
        assert not config.evidence.drone_threshold_by_fold

