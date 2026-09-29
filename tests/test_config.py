from pathlib import Path

from tade_ef.config import load_config


def test_paper_configuration_loads() -> None:
    config = load_config(Path("configs/paper.yaml"))
    assert config.window_ms == 20.0
    assert config.evidence.drone_threshold == 0.42
    assert config.evidence.classification_threshold == 0.5
    assert config.segmentation.min_duration_ms == 300.0
    assert config.segmentation.update_interval_ms == 100.0

