from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tade_ef.config import load_config
from tade_ef.features import extract_track_features, split_track
from tade_ef.replay import SequentialReplay, stream_event_windows
from tade_ef.schema import FEATURE_NAMES
from tade_ef.types import Box, Candidate, EventWindow, Track


class Model:
    def __init__(self):
        self.calls = []

    def predict_proba(self, x):
        self.calls.append(x.copy())
        return np.tile([0.1, 0.9], (len(x), 1))


def config():
    cfg = load_config(Path(__file__).parents[1] / "configs/paper.yaml", fold="fold_1")
    return replace(cfg, sensor_width=200, sensor_height=100)


def windows(count=80):
    for i in range(count):
        end = (i + 1) * 20000
        x = np.array([i + 10., i + 11., i + 12.])
        yield EventWindow(end - 20000, end, x, np.array([20., 21., 20.]),
                          np.array([end - 15000, end - 10000, end - 5000]),
                          np.array([1, -1, 1]), 200, 100)


def detector(window, _):
    x = int(window.x[0])
    return [Candidate(window.end_us, Box(x - 1, 19, x + 4, 23), 3)]


def test_prefix_equivalence_bounded_history_and_no_backfill():
    model = Model()
    cfg = config()
    replay = SequentialReplay(cfg, model, recording_id="test", candidate_detector=detector)
    output = list(replay.run(windows()))
    assert all(not item.updates for item in output[:15])
    assert output[15].updates[0]["track_age_ms"] == 300
    assert output[15].boxes[0]["identity"] == "drone"
    assert all(not item.boxes[0]["has_evidence"] for item in output[:15])
    assert all(len(call) == 1 for call in model.calls)
    reference_replay = SequentialReplay(cfg, Model(), recording_id="test", candidate_detector=detector)
    all_observations = []
    for window in windows():
        reference_replay.manager.update(window, detector(window, None))
        all_observations.append(reference_replay.manager.active[0].last)
    segments = split_track(Track(1, all_observations), min_duration_ms=300,
                           max_duration_ms=1000, update_interval_ms=100)
    expected = extract_track_features(segments, recording_id="test", track_id=1,
                                      config=cfg.features)
    actual = [feature for result in output for feature in result.features]
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal([a.values[k] for k in FEATURE_NAMES],
                                      [b.values[k] for k in FEATURE_NAMES])
    assert len(replay.manager.active[0].observations) <= 51
    assert len(replay.states[1].cache.observations) <= 51
    assert not replay.manager.finished


def test_pacing_measures_queue_lag_without_dropping():
    now = [0.0]
    class SlowModel(Model):
        def predict_proba(self, x):
            now[0] += 0.08
            return super().predict_proba(x)
    cfg = config()
    replay = SequentialReplay(cfg, SlowModel(), recording_id="test", candidate_detector=detector)
    result = list(replay.run(windows(18), paced=True, clock=lambda: now[0],
                            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    assert len(result) == 18
    assert result[15].completion_lag_ms == pytest.approx(80)
    assert result[16].waiting_ms == pytest.approx(60)
    assert result[17].waiting_ms == pytest.approx(40)


def test_missing_windows_retire_state():
    replay = SequentialReplay(config(), Model(), recording_id="test", candidate_detector=detector)
    replay.process(next(windows(1)))
    replay.detector = lambda *_: []
    for window in list(windows(11))[1:]:
        replay.process(window)
    assert not replay.states
    assert not replay.manager.active
    assert not replay.manager.finished


def test_lazy_input_and_timestamp_validation():
    x = np.array([1, 2, 3])
    source = stream_event_windows(x, x, np.array([0, 19999, 20000]), x,
                                 width=10, height=10, window_us=20000)
    result = list(source)
    assert [len(item.x) for item in result] == [2, 1]
    with pytest.raises(ValueError):
        list(stream_event_windows(x, x, np.array([1, 0, 2]), x,
                                  width=10, height=10, window_us=20000))


def test_future_suffix_cannot_change_prefix():
    a = SequentialReplay(config(), Model(), recording_id="test", candidate_detector=detector)
    b = SequentialReplay(config(), Model(), recording_id="test", candidate_detector=detector)
    short = list(a.run(windows(20)))
    long = list(b.run(windows(25)))
    assert [x.boxes for x in short] == [x.boxes for x in long[:20]]
    with pytest.raises(ValueError):
        a.process(next(windows(1)))


def test_motion_eligibility_does_not_use_future_movement():
    def late_motion(window, _):
        x = 10 if window.end_us <= 400000 else 13
        return [Candidate(window.end_us, Box(x, 20, x + 5, 24), 3)]
    replay = SequentialReplay(config(), Model(), recording_id="test", candidate_detector=late_motion)
    result = list(replay.run(windows(22)))
    assert not result[15].updates
    assert result[20].updates[0]["track_age_ms"] == 400


def test_empty_input_and_invalid_classifier_output():
    empty = np.array([], dtype=int)
    assert list(stream_event_windows(empty, empty, empty, empty,
                                     width=200, height=100, window_us=20000)) == []
    with pytest.raises(ValueError):
        list(stream_event_windows(empty, empty, empty, empty,
                                  width=200, height=100, window_us=0))
    class InvalidModel:
        def predict_proba(self, matrix):
            return np.zeros((len(matrix), 3))
    replay = SequentialReplay(config(), InvalidModel(), recording_id="test", candidate_detector=detector)
    with pytest.raises(ValueError, match="two-class"):
        list(replay.run(windows(17)))
    with pytest.raises(ValueError, match="dimensions"):
        replay.process(replace(next(windows(1)), width=100))
