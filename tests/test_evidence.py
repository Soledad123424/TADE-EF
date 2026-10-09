from tade_ef.evidence import EvidenceAccumulator, EvidenceConfig


def test_revised_non_uav_threshold_and_500ms_lock() -> None:
    accumulator = EvidenceAccumulator(EvidenceConfig())
    assert accumulator.update(0.5, 0).identity == "undecided"
    assert accumulator.update(0.5, 499999).identity == "undecided"
    record = accumulator.update(0.5, 500000)
    assert record.accumulated == 0.0
    assert record.identity == "non_drone" and record.locked_now


def test_uav_evidence_locks_and_never_changes() -> None:
    accumulator = EvidenceAccumulator(EvidenceConfig(drone_threshold=0.2))
    first = accumulator.update(0.9, 0)
    assert first.identity == "drone"
    second = accumulator.update(0.01, 1_000_000)
    assert second.identity == "drone"
    assert second.accumulated != first.accumulated


def test_non_uav_requires_low_evidence_duration() -> None:
    accumulator = EvidenceAccumulator(
        EvidenceConfig(non_drone_threshold=-0.2, non_drone_lock_ms=100.0)
    )
    assert accumulator.update(0.1, 0).identity == "undecided"
    result = accumulator.update(0.1, 100_000)
    assert result.identity == "non_drone"

