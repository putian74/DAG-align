"""Smoke tests for the AD-PHMM-align scaffold."""


def test_public_imports() -> None:
    import ad_phmm_align

    assert ad_phmm_align.__version__
    assert ad_phmm_align.InitializationTrack.LEGACY_CURRENT.value == "legacy_current"
    assert ad_phmm_align.SourceFormat.DAG_ALIGN_LEGACY.value == "dag_align_legacy"
    assert ad_phmm_align.StateIntervalSemantics.HALF_OPEN.value == "half_open"


def test_half_open_state_interval_validation() -> None:
    import numpy as np

    from ad_phmm_align.graph import validate_state_intervals

    validate_state_intervals(
        np.array([0, 2], dtype=np.int64),
        np.array([2, 5], dtype=np.int64),
        global_state_count=5,
    )


def test_packed_window_contract() -> None:
    import numpy as np

    from ad_phmm_align.graph import PackedStateWindows

    windows = PackedStateWindows(
        left=np.array([0, 2], dtype=np.int64),
        right=np.array([2, 5], dtype=np.int64),
        offset=np.array([0, 2], dtype=np.int64),
        length=np.array([2, 3], dtype=np.int64),
    )
    windows.validate(global_state_count=5)


def test_edge_overlap_contract_checks_window_bounds() -> None:
    import numpy as np

    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows

    windows = PackedStateWindows(
        left=np.array([0, 2], dtype=np.int64),
        right=np.array([2, 5], dtype=np.int64),
        offset=np.array([0, 2], dtype=np.int64),
        length=np.array([2, 3], dtype=np.int64),
    )
    overlaps = EdgeWindowOverlaps(
        edge_ids=np.array([0], dtype=np.int64),
        src_offset=np.array([0], dtype=np.int64),
        dst_offset=np.array([1], dtype=np.int64),
        length=np.array([2], dtype=np.int64),
    )
    overlaps.validate_against_windows(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        windows,
    )


def test_training_contract_imports() -> None:
    from ad_phmm_align.train import (
        LossWeights,
        PreparedTrainingBatch,
        ProfilingConfig,
        ProfilingResult,
        TrainingStepInput,
        TrainingStepResult,
    )

    assert LossWeights().negative_log_likelihood == 1.0
    assert ProfilingConfig().enabled is True
    assert ProfilingResult(wall_seconds=0.0).wall_seconds == 0.0
    assert PreparedTrainingBatch
    assert TrainingStepInput
    assert TrainingStepResult
