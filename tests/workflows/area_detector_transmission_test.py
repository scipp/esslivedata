# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.frame_batch import FrameBatchAccumulator
from ess.livedata.workflows.area_detector_transmission import (
    ImageKeyLog,
    transmission_factory,
)

SOURCE = 'detector'
KEY_STREAM = 'image_key'


def image_key_log(*changes: tuple[int, int]) -> sc.DataArray:
    """Key log from ``(time_ns, key)`` pairs."""
    times, keys = zip(*changes, strict=True)
    return sc.DataArray(
        sc.array(dims=['time'], values=list(keys), unit='dimensionless'),
        coords={
            'time': sc.epoch(unit='ns')
            + sc.array(dims=['time'], values=list(times), unit='ns', dtype='int64')
        },
    )


class Run:
    """Drives the workflow like the job layer: frames in, finalize out."""

    def __init__(self, factory=None) -> None:
        self.workflow = (factory or transmission_factory())(SOURCE)
        self.workflow.build(context_keys={KEY_STREAM: ImageKeyLog})
        self.preprocessor = FrameBatchAccumulator()

    def push(
        self, key_log: sc.DataArray, frames: list[tuple[int, float]]
    ) -> dict[str, sc.DataArray]:
        """Process one batch of ``(time_ns, pixel value)`` frames."""
        for time_ns, value in frames:
            self.preprocessor.add(
                Timestamp.from_ns(time_ns),
                sc.DataArray(
                    sc.full(dims=['x', 'y'], shape=[2, 2], value=value, unit='counts')
                ),
            )
        self.workflow.accumulate(
            {SOURCE: self.preprocessor.get(), KEY_STREAM: key_log},
            start_time=Timestamp.from_ns(0),
            end_time=Timestamp.from_ns(1),
        )
        return self.workflow.finalize()


@pytest.fixture
def run() -> Run:
    return Run()


def test_transmission_is_formed_from_mean_frames_of_each_type(run: Run) -> None:
    log = image_key_log((0, 2), (100, 1), (200, 0))
    # dark mean 1, open beam mean 5, sample mean 3 -> (3 - 1) / (5 - 1)
    run.push(log, [(10, 0.0), (20, 2.0)])
    run.push(log, [(110, 4.0), (120, 6.0)])
    result = run.push(log, [(210, 3.0)])

    np.testing.assert_allclose(result['transmission'].values, 0.5)
    assert result['transmission'].unit == 'dimensionless'


def test_frames_of_different_types_in_one_batch_are_not_mixed(run: Run) -> None:
    log = image_key_log((0, 2), (100, 1), (200, 0))
    result = run.push(log, [(10, 1.0), (110, 5.0), (210, 3.0)])

    np.testing.assert_allclose(result['dark'].values, 1.0)
    np.testing.assert_allclose(result['open_beam'].values, 5.0)
    np.testing.assert_allclose(result['sample'].values, 3.0)
    np.testing.assert_allclose(result['transmission'].values, 0.5)


def test_images_are_means_not_sums(run: Run) -> None:
    log = image_key_log((0, 0))
    result = run.push(log, [(10, 2.0), (20, 4.0)])

    np.testing.assert_allclose(result['sample'].values, 3.0)


def test_transmission_is_unavailable_until_all_types_seen(run: Run) -> None:
    log = image_key_log((0, 0), (100, 1))
    run.push(log, [(10, 3.0)])
    result = run.push(log, [(110, 5.0)])

    assert 'transmission' not in result
    assert 'dark' not in result


def test_transmission_is_nan_where_open_beam_does_not_exceed_dark(run: Run) -> None:
    log = image_key_log((0, 2), (100, 1), (200, 0))
    result = run.push(log, [(10, 5.0), (110, 5.0), (210, 3.0)])

    assert np.isnan(result['transmission'].values).all()


def test_only_outputs_affected_by_the_batch_are_published(run: Run) -> None:
    log = image_key_log((0, 2), (100, 1), (200, 0))
    run.push(log, [(10, 1.0), (110, 5.0), (210, 3.0)])
    result = run.push(log, [(220, 4.0)])

    assert set(result) == {'sample', 'transmission'}


@pytest.mark.parametrize('key', [3, 4, 7])
def test_frames_with_other_keys_are_dropped_and_counted(run: Run, key: int) -> None:
    log = image_key_log((0, key))
    result = run.push(log, [(10, 1.0), (20, 1.0)])

    assert set(result) == {'dropped_frames'}
    assert result['dropped_frames'].value == 2


def test_frames_before_first_key_are_dropped_not_taken_as_sample(run: Run) -> None:
    log = image_key_log((100, 0))
    result = run.push(log, [(10, 1.0), (110, 3.0)])

    assert result['dropped_frames'].value == 1
    np.testing.assert_allclose(result['sample'].values, 3.0)


def test_float_valued_key_log_is_accepted(run: Run) -> None:
    log = image_key_log((0, 0))
    log.data = log.data.to(dtype='float64')
    result = run.push(log, [(10, 2.0)])

    np.testing.assert_allclose(result['sample'].values, 2.0)


def test_clear_discards_accumulated_frames(run: Run) -> None:
    log = image_key_log((0, 0))
    run.push(log, [(10, 2.0)])
    run.workflow.clear()
    result = run.push(log, [(20, 8.0)])

    np.testing.assert_allclose(result['sample'].values, 8.0)


def test_transform_is_applied_to_every_frame() -> None:
    def double(da: sc.DataArray, source_name: str) -> sc.DataArray:
        assert source_name == SOURCE
        return da * 2

    run = Run(transmission_factory(transform=double))
    result = run.push(image_key_log((0, 0)), [(10, 2.0)])

    np.testing.assert_allclose(result['sample'].values, 4.0)


def test_frame_at_the_time_of_a_key_change_takes_the_new_key(run: Run) -> None:
    log = image_key_log((0, 2), (100, 0))
    result = run.push(log, [(99, 1.0), (100, 3.0)])

    np.testing.assert_allclose(result['dark'].values, 1.0)
    np.testing.assert_allclose(result['sample'].values, 3.0)


def test_batch_of_dropped_frames_keeps_accumulated_images(run: Run) -> None:
    log = image_key_log((0, 0), (100, 4))
    run.push(log, [(10, 3.0)])
    run.push(log, [(110, 9.0)])
    run.push(log, [(120, 9.0)])
    result = run.push(log, [(130, 9.0)])

    assert result['dropped_frames'].value == 3
    assert 'sample' not in result
