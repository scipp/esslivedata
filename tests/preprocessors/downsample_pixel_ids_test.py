# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from structlog.testing import capture_logs

from ess.livedata.config.instrument import DetectorDownsampling
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.downsample_pixel_ids import DownsamplePixelIds
from ess.livedata.preprocessors.group_by_pixel import GroupByPixel
from ess.livedata.preprocessors.to_nxevent_data import DetectorEvents, ToNXevent_data


class RecordingAccumulator:
    """Captures the events handed to the wrapped accumulator."""

    def __init__(self) -> None:
        self.batches: list[DetectorEvents] = []

    def add(self, timestamp: Timestamp, data: DetectorEvents) -> bool:
        self.batches.append(data)
        return True

    def get(self) -> list[DetectorEvents]:
        return self.batches

    def clear(self) -> None:
        self.batches.clear()


def events(pixel_id: list[int]) -> DetectorEvents:
    return DetectorEvents(
        time_of_arrival=np.arange(len(pixel_id), dtype='int64'),
        unit='ns',
        pixel_id=np.array(pixel_id, dtype='int64'),
    )


def ts(seconds: float = 0.0) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def downsampling(
    resolution: int, *, first_id: int = 0, declared: int | None = None
) -> DetectorDownsampling:
    grid = sc.arange('detector_number', resolution * resolution, unit=None).fold(
        dim='detector_number', sizes={'dim_0': resolution, 'dim_1': resolution}
    )
    return DetectorDownsampling(resolution, first_id, declared, grid)


class TestRemapping:
    def test_maps_each_block_of_source_pixels_to_one_target_pixel(self) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2))
        # source 4x4 -> target 2x2, block 2. id = x * 4 + y.
        # (x, y) = (0,0) (0,1) (1,0) (1,1) all fall in target pixel (0, 0) = 0.
        # (0,2) -> target (0, 1) = 1;  (2,0) -> target (1, 0) = 2;
        # (3,3) -> target (1, 1) = 3.
        acc.add(ts(), events([0, 1, 4, 5, 2, 8, 15]))
        assert acc.source_resolution == 4
        np.testing.assert_array_equal(inner.batches[0].pixel_id, [0, 0, 0, 0, 1, 2, 3])

    def test_leaves_time_of_arrival_untouched(self) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2))
        original = events([0, 15])
        acc.add(ts(), original)
        np.testing.assert_array_equal(
            inner.batches[0].time_of_arrival, original.time_of_arrival
        )
        assert inner.batches[0].unit == original.unit

    def test_does_not_mutate_incoming_events(self) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2))
        original = events([0, 15])
        acc.add(ts(), original)
        np.testing.assert_array_equal(original.pixel_id, [0, 15])


class TestSourceResolutionInference:
    def test_infers_source_resolution_from_largest_observed_id(self) -> None:
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512))
        acc.add(ts(), events([4096 * 4096 - 1]))
        assert acc.source_resolution == 4096

    def test_rounds_up_to_a_multiple_of_the_target_resolution(self) -> None:
        # Dead pixels in the last rows leave max_id short of the corner; the
        # blocks must still tile evenly, so the estimate rounds up.
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512))
        acc.add(ts(), events([4096 * 4000]))
        assert acc.source_resolution == 4096

    def test_infers_a_smaller_panel_when_the_stream_is_smaller(self) -> None:
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512))
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert acc.source_resolution == 2048

    def test_latches_on_first_batch_and_does_not_drift(self) -> None:
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512))
        acc.add(ts(), events([4096 * 4096 - 1]))
        acc.add(ts(), events([0]))
        assert acc.source_resolution == 4096

    def test_empty_batch_does_not_latch(self) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(512))
        acc.add(ts(), events([]))
        assert acc.source_resolution is None
        assert len(inner.batches) == 1

    def test_logs_the_inferred_resolution(self) -> None:
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512))
        with capture_logs() as logs:
            acc.add(ts(), events([4096 * 4096 - 1]))
        entry = next(e for e in logs if e['event'] == 'detector_resolution_inferred')
        assert entry['source_resolution'] == 4096
        assert entry['block'] == 8


class TestIdsAboveInferredResolution:
    def test_are_dropped_rather_than_raised_on(self) -> None:
        # A single corrupt id must not take the service down; grouping already
        # discards ids outside the configured detector_number.
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2))
        acc.add(ts(), events([0]))  # latches source_resolution = 2
        acc.add(ts(), events([3, 10_000]))
        remapped = inner.batches[-1].pixel_id
        assert remapped[0] < 4
        assert remapped[1] >= 4  # outside the target grid, so grouping drops it

    def test_are_counted_and_logged(self) -> None:
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(2))
        acc.add(ts(), events([0]))
        with capture_logs() as logs:
            acc.add(ts(), events([10_000, 10_001]))
        entry = next(
            e for e in logs if e['event'] == 'event_id_above_inferred_resolution'
        )
        assert entry['dropped'] == 2


class TestIdBase:
    def test_a_one_based_detector_infers_its_true_resolution(self) -> None:
        # Regression: treating a 1-based panel as 0-based makes max_id one too
        # large, and the inferred resolution comes out at 4608 instead of 4096
        # -- a wrong stride, which shears the image rather than coarsening it.
        acc = DownsamplePixelIds(RecordingAccumulator(), downsampling(512, first_id=1))
        acc.add(ts(), events([4096 * 4096]))
        assert acc.source_resolution == 4096

    def test_a_one_based_detector_maps_its_first_pixel_to_the_first_target(
        self,
    ) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2, first_id=1))
        acc.add(ts(), events([1, 16]))
        assert inner.batches[0].pixel_id[0] == 0

    def test_ids_below_the_base_are_dropped_and_logged(self) -> None:
        inner = RecordingAccumulator()
        acc = DownsamplePixelIds(inner, downsampling(2, first_id=1))
        with capture_logs() as logs:
            acc.add(ts(), events([0, 1]))
        entry = next(e for e in logs if e['event'] == 'event_id_below_first_id')
        assert entry['dropped'] == 1
        # Outside the target grid, so grouping discards it.
        assert inner.batches[0].pixel_id[0] < 0


class TestGeometryFileCrossCheck:
    def test_warns_when_the_stream_disagrees_with_the_declared_resolution(
        self,
    ) -> None:
        acc = DownsamplePixelIds(
            RecordingAccumulator(), downsampling(512, declared=4096)
        )
        with capture_logs() as logs:
            acc.add(ts(), events([2048 * 2048 - 1]))
        entry = next(
            e
            for e in logs
            if e['event'] == 'detector_resolution_differs_from_geometry_file'
        )
        assert entry['streamed_resolution'] == 2048
        assert entry['declared_resolution'] == 4096

    def test_the_stream_wins_over_the_declared_resolution(self) -> None:
        # The file is static and may be stale; it never overrides observation.
        acc = DownsamplePixelIds(
            RecordingAccumulator(), downsampling(512, declared=4096)
        )
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert acc.source_resolution == 2048

    def test_silent_when_they_agree(self) -> None:
        acc = DownsamplePixelIds(
            RecordingAccumulator(), downsampling(512, declared=4096)
        )
        with capture_logs() as logs:
            acc.add(ts(), events([4096 * 4096 - 1]))
        assert not any(
            e['event'] == 'detector_resolution_differs_from_geometry_file' for e in logs
        )


class TestEquivalenceWithFoldAndConcat:
    """Downsampling before grouping must match folding and merging after it."""

    @pytest.mark.parametrize(('source', 'target'), [(16, 4), (64, 8), (64, 16)])
    def test_matches_full_resolution_grouping_then_bins_concat(
        self, source: int, target: int
    ) -> None:
        rng = np.random.default_rng(0)
        ids = rng.integers(0, source * source, size=5000)
        toa = rng.integers(0, 71_000_000, size=5000)
        raw = DetectorEvents(
            time_of_arrival=toa, unit='ns', pixel_id=ids.astype('int64')
        )

        # Reference: group at full resolution, fold, then merge the blocks.
        full_grid = sc.arange('detector_number', source * source, unit=None)
        reference_acc = GroupByPixel(ToNXevent_data(), full_grid)
        reference_acc.add(ts(), raw)
        reference = reference_acc.get()
        reference = reference.fold('detector_number', sizes={'x': source, 'y': source})
        block = source // target
        reference = reference.fold('x', sizes={'x': target, 'x_bin': block}).fold(
            'y', sizes={'y': target, 'y_bin': block}
        )
        reference = reference.bins.concat(['x_bin', 'y_bin'])

        # Downsample the ids first, then group on the coarse grid.
        small_grid = sc.arange('detector_number', target * target, unit=None)
        acc = DownsamplePixelIds(
            GroupByPixel(ToNXevent_data(), small_grid), downsampling(target)
        )
        acc.add(ts(), raw)
        result = acc.get().fold('detector_number', sizes={'x': target, 'y': target})

        assert sc.identical(result.bins.size().data, reference.bins.size().data)
        edges = sc.linspace('event_time_offset', 0.0, 71e6, 21, unit='ns')
        assert sc.identical(
            result.hist(event_time_offset=edges).data,
            reference.hist(event_time_offset=edges).data,
        )
