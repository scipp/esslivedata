# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from structlog.testing import capture_logs

from ess.livedata.config.detector_downsampling import DetectorDownsampling
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.preprocessors.downsample_pixel_ids import (
    SOURCE_RESOLUTION,
    DownsamplePixelIds,
)
from ess.livedata.preprocessors.group_by_pixel import GroupByPixel
from ess.livedata.preprocessors.to_nxevent_data import DetectorEvents, ToNXevent_data

WINDOW = 60.0
BUCKETS = 6


class RecordingAccumulator:
    """Captures the events handed to the wrapped accumulator."""

    def __init__(self) -> None:
        self.batches: list[DetectorEvents] = []

    def add(self, timestamp: Timestamp, data: DetectorEvents) -> bool:
        self.batches.append(data)
        return True

    def get(self) -> sc.DataArray:
        return sc.DataArray(sc.scalar(len(self.batches)))

    def clear(self) -> None:
        self.batches.clear()


class Clock:
    """Advances only when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def events(pixel_id: list[int]) -> DetectorEvents:
    return DetectorEvents(
        time_of_arrival=np.arange(len(pixel_id), dtype='int64'),
        unit='ns',
        pixel_id=np.array(pixel_id, dtype='int64'),
    )


def ts(seconds: float = 0.0) -> Timestamp:
    return Timestamp.from_ns(int(seconds * 1e9))


def downsampling(
    resolution: int, *, first_id: int = 0, max_resolution: int | None = None
) -> DetectorDownsampling:
    grid = sc.arange('detector_number', resolution * resolution, unit=None).fold(
        dim='detector_number', sizes={'dim_0': resolution, 'dim_1': resolution}
    )
    if max_resolution is None:
        max_resolution = resolution * 1024
    return DetectorDownsampling(resolution, max_resolution, first_id, grid)


def make(
    resolution: int,
    *,
    inner: object | None = None,
    clock: Clock | None = None,
    min_events_to_shrink: int = 1,
    **kwargs,
) -> DownsamplePixelIds:
    return DownsamplePixelIds(
        inner if inner is not None else RecordingAccumulator(),
        downsampling(resolution, **kwargs),
        window=WINDOW,
        buckets=BUCKETS,
        min_events_to_shrink=min_events_to_shrink,
        clock=clock if clock is not None else Clock(),
    )


class TestRemapping:
    def test_maps_each_block_of_source_pixels_to_one_target_pixel(self) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner)
        # source 4x4 -> target 2x2, block 2. id = x * 4 + y.
        # (x, y) = (0,0) (0,1) (1,0) (1,1) all fall in target pixel (0, 0) = 0.
        # (0,2) -> target (0, 1) = 1;  (2,0) -> target (1, 0) = 2;
        # (3,3) -> target (1, 1) = 3.
        acc.add(ts(), events([0, 1, 4, 5, 2, 8, 15]))
        assert acc.source_resolution == 4
        np.testing.assert_array_equal(inner.batches[0].pixel_id, [0, 0, 0, 0, 1, 2, 3])

    def test_leaves_time_of_arrival_untouched(self) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner)
        original = events([0, 15])
        acc.add(ts(), original)
        np.testing.assert_array_equal(
            inner.batches[0].time_of_arrival, original.time_of_arrival
        )
        assert inner.batches[0].unit == original.unit

    def test_does_not_mutate_incoming_events(self) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner)
        original = events([0, 15])
        acc.add(ts(), original)
        np.testing.assert_array_equal(original.pixel_id, [0, 15])


class TestSourceResolutionInference:
    def test_infers_source_resolution_from_largest_observed_id(self) -> None:
        acc = make(512)
        acc.add(ts(), events([4096 * 4096 - 1]))
        assert acc.source_resolution == 4096

    def test_rounds_up_to_the_next_candidate(self) -> None:
        # Candidates double, so any id past the panel's first quarter of rows
        # pins it -- far weaker than needing the last 512 rows to fire.
        acc = make(512)
        acc.add(ts(), events([1025 * 4096]))
        assert acc.source_resolution == 4096

    def test_never_estimates_below_the_target_resolution(self) -> None:
        acc = make(512)
        acc.add(ts(), events([0]))
        assert acc.source_resolution == 512

    def test_infers_a_smaller_panel_when_the_stream_is_smaller(self) -> None:
        acc = make(512)
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert acc.source_resolution == 2048

    def test_empty_batch_does_not_latch(self) -> None:
        inner = RecordingAccumulator()
        acc = make(512, inner=inner)
        acc.add(ts(), events([]))
        assert acc.source_resolution is None
        assert len(inner.batches) == 1

    def test_candidates_are_the_target_doubled_not_powers_of_two(self) -> None:
        # 1000x1000 ingested at 250x250: the candidates are 250, 500, 1000.
        acc = make(250, max_resolution=4000)
        acc.add(ts(), events([1000 * 1000 - 1]))
        assert acc.source_resolution == 1000

    def test_a_quarter_of_the_panel_pins_it_whatever_the_resolutions(self) -> None:
        # The point of doubling: an event past the first quarter of rows is
        # enough, here row 250 of 1000, rather than one in the last 250 rows.
        acc = make(250, max_resolution=4000)
        acc.add(ts(), events([251 * 1000]))
        assert acc.source_resolution == 1000

    def test_logs_the_inferred_resolution(self) -> None:
        acc = make(512)
        with capture_logs() as logs:
            acc.add(ts(), events([4096 * 4096 - 1]))
        entry = next(e for e in logs if e['event'] == 'detector_resolution_inferred')
        assert entry['source_resolution'] == 4096
        assert entry['block'] == 8


class TestGrowth:
    """A large id proves the panel is at least that big, so growth is immediate."""

    def test_a_narrow_beam_converges_on_the_true_panel(self) -> None:
        # The failure this guards: a run whose first batch only lights the
        # first rows would otherwise map every later event with a wrong stride
        # for the lifetime of the process.
        acc = make(512, max_resolution=4096)
        acc.add(ts(), events([100 * 4096]))
        assert acc.source_resolution < 4096
        acc.add(ts(), events([4095 * 4096 + 4095]))
        assert acc.source_resolution == 4096

    def test_needs_no_supporting_event_count(self) -> None:
        acc = make(512, max_resolution=4096, min_events_to_shrink=10_000)
        acc.add(ts(), events([0]))
        acc.add(ts(), events([4095 * 4096 + 4095]))
        assert acc.source_resolution == 4096

    def test_discards_the_events_mapped_with_the_old_stride(self) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner)
        acc.add(ts(), events([0]))
        acc.add(ts(), events([300]))
        # The first batch carried the old stride; only the re-mapped one remains.
        assert len(inner.batches) == 1


class TestShrinking:
    """The readout is reconfigured downward too, and says so only by omission."""

    @staticmethod
    def large_panel(clock: Clock, **kwargs) -> DownsamplePixelIds:
        acc = make(512, max_resolution=4096, clock=clock, **kwargs)
        acc.add(ts(), events([4095 * 4096 + 4095]))
        assert acc.source_resolution == 4096
        return acc

    def test_follows_the_panel_down_once_the_old_evidence_expires(self) -> None:
        clock = Clock()
        acc = self.large_panel(clock)
        clock.advance(2 * WINDOW)
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert acc.source_resolution == 2048

    def test_holds_while_the_old_evidence_is_still_in_the_window(self) -> None:
        clock = Clock()
        acc = self.large_panel(clock)
        clock.advance(WINDOW / 2)
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert acc.source_resolution == 4096

    def test_a_few_stray_counts_do_not_drop_the_estimate(self) -> None:
        # A quiet period with a handful of low ids looks exactly like a smaller
        # panel. Absence of large ids is weak evidence, so it needs volume.
        clock = Clock()
        acc = self.large_panel(clock, min_events_to_shrink=1000)
        clock.advance(2 * WINDOW)
        acc.add(ts(), events([0, 1, 2]))
        assert acc.source_resolution == 4096

    def test_a_silent_detector_does_not_drop_the_estimate(self) -> None:
        # Evidence expires on write, so no data means no conclusion rather
        # than a reset triggered by a beam-off period.
        clock = Clock()
        acc = self.large_panel(clock)
        clock.advance(100 * WINDOW)
        assert acc.source_resolution == 4096

    def test_discards_the_events_mapped_with_the_old_stride(self) -> None:
        clock = Clock()
        inner = RecordingAccumulator()
        acc = make(512, inner=inner, max_resolution=4096, clock=clock)
        acc.add(ts(), events([4095 * 4096 + 4095]))
        clock.advance(2 * WINDOW)
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert len(inner.batches) == 1

    def test_logs_the_change(self) -> None:
        clock = Clock()
        acc = self.large_panel(clock)
        clock.advance(2 * WINDOW)
        with capture_logs() as logs:
            acc.add(ts(), events([2048 * 2048 - 1]))
        entry = next(e for e in logs if e['event'] == 'detector_resolution_changed')
        assert entry['previous_resolution'] == 4096
        assert entry['source_resolution'] == 2048


class TestIdsAboveMaxResolution:
    """Beyond what the hardware can read out, an id is corruption, not evidence."""

    def test_are_dropped_rather_than_raising_the_estimate(self) -> None:
        # A single corrupt id must not take the service down, nor ratchet the
        # estimate; grouping already discards ids outside the detector_number.
        inner = RecordingAccumulator()
        acc = make(2, inner=inner, max_resolution=4)
        acc.add(ts(), events([15]))
        assert acc.source_resolution == 4
        acc.add(ts(), events([3, 10_000]))
        assert acc.source_resolution == 4
        remapped = inner.batches[-1].pixel_id
        assert remapped[0] < 4
        assert remapped[1] >= 4  # outside the target grid, so grouping drops it

    def test_are_counted_and_logged(self) -> None:
        acc = make(2, max_resolution=4)
        acc.add(ts(), events([15]))
        with capture_logs() as logs:
            acc.add(ts(), events([10_000, 10_001]))
        entry = next(e for e in logs if e['event'] == 'event_id_above_max_resolution')
        assert entry['dropped'] == 2


class TestIdBase:
    def test_a_one_based_detector_infers_its_true_resolution(self) -> None:
        # Regression: treating a 1-based panel as 0-based makes max_id one too
        # large, and the inferred resolution doubles to 8192 -- a wrong stride,
        # which shears the image rather than coarsening it.
        acc = make(512, first_id=1)
        acc.add(ts(), events([4096 * 4096]))
        assert acc.source_resolution == 4096

    def test_a_one_based_detector_maps_its_first_pixel_to_the_first_target(
        self,
    ) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner, first_id=1)
        acc.add(ts(), events([1, 16]))
        assert inner.batches[0].pixel_id[0] == 0

    def test_ids_below_the_base_are_dropped_and_logged(self) -> None:
        inner = RecordingAccumulator()
        acc = make(2, inner=inner, first_id=1)
        with capture_logs() as logs:
            acc.add(ts(), events([0, 1]))
        entry = next(e for e in logs if e['event'] == 'event_id_below_first_id')
        assert entry['dropped'] == 1
        # Outside the target grid, so grouping discards it.
        assert inner.batches[0].pixel_id[0] < 0


class TestSourceResolutionCoord:
    """Counts remapped with different strides must not be summed together."""

    @staticmethod
    def grouping(target: int, clock: Clock) -> DownsamplePixelIds:
        grid = sc.arange('detector_number', target * target, unit=None)
        return make(
            target,
            inner=GroupByPixel(ToNXevent_data(), grid),
            max_resolution=4096,
            clock=clock,
        )

    def test_is_stamped_on_the_emitted_data(self) -> None:
        acc = self.grouping(512, Clock())
        acc.add(ts(), events([4095 * 4096 + 4095]))
        assert acc.get().coords[SOURCE_RESOLUTION].value == 4096

    def test_is_absent_before_any_event_arrives(self) -> None:
        acc = self.grouping(512, Clock())
        acc.add(ts(), events([]))
        assert SOURCE_RESOLUTION not in acc.get().coords

    def test_changes_when_the_estimate_changes(self) -> None:
        clock = Clock()
        acc = self.grouping(512, clock)
        acc.add(ts(), events([4095 * 4096 + 4095]))
        before = acc.get().coords[SOURCE_RESOLUTION]
        acc.release_buffers()
        clock.advance(2 * WINDOW)
        acc.add(ts(), events([2048 * 2048 - 1]))
        assert not sc.identical(acc.get().coords[SOURCE_RESOLUTION], before)


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
        acc = make(target, inner=GroupByPixel(ToNXevent_data(), small_grid))
        acc.add(ts(), raw)
        result = acc.get().fold('detector_number', sizes={'x': target, 'y': target})

        assert sc.identical(result.bins.size().data, reference.bins.size().data)
        edges = sc.linspace('event_time_offset', 0.0, 71e6, 21, unit='ns')
        assert sc.identical(
            result.hist(event_time_offset=edges).data,
            reference.hist(event_time_offset=edges).data,
        )
