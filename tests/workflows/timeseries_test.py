# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.livedata.preprocessors.accumulators import LogData
from ess.livedata.preprocessors.to_nxlog import ToNXlog
from ess.livedata.workflows.timeseries import TimeseriesStreamProcessor


@pytest.fixture
def nxlog() -> ToNXlog:
    return ToNXlog(attrs={'units': 'mm'})


@pytest.fixture
def processor() -> TimeseriesStreamProcessor:
    return TimeseriesStreamProcessor()


def push(nxlog: ToNXlog, *samples: tuple[int, float]) -> None:
    for time, value in samples:
        nxlog.add(0, LogData(time=time, value=value))


def run_cycle(
    processor: TimeseriesStreamProcessor, nxlog: ToNXlog
) -> dict[str, sc.DataArray]:
    processor.accumulate({'log': nxlog.get()}, start_time=0, end_time=0)
    return processor.finalize()


def values(result: dict[str, sc.DataArray]) -> list[float]:
    return list(result['delta'].values)


def test_first_finalize_returns_full_retained_history(processor, nxlog):
    push(nxlog, (10, 1.0), (20, 2.0), (30, 3.0))
    assert values(run_cycle(processor, nxlog)) == [1.0, 2.0, 3.0]


def test_subsequent_finalize_returns_only_new_samples(processor, nxlog):
    push(nxlog, (10, 1.0), (20, 2.0))
    run_cycle(processor, nxlog)

    push(nxlog, (30, 3.0), (40, 4.0))
    assert values(run_cycle(processor, nxlog)) == [3.0, 4.0]


def test_finalize_without_new_samples_raises(processor, nxlog):
    push(nxlog, (10, 1.0))
    run_cycle(processor, nxlog)

    with pytest.raises(ValueError, match="No new data"):
        run_cycle(processor, nxlog)


def test_finalize_before_accumulate_raises(processor):
    with pytest.raises(ValueError, match="No data has been added"):
        processor.finalize()


def test_accumulate_rejects_multiple_items(processor):
    with pytest.raises(ValueError, match="exactly one data item"):
        processor.accumulate(
            {'a': sc.scalar(1), 'b': sc.scalar(2)}, start_time=0, end_time=0
        )


def test_clear_resets_cursor_so_full_history_is_republished(processor, nxlog):
    push(nxlog, (10, 1.0), (20, 2.0))
    run_cycle(processor, nxlog)

    processor.clear()
    assert values(run_cycle(processor, nxlog)) == [1.0, 2.0]


class TestTrimmedBuffer:
    """Publication must stay correct when the preprocessor drops old samples.

    A positional cursor breaks under front-trimming: it either skips genuinely
    new samples or reports "no new data" until the buffer regrows past the
    stale watermark. These cases pin the timestamp-based behaviour that bounded
    retention (#1127) depends on.
    """

    def trimmed(self, nxlog: ToNXlog, drop: int) -> sc.DataArray:
        return nxlog.get()['time', drop:]

    def test_new_samples_after_trim_are_not_skipped(self, processor, nxlog):
        push(nxlog, (10, 1.0), (20, 2.0), (30, 3.0))
        run_cycle(processor, nxlog)

        push(nxlog, (40, 4.0), (50, 5.0))
        processor.accumulate(
            {'log': self.trimmed(nxlog, drop=3)}, start_time=0, end_time=0
        )
        assert values(processor.finalize()) == [4.0, 5.0]

    def test_trim_shrinking_buffer_below_published_count_still_emits(
        self, processor, nxlog
    ):
        push(nxlog, (10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0))
        run_cycle(processor, nxlog)

        push(nxlog, (50, 5.0))
        # Retain only the last two samples: fewer than were already published.
        processor.accumulate(
            {'log': self.trimmed(nxlog, drop=3)}, start_time=0, end_time=0
        )
        assert values(processor.finalize()) == [5.0]

    def test_trim_without_new_samples_still_raises(self, processor, nxlog):
        push(nxlog, (10, 1.0), (20, 2.0), (30, 3.0))
        run_cycle(processor, nxlog)

        processor.accumulate(
            {'log': self.trimmed(nxlog, drop=2)}, start_time=0, end_time=0
        )
        with pytest.raises(ValueError, match="No new data"):
            processor.finalize()

    def test_already_published_samples_are_never_re_emitted(self, processor, nxlog):
        push(nxlog, (10, 1.0), (20, 2.0), (30, 3.0))
        run_cycle(processor, nxlog)

        push(nxlog, (40, 4.0))
        # Trim keeps a sample that was already published; it must not reappear.
        processor.accumulate(
            {'log': self.trimmed(nxlog, drop=1)}, start_time=0, end_time=0
        )
        assert values(processor.finalize()) == [4.0]
