# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import threading
from typing import NewType

import dask
import pytest
import sciline
import sciline.task_graph
from sciline.scheduler import DaskScheduler, NaiveScheduler

from ess.livedata.core.sciline_scheduler import (
    SCHEDULER_MODES,
    configure_sciline_scheduler,
)

A = NewType('A', int)
B = NewType('B', int)
Result = NewType('Result', int)


@pytest.fixture(autouse=True)
def _restore_scheduler_config():
    """Undo the process-global scheduler override and dask pool config."""
    original = sciline.task_graph.DaskScheduler
    yield
    sciline.task_graph.DaskScheduler = original
    dask.config.set(pool=None)


def _make_pipeline() -> sciline.Pipeline:
    def provider_a() -> A:
        return A(1)

    def provider_b() -> B:
        return B(2)

    def provider_result(a: A, b: B) -> Result:
        return Result(a + b)

    return sciline.Pipeline(providers=[provider_a, provider_b, provider_result])


def _default_scheduler(pipeline: sciline.Pipeline):
    return pipeline.get(Result)._scheduler


def test_naive_mode_selects_naive_scheduler_by_default() -> None:
    configure_sciline_scheduler('naive')
    assert isinstance(_default_scheduler(_make_pipeline()), NaiveScheduler)


def test_dask_modes_select_dask_scheduler_by_default() -> None:
    configure_sciline_scheduler('dask-sync')
    assert isinstance(_default_scheduler(_make_pipeline()), DaskScheduler)
    configure_sciline_scheduler('dask-threaded')
    assert isinstance(_default_scheduler(_make_pipeline()), DaskScheduler)


def test_modes_can_be_switched_back_and_forth() -> None:
    configure_sciline_scheduler('naive')
    configure_sciline_scheduler('dask-sync')
    assert isinstance(_default_scheduler(_make_pipeline()), DaskScheduler)
    configure_sciline_scheduler('naive')
    assert isinstance(_default_scheduler(_make_pipeline()), NaiveScheduler)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match='Unknown scheduler mode'):
        configure_sciline_scheduler('threaded')  # type: ignore[arg-type]


def test_scheduler_modes_cover_cli_choices() -> None:
    assert SCHEDULER_MODES == ('naive', 'dask-sync', 'dask-threaded')


@pytest.mark.parametrize('mode', ['naive', 'dask-sync'])
def test_compute_runs_on_calling_thread_and_yields_correct_result(mode) -> None:
    """Both non-threaded modes execute providers on the calling thread.

    This verifies end-to-end that the configuration reaches
    ``Pipeline.compute()`` calls that do not pass a scheduler explicitly --
    the call shape used by ``ess.reduce.streaming.StreamProcessor``.
    """
    configure_sciline_scheduler(mode)

    task_threads: list[int | None] = []

    def provider_a() -> A:
        task_threads.append(threading.current_thread().ident)
        return A(1)

    def provider_b() -> B:
        task_threads.append(threading.current_thread().ident)
        return B(2)

    def provider_result(a: A, b: B) -> Result:
        task_threads.append(threading.current_thread().ident)
        return Result(a + b)

    pl = sciline.Pipeline(providers=[provider_a, provider_b, provider_result])
    result = pl.compute(Result)

    assert result == 3
    assert len(task_threads) == 3
    calling_thread = threading.current_thread().ident
    assert all(tid == calling_thread for tid in task_threads)


def test_explicit_scheduler_argument_still_wins() -> None:
    configure_sciline_scheduler('naive')
    pl = _make_pipeline()
    scheduler = DaskScheduler(dask.get)
    assert pl.get(Result, scheduler=scheduler)._scheduler is scheduler


def test_verification_raises_when_override_is_ineffective() -> None:
    """If sciline stops selecting via task_graph.DaskScheduler, fail loudly.

    Simulated by asserting a scheduler that the (unpatched) default selection
    does not produce: with dask installed sciline picks DaskScheduler, so
    verifying for NaiveScheduler must raise.
    """
    from ess.livedata.core.sciline_scheduler import _verify_default_scheduler

    sciline.task_graph.DaskScheduler = DaskScheduler  # ensure unpatched selection
    with pytest.raises(RuntimeError, match='Failed to configure'):
        _verify_default_scheduler(NaiveScheduler)
