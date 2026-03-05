# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import threading
from typing import NewType

import dask
import pytest
import sciline

from ess.livedata.service_factory import DataServiceRunner


def _make_runner() -> DataServiceRunner:
    return DataServiceRunner(pretty_name='test', make_builder=lambda **kw: None)


@pytest.fixture(autouse=True)
def _clean_dask_pool_config():
    """Ensure dask pool config is reset after each test."""
    yield
    dask.config.set(pool=None)


def test_sync_scheduler_flag_is_registered():
    runner = _make_runner()
    args = runner.parser.parse_args(['--sync-scheduler', '--instrument', 'dummy'])
    assert args.sync_scheduler is True


def test_sync_scheduler_flag_defaults_to_false():
    runner = _make_runner()
    args = runner.parser.parse_args(['--instrument', 'dummy'])
    assert args.sync_scheduler is False


A = NewType('A', int)
B = NewType('B', int)
Result = NewType('Result', int)


def test_sciline_with_sync_pool_runs_on_calling_thread():
    """Setting dask pool to SynchronousExecutor makes sciline run on the calling thread.

    This verifies that the ``--sync-scheduler`` mechanism actually takes effect
    end-to-end through sciline's DaskScheduler.
    """
    from dask.local import SynchronousExecutor

    dask.config.set(pool=SynchronousExecutor())

    task_threads: list[int] = []

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
    calling_thread = threading.current_thread().ident
    assert len(task_threads) == 3
    assert all(tid == calling_thread for tid in task_threads)
