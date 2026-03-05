# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import dask
import pytest

from ess.livedata.core.service import Service


@pytest.fixture(autouse=True)
def _clean_dask_pool_config():
    """Ensure dask pool config is reset after each test."""
    yield
    with dask.config.set(pool=None):
        pass


def test_sync_scheduler_flag_is_registered():
    parser = Service.setup_arg_parser(description='test')
    args = parser.parse_args(['--sync-scheduler'])
    assert args.sync_scheduler is True


def test_sync_scheduler_flag_defaults_to_false():
    parser = Service.setup_arg_parser(description='test')
    args = parser.parse_args([])
    assert args.sync_scheduler is False


def test_sync_scheduler_sets_dask_pool():
    from dask.local import SynchronousExecutor

    dask.config.set(pool=SynchronousExecutor())
    pool = dask.config.get("pool")
    assert isinstance(pool, SynchronousExecutor)
