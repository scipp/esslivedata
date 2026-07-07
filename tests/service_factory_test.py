# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.service_factory import DataServiceRunner, _resolve_scheduler_mode


def _make_runner() -> DataServiceRunner:
    return DataServiceRunner(pretty_name='test', make_builder=lambda **kw: None)


def test_scheduler_flag_defaults_to_none_resolving_to_naive():
    runner = _make_runner()
    args = runner.parser.parse_args(['--instrument', 'dummy'])
    assert args.scheduler is None
    assert (
        _resolve_scheduler_mode(scheduler=args.scheduler, sync_scheduler=None)
        == 'naive'
    )


@pytest.mark.parametrize('mode', ['naive', 'dask-sync', 'dask-threaded'])
def test_scheduler_flag_accepts_all_modes(mode):
    runner = _make_runner()
    args = runner.parser.parse_args(['--scheduler', mode, '--instrument', 'dummy'])
    assert args.scheduler == mode


def test_scheduler_flag_rejects_unknown_mode():
    runner = _make_runner()
    with pytest.raises(SystemExit):
        runner.parser.parse_args(['--scheduler', 'bogus', '--instrument', 'dummy'])


def test_deprecated_sync_scheduler_flag_still_parses():
    runner = _make_runner()
    args = runner.parser.parse_args(['--sync-scheduler', '--instrument', 'dummy'])
    assert args.sync_scheduler is True
    args = runner.parser.parse_args(['--no-sync-scheduler', '--instrument', 'dummy'])
    assert args.sync_scheduler is False
    args = runner.parser.parse_args(['--instrument', 'dummy'])
    assert args.sync_scheduler is None


def test_resolve_scheduler_mode_defaults_to_naive():
    assert _resolve_scheduler_mode(scheduler=None, sync_scheduler=None) == 'naive'


def test_resolve_scheduler_mode_maps_deprecated_flag_to_dask_modes():
    assert _resolve_scheduler_mode(scheduler=None, sync_scheduler=True) == 'dask-sync'
    assert (
        _resolve_scheduler_mode(scheduler=None, sync_scheduler=False) == 'dask-threaded'
    )


def test_resolve_scheduler_mode_explicit_scheduler_wins_over_deprecated_flag():
    assert _resolve_scheduler_mode(scheduler='naive', sync_scheduler=True) == 'naive'
    assert (
        _resolve_scheduler_mode(scheduler='dask-threaded', sync_scheduler=True)
        == 'dask-threaded'
    )


def test_job_threads_flag_is_registered():
    runner = _make_runner()
    args = runner.parser.parse_args(['--job-threads', '5', '--instrument', 'dummy'])
    assert args.job_threads == 5


def test_job_threads_flag_defaults_to_five():
    runner = _make_runner()
    args = runner.parser.parse_args(['--instrument', 'dummy'])
    assert args.job_threads == 5


def test_check_flag_defaults_to_false():
    runner = _make_runner()
    args = runner.parser.parse_args(['--instrument', 'dummy'])
    assert args.check is False


def test_check_flag_is_registered():
    runner = _make_runner()
    args = runner.parser.parse_args(['--check', '--instrument', 'dummy'])
    assert args.check is True
