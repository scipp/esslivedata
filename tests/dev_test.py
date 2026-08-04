# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the development stack launcher."""

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ess.livedata.scripts.dev import (
    SERVICE_GROUPS,
    Job,
    Options,
    SourceWatcher,
    Stack,
    load_dev_config,
    make_jobs,
    make_parser,
    nexus_file_for,
    resolve_services,
)


@pytest.fixture
def opts() -> Options:
    return Options(instrument='dream')


def commands(opts: Options, groups: list[str] | None = None) -> dict[str, list[str]]:
    jobs = make_jobs(resolve_services(groups or []), opts, dashboard=True)
    return {job.name: job.argv for job in jobs}


def test_no_groups_selects_all_services() -> None:
    assert set(resolve_services([])) == {
        service for group in SERVICE_GROUPS.values() for service in group
    }


def test_services_shared_between_groups_are_started_once() -> None:
    assert resolve_services(['detectors', 'reduction']).count('fake_detectors') == 1


def test_consuming_services_run_in_dev_mode_matching_the_fakes(opts: Options) -> None:
    argv = commands(opts)
    assert '--dev' in argv['detector_data']
    assert '--dev' in argv['data_reduction']
    # The fakes have no --dev flag, they always publish to the dev topic names.
    assert '--dev' not in argv['fake_detectors']


def test_instrument_is_passed_to_every_job(opts: Options) -> None:
    for argv in commands(opts).values():
        assert argv[argv.index('--instrument') + 1] == 'dream'


def test_nexus_file_is_passed_to_fake_detectors_only() -> None:
    argv = commands(Options(instrument='dream', nexus_file='/data/run.hdf'))
    fake_detectors = argv['fake_detectors']
    assert fake_detectors[fake_detectors.index('--nexus-file') + 1] == '/data/run.hdf'
    assert '--nexus-file' not in argv['fake_monitors']


def test_dashboard_allows_any_websocket_origin(opts: Options) -> None:
    (dashboard,) = make_jobs([], opts, dashboard=True)
    assert dashboard.name == 'dashboard'
    assert dashboard.env['BOKEH_ALLOW_WS_ORIGIN'] == '*'


def test_dashboard_can_be_omitted(opts: Options) -> None:
    jobs = make_jobs(resolve_services(['monitors']), opts, dashboard=False)
    assert [job.name for job in jobs] == ['fake_monitors', 'monitor_data']


def test_dev_config_is_found_in_parent_directory(tmp_path: Path) -> None:
    (tmp_path / 'dev.toml').write_text('[nexus_files]\ndream = "run.hdf"\n')
    nested = tmp_path / 'a' / 'b'
    nested.mkdir(parents=True)
    assert load_dev_config(nested) == {'nexus_files': {'dream': 'run.hdf'}}


def test_dev_config_is_empty_when_absent(tmp_path: Path) -> None:
    assert load_dev_config(tmp_path) == {}


def test_nexus_file_for_returns_none_for_unconfigured_instrument(
    tmp_path: Path,
) -> None:
    file = tmp_path / 'run.hdf'
    file.touch()
    config = {'nexus_files': {'dream': str(file)}}
    assert nexus_file_for('dream', config) == str(file)
    assert nexus_file_for('loki', config) is None


def test_configured_nexus_file_must_exist(tmp_path: Path) -> None:
    config = {'nexus_files': {'dream': str(tmp_path / 'missing.hdf')}}
    with pytest.raises(SystemExit, match=r'missing\.hdf'):
        nexus_file_for('dream', config)


def test_source_watcher_reports_edits(tmp_path: Path) -> None:
    source = tmp_path / 'mod.py'
    source.write_text('x = 1\n')
    watcher = SourceWatcher(tmp_path, interval=0.0)
    assert not watcher.changed()
    source.write_text('x = 2\n')
    os.utime(source, ns=(0, 10**9))  # avoid depending on filesystem mtime resolution
    assert watcher.changed()
    assert not watcher.changed()


def test_source_watcher_reports_new_and_removed_files(tmp_path: Path) -> None:
    (tmp_path / 'mod.py').write_text('x = 1\n')
    watcher = SourceWatcher(tmp_path, interval=0.0)
    (tmp_path / 'other.py').write_text('y = 1\n')
    assert watcher.changed()
    (tmp_path / 'other.py').unlink()
    assert watcher.changed()


def sleeper(name: str) -> Job:
    return Job(name=name, argv=[sys.executable, '-c', 'import time; time.sleep(60)'])


def wait_for_exit(stack: Stack, timeout: float = 20.0) -> str | None:
    deadline = time.monotonic() + timeout
    exited: str | None
    while (exited := stack.exited()) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return exited


def test_stack_stops_all_jobs() -> None:
    stack = Stack([sleeper('a'), sleeper('b')])
    stack.start()
    assert sorted(stack.running) == ['a', 'b']
    stack.stop(timeout=20.0)
    assert stack.running == []


def test_stack_reports_a_job_that_died() -> None:
    stack = Stack([sleeper('alive'), Job(name='dead', argv=[sys.executable, '-c', ''])])
    try:
        stack.start()
        assert wait_for_exit(stack) == 'dead'
        assert stack.running == ['alive']
    finally:
        stack.stop(timeout=20.0)


def test_stack_kills_jobs_that_ignore_sigint() -> None:
    stubborn = textwrap.dedent(
        """
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print('ready')
        time.sleep(60)
        """
    )
    stack = Stack([Job(name='stubborn', argv=[sys.executable, '-c', stubborn])])
    stack.start()
    time.sleep(1.0)  # let the child install its signal handler
    stack.stop(timeout=1.0)
    assert stack.running == []


def test_stack_prefixes_output_with_the_job_name(
    capfd: pytest.CaptureFixture[str],
) -> None:
    stack = Stack([Job(name='talker', argv=[sys.executable, '-c', 'print("hello")'])])
    stack.start()
    wait_for_exit(stack)
    stack.stop(timeout=20.0)
    out = capfd.readouterr().out
    assert 'talker' in out  # colour codes sit between the name and the pipe
    assert '| hello' in out


@pytest.mark.parametrize(
    ('argv', 'instrument', 'groups'),
    [
        ([], 'dummy', []),
        (['dream'], 'dream', []),
        (['dream', 'monitors', 'detectors'], 'dream', ['monitors', 'detectors']),
    ],
)
def test_cli_takes_instrument_and_groups_as_positionals(
    argv: list[str], instrument: str, groups: list[str]
) -> None:
    args = make_parser().parse_args(argv)
    assert args.instrument == instrument
    assert args.groups == groups


def test_cli_rejects_unknown_group() -> None:
    with pytest.raises(SystemExit):
        make_parser().parse_args(['dream', 'not-a-group'])
