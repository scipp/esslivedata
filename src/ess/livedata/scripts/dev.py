# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Run a local development stack -- fakes, backend services and dashboard.

``esslivedata-dev dream`` replaces the one-terminal-per-service workflow: it
creates the Kafka topics, starts every service the instrument needs, merges
their logs into the current terminal with a per-service prefix, and shuts the
whole stack down on Ctrl-C. ``--reload`` additionally restarts the stack
whenever a source file under ``ess.livedata`` changes.

Naming a subset of ``SERVICE_GROUPS`` limits what is started, e.g.
``esslivedata-dev dream monitors timeseries``. The dashboard always runs unless
``--no-dashboard`` is given.

Machine-local settings -- currently only the NeXus files replayed by
``fake_detectors`` -- come from a ``dev.toml`` in the current directory or any
parent, so that switching instrument stays a one-token change::

    [nexus_files]
    dream = "~/data/268227_00024779_Si_BC_offset_240_deg_wlgth.hdf"
    loki = "~/data/coda_loki_999999_00008957.hdf"
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from confluent_kafka import KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.config.instruments import available_instruments
from ess.livedata.config.streams import get_stream_mapping

SERVICE_GROUPS: dict[str, tuple[str, ...]] = {
    'detectors': ('fake_detectors', 'detector_data'),
    'monitors': ('fake_monitors', 'monitor_data'),
    # data_reduction consumes the raw streams itself, so it needs the fakes but
    # not detector_data or monitor_data.
    'reduction': ('fake_detectors', 'fake_monitors', 'data_reduction'),
    'timeseries': ('fake_logdata', 'timeseries'),
}

# The fakes publish to the dev topic names unconditionally, so every consuming
# service has to run with --dev for the topics to line up.
_FAKES = frozenset({'fake_detectors', 'fake_monitors', 'fake_logdata'})

_COLORS = ('36', '32', '33', '35', '34', '31', '96', '92')

_LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


@dataclass(frozen=True)
class Options:
    """Everything the stack needs beyond the choice of services."""

    instrument: str
    log_level: str = 'INFO'
    nexus_file: str | None = None
    monitor_mode: str = 'ev44'
    num_monitors: int = 2
    port: int = 5009


@dataclass(frozen=True)
class Job:
    """A subprocess to run as part of the stack."""

    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def resolve_services(groups: Sequence[str]) -> list[str]:
    """Return the services of the named groups, deduplicated, in start order.

    Parameters
    ----------
    groups:
        Names of :data:`SERVICE_GROUPS`. Empty selects all groups.
    """
    selected = groups or list(SERVICE_GROUPS)
    services = [service for name in selected for service in SERVICE_GROUPS[name]]
    return list(dict.fromkeys(services))


def make_jobs(services: Sequence[str], opts: Options, *, dashboard: bool) -> list[Job]:
    """Build the subprocess commands for the selected services."""
    jobs = [
        Job(name=service, argv=_service_command(service, opts)) for service in services
    ]
    if dashboard:
        jobs.append(
            Job(
                name='dashboard',
                argv=[
                    sys.executable,
                    '-m',
                    'ess.livedata.dashboard.reduction',
                    '--instrument',
                    opts.instrument,
                    '--dev',
                    '--log-level',
                    opts.log_level,
                    '--port',
                    str(opts.port),
                ],
                env={'BOKEH_ALLOW_WS_ORIGIN': '*'},
            )
        )
    return jobs


def _service_command(service: str, opts: Options) -> list[str]:
    argv = [
        sys.executable,
        '-m',
        f'ess.livedata.services.{service}',
        '--instrument',
        opts.instrument,
        '--log-level',
        opts.log_level,
    ]
    if service not in _FAKES:
        argv.append('--dev')
    if service == 'fake_detectors' and opts.nexus_file is not None:
        argv += ['--nexus-file', opts.nexus_file]
    if service == 'fake_monitors':
        argv += ['--mode', opts.monitor_mode, '--num-monitors', str(opts.num_monitors)]
    return argv


def load_dev_config(start: Path | None = None) -> dict[str, Any]:
    """Load ``dev.toml`` from ``start`` or the closest parent that has one."""
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        path = directory / 'dev.toml'
        if path.is_file():
            with path.open('rb') as file:
                return tomllib.load(file)
    return {}


def nexus_file_for(instrument: str, config: dict[str, Any]) -> str | None:
    """Return the configured NeXus file for ``instrument``, if any."""
    path = config.get('nexus_files', {}).get(instrument)
    if path is None:
        return None
    file = Path(path).expanduser()
    if not file.is_file():
        raise SystemExit(f"dev.toml: no such nexus_files[{instrument!r}]: {file}")
    return str(file)


def check_broker_reachable() -> None:
    """Fail before starting anything if the dev Kafka broker is not up."""
    servers = load_config(namespace=config_names.kafka, env='dev')['bootstrap.servers']
    host, _, port = servers.split(',')[0].rpartition(':')
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            pass
    except OSError as e:
        raise SystemExit(
            f"No Kafka broker at {servers}: {e}\n"
            "Start it with 'docker compose up -d kafka'."
        ) from e


def ensure_topics_exist(instrument: str) -> None:
    """Create the topics consumers validate at startup.

    Consumers fail fast on missing topics (``validate_topics_exist``), and
    broker-side auto-creation only triggers on produce, so a pristine broker
    (CI, fresh local setup) needs the topics created up front. Idempotent.
    """
    mapping = get_stream_mapping(instrument=instrument, dev=True)
    topics = (
        mapping.detector_topics
        | mapping.area_detector_topics
        | mapping.monitor_topics
        | mapping.log_topics
        | mapping.topics.all_topics
    )
    admin = AdminClient(load_config(namespace=config_names.kafka, env='dev'))
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=1) for topic in sorted(topics)]
    )
    for future in futures.values():
        try:
            future.result()
        except KafkaException as e:
            if e.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


class Stack:
    """A group of subprocesses that are started, logged and stopped together."""

    def __init__(self, jobs: Sequence[Job]) -> None:
        self._jobs = jobs
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._threads: list[threading.Thread] = []
        self._width = max(len(job.name) for job in jobs)
        self._write_lock = threading.Lock()

    def start(self) -> None:
        self._threads = []
        for index, job in enumerate(self._jobs):
            # Unbuffered output, else the pipe holds back log lines in 8k chunks.
            env = {**os.environ, 'PYTHONUNBUFFERED': '1', **job.env}
            proc = subprocess.Popen(  # noqa: S603
                job.argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._procs[job.name] = proc
            thread = threading.Thread(
                target=self._forward_output,
                args=(job.name, _COLORS[index % len(_COLORS)], proc.stdout),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _forward_output(self, name: str, color: str, stream: TextIO) -> None:
        prefix = f'\033[{color}m{name:>{self._width}}\033[0m | '
        with stream:  # closes the pipe once the job stops writing
            for line in stream:
                with self._write_lock:
                    sys.stdout.write(prefix + line)
                    sys.stdout.flush()

    @property
    def running(self) -> list[str]:
        """Names of the jobs that are still alive."""
        return [name for name, proc in self._procs.items() if proc.poll() is None]

    def exited(self) -> str | None:
        """Name of a job that is no longer running, if there is one."""
        return next(
            (name for name, proc in self._procs.items() if proc.poll() is not None),
            None,
        )

    def stop(self, timeout: float = 15.0) -> None:
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        deadline = time.monotonic() + timeout
        for proc in self._procs.values():
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Drain what the children logged on their way out before returning.
        for thread in self._threads:
            thread.join(timeout=1.0)


class SourceWatcher:
    """Detects edits to the package sources by polling modification times."""

    def __init__(self, root: Path, *, interval: float = 1.0) -> None:
        self._root = root
        self._interval = interval
        self._due = time.monotonic() + interval
        self._state = self._scan()

    def _scan(self) -> dict[Path, int]:
        return {path: path.stat().st_mtime_ns for path in self._root.rglob('*.py')}

    def changed(self) -> bool:
        now = time.monotonic()
        if now < self._due:
            return False
        self._due = now + self._interval
        state = self._scan()
        if state == self._state:
            return False
        # Let a burst of writes (editor save, git checkout) settle, so that it
        # triggers a single restart rather than one per file.
        while True:
            time.sleep(0.3)
            settled = self._scan()
            if settled == state:
                break
            state = settled
        self._state = state
        return True


def run(jobs: Sequence[Job], *, reload: bool) -> int:
    """Run the stack until Ctrl-C or until one of the jobs exits."""
    watcher = SourceWatcher(Path(__file__).parents[1]) if reload else None
    stack = Stack(jobs)
    stack.start()
    try:
        while True:
            time.sleep(0.2)
            if (name := stack.exited()) is not None:
                print(f'\n*** {name} exited, stopping the stack ***')
                return 1
            if watcher is not None and watcher.changed():
                print('\n*** sources changed, restarting ***')
                stack.stop()
                stack.start()
    except KeyboardInterrupt:
        # Ctrl-C reached the children via the foreground process group already;
        # stop() below only waits for them and reaps stragglers.
        print()
        return 0
    finally:
        stack.stop()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'instrument', nargs='?', default='dummy', choices=available_instruments()
    )
    parser.add_argument(
        'groups',
        nargs='*',
        # Not default=[]: argparse would validate the default against choices.
        default=None,
        choices=list(SERVICE_GROUPS),
        help='Service groups to run. Defaults to all of them.',
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Restart the stack when a source file changes.',
    )
    parser.add_argument(
        '--no-dashboard', action='store_true', help='Run backend services only.'
    )
    parser.add_argument(
        '--nexus-file', help='NeXus file to replay, overriding dev.toml.'
    )
    parser.add_argument('--log-level', choices=_LOG_LEVELS, default='INFO')
    parser.add_argument('--port', type=int, default=5009, help='Dashboard port.')
    parser.add_argument('--monitor-mode', choices=['ev44', 'da00'], default='ev44')
    parser.add_argument('--num-monitors', type=int, default=2)
    return parser


def main() -> int:
    args = make_parser().parse_args()

    services = resolve_services(args.groups)
    opts = Options(
        instrument=args.instrument,
        log_level=args.log_level,
        nexus_file=args.nexus_file
        or nexus_file_for(args.instrument, load_dev_config()),
        monitor_mode=args.monitor_mode,
        num_monitors=args.num_monitors,
        port=args.port,
    )
    check_broker_reachable()
    ensure_topics_exist(opts.instrument)

    jobs = make_jobs(services, opts, dashboard=not args.no_dashboard)
    print(f"instrument: {opts.instrument}")
    print(f"services:   {', '.join(job.name for job in jobs)}")
    if 'fake_detectors' in services:
        print(
            f"nexus file: {opts.nexus_file}"
            if opts.nexus_file is not None
            else "nexus file: none, generating random events (set "
            f"nexus_files.{opts.instrument} in dev.toml to replay a file)"
        )
    if not args.no_dashboard:
        print(f"dashboard:  http://localhost:{opts.port}")
    return run(jobs, reload=args.reload)


if __name__ == '__main__':
    raise SystemExit(main())
