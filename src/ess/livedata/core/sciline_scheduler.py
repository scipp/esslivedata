# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Process-wide selection of the scheduler sciline uses for workflow execution.

Backend services execute reduction workflows via
:class:`ess.reduce.streaming.StreamProcessor`, which calls
``sciline.Pipeline.compute()`` on internal pipeline copies without exposing a
``scheduler`` argument. Sciline picks its default scheduler at task-graph
construction time: dask's threaded scheduler when ``dask`` is importable
(always, since essreduce depends on it), otherwise
:class:`sciline.scheduler.NaiveScheduler`.

For live-data workflows the dask machinery is pure overhead: graphs are small
(tens of nodes), run once per ~1 s batch, and parallelism happens at the job
level, not within a graph. Benchmarks (monitor and detector-view workflows)
show dask adds ~1-3 ms per accumulate+finalize cycle (~40 us per graph node
per compute call) over the naive scheduler, with no benefit since production
runs dask with a synchronous executor anyway.

Since neither essreduce nor sciline offers a hook to change the default,
:func:`configure_sciline_scheduler` overrides the selection site in
``sciline.task_graph``. A startup probe verifies the override took effect so
that a sciline refactor breaks service startup loudly instead of silently
reverting to the dask scheduler.
"""

from __future__ import annotations

from typing import Literal, NewType, get_args

import sciline
import sciline.task_graph
from sciline.scheduler import DaskScheduler, NaiveScheduler, Scheduler

SchedulerMode = Literal['naive', 'dask-sync', 'dask-threaded']
SCHEDULER_MODES: tuple[str, ...] = get_args(SchedulerMode)

# The factory sciline's TaskGraph instantiates when no scheduler is given.
# Captured at import time so the override can be applied repeatedly (tests)
# and composed from a known-good baseline.
_ORIGINAL_DASK_SCHEDULER = sciline.task_graph.DaskScheduler


def configure_sciline_scheduler(mode: SchedulerMode) -> None:
    """Select the scheduler sciline uses when none is passed explicitly.

    Applies to every ``Pipeline.compute()`` in this process, in particular the
    calls ``StreamProcessor`` makes internally, which cannot be reached via a
    ``scheduler`` argument.

    Parameters
    ----------
    mode:
        - ``'naive'``: sciline's :class:`~sciline.scheduler.NaiveScheduler`.
          Sequential, no dask involvement. Fastest for the small graphs used
          in live-data workflows.
        - ``'dask-sync'``: dask's threaded scheduler with a synchronous
          executor, i.e. task execution on the calling thread (the previous
          production default, ``--sync-scheduler``).
        - ``'dask-threaded'``: dask's threaded scheduler with its regular
          thread pool. Subject to GIL contention with job-level threading.
    """
    if mode not in SCHEDULER_MODES:
        raise ValueError(f"Unknown scheduler mode: {mode!r}. Use {SCHEDULER_MODES}.")

    import dask

    if mode == 'naive':
        sciline.task_graph.DaskScheduler = NaiveScheduler
        _verify_default_scheduler(NaiveScheduler)
    elif mode == 'dask-sync':
        from dask.local import SynchronousExecutor

        sciline.task_graph.DaskScheduler = _ORIGINAL_DASK_SCHEDULER
        dask.config.set(pool=SynchronousExecutor())
        _verify_default_scheduler(DaskScheduler)
    else:  # 'dask-threaded'
        sciline.task_graph.DaskScheduler = _ORIGINAL_DASK_SCHEDULER
        dask.config.set(pool=None)
        _verify_default_scheduler(DaskScheduler)


_Probe = NewType('_Probe', int)


def _verify_default_scheduler(expected: type[Scheduler]) -> None:
    """Check that a pipeline built without an explicit scheduler uses `expected`.

    The override patches a name in a sciline-internal module; if a sciline
    update stops routing default-scheduler selection through that name (or
    renames the attribute inspected here), this raises at service startup
    rather than letting workflows silently fall back to the dask scheduler.
    """
    task_graph = sciline.Pipeline([], params={_Probe: _Probe(0)}).get(_Probe)
    scheduler = getattr(task_graph, '_scheduler', None)
    if not isinstance(scheduler, expected):
        raise RuntimeError(
            f"Failed to configure sciline's default scheduler: expected "
            f"{expected.__name__}, but a probe pipeline selected "
            f"{type(scheduler).__name__}. The sciline internals targeted by "
            f"ess.livedata.core.sciline_scheduler have likely changed; update "
            f"the override to match the installed sciline version."
        )
