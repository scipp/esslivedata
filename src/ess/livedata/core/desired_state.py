# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Desired state: the set of jobs that should run, per workflow.

The desired state lives on the compacted ``{instrument}_livedata_desired`` topic,
one record per workflow. The Kafka key is the workflow id in its string form
(``instrument/name/version``); the value is a :class:`DesiredWorkflow` as JSON.
A record with a null value (a *tombstone*) removes the workflow, i.e. none of its
jobs should run. Reading the topic from offset 0 and folding its records with
:func:`fold_updates` gives the *desired table*.

The Kafka implementations of :class:`DesiredStateReader` and
:class:`DesiredStateWriter` are in :mod:`ess.livedata.kafka.desired_state`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import pydantic

from ..config.workflow_spec import JobNumber, WorkflowId


class DesiredJob(pydantic.BaseModel):
    """Desired job for one source of a workflow."""

    params: dict[str, Any]
    aux_source_names: dict[str, str]


class DesiredWorkflow(pydantic.BaseModel):
    """Desired jobs of one workflow, written as one record per commit.

    Every commit mints a new ``job_number``; the jobs of a workflow are
    ``JobId(source_name, job_number)`` for each key of ``jobs``.
    """

    workflow_id: WorkflowId
    job_number: JobNumber
    jobs: dict[str, DesiredJob] = pydantic.Field(
        description="Desired jobs keyed by source name."
    )


DesiredTable = dict[WorkflowId, DesiredWorkflow]
#: One record of the desired-state topic; ``None`` is a tombstone.
DesiredUpdate = tuple[WorkflowId, DesiredWorkflow | None]


class DesiredStateDecodeError(Exception):
    """Raised when a desired-state record cannot be decoded."""


class DesiredStateWriteError(Exception):
    """Raised when a desired-state record was not confirmed as stored."""


def encode_desired_record(
    workflow_id: WorkflowId, desired: DesiredWorkflow | None
) -> tuple[bytes, bytes | None]:
    """Encode a desired-state record as Kafka ``(key, value)``.

    Parameters
    ----------
    workflow_id:
        Workflow the record is for.
    desired:
        Desired jobs of the workflow, or ``None`` for a tombstone.

    Returns
    -------
    :
        UTF-8 workflow id string as key, and the JSON value or ``None``.
    """
    if desired is not None and desired.workflow_id != workflow_id:
        raise ValueError(
            f"Record for '{workflow_id}' carries workflow_id '{desired.workflow_id}'"
        )
    value = None if desired is None else desired.model_dump_json().encode('utf-8')
    return str(workflow_id).encode('utf-8'), value


def decode_desired_record(key: bytes | None, value: bytes | None) -> DesiredUpdate:
    """Decode a Kafka ``(key, value)`` desired-state record.

    Raises
    ------
    DesiredStateDecodeError
        If the key is not a workflow id, the value is not a
        :class:`DesiredWorkflow`, or the two disagree on the workflow id.
    """
    if key is None:
        raise DesiredStateDecodeError("Record has no key")
    try:
        workflow_id = WorkflowId.from_string(key.decode('utf-8'))
    except (UnicodeDecodeError, pydantic.ValidationError) as e:
        raise DesiredStateDecodeError(f"Invalid key {key!r}: {e}") from e
    if value is None:
        return workflow_id, None
    try:
        desired = DesiredWorkflow.model_validate_json(value)
    except pydantic.ValidationError as e:
        raise DesiredStateDecodeError(f"Invalid value for '{workflow_id}': {e}") from e
    if desired.workflow_id != workflow_id:
        raise DesiredStateDecodeError(
            f"Key '{workflow_id}' disagrees with value '{desired.workflow_id}'"
        )
    return workflow_id, desired


def fold_updates(
    table: Mapping[WorkflowId, DesiredWorkflow], updates: Iterable[DesiredUpdate]
) -> DesiredTable:
    """Return ``table`` with ``updates`` applied in order; a tombstone deletes."""
    result = dict(table)
    for workflow_id, desired in updates:
        if desired is None:
            result.pop(workflow_id, None)
        else:
            result[workflow_id] = desired
    return result


class DesiredStateReader(Protocol):
    """Reads the desired-state topic from offset 0, then incrementally."""

    def bootstrap(self, timeout_s: float) -> list[DesiredUpdate]:
        """Read every record up to the current end of the topic.

        Raises if the end is not reached within ``timeout_s``: a caller must
        never act on a partial table.
        """

    def poll(self) -> list[DesiredUpdate]:
        """Return the records that arrived since the last call, without blocking."""


class DesiredStateWriter(Protocol):
    """Writes desired-state records."""

    def write(self, workflow_id: WorkflowId, desired: DesiredWorkflow | None) -> None:
        """Write one record (``None`` is a tombstone) and wait until it is stored.

        Raises
        ------
        DesiredStateWriteError
            If the record was not confirmed as stored.
        """


class InMemoryDesiredStateLog:
    """In-process stand-in for the desired-state topic, without compaction.

    Acts as the writer. Any number of readers from :meth:`reader` read the log
    from offset 0. Records are stored encoded, so the codec is exercised as
    with Kafka. Thread-safe.
    """

    def __init__(self) -> None:
        self._records: list[tuple[bytes, bytes | None]] = []
        self._lock = threading.Lock()

    def write(self, workflow_id: WorkflowId, desired: DesiredWorkflow | None) -> None:
        record = encode_desired_record(workflow_id, desired)
        with self._lock:
            self._records.append(record)

    def read(self, offset: int) -> list[DesiredUpdate]:
        """Return the decoded records from ``offset`` to the end of the log."""
        with self._lock:
            records = self._records[offset:]
        return [decode_desired_record(key, value) for key, value in records]

    def reader(self) -> InMemoryDesiredStateReader:
        return InMemoryDesiredStateReader(self)


class InMemoryDesiredStateReader:
    """Reader of an :class:`InMemoryDesiredStateLog`, starting at offset 0."""

    def __init__(self, log: InMemoryDesiredStateLog) -> None:
        self._log = log
        self._offset = 0

    def bootstrap(self, timeout_s: float) -> list[DesiredUpdate]:
        return self.poll()

    def poll(self) -> list[DesiredUpdate]:
        updates = self._log.read(self._offset)
        self._offset += len(updates)
        return updates
