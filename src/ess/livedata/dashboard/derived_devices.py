# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Live derivation of currently-exposed NICOS derived devices.

A NICOS derived device is "live" iff its owning ``(workflow, source)`` job is
running: the backend projects the registry-declared device outputs of running
jobs. The dashboard is read-only -- it derives device-bearing status each refresh
by intersecting the orchestrator's running ``(workflow, source)`` jobs with the
per-instrument :class:`DeviceContract`.

This module is the single source of truth for that derivation, reused by the
device badge, per-output marker, confirmation gate, and overview modal. It is
pure: callers pass the running-sources snapshot (from
:meth:`JobOrchestrator.get_running_workflow_sources`) plus the contract and the
workflow registry; nothing here reads mutable state.

The contract keys outputs by pydantic *field name*, whereas the dashboard
renders outputs as user-facing :class:`OutputView` s. A view maps to one or
more backing fields via ``OutputView.streams``; an output view "is a device"
when any of its backing fields is in the contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ess.livedata.config.device_contract import DeviceContract
from ess.livedata.config.workflow_spec import OutputView, WorkflowId


@dataclass(frozen=True)
class ContractDevice:
    """A NICOS derived device declared in the contract.

    Identity only -- whether it is currently *live* (its owning job running) is
    derived separately via :func:`is_running_in`.

    Parameters
    ----------
    device_name:
        NICOS device name from the contract.
    workflow_id:
        Owning workflow.
    source_name:
        Data source whose running job exposes the device.
    output_name:
        Pydantic field name of the backing output (the contract key).
    """

    device_name: str
    workflow_id: WorkflowId
    source_name: str
    output_name: str

    def is_running_in(self, running_sources: Mapping[WorkflowId, set[str]]) -> bool:
        """Whether this device's owning ``(workflow, source)`` job is running."""
        return self.source_name in running_sources.get(self.workflow_id, set())


def all_devices(contract: DeviceContract) -> list[ContractDevice]:
    """Every device the contract declares, in declaration order.

    Static: independent of run state.
    """
    return [
        ContractDevice(
            device_name=entry.device_name,
            workflow_id=WorkflowId.from_string(entry.workflow_id),
            source_name=entry.source_name,
            output_name=entry.output_name,
        )
        for entry in contract
    ]


def exposed_devices(
    running_sources: Mapping[WorkflowId, set[str]],
    contract: DeviceContract,
) -> list[ContractDevice]:
    """Return the devices currently exposed = contract ∩ running jobs.

    A contract entry is exposed when its owning ``(workflow_id, source_name)``
    has a running job. Results are returned in contract declaration order.

    Parameters
    ----------
    running_sources:
        Mapping from workflow ID to the set of source names with a running job
        (from :meth:`JobOrchestrator.get_running_workflow_sources`).
    contract:
        The instrument's device contract.

    Returns
    -------
    :
        Exposed devices, in contract declaration order.
    """
    return [
        device
        for device in all_devices(contract)
        if device.is_running_in(running_sources)
    ]


def declared_device_names(
    workflow_id: WorkflowId, contract: DeviceContract
) -> list[str]:
    """All device names the contract declares for ``workflow_id`` (any source).

    Static: independent of run state. Used by the header badge, which marks that
    a workflow feeds NICOS regardless of whether its job is currently running.
    """
    return [
        device.device_name
        for device in all_devices(contract)
        if device.workflow_id == workflow_id
    ]


def affected_device_names(
    workflow_id: WorkflowId,
    running_sources: Mapping[WorkflowId, set[str]],
    contract: DeviceContract,
) -> list[str]:
    """Names of the workflow's currently-exposed devices, for naming in the gate.

    Returns the device names a disruptive action (commit/stop/reset) on
    ``workflow_id`` would affect: the exposed devices owned by this workflow,
    in contract declaration order.
    """
    return [
        device.device_name
        for device in exposed_devices(running_sources, contract)
        if device.workflow_id == workflow_id
    ]


def view_device_names(
    workflow_id: WorkflowId,
    source_names: set[str],
    view: OutputView,
    contract: DeviceContract,
) -> list[str]:
    """Device names a given output view maps to, across the given sources.

    A view "is a device" on a source when any of its backing fields (the
    pydantic field names in ``view.streams``) is an in-contract output for that
    ``(workflow, source)``. Used by the per-output marker: a non-empty result
    means the output chip carries a device.

    Parameters
    ----------
    workflow_id:
        Owning workflow.
    source_names:
        Sources to test (e.g. a workflow's running sources).
    view:
        The output view being rendered.
    contract:
        The instrument's device contract.

    Returns
    -------
    :
        Device names this view corresponds to, in sorted order without
        duplicates.
    """
    names = {
        name
        for source_name in source_names
        for field_name in view.streams.values()
        if (name := contract.device_name(workflow_id, source_name, field_name))
        is not None
    }
    return sorted(names)
