# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-instrument NICOS derived-device contract, derived from the registry.

A NICOS *derived device* is a scalar, cumulative workflow output (e.g.
``counts_total_cumulative``) exposed to NICOS as a device. The mapping

    (WorkflowId, source_name, output_name) -> device_name

is **derived from the workflow registry**: an output is a device iff its
:class:`~ess.livedata.config.workflow_spec.WorkflowSpec` declares it in
``device_outputs``, and the device name is rendered from the declared template
for each of the spec's source names. The registry is the single source of truth,
read directly by the backend (which decides what to project) and the dashboard
(which decides which outputs are devices).

Because the contract is generated from the registry it cannot drift from it: an
output that does not exist, or a source the spec does not declare, simply cannot
appear. The one remaining failure mode -- a template that renders the same
device name twice -- is checked at construction and fails loud.

The per-instrument ``device_contract.yaml`` is a *generated, committed export*
for NICOS, which wants a static, git-tracked device list. Nothing reads it back;
:func:`write_exports` regenerates it and a test guards that it stays in sync.
Run ``python -m ess.livedata.config.device_contract`` to regenerate.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

from .workflow_spec import WorkflowId, WorkflowSpec

if TYPE_CHECKING:
    from .instrument import Instrument

CONTRACT_FILENAME = 'device_contract.yaml'


class DeviceContractError(ValueError):
    """Raised when a device contract renders inconsistent device names."""


def _render_device_name(template: str, source_name: str) -> str:
    """Render a device-name template for a source, failing loud on bad placeholders."""
    try:
        return template.format(source_name=source_name)
    except (KeyError, IndexError) as exc:
        raise DeviceContractError(
            f"Invalid device_name template {template!r}: unknown placeholder {exc}"
        ) from exc


class DeviceContractEntry(BaseModel, frozen=True):
    """A single ``(WorkflowId, source_name, output_name) -> device_name`` mapping.

    Parameters
    ----------
    workflow_id:
        String form of the :class:`WorkflowId`, ``"instrument/name/version"``.
    source_name:
        Data source the workflow runs on (e.g. a monitor or detector name).
    output_name:
        Pydantic field name of the workflow output (e.g.
        ``counts_total_cumulative``), not the user-facing view name.
    device_name:
        NICOS device name this output is exposed as.
    """

    workflow_id: str
    source_name: str
    output_name: str
    device_name: str

    @property
    def key(self) -> tuple[str, str, str]:
        """The ``(workflow_id, source_name, output_name)`` identity tuple."""
        return (self.workflow_id, self.source_name, self.output_name)


class DeviceContract:
    """Validated, immutable per-instrument device contract.

    Construct via :meth:`from_instrument` or :meth:`from_registry`, which derive
    the entries from the workflow registry. The constructor validates device-name
    uniqueness and is the low-level entry point used by those classmethods and by
    tests that build a contract from explicit entries.
    """

    def __init__(self, entries: Iterable[DeviceContractEntry]) -> None:
        self._entries = tuple(entries)
        self._by_key: dict[tuple[str, str, str], DeviceContractEntry] = {}
        seen_devices: dict[str, DeviceContractEntry] = {}
        for entry in self._entries:
            if entry.key in self._by_key:
                raise DeviceContractError(
                    f"Duplicate device-contract key {entry.key} "
                    f"(device names {self._by_key[entry.key].device_name!r} and "
                    f"{entry.device_name!r})"
                )
            if entry.device_name in seen_devices:
                raise DeviceContractError(
                    f"Duplicate device_name {entry.device_name!r} for keys "
                    f"{seen_devices[entry.device_name].key} and {entry.key}"
                )
            self._by_key[entry.key] = entry
            seen_devices[entry.device_name] = entry

    @property
    def entries(self) -> tuple[DeviceContractEntry, ...]:
        """All contract entries, in declaration order."""
        return self._entries

    def __iter__(self) -> Iterator[DeviceContractEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def is_device(
        self, workflow_id: WorkflowId, source_name: str, output_name: str
    ) -> bool:
        """Whether the given output is exposed as a NICOS device."""
        return (str(workflow_id), source_name, output_name) in self._by_key

    def device_name(
        self, workflow_id: WorkflowId, source_name: str, output_name: str
    ) -> str | None:
        """NICOS device name for the given output, or None if not a device."""
        entry = self._by_key.get((str(workflow_id), source_name, output_name))
        return entry.device_name if entry is not None else None

    @classmethod
    def from_registry(
        cls, registry: Mapping[WorkflowId, WorkflowSpec]
    ) -> DeviceContract:
        """Derive the contract from a workflow registry.

        For every spec that declares ``device_outputs``, emit one entry per
        ``(output_name, source_name)`` with the device name rendered from the
        declared template. The result is validated for device-name uniqueness.

        Parameters
        ----------
        registry:
            Workflow registry mapping :class:`WorkflowId` to
            :class:`WorkflowSpec`. The instrument's ``workflow_factory`` satisfies
            this mapping.

        Returns
        -------
        :
            A validated, immutable contract.
        """
        entries = [
            DeviceContractEntry(
                workflow_id=str(workflow_id),
                source_name=source_name,
                output_name=output_name,
                device_name=_render_device_name(template, source_name),
            )
            for workflow_id, spec in registry.items()
            for output_name, template in spec.device_outputs.items()
            for source_name in spec.source_names
        ]
        return cls(entries)

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> DeviceContract:
        """Derive the contract for an :class:`Instrument` from its registry."""
        return cls.from_registry(instrument.workflow_factory)

    def as_yaml(self, instrument_name: str) -> str:
        """Render the NICOS device-list export for ``instrument_name``.

        The export is a flat list of ``device_name -> identity`` entries -- the
        static device list NICOS consumes. It is generated, not authored; see the
        module docstring.
        """
        header = (
            "# SPDX-License-Identifier: BSD-3-Clause\n"
            "# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)\n"
            "#\n"
            f"# GENERATED -- do not edit. NICOS derived-device list for "
            f"{instrument_name}.\n"
            "# Regenerate: python -m ess.livedata.config.device_contract\n"
        )
        body = yaml.safe_dump(
            {
                'devices': [
                    {
                        'device_name': entry.device_name,
                        'workflow_id': entry.workflow_id,
                        'source_name': entry.source_name,
                        'output_name': entry.output_name,
                    }
                    for entry in self._entries
                ]
            },
            sort_keys=False,
        )
        return header + body


def write_exports() -> list[str]:
    """Regenerate the committed ``device_contract.yaml`` export for each instrument.

    Writes one file per instrument that declares any device, into the instrument
    package, and removes a stale file for an instrument that no longer declares
    devices. Returns the instrument names that have a contract.

    Returns
    -------
    :
        Names of instruments with a non-empty contract, in registration order.
    """
    from importlib import resources

    from .instrument import instrument_registry
    from .instruments import available_instruments, get_config

    written: list[str] = []
    for name in available_instruments():
        get_config(name)
        instrument = instrument_registry[name]
        contract = DeviceContract.from_instrument(instrument)
        path = resources.files(f'ess.livedata.config.instruments.{name}').joinpath(
            CONTRACT_FILENAME
        )
        if len(contract) == 0:
            if path.is_file():
                path.unlink()  # type: ignore[attr-defined]
            continue
        path.write_text(contract.as_yaml(name))  # type: ignore[attr-defined]
        written.append(name)
    return written


if __name__ == '__main__':
    for instrument_name in write_exports():
        print(f'Wrote {CONTRACT_FILENAME} for {instrument_name}')  # noqa: T201
