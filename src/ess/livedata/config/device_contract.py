# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-instrument NICOS derived-device contract.

A NICOS *derived device* is a scalar, cumulative workflow output (e.g.
``counts_total_cumulative``) exposed to NICOS as a device. The mapping

    (WorkflowId, source_name, output_name) -> device_name

is a versioned, per-instrument YAML contract shared by the backend (which
decides what to project onto a Kafka topic) and the dashboard (which decides
which outputs are "devices"). This module owns the loader and the validated
read API.

The contract is validated against the live workflow registry at load time and
fails loud on any inconsistency, so a malformed or stale contract cannot
silently mis-route a device. Which outputs are sensible as devices is the
contract author's responsibility; the loader only checks that each entry
resolves to a real workflow output.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

from .workflow_spec import WorkflowId, WorkflowSpec

if TYPE_CHECKING:
    from .instrument import Instrument

CONTRACT_FILENAME = 'device_contract.yaml'


class DeviceContractError(ValueError):
    """Raised when a device contract is inconsistent with the registry."""


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


class DeviceContractFileEntry(BaseModel):
    """A grouped contract entry: one workflow output across several sources.

    Expands to one :class:`DeviceContractEntry` per source. ``device_name`` is a
    template formatted with ``source_name`` for each source; include the
    ``{source_name}`` placeholder to derive a distinct device name per source
    (required when more than one source is listed, otherwise the names collide).

    Parameters
    ----------
    workflow_id:
        String form of the :class:`WorkflowId`, ``"instrument/name/version"``.
    output_name:
        Pydantic field name of the workflow output (e.g.
        ``counts_total_cumulative``), not the user-facing view name.
    source_names:
        Data sources the workflow runs on (e.g. monitor or detector names).
    device_name:
        NICOS device-name template. A literal (no ``{source_name}``) is allowed
        only for a single source.
    """

    workflow_id: str
    output_name: str
    source_names: list[str] = Field(min_length=1)
    device_name: str


class DeviceContractFile(BaseModel):
    """On-disk model of a device-contract YAML file."""

    entries: list[DeviceContractFileEntry] = []


class DeviceContract:
    """Validated, immutable per-instrument device contract.

    Construct via :meth:`load` or :meth:`from_instrument`; both validate the
    entries against the workflow registry and raise :class:`DeviceContractError`
    on any inconsistency. The constructor is the low-level entry point used by
    the loaders and assumes its arguments are already validated.
    """

    def __init__(self, entries: tuple[DeviceContractEntry, ...]) -> None:
        self._entries = entries
        self._by_key: dict[tuple[str, str, str], DeviceContractEntry] = {
            entry.key: entry for entry in entries
        }

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
    def load(
        cls,
        instrument_name: str,
        registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> DeviceContract:
        """Load and validate the contract for an instrument.

        Reads ``device_contract.yaml`` from the instrument package
        (``ess.livedata.config.instruments.<instrument_name>``). A missing file
        yields an empty contract; a present-but-invalid file raises.

        Parameters
        ----------
        instrument_name:
            Instrument name, used to locate the package and the YAML file.
        registry:
            Workflow registry mapping :class:`WorkflowId` to
            :class:`WorkflowSpec`, used to validate every entry. The
            instrument's ``workflow_factory`` satisfies this mapping.

        Returns
        -------
        :
            A validated, immutable contract.
        """
        from importlib import resources

        package = f'ess.livedata.config.instruments.{instrument_name}'
        resource = resources.files(package).joinpath(CONTRACT_FILENAME)
        if not resource.is_file():
            return cls(())
        contract_file = DeviceContractFile.model_validate(
            yaml.safe_load(resource.read_text()) or {}
        )
        return cls.from_file_entries(contract_file.entries, registry)

    @classmethod
    def from_file_entries(
        cls,
        file_entries: list[DeviceContractFileEntry],
        registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> DeviceContract:
        """Expand grouped file entries to per-source entries and validate.

        Each :class:`DeviceContractFileEntry` expands to one
        :class:`DeviceContractEntry` per source, with ``device_name`` rendered
        from its template; the result is validated by :meth:`from_entries`.
        """
        resolved = [
            DeviceContractEntry(
                workflow_id=file_entry.workflow_id,
                source_name=source_name,
                output_name=file_entry.output_name,
                device_name=_render_device_name(file_entry.device_name, source_name),
            )
            for file_entry in file_entries
            for source_name in file_entry.source_names
        ]
        return cls.from_entries(resolved, registry)

    @classmethod
    def from_instrument(cls, instrument: Instrument) -> DeviceContract:
        """Load and validate the contract for an :class:`Instrument`.

        Convenience wrapper reading ``instrument.name`` and
        ``instrument.workflow_factory`` so callers holding an instrument need
        not unpack them.
        """
        return cls.load(instrument.name, instrument.workflow_factory)

    @classmethod
    def from_entries(
        cls,
        entries: list[DeviceContractEntry],
        registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> DeviceContract:
        """Validate explicit entries against a registry and build a contract.

        Validation fails loud, naming the offending entry, on:

        - a ``workflow_id`` that does not parse or is absent from ``registry``;
        - a ``source_name`` not declared by the spec's ``source_names``;
        - an ``output_name`` absent from the spec's outputs;
        - a duplicate ``(workflow_id, source_name, output_name)`` key;
        - a duplicate ``device_name``.

        Parameters
        ----------
        entries:
            Contract entries to validate.
        registry:
            Workflow registry mapping :class:`WorkflowId` to
            :class:`WorkflowSpec`.

        Returns
        -------
        :
            A validated, immutable contract.
        """
        seen_keys: dict[tuple[str, str, str], DeviceContractEntry] = {}
        seen_devices: dict[str, DeviceContractEntry] = {}
        for entry in entries:
            cls._validate_entry(entry, registry)
            if entry.key in seen_keys:
                raise DeviceContractError(
                    f"Duplicate device-contract key {entry.key} "
                    f"(device names {seen_keys[entry.key].device_name!r} and "
                    f"{entry.device_name!r})"
                )
            if entry.device_name in seen_devices:
                raise DeviceContractError(
                    f"Duplicate device_name {entry.device_name!r} for keys "
                    f"{seen_devices[entry.device_name].key} and {entry.key}"
                )
            seen_keys[entry.key] = entry
            seen_devices[entry.device_name] = entry
        return cls(tuple(entries))

    @staticmethod
    def _validate_entry(
        entry: DeviceContractEntry,
        registry: Mapping[WorkflowId, WorkflowSpec],
    ) -> None:
        try:
            workflow_id = WorkflowId.from_string(entry.workflow_id)
        except ValueError as exc:
            raise DeviceContractError(
                f"Invalid workflow_id {entry.workflow_id!r} in device-contract "
                f"entry for device {entry.device_name!r}: {exc}"
            ) from exc

        spec = registry.get(workflow_id)
        if spec is None:
            raise DeviceContractError(
                f"Unknown workflow_id {entry.workflow_id!r} in device-contract "
                f"entry for device {entry.device_name!r}; not in the registry"
            )

        if spec.source_names and entry.source_name not in spec.source_names:
            raise DeviceContractError(
                f"Unknown source_name {entry.source_name!r} for workflow "
                f"{entry.workflow_id!r} (device {entry.device_name!r}); "
                f"declared sources: {spec.source_names}"
            )

        if entry.output_name not in spec.outputs.model_fields:
            raise DeviceContractError(
                f"Unknown output_name {entry.output_name!r} for workflow "
                f"{entry.workflow_id!r} (device {entry.device_name!r}); "
                f"declared outputs: {sorted(spec.outputs.model_fields)}"
            )
