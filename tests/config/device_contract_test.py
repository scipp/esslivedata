# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the registry-derived NICOS device contract.

Uses real instrument registries and real workflow specs; no mocking. The
contract is derived from ``WorkflowSpec.device_outputs``, so the failure modes
exercised here are device-name collisions and bad templates -- registry drift is
impossible by construction.
"""

from __future__ import annotations

from importlib import resources

import pytest
import scipp as sc
from pydantic import Field, ValidationError

from ess.livedata.config.device_contract import (
    CONTRACT_FILENAME,
    DeviceContract,
    DeviceContractEntry,
    DeviceContractError,
)
from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import (
    REDUCTION,
    WorkflowId,
    WorkflowOutputsBase,
    WorkflowSpec,
)

MONITOR_WORKFLOW = 'dummy/monitor_histogram/1'


@pytest.fixture(scope='module')
def dummy_instrument() -> Instrument:
    """Real dummy instrument with workflow specs registered."""
    get_config('dummy')  # Imports spec module, registering workflow specs.
    return instrument_registry['dummy']


def _entry(**overrides: str) -> DeviceContractEntry:
    fields = {
        'workflow_id': MONITOR_WORKFLOW,
        'source_name': 'monitor1',
        'output_name': 'counts_total_cumulative',
        'device_name': 'monitor1_counts_total',
    }
    fields.update(overrides)
    return DeviceContractEntry(**fields)


def test_from_instrument_derives_monitor_devices(
    dummy_instrument: Instrument,
) -> None:
    # Every instrument's cumulative monitor total is a device by the shared
    # default on the monitor workflow registration.
    contract = DeviceContract.from_instrument(dummy_instrument)
    wid = WorkflowId.from_string(MONITOR_WORKFLOW)
    assert contract.is_device(wid, 'monitor1', 'counts_total_cumulative')
    assert (
        contract.device_name(wid, 'monitor1', 'counts_total_cumulative')
        == 'monitor1_counts_total'
    )


def test_is_device_negative(dummy_instrument: Instrument) -> None:
    contract = DeviceContract.from_instrument(dummy_instrument)
    wid = WorkflowId.from_string(MONITOR_WORKFLOW)
    # 'cumulative' is a real output but not a declared device.
    assert not contract.is_device(wid, 'monitor1', 'cumulative')
    assert contract.device_name(wid, 'monitor1', 'cumulative') is None
    # Source the workflow does not run on.
    assert not contract.is_device(wid, 'no_such_monitor', 'counts_total_cumulative')


def test_from_registry_empty() -> None:
    assert len(DeviceContract.from_registry({})) == 0


def test_entries_preserve_order() -> None:
    contract = DeviceContract(
        [_entry(), _entry(source_name='monitor2', device_name='monitor2_counts_total')]
    )
    assert [e.device_name for e in contract] == [
        'monitor1_counts_total',
        'monitor2_counts_total',
    ]


def test_duplicate_key_raises() -> None:
    with pytest.raises(DeviceContractError, match='Duplicate device-contract key'):
        DeviceContract([_entry(), _entry(device_name='other_name')])


def test_duplicate_device_name_raises() -> None:
    with pytest.raises(DeviceContractError, match='Duplicate device_name'):
        DeviceContract([_entry(), _entry(source_name='monitor2')])


class _ScalarOutputs(WorkflowOutputsBase):
    counts_total_cumulative: sc.DataArray = Field(
        default_factory=lambda: sc.DataArray(sc.scalar(0, unit='counts')),
        title='Total',
        description='Total counts.',
    )


def _spec(*, source_names: list[str], device_outputs: dict[str, str]) -> WorkflowSpec:
    return WorkflowSpec(
        instrument='test',
        name='wf',
        version=1,
        title='WF',
        description='',
        params=None,
        outputs=_ScalarOutputs,
        device_outputs=device_outputs,
        source_names=source_names,
        group=REDUCTION,
    )


def test_device_outputs_validator_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match='unknown output field'):
        _spec(source_names=['a'], device_outputs={'no_such': '{source_name}_x'})


def test_literal_template_over_multiple_sources_collides() -> None:
    spec = _spec(
        source_names=['a', 'b'],
        device_outputs={'counts_total_cumulative': 'fixed_name'},
    )
    with pytest.raises(DeviceContractError, match='Duplicate device_name'):
        DeviceContract.from_registry({spec.get_id(): spec})


def test_bad_placeholder_in_template_raises() -> None:
    spec = _spec(
        source_names=['a'],
        device_outputs={'counts_total_cumulative': '{nope}_x'},
    )
    with pytest.raises(DeviceContractError, match='placeholder'):
        DeviceContract.from_registry({spec.get_id(): spec})


@pytest.mark.parametrize('name', available_instruments())
def test_committed_export_is_in_sync(name: str) -> None:
    # The committed device_contract.yaml is a generated export; a drift here means
    # someone changed device_outputs without rerunning
    # ``python -m ess.livedata.config.device_contract``.
    get_config(name)
    contract = DeviceContract.from_instrument(instrument_registry[name])
    path = resources.files(f'ess.livedata.config.instruments.{name}').joinpath(
        CONTRACT_FILENAME
    )
    if len(contract) == 0:
        assert not path.is_file()
    else:
        assert path.read_text() == contract.as_yaml(name)
