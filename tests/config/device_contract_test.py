# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the per-instrument NICOS device contract.

Uses the real ``dummy`` instrument registry and real workflow specs; no
mocking. Validation failure modes are exercised against this live registry.
"""

from __future__ import annotations

import pytest
import scipp as sc

from ess.livedata.config.device_contract import (
    DeviceContract,
    DeviceContractEntry,
    DeviceContractError,
    is_scalar_cumulative,
)
from ess.livedata.config.instrument import Instrument, instrument_registry
from ess.livedata.config.instruments import get_config
from ess.livedata.config.workflow_spec import WorkflowId

MONITOR_WORKFLOW = 'dummy/monitor_histogram/1'
TOTAL_COUNTS_WORKFLOW = 'dummy/total_counts/1'


@pytest.fixture(scope='module')
def dummy_instrument() -> Instrument:
    """Real dummy instrument with workflow specs registered."""
    get_config('dummy')  # Imports spec module, registering workflow specs.
    return instrument_registry['dummy']


@pytest.fixture
def registry(dummy_instrument: Instrument):
    return dummy_instrument.workflow_factory


def _entry(**overrides: str) -> DeviceContractEntry:
    fields = {
        'workflow_id': MONITOR_WORKFLOW,
        'source_name': 'monitor1',
        'output_name': 'counts_total_cumulative',
        'device_name': 'monitor1_counts_total',
    }
    fields.update(overrides)
    return DeviceContractEntry(**fields)


def test_is_scalar_cumulative_true_for_0d_without_time() -> None:
    da = sc.DataArray(sc.scalar(0, unit='counts'))
    assert is_scalar_cumulative(da)


def test_is_scalar_cumulative_false_for_0d_with_time() -> None:
    da = sc.DataArray(sc.scalar(0, unit='counts'), coords={'time': sc.scalar(0)})
    assert not is_scalar_cumulative(da)


def test_is_scalar_cumulative_false_for_1d() -> None:
    da = sc.DataArray(sc.zeros(dims=['x'], shape=[3], unit='counts'))
    assert not is_scalar_cumulative(da)


@pytest.fixture
def sample_contract(registry) -> DeviceContract:
    """A two-device contract built against the real dummy registry."""
    entries = [
        _entry(),
        _entry(
            workflow_id=TOTAL_COUNTS_WORKFLOW,
            source_name='panel_0',
            output_name='total_counts',
            device_name='panel_0_counts_total',
        ),
    ]
    return DeviceContract.from_entries(entries, registry)


def test_dummy_ships_no_contract(dummy_instrument: Instrument) -> None:
    # The dummy instrument ships no device_contract.yaml; a contract is a
    # deliberate per-instrument act, not a default.
    contract = DeviceContract.from_instrument(dummy_instrument)
    assert len(contract) == 0


def test_is_device_and_device_name_positive(sample_contract: DeviceContract) -> None:
    contract = sample_contract
    wid = WorkflowId.from_string(MONITOR_WORKFLOW)
    assert contract.is_device(wid, 'monitor1', 'counts_total_cumulative')
    assert (
        contract.device_name(wid, 'monitor1', 'counts_total_cumulative')
        == 'monitor1_counts_total'
    )


def test_is_device_and_device_name_negative(sample_contract: DeviceContract) -> None:
    contract = sample_contract
    wid = WorkflowId.from_string(MONITOR_WORKFLOW)
    # 'cumulative' is a real output but not a contracted device.
    assert not contract.is_device(wid, 'monitor1', 'cumulative')
    assert contract.device_name(wid, 'monitor1', 'cumulative') is None
    # Wrong source.
    assert not contract.is_device(wid, 'monitor2', 'counts_total_cumulative')


def test_entries_preserve_order(registry) -> None:
    entries = [
        _entry(),
        _entry(
            workflow_id=TOTAL_COUNTS_WORKFLOW,
            source_name='panel_0',
            output_name='total_counts',
            device_name='panel_0_counts_total',
        ),
    ]
    contract = DeviceContract.from_entries(entries, registry)
    assert [e.device_name for e in contract] == [
        'monitor1_counts_total',
        'panel_0_counts_total',
    ]


def test_unknown_workflow_name_raises(registry) -> None:
    entry = _entry(workflow_id='dummy/does_not_exist/1')
    with pytest.raises(DeviceContractError, match='Unknown workflow_id'):
        DeviceContract.from_entries([entry], registry)


def test_unknown_workflow_version_raises(registry) -> None:
    entry = _entry(workflow_id='dummy/monitor_histogram/99')
    with pytest.raises(DeviceContractError, match='Unknown workflow_id'):
        DeviceContract.from_entries([entry], registry)


def test_malformed_workflow_id_raises(registry) -> None:
    entry = _entry(workflow_id='not-a-valid-id')
    with pytest.raises(DeviceContractError, match='Invalid workflow_id'):
        DeviceContract.from_entries([entry], registry)


def test_unknown_source_raises(registry) -> None:
    entry = _entry(source_name='not_a_monitor')
    with pytest.raises(DeviceContractError, match='Unknown source_name'):
        DeviceContract.from_entries([entry], registry)


def test_unknown_output_raises(registry) -> None:
    entry = _entry(output_name='no_such_output')
    with pytest.raises(DeviceContractError, match='Unknown output_name'):
        DeviceContract.from_entries([entry], registry)


def test_non_scalar_output_raises(registry) -> None:
    # 'cumulative' is a 1-D histogram output: not scalar-cumulative.
    entry = _entry(output_name='cumulative')
    with pytest.raises(DeviceContractError, match='not scalar-cumulative'):
        DeviceContract.from_entries([entry], registry)


def test_time_coord_output_raises(registry) -> None:
    # 'counts_total' is 0-D but carries a 'time' coord (per-update timeseries).
    entry = _entry(output_name='counts_total')
    with pytest.raises(DeviceContractError, match='not scalar-cumulative'):
        DeviceContract.from_entries([entry], registry)


def test_duplicate_key_raises(registry) -> None:
    entries = [_entry(), _entry(device_name='other_name')]
    with pytest.raises(DeviceContractError, match='Duplicate device-contract key'):
        DeviceContract.from_entries(entries, registry)


def test_duplicate_device_name_raises(registry) -> None:
    entries = [
        _entry(),
        _entry(
            workflow_id=TOTAL_COUNTS_WORKFLOW,
            source_name='panel_0',
            output_name='total_counts',
        ),
    ]
    with pytest.raises(DeviceContractError, match='Duplicate device_name'):
        DeviceContract.from_entries(entries, registry)


def test_missing_contract_file_yields_empty(registry) -> None:
    # 'dream' has no device_contract.yaml; loading must yield an empty contract.
    get_config('dream')
    instrument = instrument_registry['dream']
    contract = DeviceContract.from_instrument(instrument)
    assert len(contract) == 0


def test_bifrost_contract_loads_and_validates() -> None:
    # Guards the shipped bifrost contract against workflow-spec drift: every
    # entry must resolve against the live registry and be scalar-cumulative.
    get_config('bifrost')
    instrument = instrument_registry['bifrost']
    contract = DeviceContract.from_instrument(instrument)
    wid = WorkflowId.from_string('bifrost/monitor_histogram/1')
    monitors = {entry.source_name for entry in contract}
    assert monitors == {
        'psc_monitor',
        'overlap_monitor',
        'bandwidth_monitor',
        'normalization_monitor',
        'elastic_monitor',
    }
    assert all(
        contract.is_device(wid, monitor, 'counts_total_cumulative')
        for monitor in monitors
    )
