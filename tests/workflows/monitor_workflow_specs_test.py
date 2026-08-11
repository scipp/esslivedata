# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest

from ess.livedata.config.device_contract import DeviceContract
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import AuxSources
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    WavelengthMonitorDataParams,
    register_monitor_wavelength_workflow_specs,
    register_monitor_workflow_specs,
)
from ess.livedata.workflows.workflow_factory import SpecHandle


class TestRegisterMonitorWorkflowSpecs:
    def test_returns_none_for_empty_source_names(self):
        instrument = Instrument(name="test")
        assert register_monitor_workflow_specs(instrument, []) is None

    def test_returns_spec_handle(self):
        instrument = Instrument(name="test")
        handle = register_monitor_workflow_specs(instrument, ['monitor_1'])
        assert isinstance(handle, SpecHandle)

    def test_registers_without_aux_sources_by_default(self):
        instrument = Instrument(name="test")
        handle = register_monitor_workflow_specs(instrument, ['monitor_1'])
        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is None

    def test_registers_with_aux_sources(self):
        instrument = Instrument(name="test")
        aux = AuxSources({'position': 'trans_20'})
        handle = register_monitor_workflow_specs(
            instrument, ['monitor_1'], aux_sources=aux
        )
        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.aux_sources is aux


class TestRegisterMonitorWavelengthWorkflowSpecs:
    def test_returns_none_for_empty_source_names(self):
        instrument = Instrument(name="test")
        assert register_monitor_wavelength_workflow_specs(instrument, []) is None

    def test_registers_under_a_distinct_name(self):
        instrument = Instrument(name="test")
        toa = register_monitor_workflow_specs(instrument, ['monitor_1'])
        wavelength = register_monitor_wavelength_workflow_specs(
            instrument, ['monitor_1']
        )
        assert wavelength.workflow_id != toa.workflow_id

    def test_declares_no_device_outputs(self):
        """The NICOS monitor-total device belongs to the TOA spec.

        Declaring it on both would render the same device name twice, which
        DeviceContract rejects.
        """
        instrument = Instrument(name="test")
        handle = register_monitor_wavelength_workflow_specs(instrument, ['monitor_1'])
        spec = instrument.workflow_factory[handle.workflow_id]
        assert spec.device_outputs == {}

    def test_both_specs_coexist_in_the_device_contract(self):
        instrument = Instrument(name="test")
        register_monitor_workflow_specs(instrument, ['monitor_1'])
        register_monitor_wavelength_workflow_specs(instrument, ['monitor_1'])
        contract = DeviceContract.from_instrument(instrument)
        assert [entry.device_name for entry in contract] == ['monitor_1_counts_total']


class TestMonitorCoordinateModeRestriction:
    def test_toa_only_params_reject_wavelength(self):
        with pytest.raises(pydantic.ValidationError):
            TOAOnlyMonitorDataParams(coordinate_mode={'mode': 'wavelength'})

    def test_wavelength_params_reject_toa(self):
        with pytest.raises(pydantic.ValidationError):
            WavelengthMonitorDataParams(coordinate_mode={'mode': 'toa'})

    def test_wavelength_params_default_to_wavelength(self):
        params = WavelengthMonitorDataParams()
        assert params.get_coordinate_mode() == 'wavelength'

    def test_wavelength_params_use_wavelength_edges(self):
        params = WavelengthMonitorDataParams()
        assert params.get_active_edges().unit == 'angstrom'

    def test_wavelength_params_carry_no_toa_fields(self):
        assert 'toa_edges' not in WavelengthMonitorDataParams.model_fields
        assert 'toa_range' not in WavelengthMonitorDataParams.model_fields
