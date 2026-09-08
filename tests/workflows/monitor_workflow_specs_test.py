# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest

from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import AuxSources
from ess.livedata.workflows.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
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


class TestMonitorCoordinateModeRestriction:
    def test_toa_only_params_reject_wavelength(self):
        with pytest.raises(pydantic.ValidationError):
            TOAOnlyMonitorDataParams(coordinate_mode={'mode': 'wavelength'})

    def test_toa_only_params_carry_no_wavelength_fields(self):
        assert 'wavelength_edges' not in TOAOnlyMonitorDataParams.model_fields
        assert 'wavelength_range' not in TOAOnlyMonitorDataParams.model_fields
