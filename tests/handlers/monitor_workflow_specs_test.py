# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.workflow_spec import AuxSources
from ess.livedata.handlers.monitor_workflow_specs import (
    register_monitor_workflow_specs,
)
from ess.livedata.handlers.workflow_factory import SpecHandle


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
