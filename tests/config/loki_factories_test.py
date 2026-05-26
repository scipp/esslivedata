# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI ContextInput declarations."""

from ess.livedata.config.instruments.loki import factories, specs
from ess.livedata.config.stream import ChainPatchContextInput


def test_setup_factories_declares_detector_carriage_context_input() -> None:
    """The rear bank's f144 carriage stream is declared at instrument scope.

    Declared as a :class:`ChainPatchContextInput` on ``loki_detector_0`` only;
    specs consuming that source pick it up by default and non-consumers
    (tube_view, i_of_q) opt out via ``skip_motion``.
    """
    instrument = specs.instrument
    factories.setup_factories(instrument)

    matching = [
        ci
        for ci in instrument.context_inputs
        if isinstance(ci, ChainPatchContextInput)
        and ci.stream_name == 'detector_carriage'
        and ci.transform_path == '/entry/instrument/detector_carriage/value'
        and ci.dependent_sources == frozenset({'loki_detector_0'})
        and ci.stream_resolver is None
        and ci.seed_factory is None
    ]
    assert matching, instrument.context_inputs


def test_motion_independent_specs_opt_out_via_skip_motion() -> None:
    """tube_view consumes ``loki_detector_0`` but not its position.

    It must declare ``skip_motion`` so it's not gated on the instrument-scope
    carriage stream. ``xy_projection`` and ``i_of_q`` do consume position and
    must not opt out.
    """
    instrument = specs.instrument

    tube_view_spec = instrument.workflow_factory[
        next(
            wf_id for wf_id in instrument.workflow_factory if wf_id.name == 'tube_view'
        )
    ]
    assert tube_view_spec.skip_motion

    xy_projection_spec = instrument.workflow_factory[
        specs.xy_projection_handle.workflow_id
    ]
    i_of_q_spec = instrument.workflow_factory[specs.i_of_q_handle.workflow_id]
    assert not xy_projection_spec.skip_motion
    assert not i_of_q_spec.skip_motion
