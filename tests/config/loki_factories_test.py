# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI ContextInput declarations."""

from ess.livedata.config.instruments.loki import factories, specs


def test_setup_factories_declares_detector_carriage_context_input() -> None:
    """The rear bank's f144 carriage stream is declared at instrument scope.

    Declared as a chain-patch :class:`ContextInput` on ``loki_detector_0`` only
    (``workflow_key`` is :class:`specs.DetectorCarriageLog`, a :class:`ValueLog`
    subclass); specs consuming that source pick it up by default and
    non-consumers (tube_view, i_of_q) opt out via ``skip_instrument_contexts``.
    """
    instrument = specs.instrument
    factories.setup_factories(instrument)

    matching = [
        ci
        for ci in instrument.context_inputs
        if ci.stream_name == 'detector_carriage'
        and ci.workflow_key is specs.DetectorCarriageLog
        and ci.dependent_sources == frozenset({'loki_detector_0'})
    ]
    assert matching, instrument.context_inputs
    assert (
        specs.DetectorCarriageLog.transform_path
        == '/entry/instrument/detector_carriage/value'
    )


def test_motion_independent_specs_opt_out_via_skip_instrument_contexts() -> None:
    """tube_view consumes ``loki_detector_0`` but not its position.

    It must declare ``skip_instrument_contexts`` so it's not gated on the
    instrument-scope carriage stream. ``xy_projection`` and ``i_of_q`` do
    consume position and must not opt out.
    """
    instrument = specs.instrument
    factory = instrument.workflow_factory

    tube_view_id = next(wf_id for wf_id in factory if wf_id.name == 'tube_view')
    assert factory.registration(tube_view_id).skip_instrument_contexts

    xy_projection_reg = factory.registration(specs.xy_projection_handle.workflow_id)
    i_of_q_reg = factory.registration(specs.i_of_q_handle.workflow_id)
    assert not xy_projection_reg.skip_instrument_contexts
    assert not i_of_q_reg.skip_instrument_contexts
