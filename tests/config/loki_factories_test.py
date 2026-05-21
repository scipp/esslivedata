# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI spec-level ContextInput declarations."""

from ess.livedata.config.instruments.loki import factories, specs


def test_setup_factories_declares_detector_carriage_context_input() -> None:
    """The LOKI rear bank's f144 carriage stream is declared as a context input.

    Per ADR 0003 § "Instrument bindings are source-scoped", the carriage stream
    is declared on the ``xy_projection`` spec rather than at instrument scope:
    ``loki_detector_0`` also feeds ``tube_view`` (logical sum, no absolute-
    position need) and other workflows that do not consume carriage. The
    spec-level declaration scopes the gate precisely to the geometric view.
    """
    from ess.livedata.handlers.detector_view.types import TransformValueLog

    instrument = specs.instrument
    factories.setup_factories(instrument)

    spec = instrument.workflow_factory[specs.xy_projection_handle.workflow_id]
    # setup_factories may run more than once across the test session (the
    # singleton instrument is shared); assert presence rather than count.
    matching = [
        ci
        for ci in spec.context_inputs
        if ci.stream_name == 'detector_carriage'
        and ci.workflow_key is TransformValueLog
        and ci.dependent_sources == frozenset({'loki_detector_0'})
        and ci.stream_resolver is None
        and ci.seed_factory is None
    ]
    assert matching, spec.context_inputs
