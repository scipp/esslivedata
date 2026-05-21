# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for LOKI instrument-level ContextInput declarations."""

from ess.livedata.config.instruments.loki import factories, specs


def test_setup_factories_declares_detector_carriage_context_input() -> None:
    """The LOKI rear bank's f144 carriage stream is declared as a context input.

    ADR 0003 §"LOKI transform_name carrier" requires the f144 carriage readback
    to be declared at the instrument level so the routing-pickup extension (B7)
    subscribes it and the gate (B4 onwards) waits on it for ``loki_detector_0``.
    """
    from ess.livedata.handlers.detector_view.types import TransformValueLog

    instrument = specs.instrument
    factories.setup_factories(instrument)

    # setup_factories may run more than once across the test session (the
    # singleton instrument is shared); assert presence rather than count.
    matching = [
        ci
        for ci in instrument.context_inputs
        if ci.stream_name == 'detector_carriage'
        and ci.workflow_key is TransformValueLog
        and ci.dependent_sources == frozenset({'loki_detector_0'})
        and ci.stream_resolver is None
        and ci.seed_factory is None
    ]
    assert matching, instrument.context_inputs
