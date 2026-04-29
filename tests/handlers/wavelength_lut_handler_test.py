# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the wavelength-lut preprocessor factory."""

from __future__ import annotations

from ess.livedata.config.instrument import Instrument
from ess.livedata.core.message import StreamId, StreamKind
from ess.livedata.handlers.to_nxlog import ToNXlog
from ess.livedata.handlers.wavelength_lut_handler import WavelengthLutHandlerFactory
from ess.livedata.handlers.wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE


def _factory() -> WavelengthLutHandlerFactory:
    return WavelengthLutHandlerFactory(instrument=Instrument(name='test'))


def test_make_preprocessor_for_chopper_cascade_returns_context_nxlog():
    acc = _factory().make_preprocessor(
        StreamId(kind=StreamKind.LOG, name=CHOPPER_CASCADE_SOURCE)
    )
    assert isinstance(acc, ToNXlog)
    assert acc.is_context is True


def test_make_preprocessor_for_other_log_stream_returns_none():
    acc = _factory().make_preprocessor(
        StreamId(kind=StreamKind.LOG, name='something_else')
    )
    assert acc is None


def test_make_preprocessor_for_chopper_cascade_with_wrong_kind_returns_none():
    acc = _factory().make_preprocessor(
        StreamId(kind=StreamKind.MONITOR_COUNTS, name=CHOPPER_CASCADE_SOURCE)
    )
    assert acc is None
