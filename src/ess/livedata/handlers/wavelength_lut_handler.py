# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Preprocessor factory for the wavelength lookup-table service."""

from __future__ import annotations

from ..core.handler import Accumulator, JobBasedPreprocessorFactoryBase
from ..core.message import StreamId, StreamKind
from .to_nxlog import ToNXlog
from .wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE

#: Attribute hints used by ToNXlog for the synthetic chopper-cascade signal.
#: Treated as a unitless 0/1 ("at setpoint") f144-style log.
_CHOPPER_CASCADE_ATTRS = {'units': ''}


class WavelengthLutHandlerFactory(JobBasedPreprocessorFactoryBase):
    """Preprocessors for the ``wavelength_lut`` service.

    The only stream consumed is the synthetic ``chopper_cascade`` log emitted
    by :class:`ChopperSynthesizer`. It is accumulated as a context-style
    NXlog so the latest value is replayed when a job activates.
    """

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        if key.kind == StreamKind.LOG and key.name == CHOPPER_CASCADE_SOURCE:
            return ToNXlog(attrs=_CHOPPER_CASCADE_ATTRS)
        return None
