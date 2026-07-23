# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TYPE_CHECKING

import scipp as sc
import structlog

from ..core.message import StreamId, StreamKind
from ..core.preprocessor import Accumulator, JobBasedPreprocessorFactoryBase
from ..workflows.wavelength_lut_workflow_specs import CHOPPER_CASCADE_SOURCE
from .accumulators import LogData
from .to_nxlog import ToNXlog, nxlog_for_stream

if TYPE_CHECKING:
    from ..config.instrument import Instrument

logger = structlog.get_logger(__name__)

#: Source names for synthetic LOG streams that are not real upstream f144s and
#: are therefore not declared in :attr:`Instrument.streams`. The
#: ``chopper_cascade`` tick from :class:`ChopperSynthesizer` is the only such
#: stream today; its value is a boolean "at setpoint" indicator and carries no
#: unit (distinct from ``'dimensionless'``, which a number with cancelled units
#: would have).
_SYNTHETIC_LOG_SOURCES: frozenset[str] = frozenset({CHOPPER_CASCADE_SOURCE})


class LogdataPreprocessorFactory(
    JobBasedPreprocessorFactoryBase[LogData, sc.DataArray]
):
    """
    Factory for creating preprocessors for log data.

    Stream metadata (units etc.) is read from :attr:`Instrument.streams`. Sources
    in :data:`_SYNTHETIC_LOG_SOURCES` are accepted as unit-less streams without a
    corresponding stream record.
    """

    def __init__(self, *, instrument: Instrument) -> None:
        self._instrument = instrument

    def make_preprocessor(self, key: StreamId) -> Accumulator | None:
        match key.kind:
            case StreamKind.DEVICE:
                return nxlog_for_stream(self._instrument.streams.get(key.name))
            case StreamKind.LOG:
                accumulator = nxlog_for_stream(self._instrument.streams.get(key.name))
                if accumulator is not None:
                    return accumulator
                if key.name in _SYNTHETIC_LOG_SOURCES:
                    return ToNXlog(attrs={})
                logger.warning(
                    "No attributes found for source name '%s'. "
                    "Messages will be dropped.",
                    key.name,
                )
                return None
            case _:
                return None
