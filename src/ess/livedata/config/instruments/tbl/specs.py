# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument spec registration (lightweight).

This module registers the TBL instrument WITHOUT heavy dependencies.
Frontend loads this module to access instrument specs.
Backend services must also import .streams for stream mapping configuration.
"""

from ess.livedata.config import Instrument, instrument_registry

# Create instrument
instrument = Instrument(name='tbl')

# Register instrument
instrument_registry.register(instrument)
