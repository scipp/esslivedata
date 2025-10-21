# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument spec registration.
"""

from ess.livedata.config import Instrument, instrument_registry

# Create instrument
instrument = Instrument(name='tbl')

# Register instrument
instrument_registry.register(instrument)
