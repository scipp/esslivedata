# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL instrument factory implementations.
"""

from ess.livedata.config import instrument_registry

# Get instrument from registry (already registered by specs.py)
instrument = instrument_registry['tbl']

# Future workflow factories can be added here
