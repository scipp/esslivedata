# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL (Test Beamline) instrument configuration.
"""

from . import streams
from .factories import setup_factories

__all__ = ['setup_factories', 'streams']
