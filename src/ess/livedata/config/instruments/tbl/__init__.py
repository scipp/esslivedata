# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL (Test Beamline) instrument configuration.

This module provides lightweight spec registration for frontend use.
Backend services must explicitly import .streams for stream mapping configuration.
"""

from . import specs

__all__ = ['specs']
