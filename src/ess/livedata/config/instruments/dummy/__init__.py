# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument configuration package.
"""

from . import streams
from .factories import setup_factories

__all__ = ['setup_factories', 'streams']
