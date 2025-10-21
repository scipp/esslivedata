# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM instrument configuration.

This module provides lightweight spec registration for frontend use.
Backend services must explicitly import .factories to attach implementations.
"""

from . import specs

__all__ = ['specs']
