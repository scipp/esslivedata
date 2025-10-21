# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
LOKI instrument configuration.

This module provides lightweight spec registration for frontend use.
Backend services must explicitly import .factories to attach implementations.
"""

from . import specs, streams

# Re-export stream configuration for backward compatibility
detectors_config = streams.detectors_config
stream_mapping = streams.stream_mapping

__all__ = ['detectors_config', 'specs', 'stream_mapping', 'streams']
