# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL (Test Beamline) instrument configuration.
"""

from . import specs, streams

# Re-export stream configuration for backward compatibility
detectors_config = streams.detectors_config
stream_mapping = streams.stream_mapping

__all__ = ['detectors_config', 'specs', 'stream_mapping', 'streams']
