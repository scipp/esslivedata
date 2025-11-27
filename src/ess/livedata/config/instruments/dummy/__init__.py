# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Dummy instrument configuration package.
"""

from .factories import setup_factories
from .streams import area_detector_fakes, detector_fakes, stream_mapping

__all__ = ['area_detector_fakes', 'detector_fakes', 'setup_factories', 'stream_mapping']
