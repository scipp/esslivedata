# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument configuration package.
"""

from .factories import setup_factories
from .specs import multiblade_view_handle
from .streams import detector_fakes, stream_mapping

__all__ = [
    'detector_fakes',
    'multiblade_view_handle',
    'setup_factories',
    'stream_mapping',
]
