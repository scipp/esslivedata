# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
FREIA instrument configuration package.
"""

from .factories import setup_factories
from .streams import detector_fakes, stream_mapping

__all__ = ['detector_fakes', 'setup_factories', 'stream_mapping']
