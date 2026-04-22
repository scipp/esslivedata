# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""

from ess.livedata.config import Instrument


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows.

    ESTIA currently has no bespoke workflows: the multiblade detector view
    (with its spectrum output) is wired entirely via ``add_logical_view`` in
    ``specs.py``.
    """
