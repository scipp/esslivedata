# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow workflow factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs
from .views import fold_image


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    from ess.livedata.workflows.area_detector_transmission import (
        ImageKeyLog,
        transmission_factory,
    )
    from ess.livedata.workflows.area_detector_view import AreaDetectorView

    specs.orca_view_handle.attach_factory()(
        AreaDetectorView.view_factory(
            transform=fold_image, reduction_dim=['x_bin', 'y_bin']
        )
    )

    specs.orca_transmission_handle.attach_factory()(
        transmission_factory(transform=fold_image, reduction_dim=['x_bin', 'y_bin'])
    )
    specs.orca_transmission_handle.add_context_binding(
        stream_name='orca_detector/image_key', workflow_key=ImageKeyLog
    )
