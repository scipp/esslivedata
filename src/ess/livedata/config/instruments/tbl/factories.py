# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
TBL workflow workflow factory implementations.
"""

from ess.livedata.config import Instrument

from . import specs
from .specs import Timepix3DetectorViewParams
from .views import fold_image, fold_timepix3_image


def setup_factories(instrument: Instrument) -> None:
    """Initialize TBL-specific factories and workflows."""
    from ess.livedata.workflows.area_detector_view import AreaDetectorView
    from ess.livedata.workflows.detector_view import (
        DetectorViewFactory,
        InstrumentDetectorSource,
        LogicalViewConfig,
    )
    from ess.livedata.workflows.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    specs.orca_view_handle.attach_factory()(
        AreaDetectorView.view_factory(
            transform=fold_image, reduction_dim=['x_bin', 'y_bin']
        )
    )

    @specs.timepix3_view_handle.attach_factory()
    def _timepix3_view_factory(
        source_name: str,
        params: Timepix3DetectorViewParams,
        aux_source_names: dict[str, str],
    ) -> StreamProcessorWorkflow:
        """Timepix3 view, cut to the resolution mode selected in the params.

        The resolution shapes the transform, so the view config cannot be shared
        across jobs the way the other instruments' detector views share theirs.
        """
        resolution = params.resolution.value
        factory = DetectorViewFactory(
            data_source=InstrumentDetectorSource(instrument),
            view_config=LogicalViewConfig(
                transform=lambda da, _source_name: fold_timepix3_image(da, resolution),
                reduction_dim=['x_bin', 'y_bin'],
            ),
        )
        return factory.make_workflow(source_name, params, aux_source_names)
