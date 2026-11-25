# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

from .views import DETECTOR_BANK_SIZES


def setup_factories(instrument: Instrument) -> None:
    """Initialize MAGIC-specific factories and workflows.

    Logical view factories are attached automatically by ``load_factories``,
    which also reads each bank's ``detector_number`` from the geometry file;
    here we supply the cylinder-Y mantle projection factory.
    """
    from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
    from ess.livedata.workflows.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.workflows.detector_view_specs import (
        TOAOnlyDetectorViewParams,
    )
    from ess.livedata.workflows.stream_processor_workflow import StreamProcessorWorkflow

    from . import specs

    _pixel_noise = sc.scalar(2.0, unit='mm')
    # One screen bin per strip and per segment. The banks are radially deep (32
    # wires over ~0.5 m), so the mantle projection scales each pixel's y by
    # r_min/r and the wires land at slightly different arc positions; bin counts
    # that are not commensurate with the strip/segment structure alias into a
    # moire of empty bins.
    _view_config = {
        source_name: GeometricViewConfig(
            projection_type='cylinder_mantle_y',
            resolution={'y': sizes['strip'], 'arc_length': sizes['segment']},
            pixel_noise=_pixel_noise,
        )
        for source_name, sizes in DETECTOR_BANK_SIZES.items()
    }

    @specs.projection_handle.attach_factory()
    def _projection_factory(
        source_name: str, params: TOAOnlyDetectorViewParams
    ) -> StreamProcessorWorkflow:
        # Resolve the geometry file lazily so logical views and service startup
        # do not depend on it being registered yet.
        factory = DetectorViewFactory(
            data_source=NeXusDetectorSource(get_nexus_geometry_filename('magic')),
            view_config=_view_config,
        )
        return factory.make_workflow(source_name, params)
