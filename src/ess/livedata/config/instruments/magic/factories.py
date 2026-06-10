# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument

#: Contiguous detector_number ranges (inclusive) per bank, matching the
#: NeXus files. The logical views need only this pixel structure, so we provide
#: it directly instead of loading a geometry file.
_detector_number_ranges: dict[str, tuple[int, int]] = {
    'magic_detector_a': (1, 491520),
    'magic_detector_b': (491521, 622592),
}


def setup_factories(instrument: Instrument) -> None:
    """Initialize MAGIC-specific factories and workflows.

    Logical view factories are attached automatically by ``load_factories``;
    here we supply the detector_number arrays the views fold over and the
    cylinder-Y mantle projection factory.
    """
    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    from . import specs
    from .specs import DetectorViewParams

    for name, (start, stop) in _detector_number_ranges.items():
        instrument.configure_detector(
            name,
            detector_number=sc.arange(
                'detector_number', start, stop + 1, unit=None, dtype='int32'
            ),
        )

    _pixel_noise = sc.scalar(2.0, unit='mm')
    _view_config = {
        'magic_detector_a': GeometricViewConfig(
            projection_type='cylinder_mantle_y',
            resolution={'arc_length': 256, 'y': 220},
            pixel_noise=_pixel_noise,
        ),
        'magic_detector_b': GeometricViewConfig(
            projection_type='cylinder_mantle_y',
            resolution={'arc_length': 256, 'y': 40},
            pixel_noise=_pixel_noise,
        ),
    }

    @specs.projection_handle.attach_factory()
    def _projection_factory(
        source_name: str, params: DetectorViewParams
    ) -> StreamProcessorWorkflow:
        # Resolve the geometry file lazily so logical views and service startup
        # do not depend on it being registered yet.
        factory = DetectorViewFactory(
            data_source=NeXusDetectorSource(get_nexus_geometry_filename('magic')),
            view_config=_view_config,
        )
        return factory.make_workflow(source_name, params)
