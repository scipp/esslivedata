# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
MAGIC instrument factory implementations.
"""

import scipp as sc

from ess.livedata.config import Instrument
from ess.livedata.config.value_log import ValueLog

from .specs import detector_pixel_ranges
from .views import DETECTOR_BANK_SIZES


class DetectorARotationLog(ValueLog):
    """Per-binding Sciline key for the ``magic_detector_a`` rotation readback."""


class DetectorBRotationLog(ValueLog):
    """Per-binding Sciline key for the ``magic_detector_b`` rotation readback."""


#: Each bank hangs off its own rotation stage, so its placement is only known
#: once the live motor readback arrives. Distinct ``ValueLog`` subclasses keep
#: the two dynamic transforms distinguishable in Sciline.
_rotation_logs = {
    'magic_detector_a': ('detector_a_rotation', DetectorARotationLog),
    'magic_detector_b': ('detector_b_rotation', DetectorBRotationLog),
}


def setup_factories(instrument: Instrument) -> None:
    """Initialize MAGIC-specific factories and workflows.

    Logical view factories are attached automatically by ``load_factories``;
    here we supply the detector_number arrays the views fold over and the
    cylinder-Y mantle projection factory.
    """
    from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
    from ess.livedata.workflows.detector_view import (
        DetectorViewFactory,
        GeometricViewConfig,
        NeXusDetectorSource,
    )
    from ess.livedata.workflows.detector_view_specs import DetectorViewParams
    from ess.livedata.workflows.stream_processor_workflow import StreamProcessorWorkflow

    from . import specs

    for name, (first, last) in detector_pixel_ranges.items():
        instrument.configure_detector(
            name,
            detector_number=sc.arange(
                'detector_number', first, last + 1, unit=None, dtype='int32'
            ),
        )

    for source_name, (stream_name, workflow_key) in _rotation_logs.items():
        instrument.add_context_binding(
            stream_name=stream_name,
            dependent_sources={source_name},
            workflow_key=workflow_key,
        )
    # The wire and strip views fold `detector_number` and never consume a
    # position, so they must not wait on the rotation readback.
    specs.wire_view_handle.skip_instrument_contexts()
    specs.strip_view_handle.skip_instrument_contexts()

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
        source_name: str,
        params: DetectorViewParams,
        aux_source_names: dict[str, str],
    ) -> StreamProcessorWorkflow:
        # Resolve the geometry file lazily so logical views and service startup
        # do not depend on it being registered yet.
        factory = DetectorViewFactory(
            data_source=NeXusDetectorSource(get_nexus_geometry_filename('magic')),
            view_config=_view_config,
        )
        return factory.make_workflow(source_name, params, aux_source_names)
