# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""

from typing import NewType

import scipp as sc

from ess.livedata.config import Instrument

from . import specs

SpectrumView = NewType('SpectrumView', sc.DataArray)
SpectrumViewTOAEdges = NewType('SpectrumViewTOAEdges', sc.Variable)


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows."""
    from ess.estia import EstiaWorkflow
    from ess.reduce.nexus.types import NeXusData, RawDetector, SampleRun
    from ess.reduce.streaming import EternalAccumulator
    from scippnexus import NXdetector

    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    def _make_spectrum_view(
        data: RawDetector[SampleRun],
        toa_edges: SpectrumViewTOAEdges,
    ) -> SpectrumView:
        """Create spectrum view with over strip, which has constant scattering angle."""
        edges_ns = toa_edges.to(unit='ns')
        return SpectrumView(
            data.bins.concat('strip')
            .hist(event_time_offset=edges_ns)
            .assign_coords(event_time_offset=toa_edges)
        )

    from ess.reduce.nexus.types import Filename

    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename

    reduction_workflow = EstiaWorkflow()
    reduction_workflow[Filename[SampleRun]] = get_nexus_geometry_filename('estia')
    reduction_workflow.insert(_make_spectrum_view)

    @specs.spectrum_view_handle.attach_factory()
    def _spectrum_view_workflow(
        params: specs.EstiaSpectrumViewParams,
    ) -> StreamProcessorWorkflow:
        wf = reduction_workflow.copy()
        edges = params.toa_edges.get_edges().rename_dims(
            time_of_arrival='event_time_offset'
        )
        wf[SpectrumViewTOAEdges] = edges
        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={'multiblade_detector': NeXusData[NXdetector, SampleRun]},
            target_keys={'spectrum_view': SpectrumView},
            accumulators={SpectrumView: EternalAccumulator},
        )
