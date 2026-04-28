# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument factory implementations.
"""








from ess.livedata.config import Instrument

from . import specs


def setup_factories(instrument: Instrument) -> None:
    """Initialize ESTIA-specific factories and workflows.

    The multiblade detector view (with its spectrum output) is wired via
    ``add_logical_view`` in ``specs.py``. The generic ``cbm`` monitor workflow
    factory and live reflectometry diagnostics are attached here.
    """
    from typing import NewType

    import sciline.typing
    import scipp as sc
    from ess.estia import EstiaWorkflow, data as estia_data
    from ess.reduce.nexus.types import NeXusData
    from ess.reflectometry.types import (
        BeamDivergenceLimits,
        CorrectionsToApply,
        DetectorRotation,
        Filename,
        LookupTableFilename,
        LookupTableRelativeErrorThreshold,
        NeXusDetectorName,
        ProtonCurrent,
        QBins,
        ReducibleData,
        SampleRun,
        SampleRotation,
        ThetaBins,
        WavelengthBins,
        YIndexLimits,
        ZIndexLimits,
    )
    from scippnexus import NXdetector

    from ess.livedata.handlers.detector_data_handler import get_nexus_geometry_filename
    from ess.livedata.handlers.monitor_workflow import create_monitor_workflow
    from ess.livedata.handlers.monitor_workflow_specs import TOAOnlyMonitorDataParams
    from ess.livedata.handlers.stream_processor_workflow import StreamProcessorWorkflow

    IntensityWavelength = NewType('IntensityWavelength', sc.DataArray)
    IntensityQ = NewType('IntensityQ', sc.DataArray)
    IntensityThetaWavelength = NewType('IntensityThetaWavelength', sc.DataArray)

    def _intensity_over_wavelength(
        events: ReducibleData[SampleRun],
        wavelength_bins: WavelengthBins,
    ) -> IntensityWavelength:
        return IntensityWavelength(events.hist(wavelength=wavelength_bins, dim=events.dims))

    def _intensity_over_q(
        events: ReducibleData[SampleRun],
        q_bins: QBins,
    ) -> IntensityQ:
        return IntensityQ(events.hist(Q=q_bins, dim=events.dims))

    def _intensity_over_theta_wavelength(
        events: ReducibleData[SampleRun],
        wavelength_bins: WavelengthBins,
        theta_bins: ThetaBins[SampleRun],
    ) -> IntensityThetaWavelength:
        return IntensityThetaWavelength(
            events.hist(theta=theta_bins, wavelength=wavelength_bins, dim=events.dims).transpose(
                ['theta', 'wavelength']
            )
        )

    @specs.monitor_handle.attach_factory()
    def _monitor_workflow_factory(source_name: str, params: TOAOnlyMonitorDataParams):
        return create_monitor_workflow(
            source_name=source_name,
            edges=params.get_active_edges(),
            range_filter=params.get_active_range(),
            coordinate_mode='toa',
        )

    @specs.live_diagnostics_handle.attach_factory()
    def _live_diagnostics_workflow_factory(
        source_name: str,
        params: specs.EstiaLiveDiagnosticsParams,
    ) -> StreamProcessorWorkflow:
        wf = EstiaWorkflow()
        wf.insert(_intensity_over_wavelength)
        wf.insert(_intensity_over_q)
        wf.insert(_intensity_over_theta_wavelength)
        wf[Filename[SampleRun]] = get_nexus_geometry_filename('estia')
        wf[NeXusDetectorName] = source_name
        wf[LookupTableFilename] = estia_data.estia_wavelength_lookup_table()
        wf[LookupTableRelativeErrorThreshold] = {source_name: float('inf')}
        wf[WavelengthBins] = params.wavelength_edges.get_edges()
        wf[QBins] = params.q_edges.get_edges()
        wf[ThetaBins[SampleRun]] = params.theta_edges.get_edges()
        wf[YIndexLimits] = (sc.scalar(0), sc.scalar(63))
        wf[ZIndexLimits] = (sc.scalar(0), sc.scalar(1535))
        wf[BeamDivergenceLimits] = (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        )

        target_keys: dict[str, sciline.typing.Key] = {
            'i_of_wavelength': IntensityWavelength,
            'i_of_q': IntensityQ,
            'i_of_theta_wavelength': IntensityThetaWavelength,
        }

        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            target_keys=target_keys,
            accumulators=(ReducibleData[SampleRun],),
        )
