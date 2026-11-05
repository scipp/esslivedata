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
    factory and live reflectometry reduction are attached here.
    """
    from typing import NewType

    import sciline.typing
    import scipp as sc
    from ess.estia import EstiaWorkflow
    from ess.estia import corrections as estia_corrections
    from ess.reduce.nexus.types import NeXusData
    from ess.reflectometry.types import (
        BeamDivergenceLimits,
        CoordTransformationGraph,
        CorrectionsToApply,
        Filename,
        LookupTableRelativeErrorThreshold,
        NeXusDetectorName,
        ProtonCharge,
        QBins,
        ReducibleData,
        SampleRun,
        ThetaBins,
        WavelengthBins,
        YIndexLimits,
        ZIndexLimits,
    )
    from scippnexus import NXdetector

    from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
    from ess.livedata.workflows.stream_processor_workflow import (
        StreamProcessorWorkflow,
    )

    from .live_geometry import insert_live_geometry, load_live_geometry

    geometry_filename = get_nexus_geometry_filename('estia')
    live_geometry = load_live_geometry(geometry_filename, instrument)
    for stream, key in live_geometry.context_keys.items():
        instrument.offer_context_stream(stream_name=stream, workflow_key=key)
    specs.reflectometry_reduction_handle.add_context_binding(
        stream_name='source/pulse_charge', workflow_key=ProtonCharge[SampleRun]
    )

    @specs.reflectometry_reduction_handle.attach_factory()
    def _reflectometry_reduction_workflow_factory(
        source_name: str,
        params: specs.EstiaReflectometryReductionParams,
    ) -> StreamProcessorWorkflow:
        wf = EstiaWorkflow()
        wf[Filename[SampleRun]] = geometry_filename
        insert_live_geometry(wf, live_geometry)
        wf[NeXusDetectorName] = source_name
        wf[LookupTableRelativeErrorThreshold] = {source_name: float('inf')}
        wf[WavelengthBins] = params.wavelength_edges.get_edges().to(unit='angstrom')
        wf[QBins] = params.q_edges.get_edges().to(unit='1/angstrom')
        wf[ThetaBins[SampleRun]] = params.theta_edges.get_edges()

        wf[YIndexLimits] = params.y_index_limits.get_limits()
        wf[ZIndexLimits] = params.z_index_limits.get_limits()
        wf[BeamDivergenceLimits] = [
            edge.to(unit='rad') for edge in params.beam_divergence_limits.get_limits()
        ]
        wf[CorrectionsToApply] = estia_corrections.default_corrections
        IntensityThetaWavelength = NewType('IntensityThetaWavelength', sc.DataArray)
        IntensityWavelength = NewType('IntensityWavelength', sc.DataArray)
        IntensityQ = NewType('IntensityQ', sc.DataArray)

        def _intensity_over_theta_wavelength(
            events: ReducibleData[SampleRun],
            wavelength_bins: WavelengthBins,
            theta_bins: ThetaBins[SampleRun],
        ) -> IntensityThetaWavelength:
            da = events.hist(
                theta=theta_bins.to(unit='rad'),
                wavelength=wavelength_bins,
                dim=events.dims,
            ).transpose(['theta', 'wavelength'])
            da = da.assign_coords(theta=da.coords['theta'].to(unit='deg'))
            # Per-chunk geometry cannot describe an accumulated histogram.
            da = da.drop_coords(
                [name for name in da.coords if name not in ('theta', 'wavelength')]
            )
            return IntensityThetaWavelength(da)

        def _intensity_over_q(
            intensity: IntensityThetaWavelength,
            q_bins: QBins,
            graph: CoordTransformationGraph[SampleRun],
        ) -> IntensityQ:
            return IntensityQ(
                intensity.assign_coords(
                    wavelength=sc.midpoints(intensity.coords['wavelength']),
                    theta=sc.midpoints(intensity.coords['theta'].to(unit='rad')),
                )
                .transform_coords('Q', graph=graph)
                .hist(Q=q_bins)
            )

        def _intensity_over_wavelength(
            intensity: IntensityThetaWavelength,
        ) -> IntensityWavelength:
            return IntensityWavelength(intensity.sum('theta'))

        wf.insert(_intensity_over_wavelength)
        wf.insert(_intensity_over_q)
        wf.insert(_intensity_over_theta_wavelength)

        target_keys: dict[str, sciline.typing.Key] = {
            'i_of_wavelength': IntensityWavelength,
            'i_of_q': IntensityQ,
            'i_of_theta_wavelength': IntensityThetaWavelength,
        }

        return StreamProcessorWorkflow(
            wf,
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            target_keys=target_keys,
            accumulators=(IntensityThetaWavelength,),
        )
