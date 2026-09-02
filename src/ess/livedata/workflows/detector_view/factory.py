# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Factory for detector view Sciline workflow creation.

This module provides the DetectorViewScilineFactory for creating detector view
workflows with configurable projection types and parameters.
"""

from __future__ import annotations

import scipp as sc
from ess.reduce.nexus.types import NeXusData, SampleRun
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold
from scippnexus import NXdetector

# Import types unconditionally for runtime type hint resolution
# (used by workflow_factory.attach_factory to inspect parameter types)
from ...config.roi_names import get_roi_mapper, roi_stream_name
from ...config.workflow_spec import Temporality
from ...preprocessors.accumulators import make_no_copy_accumulator_pair
from ...preprocessors.downsample_pixel_ids import SOURCE_RESOLUTION
from ..detector_view_specs import (
    DetectorViewOutputs,
    DetectorViewOutputsBase,
    DetectorViewParamsBase,
)
from ..lut_context import detector_lookup_table
from ..stream_processor_workflow import StreamProcessorWorkflow
from ..workflow_factory import WorkflowFactory
from .data_source import DetectorDataSource, DetectorNumberSource
from .providers import spectrum_view
from .types import (
    DETECTOR_TRANSFORM,
    AccumulatedHistogram,
    CountsInRange,
    CountsTotal,
    Cumulative,
    Current,
    DetectorImage,
    GeometricViewConfig,
    LogicalViewConfig,
    ROIPolygonReadback,
    ROIPolygonRequest,
    ROIRectangleReadback,
    ROIRectangleRequest,
    ROISpectra,
    SpectrumView,
    SpectrumViewTransform,
    UsePixelWeighting,
    ViewConfig,
)
from .workflow import (
    add_geometric_projection,
    add_logical_projection,
    create_base_workflow,
)


class DetectorViewFactory:
    """
    Factory for creating Sciline-based detector view workflows.

    This factory creates StreamProcessorWorkflow instances that use the
    Sciline-based detector view workflow for accumulating detector data
    and producing cumulative and current detector images.

    Supports two projection modes via ViewConfig:
    1. GeometricViewConfig: For xy_plane/cylinder_mantle_z projections
    2. LogicalViewConfig: For fold/slice transforms

    Parameters
    ----------
    data_source:
        Detector data source configuration. Use NeXusDetectorSource for
        loading geometry from a file, or DetectorNumberSource for fast
        file-less startup with logical views.
    view_config:
        View configuration. Can be a single config (applied to all sources)
        or a dict mapping source names to configs (for per-detector settings).
    """

    def __init__(
        self,
        *,
        data_source: DetectorDataSource,
        view_config: ViewConfig | dict[str, ViewConfig],
    ) -> None:
        self._data_source = data_source
        self._view_config = view_config

    def _get_config(self, source_name: str) -> ViewConfig:
        """Get the view config for a given source."""
        if isinstance(self._view_config, dict):
            return self._view_config[source_name]
        return self._view_config

    def make_workflow(
        self,
        source_name: str,
        params: DetectorViewParamsBase,
    ) -> StreamProcessorWorkflow:
        """
        Factory method that creates a detector view workflow.

        Parameters
        ----------
        source_name:
            Name of the detector source (e.g., 'panel_0').
        params:
            Workflow parameters containing coordinate mode, edges, and ranges.

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
            Every context input, the ROI request streams included (see
            :func:`bind_roi_requests`), is injected by the routing layer after
            creation.
        """
        mode = params.coordinate_mode.mode

        # Validate wavelength mode requirements
        if mode == 'wavelength':
            if isinstance(self._data_source, DetectorNumberSource):
                raise ValueError(
                    f"{mode} mode requires geometry for Ltotal computation; "
                    "use NeXusDetectorSource instead of DetectorNumberSource"
                )

        # Get mode-specific event coordinate
        event_coord = {
            'toa': 'event_time_offset',
            'wavelength': 'wavelength',
        }[mode]

        # Get active edges and range for current mode
        bins = params.get_active_edges()
        histogram_slice = params.get_active_range()

        # Get pixel weighting setting from params
        use_pixel_weighting = params.pixel_weighting.enabled

        # Create base workflow with appropriate mode
        workflow = create_base_workflow(
            bins=bins,
            event_coord=event_coord,
            histogram_slice=histogram_slice,
            coordinate_mode=mode,
        )

        # The lookup table arrives as context from the LUT workflow. Inserting
        # the provider is the whole of the wiring: the instrument declares which
        # stream carries its key, and the build gates the job on that stream
        # only if the targets reach the provider. Hence the unconditional insert
        # -- in TOA mode nothing reaches it, so the job neither waits for the
        # table nor needs one (ADR 0010).
        workflow.insert(detector_lookup_table)
        if mode == 'wavelength':
            workflow[LookupTableRelativeErrorThreshold] = {source_name: float('inf')}

        # Configure detector data source (EmptyDetector)
        self._data_source.configure_workflow(workflow, source_name)

        # Set pixel weighting configuration
        workflow[UsePixelWeighting] = use_pixel_weighting

        # Add projection based on config type
        config = self._get_config(source_name)
        match config:
            case GeometricViewConfig():
                add_geometric_projection(
                    workflow,
                    projection_type=config.projection_type,
                    resolution=config.resolution,
                    pixel_noise=config.pixel_noise,
                    flip_x=config.flip_x,
                )
                roi_support = True  # Geometric views always support ROI
            case LogicalViewConfig():
                # Bind source_name to the transform if provided
                if config.transform is not None:

                    def bound_transform(
                        da: sc.DataArray, transform=config.transform
                    ) -> sc.DataArray:
                        return transform(da, source_name)

                else:
                    bound_transform = None

                add_logical_projection(
                    workflow,
                    transform=bound_transform,
                    reduction_dim=config.reduction_dim,
                )
                roi_support = config.roi_support

        # Build target keys - conditionally include ROI outputs
        target_keys: dict[str, type] = {
            'cumulative': DetectorImage[Cumulative],
            'current': DetectorImage[Current],
            'counts_total': CountsTotal[Current],
            'counts_in_toa_range': CountsInRange[Current],
            'counts_total_cumulative': CountsTotal[Cumulative],
            'counts_in_toa_range_cumulative': CountsInRange[Cumulative],
        }

        if config.spectrum_view is not None:
            workflow.insert(spectrum_view)
            raw_transform = config.spectrum_view.transform
            if config.spectrum_view.params_model is not None:
                spectrum_params = params.spectrum_params  # type: ignore[attr-defined]

                def bound_spectrum_transform(
                    histogram: sc.DataArray,
                    _transform=raw_transform,
                    _params=spectrum_params,
                ) -> sc.DataArray:
                    return _transform(histogram, _params)

                workflow[SpectrumViewTransform] = bound_spectrum_transform
            else:
                workflow[SpectrumViewTransform] = raw_transform
            target_keys['spectrum_view'] = SpectrumView
        if roi_support:
            # Add ROI-related outputs only when supported
            target_keys.update(
                {
                    'roi_spectra_cumulative': ROISpectra[Cumulative],
                    'roi_spectra_current': ROISpectra[Current],
                    'roi_rectangle': ROIRectangleReadback,
                    'roi_polygon': ROIPolygonReadback,
                }
            )

        # Reset the cumulative histogram when the detector moves: summing across a
        # move mixes incompatible geometries (geometric views shift screen bins;
        # wavelength views shift the per-pixel wavelength calibration). The coord is
        # stamped only for file-based sources, so this is a no-op for file-less
        # logical TOA views, which stay valid across a move.
        #
        # Reset it likewise when a downsampled detector's source resolution changes:
        # the target grid is unchanged, but counts remapped with a different stride
        # land in different pixels. Stamped only by DownsamplePixelIds, so this is a
        # no-op for detectors ingested at full resolution.
        cumulative, window = make_no_copy_accumulator_pair(
            reset_coords=(DETECTOR_TRANSFORM, SOURCE_RESOLUTION)
        )
        return StreamProcessorWorkflow(
            workflow,
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            target_keys=target_keys,
            window_outputs=(
                DetectorViewOutputs if roi_support else DetectorViewOutputsBase
            ).fields_with(Temporality.window),
            accumulators={
                AccumulatedHistogram[Cumulative]: cumulative,
                AccumulatedHistogram[Current]: window,
            },
        )


_ROI_REQUEST_KEYS = {
    'rectangle': ROIRectangleRequest,
    'polygon': ROIPolygonRequest,
}


def bind_roi_requests(workflow_factory: WorkflowFactory) -> None:
    """Bind the ROI request streams of every spec that publishes ROI readbacks.

    Called once from :meth:`~ess.livedata.config.instrument.Instrument.load_factories`.
    A spec whose outputs model carries the ROI readback fields is served by a
    detector-view graph with the ROI providers inserted, which cannot build
    without the request keys -- so the bindings follow from the outputs
    declaration and are derived here rather than restated next to each factory.
    The workflow-key import that forces this to live on the factory side, away
    from the ``specs.py`` the dashboard imports, is the only reason the two are
    declared apart at all.

    One spec-scope context binding per source and ROI geometry, naming the
    stream by :func:`~ess.livedata.config.roi_names.roi_stream_name`. The
    routing layer thereby subscribes each job to its view's requests and
    delivers them to ``set_context`` under the request key
    :func:`make_workflow` wired the ROI providers by.

    The bindings do not gate (ADR 0002): the providers read a missing request
    as "no ROI selected", so a job runs without one and picks up whatever
    arrives. The stream is nonetheless latched like any other context input,
    which together with the job-free name lets a selection be published
    before its job exists -- ``JobManager.peek_pending_streams`` hands the
    latched value to the job as it activates -- and survive a restart of it.

    Two concurrent jobs of one view would therefore read one selection. The
    backend does not rule that out -- ``JobManager`` keys jobs by ``JobId``
    and never supersedes -- and multiple jobs per workflow remain supported;
    it is the dashboard that commits one generation at a time, stopping the
    previous job before starting the next.
    """
    mapper = get_roi_mapper()
    readback_keys = set(mapper.readback_keys)
    for registration in workflow_factory.registrations():
        spec = registration.spec
        if not readback_keys <= set(spec.outputs.model_fields):
            continue
        handle = workflow_factory.handle(spec.get_id())
        for source_name in spec.source_names:
            for geometry in mapper.geometries:
                handle.add_context_binding(
                    stream_name=roi_stream_name(
                        handle.workflow_id, source_name, geometry.readback_key
                    ),
                    workflow_key=_ROI_REQUEST_KEYS[geometry.geometry_type],
                    dependent_sources={source_name},
                    gating=False,
                )
