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
from ess.reduce.unwrap import LookupTableFilename
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold
from scippnexus import NXdetector

from ..accumulators import make_no_copy_accumulator_pair

# Import types unconditionally for runtime type hint resolution
# (used by workflow_factory.attach_factory to inspect parameter types)
from ..detector_view_specs import DetectorViewParams, SpectrumViewRebin
from ..stream_processor_workflow import StreamProcessorWorkflow
from .data_source import DetectorDataSource, DetectorNumberSource
from .providers import spectrum_view
from .types import (
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
    TransformValueLog,
    TransformValueStream,
    UsePixelWeighting,
    ViewConfig,
)
from .workflow import (
    add_dynamic_transform,
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
    dynamic_transforms:
        Optional mapping ``source_name -> TransformValueStream`` binding a
        NeXus transformation entry of the detector's chain to the f144
        stream that supplies its live values. ``aux_stream`` is the
        logical name of the auxiliary input delivering the NXlog
        DataArray (must match the corresponding ``AuxSources`` entry).
        ``transform_name`` is the entry of ``chain.transformations``
        whose ``.value`` will be replaced with the latest sample.
    """

    def __init__(
        self,
        *,
        data_source: DetectorDataSource,
        view_config: ViewConfig | dict[str, ViewConfig],
        dynamic_transforms: dict[str, TransformValueStream] | None = None,
    ) -> None:
        self._data_source = data_source
        self._view_config = view_config
        self._dynamic_transforms = dynamic_transforms or {}

    def _get_config(self, source_name: str) -> ViewConfig:
        """Get the view config for a given source."""
        if isinstance(self._view_config, dict):
            return self._view_config[source_name]
        return self._view_config

    def make_workflow(
        self,
        source_name: str,
        params: DetectorViewParams,
        lookup_table_filename: str | None = None,
    ) -> StreamProcessorWorkflow:
        """
        Factory method that creates a detector view workflow.

        Parameters
        ----------
        source_name:
            Name of the detector source (e.g., 'panel_0').
        params:
            Workflow parameters containing coordinate mode, edges, and ranges.
        lookup_table_filename:
            Path to lookup table file. Required for 'wavelength' coordinate mode.
            The caller (instrument factory) is responsible for resolving this
            from instrument-specific params.

        Returns
        -------
        :
            StreamProcessorWorkflow wrapping the Sciline-based detector view.
        """
        mode = params.coordinate_mode.mode

        # Validate wavelength mode requirements
        if mode == 'wavelength':
            if lookup_table_filename is None:
                raise ValueError(f"{mode} mode requires lookup_table_filename")
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

        # Set lookup table filename and error threshold for wavelength mode
        if mode == 'wavelength':
            workflow[LookupTableFilename] = lookup_table_filename
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
            workflow[SpectrumViewTransform] = config.spectrum_view.transform
            workflow[SpectrumViewRebin] = params.spectrum_rebin
            target_keys['spectrum_view'] = SpectrumView
        context_keys: dict[str, type] = {}
        window_outputs = (
            'current',
            'counts_total',
            'counts_in_toa_range',
        )

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
            context_keys.update(
                {
                    'roi_rectangle': ROIRectangleRequest,
                    'roi_polygon': ROIPolygonRequest,
                }
            )
            window_outputs = (
                'current',
                'counts_total',
                'counts_in_toa_range',
                'roi_spectra_current',
            )

        # Wire dynamic detector geometry (f144 NXlog stream) if configured for
        # this source.
        value_stream = self._dynamic_transforms.get(source_name)
        if value_stream is not None:
            add_dynamic_transform(workflow, transform_name=value_stream.transform_name)
            context_keys[value_stream.aux_stream] = TransformValueLog

        cumulative, window = make_no_copy_accumulator_pair()
        return StreamProcessorWorkflow(
            workflow,
            dynamic_keys={source_name: NeXusData[NXdetector, SampleRun]},
            context_keys=context_keys,
            target_keys=target_keys,
            window_outputs=window_outputs,
            accumulators={
                AccumulatedHistogram[Cumulative]: cumulative,
                AccumulatedHistogram[Current]: window,
            },
        )
