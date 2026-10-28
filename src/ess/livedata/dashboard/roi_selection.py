# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Selection of a single ROI from a per-ROI output.

Per-ROI outputs stack the values of all drawn ROIs along a ``roi`` dim, with
the ROI index as the ``roi`` coord. A plot layer may select one ROI instead
(:attr:`~ess.livedata.dashboard.plot_topology.DataSourceConfig.roi`); its
plotter then receives the data of that ROI without the ``roi`` dim, as from an
output of a single ROI. Coords along ``roi``, such as the number of detector
pixels in the ROI, become scalar coords.
"""

from __future__ import annotations

import numpy as np
import scipp as sc

from ess.livedata.config.roi_names import get_roi_mapper
from ess.livedata.config.workflow_spec import WorkflowSpec

ROI_DIM = 'roi'


def roi_label(roi: int) -> str:
    """Label of an ROI in plots, the same as its legend entry in per-ROI plots."""
    return f'{ROI_DIM}={roi}'


def roi_choices() -> dict[str, int]:
    """The selectable ROIs, by display label, e.g. ``'roi=4 (polygon)'``."""
    return {
        f'{roi_label(index)} ({geometry.geometry_type})': index
        for geometry in get_roi_mapper().geometries
        for index in geometry.index_range
    }


def select_roi(data: sc.DataArray, roi: int) -> sc.DataArray | None:
    """Return the data of one ROI, or None if the ROI is not drawn."""
    (positions,) = np.nonzero(data.coords[ROI_DIM].values == roi)
    if positions.size == 0:
        return None
    # Copy: binning, e.g. in correlation histograms, rejects the strided view of
    # a slice along an inner dim.
    return data[ROI_DIM, int(positions[0])].copy()


def selected_output_template(
    workflow_spec: WorkflowSpec, view_name: str, roi: int | None
) -> sc.DataArray | None:
    """Return the template of an output view as a plotter receives it.

    With an ROI selected, this is the template of a single ROI. Plotters and
    their options are offered by this template.
    """
    template = workflow_spec.get_output_template(view_name)
    if template is None or roi is None:
        return template

    # The template holds no ROI to slice, so drop the dim instead.
    def drop_roi(var: sc.Variable) -> sc.Variable:
        sizes = {dim: size for dim, size in var.sizes.items() if dim != ROI_DIM}
        return sc.zeros(sizes=sizes, unit=var.unit, dtype=var.dtype)

    return sc.DataArray(
        drop_roi(template.data),
        coords={
            name: drop_roi(coord) if ROI_DIM in coord.dims else coord
            for name, coord in template.coords.items()
        },
    )


def has_roi_dim(workflow_spec: WorkflowSpec, view_name: str) -> bool:
    """Whether an output view stacks the values of several ROIs."""
    template = workflow_spec.get_output_template(view_name)
    return template is not None and ROI_DIM in template.dims
