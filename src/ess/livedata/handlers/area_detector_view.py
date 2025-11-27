# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Area detector view for dense image data.

This module provides view classes for area detector data (ad00 schema).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import scipp as sc

from ess.reduce.live import raw

from .workflow_factory import Workflow


class AreaDetectorView(Workflow):
    """
    Workflow for area detector image visualization.

    Accumulates image data and provides cumulative and delta (current) outputs.
    Optionally applies a transform via LogicalView (e.g., for downsampling).

    Parameters
    ----------
    logical_view:
        LogicalView instance for transforming images. Use identity for no transform.
    """

    def __init__(self, logical_view: raw.LogicalView) -> None:
        self._logical_view = logical_view
        self._cumulative: sc.DataArray | None = None
        self._previous: sc.DataArray | None = None
        self._current_start_time: int | None = None

    def accumulate(
        self, data: dict[str, Any], *, start_time: int, end_time: int
    ) -> None:
        """
        Add data to the accumulator.

        Parameters
        ----------
        data:
            Data to be added. Expected to contain exactly one area detector image.
        start_time:
            Start time of the data window in nanoseconds since epoch.
        end_time:
            End time of the data window in nanoseconds since epoch.
        """
        _ = end_time  # unused
        if len(data) != 1:
            raise ValueError("AreaDetectorView expects exactly one detector data item.")

        if self._current_start_time is None:
            self._current_start_time = start_time

        image = next(iter(data.values()))
        transformed = self._logical_view(image)
        if self._cumulative is None:
            self._cumulative = transformed.copy()
        else:
            self._cumulative += transformed

    def finalize(self) -> dict[str, sc.DataArray]:
        """
        Finalize accumulated data and return results.

        Returns
        -------
        :
            Dictionary with 'cumulative' (total so far) and 'current' (delta since
            last finalize) entries.
        """
        if self._current_start_time is None:
            raise RuntimeError(
                "finalize called without any detector data accumulated via accumulate"
            )

        cumulative = self._cumulative.copy()
        current = cumulative
        if self._previous is not None:
            current = current - self._previous
        self._previous = cumulative

        time_coord = sc.scalar(self._current_start_time, unit='ns')
        current = current.assign_coords(time=time_coord)
        self._current_start_time = None

        return {'cumulative': cumulative, 'current': current}

    def clear(self) -> None:
        """Clear all accumulated state."""
        self._cumulative = None
        self._previous = None
        self._current_start_time = None


class AreaDetectorViewFactory:
    """
    Factory for area detector views with optional transform and reduction.

    Creates AreaDetectorView workflows that use LogicalView for data transformation.

    Parameters
    ----------
    input_sizes:
        Dictionary defining the input dimension sizes
        (e.g., {'dim_0': 512, 'dim_1': 512}).
    transform:
        Callable that transforms input data (e.g., fold or slice operations).
        If None, identity transform is used.
    reduction_dim:
        Dimension(s) to sum over after applying transform. Enables downsampling.
    """

    def __init__(
        self,
        *,
        input_sizes: dict[str, int],
        transform: Callable[[sc.DataArray], sc.DataArray] | None = None,
        reduction_dim: str | list[str] | None = None,
    ) -> None:
        self._input_sizes = input_sizes
        self._transform = transform if transform is not None else _identity
        self._reduction_dim = reduction_dim

    def make_view(self, source_name: str) -> AreaDetectorView:
        """
        Factory method that creates an area detector view for the given source.

        Parameters
        ----------
        source_name:
            Name of the detector source (used for identification).

        Returns
        -------
        :
            AreaDetectorView workflow instance.
        """
        _ = source_name  # Not used currently, but kept for API consistency
        logical_view = raw.LogicalView(
            transform=self._transform,
            reduction_dim=self._reduction_dim,
            input_sizes=self._input_sizes,
        )
        return AreaDetectorView(logical_view)


def _identity(da: sc.DataArray) -> sc.DataArray:
    return da
