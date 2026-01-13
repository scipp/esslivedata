# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Projector implementations for detector view workflow.

This module provides the ProjectorProtocol and concrete implementations for
projecting detector events from pixel coordinates to screen coordinates.

Two projection strategies are supported:
1. GeometricProjector: Uses calibrated positions to project to xy_plane or
   cylinder_mantle_z coordinates.
2. LogicalProjector: Reshapes detector data using fold/slice transforms,
   optionally reducing dimensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NewType, Protocol

import numpy as np
import scipp as sc

from ess.reduce.live.raw import (
    CalibratedPositionWithNoisyReplicas,
    DetectorViewResolution,
    make_cylinder_mantle_coords,
    make_xy_plane_coords,
)
from ess.reduce.nexus.types import EmptyDetector, SampleRun

from .types import LogicalTransform, ProjectionType, ReductionDim

# Type for Projector param in Sciline workflow
Projector = NewType('Projector', 'ProjectorProtocol')
"""Projector instance set as workflow param."""


class ProjectorProtocol(Protocol):
    """Protocol for event projection strategies.

    All projectors must provide:
    - project_events(): Transform events from detector pixels to screen coordinates
    - Coordinate info properties: y_dim, x_dim, y_edges, x_edges
    """

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """Project events from detector pixels to screen coordinates."""
        ...

    @property
    def y_dim(self) -> str:
        """Name of the y (vertical) screen dimension."""
        ...

    @property
    def x_dim(self) -> str:
        """Name of the x (horizontal) screen dimension."""
        ...

    @property
    def y_edges(self) -> sc.Variable | None:
        """Bin edges for y dimension. None if logical projection has no coords."""
        ...

    @property
    def x_edges(self) -> sc.Variable | None:
        """Bin edges for x dimension. None if logical projection has no coords."""
        ...


class GeometricProjector:
    """
    Projects events using geometric coordinate transformation.

    Uses calibrated positions (with optional noise replicas) to project
    detector pixels to screen coordinates (xy_plane or cylinder_mantle_z).
    Bins events instead of histogramming counts, preserving TOF information.

    Parameters
    ----------
    coords:
        Projected coordinates for each detector pixel, with shape
        (replica, detector_number). Created by make_xy_plane_coords or
        make_cylinder_mantle_coords.
    edges:
        Bin edges for screen coordinates, keyed by dimension name.
    """

    def __init__(self, coords: sc.DataGroup, edges: sc.DataGroup) -> None:
        self._coords = coords
        self._edges = edges
        self._replica_dim = 'replica'
        self._replicas = coords.sizes.get(self._replica_dim, 1)
        self._current = 0

        # Extract dimension names from edges (typically 'screen_x', 'screen_y')
        dims = list(edges.keys())
        if len(dims) != 2:
            raise ValueError(f"Expected 2 spatial dims from projector, got {len(dims)}")
        # Order: first dim is y, second is x (matching histogram convention)
        self._y_dim, self._x_dim = dims[0], dims[1]

    @property
    def y_dim(self) -> str:
        """Name of the y (vertical) screen dimension."""
        return self._y_dim

    @property
    def x_dim(self) -> str:
        """Name of the x (horizontal) screen dimension."""
        return self._x_dim

    @property
    def y_edges(self) -> sc.Variable:
        """Bin edges for y dimension."""
        return self._edges[self._y_dim]

    @property
    def x_edges(self) -> sc.Variable:
        """Bin edges for x dimension."""
        return self._edges[self._x_dim]

    @property
    def edges(self) -> sc.DataGroup:
        """Bin edges for screen coordinates (legacy property)."""
        return self._edges

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """
        Project events from detector pixels to screen coordinates.

        This method broadcasts per-pixel screen coordinates to individual events,
        then re-bins events by screen position. We use manual numpy indexing rather
        than ``sc.bins_like`` for performance reasons:

        - ``sc.bins_like`` has O(n_pixels) overhead that dominates when pixels >> events
        - With 1M pixels and <1M events (typical for live streaming), numpy is 2-10x
          faster
        - ``sc.bins_like`` only becomes faster at high event density (>10 events/pixel)

        Parameters
        ----------
        events:
            Binned event data with detector pixels as the outer dimension.
            Events should have 'event_time_offset' coordinate.

        Returns
        -------
        :
            Binned data with screen coordinates as outer dimensions,
            preserving events with TOF information.
        """
        # Cycle through replicas for smoother visualization
        replica = self._current % self._replicas
        self._current += 1

        # Get coords for this replica
        replica_coords = {
            key: self._coords[key][self._replica_dim, replica]
            for key in self._edges.keys()
        }

        # Extract flat event table from bins, discarding the pixel binning structure.
        # This is more efficient than using bin(dim='detector_number', ...) which
        # would process the bin structure before flattening.
        constituents = events.data.bins.constituents
        begin = constituents['begin'].values
        end = constituents['end'].values
        event_table = constituents['data']

        # Compute event-to-pixel mapping: for each event, which pixel did it come from?
        # This allows broadcasting per-pixel coordinates to per-event coordinates.
        n_events_per_pixel = end - begin
        event_to_pixel = np.repeat(
            np.arange(len(n_events_per_pixel)), n_events_per_pixel
        )

        # Build coordinates dict for flat events
        event_coords = {}

        # Copy existing event coordinates (event_time_offset, etc.)
        for name in event_table.coords:
            event_coords[name] = event_table.coords[name]

        # Add screen coordinates by indexing pixel coords with the event-to-pixel map
        for key, coord in replica_coords.items():
            event_coords[key] = sc.array(
                dims=[event_table.dim],
                values=coord.values[event_to_pixel],
                unit=coord.unit,
            )

        # Create flat event table with screen coordinates
        flat_events = sc.DataArray(
            data=event_table.data,
            coords=event_coords,
        )

        # Bin by screen coordinates (preserving events)
        return flat_events.bin(self._edges)


class LogicalProjector:
    """
    Projects events using logical reshape and optional reduction.

    Applies a transform function to reshape detector data (e.g., fold by panel/tube),
    then optionally reduces dimensions by concatenating bins.

    Parameters
    ----------
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.
    reduction_dim:
        Dimension(s) to merge events over via bins.concat. None means no reduction.
    y_dim:
        Name of the y (vertical) output dimension.
    x_dim:
        Name of the x (horizontal) output dimension.
    y_edges:
        Bin edges for y dimension. None if no coordinates available.
    x_edges:
        Bin edges for x dimension. None if no coordinates available.
    """

    def __init__(
        self,
        *,
        transform: Callable[[sc.DataArray], sc.DataArray] | None,
        reduction_dim: str | list[str] | None,
        y_dim: str,
        x_dim: str,
        y_edges: sc.Variable | None,
        x_edges: sc.Variable | None,
    ) -> None:
        self._transform = transform
        self._reduction_dim = reduction_dim
        self._y_dim = y_dim
        self._x_dim = x_dim
        self._y_edges = y_edges
        self._x_edges = x_edges

    @property
    def y_dim(self) -> str:
        """Name of the y (vertical) screen dimension."""
        return self._y_dim

    @property
    def x_dim(self) -> str:
        """Name of the x (horizontal) screen dimension."""
        return self._x_dim

    @property
    def y_edges(self) -> sc.Variable | None:
        """Bin edges for y dimension. None if no coordinates available."""
        return self._y_edges

    @property
    def x_edges(self) -> sc.Variable | None:
        """Bin edges for x dimension. None if no coordinates available."""
        return self._x_edges

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """
        Project events using logical view (reshape + optional reduction).

        Parameters
        ----------
        events:
            Detector data with events binned by detector pixel.

        Returns
        -------
        :
            Events binned by logical coordinates with TOF preserved.
        """
        if self._transform is None:
            result = events
        else:
            # Apply transform to reshape bin structure
            # e.g., fold('detector_number', {'y': 100, 'x': 100})
            result = self._transform(events)

        # Merge events along reduction dimensions (if any)
        if self._reduction_dim is not None:
            dims_to_reduce = (
                [self._reduction_dim]
                if isinstance(self._reduction_dim, str)
                else list(self._reduction_dim)
            )
            for dim in dims_to_reduce:
                result = result.bins.concat(dim)

        return result


def make_geometric_projector(
    coords: CalibratedPositionWithNoisyReplicas,
    projection_type: ProjectionType,
    resolution: DetectorViewResolution,
) -> Projector:
    """
    Sciline provider: Create a geometric projector.

    Parameters
    ----------
    coords:
        Calibrated position with noisy replicas from NeXus workflow.
    projection_type:
        Type of geometric projection ('xy_plane' or 'cylinder_mantle_z').
    resolution:
        Resolution (number of bins) for each screen dimension.

    Returns
    -------
    :
        Projector configured for the specified geometric projection.
    """
    # Use existing projection functions from essreduce
    if projection_type == 'xy_plane':
        projected_coords = make_xy_plane_coords(coords)
    elif projection_type == 'cylinder_mantle_z':
        projected_coords = make_cylinder_mantle_coords(coords)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    # Create bin edges from coordinates and resolution
    edges = sc.DataGroup(
        {
            dim: projected_coords[dim].hist({dim: res}).coords[dim]
            for dim, res in resolution.items()
        }
    )

    return Projector(GeometricProjector(projected_coords, edges))


def make_logical_projector(
    empty_detector: EmptyDetector[SampleRun],
    transform: LogicalTransform,
    reduction_dim: ReductionDim,
) -> Projector:
    """
    Sciline provider: Create a logical projector.

    Parameters
    ----------
    empty_detector:
        Detector structure without events (used to compute output dimensions).
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.
    reduction_dim:
        Dimension(s) to merge events over. None means no reduction.

    Returns
    -------
    :
        Projector configured for the specified logical transform.
    """
    # Apply transform to get output structure
    if transform is None:
        transformed = empty_detector
    else:
        transformed = transform(empty_detector)

    # Extract spatial dimensions
    dims = list(transformed.dims)
    if len(dims) < 2:
        raise ValueError(f"Expected at least 2 dims from transform, got {dims}")

    # Assume first two dims are spatial (y, x)
    y_dim, x_dim = dims[0], dims[1]

    # Get coordinates if available from transform
    y_edges = transformed.coords.get(y_dim)
    x_edges = transformed.coords.get(x_dim)

    return Projector(
        LogicalProjector(
            transform=transform,
            reduction_dim=reduction_dim,
            y_dim=y_dim,
            x_dim=x_dim,
            y_edges=y_edges,
            x_edges=x_edges,
        )
    )


# Backwards compatibility aliases
EventProjector = GeometricProjector
make_event_projector = make_geometric_projector
