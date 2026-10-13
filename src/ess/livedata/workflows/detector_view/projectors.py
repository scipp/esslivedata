# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Projector implementations for detector view workflow.

This module provides the Projector protocol and concrete implementations for
projecting detector events from pixel coordinates to screen coordinates.

Two projection strategies are supported:
1. GeometricProjector: Uses calibrated positions to project to a plane or a
   cylinder mantle.
2. LogicalProjector: Reshapes detector data using fold/slice transforms,
   optionally reducing dimensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipp as sc
from ess.reduce.live.raw import (
    CalibratedPositionWithNoisyReplicas,
    DetectorViewResolution,
    make_cylinder_mantle_coords,
    make_xy_plane_coords,
)

from .types import FlipX, LogicalTransform, ProjectionType, ReductionDim, ScreenMetadata


class Projector(Protocol):
    """Protocol for event projection strategies.

    Projectors transform events from detector pixels to screen coordinates.
    Screen metadata (coords, sizes) is obtained separately via get_screen_metadata
    provider to make dependencies explicit in the Sciline DAG.
    """

    def project_events(self, events: sc.DataArray) -> sc.DataArray:
        """Project events from detector pixels to screen coordinates."""
        ...


class GeometricProjector:
    """
    Projects events using geometric coordinate transformation.

    Uses calibrated positions (with position-noise replicas) to project
    detector pixels to screen coordinates (a plane or a cylinder mantle).
    Bins events instead of histogramming counts, preserving TOF information.

    Each detector pixel has one screen position per position-noise replica. The
    replicas sample the pixel's extent, so that pixel centers do not alias onto the
    image grid. Every event uses one replica, and the events of each pixel are
    spread evenly over its replicas within every batch. Dividing a single-batch image
    by :meth:`compute_weights` therefore gives counts per detector pixel, up to
    counting noise.

    Parameters
    ----------
    coords:
        Projected coordinates for each detector pixel, with shape
        (replica, detector_number). The replica dim has length 1 without position
        noise. Created by make_xy_plane_coords or make_cylinder_mantle_coords.
    edges:
        Bin edges for screen coordinates, keyed by dimension name.
    """

    def __init__(self, coords: sc.DataGroup, edges: sc.DataGroup) -> None:
        self._edges = edges
        self._replicas = coords.sizes['replica']
        # Screen coordinates of all (pixel, replica) pairs, flattened pixel-major:
        # index = pixel * replicas + replica. Events come in pixel order, so looking
        # up their coordinates reads memory in order.
        self._coords = sc.DataGroup(
            {
                key: coords[key]
                .transpose(['detector_number', 'replica'])
                .flatten(to='pixel_replica')
                for key in edges.keys()
            }
        )
        # Replica of the first event of the next batch, see project_events.
        self._first_replica = 0
        self._screen_metadata = ScreenMetadata(
            coords={dim: sc.midpoints(edges[dim]) for dim in edges.keys()},
            sizes={dim: len(edges[dim]) - 1 for dim in edges.keys()},
        )

    @property
    def screen_metadata(self) -> ScreenMetadata:
        """Screen metadata with coordinate bin centers and sizes."""
        return self._screen_metadata

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
            Events should have 'event_time_offset' coordinate. The event table must
            be contiguous and in pixel order, as produced by grouping events by
            detector pixel.

        Returns
        -------
        :
            Binned data with screen coordinates as outer dimensions,
            preserving events with TOF information.
        """
        # Extract flat event table from bins, discarding the pixel binning structure.
        # This is more efficient than using bin(dim='detector_number', ...) which
        # would process the bin structure before flattening.
        constituents = events.data.bins.constituents
        begin = constituents['begin'].values
        end = constituents['end'].values
        event_table = constituents['data']

        # For each event, the index of its (pixel, replica) pair in self._coords.
        # Consecutive events take consecutive replicas, so each pixel's events are
        # spread evenly over its replicas. The first replica rotates from batch to
        # batch, so a pixel with the same number of events in every batch does not
        # always use the same replicas.
        n_events_per_pixel = end - begin
        replicas = self._replicas
        index = np.repeat(
            np.arange(len(n_events_per_pixel)) * replicas, n_events_per_pixel
        )
        cycle = (np.arange(replicas) + self._first_replica) % replicas
        self._first_replica = (self._first_replica + 1) % replicas
        # Add the repeating cycle in place. np.arange(n_events) % replicas would add
        # about 40% to the run time of this method.
        n_tail = len(index) % replicas
        n_full = len(index) - n_tail
        full_cycles = index[:n_full].reshape(-1, replicas)  # view, index is contiguous
        full_cycles += cycle
        index[n_full:] += cycle[:n_tail]

        # Build coordinates dict for flat events
        event_coords = {}

        # Copy existing event coordinates (event_time_offset, etc.)
        for name in event_table.coords:
            event_coords[name] = event_table.coords[name]

        # Add screen coordinates of each event's (pixel, replica) pair
        for key, coord in self._coords.items():
            event_coords[key] = sc.array(
                dims=[event_table.dim],
                values=coord.values[index],
                unit=coord.unit,
            )

        # Create flat event table with screen coordinates
        flat_events = sc.DataArray(
            data=event_table.data,
            coords=event_coords,
        )

        # Bin by screen coordinates (preserving events)
        return flat_events.bin(self._edges)

    def compute_weights(self) -> sc.DataArray:
        """
        Compute pixel weights by histogramming ones through all replicas.

        Returns the number of detector pixels contributing to each screen bin,
        averaged over all noise replicas. Used to normalize screen pixels when
        pixel weighting is enabled.

        Returns
        -------
        :
            2D array with shape matching screen dimensions, containing the
            average number of detector pixels per screen bin.
        """
        ones = sc.DataArray(
            sc.ones(sizes=self._coords.sizes, dtype='float32'), coords=self._coords
        )
        return ones.hist(self._edges) / self._replicas


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
    """

    def __init__(
        self,
        *,
        transform: Callable[[sc.DataArray], sc.DataArray] | None,
        reduction_dim: str | list[str] | None,
    ) -> None:
        self._transform = transform
        self._reduction_dim = reduction_dim

    def _get_output_dims(self, all_dims: tuple[str, ...]) -> list[str]:
        """Determine output dimensions after reduction."""
        if self._reduction_dim is None:
            dims_to_reduce: set[str] = set()
        elif isinstance(self._reduction_dim, str):
            dims_to_reduce = {self._reduction_dim}
        else:
            dims_to_reduce = set(self._reduction_dim)
        return [d for d in all_dims if d not in dims_to_reduce]

    def get_screen_metadata(self, empty_detector: sc.DataArray) -> ScreenMetadata:
        """
        Compute screen metadata by applying transform to detector structure.

        Parameters
        ----------
        empty_detector:
            Detector structure without events.

        Returns
        -------
        :
            Screen metadata with output dimensions and coordinates.
        """
        if self._transform is None:
            transformed = empty_detector
        else:
            transformed = self._transform(empty_detector)

        output_dims = self._get_output_dims(transformed.dims)

        def get_bin_centers(dim: str) -> sc.Variable | None:
            coord = transformed.coords.get(dim)
            if coord is None:
                return None
            if transformed.coords.is_edges(dim):
                return sc.midpoints(coord)
            return coord

        return ScreenMetadata(
            coords={dim: get_bin_centers(dim) for dim in output_dims},
            sizes={dim: transformed.sizes[dim] for dim in output_dims},
        )

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
        # Apply transform to reshape bin structure
        # e.g., fold('detector_number', {'y': 100, 'x': 100})
        result = events if self._transform is None else self._transform(events)

        # Merge events along reduction dimensions (if any)
        if self._reduction_dim is not None:
            dims_to_reduce = (
                [self._reduction_dim]
                if isinstance(self._reduction_dim, str)
                else list(self._reduction_dim)
            )
            result = result.bins.concat(dims_to_reduce)

        return result

    def compute_weights(self, empty_detector: sc.DataArray) -> sc.DataArray:
        """
        Compute pixel weights for logical projection.

        Returns the number of detector pixels contributing to each output pixel.
        For logical views without reduction, this is 1 everywhere. With reduction,
        it equals the number of pixels summed over.

        Parameters
        ----------
        empty_detector:
            Detector structure without events (used to determine input shape).

        Returns
        -------
        :
            Array with shape matching output dimensions, containing the
            number of detector pixels per output pixel.
        """
        ones = sc.DataArray(
            sc.ones(sizes=empty_detector.sizes, dtype='float32', unit='dimensionless')
        )
        if self._transform is not None:
            ones = self._transform(ones)
        if self._reduction_dim is not None:
            dims_to_reduce = (
                [self._reduction_dim]
                if isinstance(self._reduction_dim, str)
                else list(self._reduction_dim)
            )
            ones = ones.sum(dims_to_reduce)
        return ones


def make_geometric_projector(
    coords: CalibratedPositionWithNoisyReplicas,
    projection_type: ProjectionType,
    resolution: DetectorViewResolution,
    flip_x: FlipX,
) -> Projector:
    """
    Sciline provider: Create a geometric projector.

    Parameters
    ----------
    coords:
        Calibrated position with noisy replicas from NeXus workflow.
    projection_type:
        Type of geometric projection; see ``GeometricViewConfig``.
    resolution:
        Resolution (number of bins) for each screen dimension.
    flip_x:
        Whether to negate x-coordinates for 'view from sample' orientation.
        See :class:`GeometricViewConfig` for details on when to enable this.

    Returns
    -------
    :
        Projector configured for the specified geometric projection.
    """
    # Use existing projection functions from essreduce
    if projection_type == 'xy_plane':
        projected_coords = make_xy_plane_coords(coords)
    elif projection_type in ('cylinder_mantle_y', 'cylinder_mantle_z'):
        # The cylinder axis ('y' or 'z') is the last character of the name.
        projected_coords = make_cylinder_mantle_coords(coords, axis=projection_type[-1])
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")

    if flip_x:
        projected_coords['x'] = -projected_coords['x']

    # Create bin edges from coordinates and resolution
    edges = sc.DataGroup(
        {
            dim: projected_coords[dim].hist({dim: res}).coords[dim]
            for dim, res in resolution.items()
        }
    )

    return GeometricProjector(projected_coords, edges)


def make_logical_projector(
    transform: LogicalTransform,
    reduction_dim: ReductionDim,
) -> Projector:
    """
    Sciline provider: Create a logical projector.

    Parameters
    ----------
    transform:
        Callable that reshapes detector data (fold/slice). If None, identity.
    reduction_dim:
        Dimension(s) to merge events over. None means no reduction.

    Returns
    -------
    :
        Projector configured for the specified logical transform.
    """
    return LogicalProjector(transform=transform, reduction_dim=reduction_dim)
