# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Event projector implementations for detector view workflow.

This module provides classes and functions for projecting detector events
from pixel coordinates to screen coordinates.
"""

from __future__ import annotations

import numpy as np
import scipp as sc

from ess.reduce.live.raw import (
    CalibratedPositionWithNoisyReplicas,
    DetectorViewResolution,
    make_cylinder_mantle_coords,
    make_xy_plane_coords,
)


class EventProjector:
    """
    Projects events from detector pixels to screen coordinates.

    Reuses Histogrammer's coordinate infrastructure (including noise replicas)
    but bins events instead of histogramming counts, preserving TOF information.

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

    @property
    def edges(self) -> sc.DataGroup:
        """Bin edges for screen coordinates."""
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


def make_event_projector(
    coords: CalibratedPositionWithNoisyReplicas,
    projection_type: str | None,
    resolution: DetectorViewResolution,
) -> EventProjector:
    """
    Create an EventProjector for geometric projection.

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
        EventProjector configured for the specified projection.
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

    return EventProjector(projected_coords, edges)
