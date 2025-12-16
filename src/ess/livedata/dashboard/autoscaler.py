# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections import defaultdict

import scipp as sc


class Autoscaler:
    """
    A helper class that automatically adjusts bounds based on data.

    Maybe I missed something in the Holoviews docs, but looking, e.g., at
    https://holoviews.org/FAQ.html we need framewise=True to autoscale for streaming
    data. However, this leads to losing the current pan/zoom state when new data
    arrives, making it unusable for interactive exploration.
    Instead, we use this class to track the bounds of the data and update the plot with
    framewise=True only when the bounds change significantly. This way, we keep the
    current pan/zoom state most of the time while still allowing the plot to adjust
    as data changes. This is especially important since there seems to be no way of
    initializing holoviews.streams.Pipe without initial dummy data (such as `None`),
    i.e., we need to return an empty plot with no good starting guess of bounds.
    """

    def __init__(
        self,
        grow_threshold: float = 0.01,
        shrink_threshold: float = 0.1,
    ):
        """
        Initialize the autoscaler with empty bounds.

        Parameters
        ----------
        grow_threshold:
            Threshold for growing bounds (both coordinate and value), as fraction
            of current extent. Default 0.01 means bounds grow when data exceeds
            them by more than 1%. This prevents jumping on small fluctuations.
        shrink_threshold:
            Threshold for shrinking bounds (both coordinate and value), as fraction
            of current extent. Default 0.1 means bounds only shrink when data range
            is more than 10% inside the current bounds. Set higher to reduce jumping
            when data fluctuates, or lower to track data more closely.
        """
        self._grow_threshold = grow_threshold
        self._shrink_threshold = shrink_threshold
        self.coord_bounds: dict[str, tuple[float | None, float | None]] = defaultdict(
            lambda: (None, None)
        )
        self.value_bounds: tuple[float | None, float | None] = (None, None)

    def update_bounds(
        self, data: sc.DataArray, *, coord_data: sc.DataArray | None = None
    ) -> bool:
        """
        Update bounds based on the data, return True if bounds changed.

        Parameters
        ----------
        data:
            Data to use for value (color) bounds.
        coord_data:
            Optional separate data to use for coordinate (axis) bounds.
            If None, uses `data` for both. Useful when you want consistent
            value scaling across all slices but axis ranges from the current slice.
        """
        if coord_data is None:
            coord_data = data

        changed = False
        for dim in coord_data.dims:
            coord = coord_data.coords.get(dim)
            if coord is not None and coord.ndim == 1:
                changed |= self._update_coord_bounds(coord)
            else:
                # No coord, or coord is multi-dimensional (e.g., 2D detector coords)
                changed |= self._update_from_size(dim, coord_data.sizes[dim])
        changed |= self._update_value_bounds(data.data)
        return changed

    def _update_coord_bounds(self, coord: sc.Variable) -> bool:
        """
        Update bounds for a single coordinate with bidirectional threshold logic.

        Bounds can both grow and shrink, with separate thresholds controlling
        how much change is needed before triggering an update.
        """
        name = coord.dim
        new_low = coord[0].value
        new_high = coord[-1].value
        current_low, current_high = self.coord_bounds[name]

        # First update: always set bounds
        if current_low is None or current_high is None:
            self.coord_bounds[name] = (new_low, new_high)
            return True

        extent = current_high - current_low
        if extent <= 0:
            # Degenerate case: no extent, just check for any difference
            if new_low != current_low or new_high != current_high:
                self.coord_bounds[name] = (new_low, new_high)
                return True
            return False

        grow_margin = self._grow_threshold * extent
        shrink_margin = self._shrink_threshold * extent

        changed = False
        updated_low, updated_high = current_low, current_high

        # Check low bound
        if new_low < current_low - grow_margin:
            # Growing: data extends below current low
            updated_low = new_low
            changed = True
        elif new_low > current_low + shrink_margin:
            # Shrinking: data is significantly above current low
            updated_low = new_low
            changed = True

        # Check high bound
        if new_high > current_high + grow_margin:
            # Growing: data extends above current high
            updated_high = new_high
            changed = True
        elif new_high < current_high - shrink_margin:
            # Shrinking: data is significantly below current high
            updated_high = new_high
            changed = True

        if changed:
            self.coord_bounds[name] = (updated_low, updated_high)
        return changed

    def _update_from_size(self, dim: str, size: int) -> bool:
        """Update bounds for a dimension without coordinates, using its size."""
        changed = False
        if self.coord_bounds[dim] != (0, size):
            self.coord_bounds[dim] = (0, size)
            changed = True
        return changed

    def _update_value_bounds(self, data: sc.Variable) -> bool:
        """
        Update value bounds based on the data with bidirectional threshold logic.

        Uses the same grow/shrink thresholds as coordinate bounds.
        """
        new_low = data.nanmin().value
        new_high = data.nanmax().value
        if new_high <= new_low:
            # Avoid updating bounds if data is constant or empty
            return False

        current_low, current_high = self.value_bounds

        # First update: always set bounds
        if current_low is None or current_high is None:
            self.value_bounds = (new_low, new_high)
            return True

        extent = current_high - current_low
        if extent <= 0:
            # Degenerate case: no extent, just check for any difference
            if new_low != current_low or new_high != current_high:
                self.value_bounds = (new_low, new_high)
                return True
            return False

        grow_margin = self._grow_threshold * extent
        shrink_margin = self._shrink_threshold * extent

        changed = False
        updated_low, updated_high = current_low, current_high

        # Check low bound
        if new_low < current_low - grow_margin:
            updated_low = new_low
            changed = True
        elif new_low > current_low + shrink_margin:
            updated_low = new_low
            changed = True

        # Check high bound
        if new_high > current_high + grow_margin:
            updated_high = new_high
            changed = True
        elif new_high < current_high - shrink_margin:
            updated_high = new_high
            changed = True

        if changed:
            self.value_bounds = (updated_low, updated_high)
        return changed
