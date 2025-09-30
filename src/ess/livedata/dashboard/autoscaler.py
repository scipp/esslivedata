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
    framewise=True only when the bounds *increase*, i.e., new data extends the
    existing bounds. This way, we keep the current pan/zoom state most of the time while
    still allowing the plot to grow as new data comes in. This is especially important
    since there seems to be no way of initializing holoviews.streams.Pipe without
    initial dummy data (such as `None`), i.e., we need to return an empty plot with no
    good starting guess of bounds.
    """

    def __init__(self, value_margin_factor: float = 0.01):
        """
        Initialize the autoscaler with empty bounds.

        Parameters
        ----------
        value_margin_factor:
            Factor by which to extend the value bounds when updating, by default 0.01.
            This prevents the plot from jumping around when new data arrives that only
            slightly extends the bounds. The value bounds are updated to be 99% of the
            new minimum and 101% of the new maximum when set to 0.01, for example.
        """
        self._value_margin_factor = value_margin_factor
        self.coord_bounds: dict[str, tuple[float | None, float | None]] = defaultdict(
            lambda: (None, None)
        )
        self.value_bounds = (None, None)

    def update_bounds(self, data: sc.DataArray) -> bool:
        """Update bounds based on the data, return True if bounds changed."""
        changed = False
        for dim in data.dims:
            if (coord := data.coords.get(dim)) is not None:
                changed |= self._update_coord_bounds(coord)
            else:
                changed |= self._update_from_size(dim, data.sizes[dim])
        changed |= self._update_value_bounds(data.data)
        return changed

    def _update_coord_bounds(self, coord: sc.Variable) -> bool:
        """Update bounds for a single coordinate."""
        name = coord.dim
        low = coord[0].value
        high = coord[-1].value
        changed = False

        if self.coord_bounds[name][0] is None or low < self.coord_bounds[name][0]:
            self.coord_bounds[name] = (low, self.coord_bounds[name][1])
            changed = True
        if self.coord_bounds[name][1] is None or high > self.coord_bounds[name][1]:
            self.coord_bounds[name] = (self.coord_bounds[name][0], high)
            changed = True

        return changed

    def _update_from_size(self, dim: str, size: int) -> bool:
        """Update bounds for a dimension without coordinates, using its size."""
        changed = False
        if self.coord_bounds[dim] != (0, size):
            self.coord_bounds[dim] = (0, size)
            changed = True
        return changed

    def _update_value_bounds(self, data: sc.Variable) -> bool:
        """Update value bounds based on the data, return True if bounds changed."""
        low = data.nanmin().value
        high = data.nanmax().value
        if high <= low:
            # Avoid updating bounds if data is constant or empty
            return False

        changed = False
        if self.value_bounds[0] is None or low < self.value_bounds[0]:
            self.value_bounds = (
                low * (1 - self._value_margin_factor),
                self.value_bounds[1],
            )
            changed = True
        if self.value_bounds[1] is None or high > self.value_bounds[1]:
            self.value_bounds = (
                self.value_bounds[0],
                high * (1 + self._value_margin_factor),
            )
            changed = True

        return changed
