# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Names of the spectral axes and of the coords describing a spectral range.

Monitor and detector-view workflows histogram events along a spectral axis, time
of arrival or wavelength. Outputs that sum over that axis carry the summed
interval as two scalar coords, so the dashboard can divide by its width and show
counts per unit of the axis, e.g. counts/µs or counts/Å.
"""

import scipp as sc

SPECTRAL_DIMS = ('time_of_arrival', 'wavelength')
"""Dims of the spectral axis of monitor and detector-view histograms."""

SPECTRAL_RANGE_LOW_COORD = 'spectral_range_low'
"""Scalar coord: lower edge of the spectral interval an output was summed over.

The interval spans whole histogram bins: the bins that overlap the range filter,
or all bins without a filter. It is therefore wider than the range filter
unless the filter bounds fall on bin edges.
"""

SPECTRAL_RANGE_HIGH_COORD = 'spectral_range_high'
"""Scalar coord: upper edge of the spectral interval an output was summed over.

See :data:`SPECTRAL_RANGE_LOW_COORD`.
"""


def spectral_range_template_coords() -> dict[str, sc.Variable]:
    """Spectral-range coords for the empty template of an output.

    A template declaring these coords tells the dashboard that the output carries
    them, so it offers to normalize by the width of the interval. The values are
    placeholders: the unit follows the histogram edges at runtime.
    """
    return {
        SPECTRAL_RANGE_LOW_COORD: sc.scalar(0.0, unit='ms'),
        SPECTRAL_RANGE_HIGH_COORD: sc.scalar(0.0, unit='ms'),
    }
