# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
ROI stream naming and configuration.

This module provides centralized naming conventions for ROI-related streams,
ensuring consistency between backend handlers and frontend components.
"""

import re
from dataclasses import dataclass
from typing import Literal

ROIGeometryType = Literal["rectangle", "polygon"]


@dataclass(frozen=True)
class ROIGeometry:
    """
    Configuration for a specific ROI geometry type.

    Parameters
    ----------
    geometry_type:
        The type of ROI geometry (e.g., "rectangle", "polygon").
    num_rois:
        Number of ROIs of this geometry type.
    index_offset:
        Starting index for ROIs of this type in the global numbering scheme.
    """

    geometry_type: ROIGeometryType
    num_rois: int
    index_offset: int = 0

    @property
    def readback_key(self) -> str:
        """Readback stream key for this geometry."""
        return f"roi_{self.geometry_type}"

    @property
    def index_range(self) -> range:
        """Range of ROI indices for this geometry."""
        return range(self.index_offset, self.index_offset + self.num_rois)


class ROIStreamMapper:
    """
    Manages ROI stream naming across multiple geometry types.

    Supports multiple ROI geometry streams (rectangle, polygon, etc.)
    with index offsets for unified histogram numbering.

    Parameters
    ----------
    geometries:
        List of ROI geometry configurations.
    """

    def __init__(self, geometries: list[ROIGeometry]):
        self._geometries = geometries
        self._total_rois = sum(g.num_rois for g in geometries)
        self._roi_index_pattern = re.compile(r"^roi_(?:current|cumulative)_(\d+)$")

    @property
    def geometries(self) -> list[ROIGeometry]:
        """All configured ROI geometries."""
        return self._geometries

    @property
    def total_rois(self) -> int:
        """Total number of ROIs across all geometries."""
        return self._total_rois

    @property
    def readback_keys(self) -> list[str]:
        """All readback stream keys."""
        return [g.readback_key for g in self._geometries]

    def current_key(self, index: int) -> str:
        """Generate ROI current histogram key."""
        return f"roi_current_{index}"

    def cumulative_key(self, index: int) -> str:
        """Generate ROI cumulative histogram key."""
        return f"roi_cumulative_{index}"

    def all_current_keys(self) -> list[str]:
        """Generate all current histogram keys across all geometries."""
        return [self.current_key(i) for i in range(self._total_rois)]

    def all_cumulative_keys(self) -> list[str]:
        """Generate all cumulative histogram keys across all geometries."""
        return [self.cumulative_key(i) for i in range(self._total_rois)]

    def all_histogram_keys(self) -> list[str]:
        """Generate all histogram keys (current + cumulative)."""
        return self.all_current_keys() + self.all_cumulative_keys()

    def parse_roi_index(self, key: str) -> int | None:
        """
        Extract ROI index from histogram key.

        Parameters
        ----------
        key:
            Key string like 'roi_current_0' or 'roi_cumulative_2'.

        Returns
        -------
        :
            The ROI index, or None if the key doesn't match the pattern.
        """
        match = self._roi_index_pattern.match(key)
        return int(match.group(1)) if match else None

    def geometry_for_index(self, index: int) -> ROIGeometry | None:
        """
        Find which geometry owns a given ROI index.

        Parameters
        ----------
        index:
            ROI index to look up.

        Returns
        -------
        :
            The geometry that owns this index, or None if out of range.
        """
        for geom in self._geometries:
            if index in geom.index_range:
                return geom
        return None


# Default ROI configuration - single source of truth
DEFAULT_ROI_GEOMETRIES = [
    ROIGeometry(geometry_type="rectangle", num_rois=4, index_offset=0),
    ROIGeometry(geometry_type="polygon", num_rois=4, index_offset=4),
]


def get_roi_mapper(instrument: str | None = None) -> ROIStreamMapper:
    """
    Get the ROI stream mapper configuration.

    This is the single source of truth for ROI configuration
    shared between backend handlers and frontend factories.

    Parameters
    ----------
    instrument:
        Instrument name. Currently unused, but provided for future
        instrument-specific configurations.

    Returns
    -------
    :
        Configured ROI stream mapper.
    """
    # For now, all instruments use the default configuration
    # Future: could load instrument-specific configuration
    return ROIStreamMapper(DEFAULT_ROI_GEOMETRIES)
