# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

from ess.livedata.dashboard.autoscaler import Autoscaler


class TestAutoscaler:
    """Tests for the Autoscaler class."""

    def test_update_bounds_with_1d_coords(self):
        """Test basic bounds update with 1D coordinates."""
        data = sc.DataArray(
            data=sc.array(dims=['x', 'y'], values=[[1.0, 2.0], [3.0, 4.0]]),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1.0]),
                'y': sc.array(dims=['y'], values=[0.0, 2.0]),
            },
        )
        autoscaler = Autoscaler()
        changed = autoscaler.update_bounds(data)

        assert changed is True
        assert autoscaler.coord_bounds['x'] == (0.0, 1.0)
        assert autoscaler.coord_bounds['y'] == (0.0, 2.0)

    def test_update_bounds_without_coords(self):
        """Test bounds update when no coordinates are present."""
        data = sc.DataArray(
            data=sc.array(dims=['x', 'y'], values=[[1.0, 2.0], [3.0, 4.0]])
        )
        autoscaler = Autoscaler()
        changed = autoscaler.update_bounds(data)

        assert changed is True
        # Falls back to size-based bounds
        assert autoscaler.coord_bounds['x'] == (0, 2)
        assert autoscaler.coord_bounds['y'] == (0, 2)

    def test_update_bounds_with_2d_dimension_coord(self):
        """Test that 2D dimension coordinates don't cause errors."""
        # Create data where a dimension coordinate depends on multiple dimensions
        # This can happen with detector pixel coordinates
        data = sc.DataArray(
            data=sc.array(
                dims=['z', 'y', 'x'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
            ),
        )
        # Make 'y' a 2D coordinate (depends on both z and y)
        y_2d = sc.array(
            dims=['z', 'y'],
            values=np.arange(6).reshape(2, 3).astype('float64'),
        )
        data = data.assign_coords(
            {
                'z': sc.array(dims=['z'], values=[0.0, 1.0]),
                'y': y_2d,  # 2D coord with same name as dimension
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0, 3.0]),
            }
        )

        autoscaler = Autoscaler()
        # Should not raise despite 2D coord
        changed = autoscaler.update_bounds(data)

        assert changed is True
        # z and x have 1D coords, should use coord bounds
        assert autoscaler.coord_bounds['z'] == (0.0, 1.0)
        assert autoscaler.coord_bounds['x'] == (0.0, 3.0)
        # y has 2D coord, should fall back to size-based bounds
        assert autoscaler.coord_bounds['y'] == (0, 3)

    def test_update_bounds_with_mixed_coord_types(self):
        """Test bounds update with a mix of 1D coords, 2D coords, and no coords."""
        data = sc.DataArray(
            data=sc.array(
                dims=['a', 'b', 'c'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
            ),
        )
        # a: 1D coord
        # b: 2D coord (depends on a and b)
        # c: no coord
        b_2d = sc.array(
            dims=['a', 'b'],
            values=np.arange(6).reshape(2, 3).astype('float64'),
        )
        data = data.assign_coords(
            {
                'a': sc.array(dims=['a'], values=[10.0, 20.0]),
                'b': b_2d,
            }
        )

        autoscaler = Autoscaler()
        changed = autoscaler.update_bounds(data)

        assert changed is True
        assert autoscaler.coord_bounds['a'] == (10.0, 20.0)  # from 1D coord
        assert autoscaler.coord_bounds['b'] == (0, 3)  # size fallback for 2D coord
        assert autoscaler.coord_bounds['c'] == (0, 4)  # size fallback for no coord

    def test_bounds_not_changed_on_repeated_same_data(self):
        """Test that bounds don't change when updating with same data."""
        data = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0])},
        )
        autoscaler = Autoscaler()

        # First update
        changed1 = autoscaler.update_bounds(data)
        assert changed1 is True

        # Second update with same data
        changed2 = autoscaler.update_bounds(data)
        assert changed2 is False

    def test_bounds_expand_with_larger_data(self):
        """Test that bounds expand when new data extends range."""
        autoscaler = Autoscaler()

        # First update
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0]),
            coords={'x': sc.array(dims=['x'], values=[0.0, 1.0])},
        )
        autoscaler.update_bounds(data1)
        assert autoscaler.coord_bounds['x'] == (0.0, 1.0)

        # Second update with extended range
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0, 2.0, 3.0]),
            coords={'x': sc.array(dims=['x'], values=[-1.0, 0.0, 2.0])},
        )
        changed = autoscaler.update_bounds(data2)
        assert changed is True
        assert autoscaler.coord_bounds['x'] == (-1.0, 2.0)

    def test_bounds_shrink_when_data_significantly_smaller(self):
        """Test that bounds shrink when data range is significantly smaller."""
        # Use shrink_threshold=0.1 (default), so bounds shrink if data is >10% inside
        autoscaler = Autoscaler(shrink_threshold=0.1)

        # Initial range [0, 100]
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 101),
            coords={'x': sc.array(dims=['x'], values=np.linspace(0, 100, 101))},
        )
        autoscaler.update_bounds(data1)
        assert autoscaler.coord_bounds['x'] == (0.0, 100.0)

        # Small shrink: [5, 95] - only 5% inside on each end, below threshold
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 91),
            coords={'x': sc.array(dims=['x'], values=np.linspace(5, 95, 91))},
        )
        changed = autoscaler.update_bounds(data2)
        assert changed is False  # Below 10% threshold
        assert autoscaler.coord_bounds['x'] == (0.0, 100.0)

        # Large shrink: [20, 80] - 20% inside on each end, above threshold
        data3 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 61),
            coords={'x': sc.array(dims=['x'], values=np.linspace(20, 80, 61))},
        )
        changed = autoscaler.update_bounds(data3)
        assert changed is True  # Above 10% threshold
        assert autoscaler.coord_bounds['x'] == (20.0, 80.0)

    def test_shrink_threshold_zero_always_matches_data(self):
        """Test that shrink_threshold=0 makes bounds always match data exactly."""
        autoscaler = Autoscaler(shrink_threshold=0.0)

        # Initial range [0, 100]
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 101),
            coords={'x': sc.array(dims=['x'], values=np.linspace(0, 100, 101))},
        )
        autoscaler.update_bounds(data1)
        assert autoscaler.coord_bounds['x'] == (0.0, 100.0)

        # Any shrink should trigger update
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 99),
            coords={'x': sc.array(dims=['x'], values=np.linspace(1, 99, 99))},
        )
        changed = autoscaler.update_bounds(data2)
        assert changed is True
        assert autoscaler.coord_bounds['x'] == (1.0, 99.0)

    def test_grow_threshold_delays_expansion(self):
        """Test that grow_threshold delays bounds expansion."""
        autoscaler = Autoscaler(grow_threshold=0.1)

        # Initial range [0, 100]
        data1 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 101),
            coords={'x': sc.array(dims=['x'], values=np.linspace(0, 100, 101))},
        )
        autoscaler.update_bounds(data1)
        assert autoscaler.coord_bounds['x'] == (0.0, 100.0)

        # Small grow: [0, 105] - only 5% beyond, below threshold
        data2 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 106),
            coords={'x': sc.array(dims=['x'], values=np.linspace(0, 105, 106))},
        )
        changed = autoscaler.update_bounds(data2)
        assert changed is False  # Below 10% threshold
        assert autoscaler.coord_bounds['x'] == (0.0, 100.0)

        # Large grow: [0, 115] - 15% beyond, above threshold
        data3 = sc.DataArray(
            data=sc.array(dims=['x'], values=[1.0] * 116),
            coords={'x': sc.array(dims=['x'], values=np.linspace(0, 115, 116))},
        )
        changed = autoscaler.update_bounds(data3)
        assert changed is True
        assert autoscaler.coord_bounds['x'] == (0.0, 115.0)

    def test_separate_coord_data_for_axis_bounds(self):
        """Test using separate data for value vs coordinate bounds."""
        autoscaler = Autoscaler()

        # 3D data for value bounds
        data_3d = sc.DataArray(
            data=sc.array(
                dims=['z', 'y', 'x'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
            ),
        )
        # 2D slice for coordinate bounds
        coord_data = sc.DataArray(
            data=sc.array(
                dims=['y', 'x'],
                values=np.arange(12).reshape(3, 4).astype('float64'),
            ),
            coords={
                'y': sc.array(dims=['y'], values=[0.0, 1.0, 2.0]),
                'x': sc.array(dims=['x'], values=[0.0, 1.0, 2.0, 3.0]),
            },
        )

        changed = autoscaler.update_bounds(data_3d, coord_data=coord_data)

        assert changed is True
        # Coord bounds should come from coord_data (2D)
        assert autoscaler.coord_bounds['y'] == (0.0, 2.0)
        assert autoscaler.coord_bounds['x'] == (0.0, 3.0)
        # Value bounds should come from data_3d (3D) - range is 0 to 23
        assert autoscaler.value_bounds == (0.0, 23.0)

    def test_coord_data_allows_tracking_2d_coords_after_slicing(self):
        """Test that coord_data enables proper bounds tracking for 2D coords."""
        # Use shrink_threshold=0 so bounds always match current slice exactly
        autoscaler = Autoscaler(shrink_threshold=0.0)

        # Create 3D data with a 2D coord
        data_3d = sc.DataArray(
            data=sc.array(
                dims=['z', 'y', 'x'],
                values=np.arange(24).reshape(2, 3, 4).astype('float64'),
            ),
        )
        # 2D coord: y values depend on z
        y_2d = sc.array(
            dims=['z', 'y'],
            values=[[0, 1, 2], [10, 11, 12]],
            dtype='float64',
        )
        data_3d = data_3d.assign_coords({'y': y_2d})

        # Slice at z=0 - y coord becomes 1D with values [0, 1, 2]
        slice_z0 = data_3d['z', 0]
        changed1 = autoscaler.update_bounds(data_3d, coord_data=slice_z0)
        assert changed1 is True
        assert autoscaler.coord_bounds['y'] == (0.0, 2.0)

        # Slice at z=1 - y coord becomes 1D with values [10, 11, 12]
        # Bounds update to match new slice (bidirectional)
        slice_z1 = data_3d['z', 1]
        changed2 = autoscaler.update_bounds(data_3d, coord_data=slice_z1)
        assert changed2 is True
        assert autoscaler.coord_bounds['y'] == (10.0, 12.0)

        # Slice at z=0 again - bounds update back to match
        changed3 = autoscaler.update_bounds(data_3d, coord_data=slice_z0)
        assert changed3 is True  # Bounds changed to match new slice
        assert autoscaler.coord_bounds['y'] == (0.0, 2.0)

        # Same slice again - no change
        changed4 = autoscaler.update_bounds(data_3d, coord_data=slice_z0)
        assert changed4 is False  # No change - zoom state preserved!
