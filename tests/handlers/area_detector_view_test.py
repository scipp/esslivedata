# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.livedata.handlers.area_detector_view import (
    AreaDetectorLogicalView,
    AreaDetectorView,
    DenseDetectorView,
)
from ess.livedata.handlers.detector_view_specs import DetectorViewParams


class TestDenseDetectorView:
    def test_add_image_accumulates(self):
        """Test that add_image accumulates frames."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        view = DenseDetectorView(logical_view)

        frame1 = sc.ones(dims=['y', 'x'], shape=[4, 4])
        frame2 = sc.ones(dims=['y', 'x'], shape=[4, 4]) * 2

        view.add_image(sc.DataArray(frame1))
        assert view.cumulative is not None
        assert sc.allclose(view.cumulative.data, frame1)

        view.add_image(sc.DataArray(frame2))
        assert sc.allclose(view.cumulative.data, frame1 + frame2)

    def test_add_image_with_transform(self):
        """Test that transform is applied during accumulation."""

        def fold_transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='x', sizes={'x': 2, 'x_bin': 2})

        logical_view = _make_logical_view(
            sizes={'y': 4, 'x': 4},
            transform=fold_transform,
            reduction_dim='x_bin',
        )
        view = DenseDetectorView(logical_view)

        # 4x4 image with values 1-16
        data = np.arange(16).reshape(4, 4).astype(float)
        frame = sc.DataArray(sc.array(dims=['y', 'x'], values=data))

        view.add_image(frame)

        # After folding x into (x:2, x_bin:2) and summing x_bin:
        # Shape should be (y:4, x:2)
        assert view.cumulative.shape == (4, 2)

    def test_clear_counts(self):
        """Test that clear_counts resets accumulation."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        view = DenseDetectorView(logical_view)

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        view.add_image(frame)
        assert view.cumulative is not None

        view.clear_counts()
        assert view.cumulative is None

    def test_make_roi_filter(self):
        """Test that make_roi_filter returns a valid ROIFilter."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        view = DenseDetectorView(logical_view)

        roi_filter = view.make_roi_filter()
        assert roi_filter is not None
        assert roi_filter.spatial_dims == ('y', 'x')

    def test_from_transform_factory(self):
        """Test the from_transform static factory method."""

        def my_transform(da: sc.DataArray) -> sc.DataArray:
            return da

        view = DenseDetectorView.from_transform(
            transform=my_transform,
            input_sizes={'y': 8, 'x': 8},
        )

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[8, 8]))
        view.add_image(frame)
        assert view.cumulative.shape == (8, 8)


class TestAreaDetectorLogicalView:
    def test_make_view_creates_workflow(self):
        """Test that make_view creates an AreaDetectorView workflow."""
        factory = AreaDetectorLogicalView(
            input_sizes={'dim_0': 8, 'dim_1': 8},
        )

        params = _make_default_params()
        workflow = factory.make_view(source_name='detector', params=params)

        assert isinstance(workflow, AreaDetectorView)

    def test_make_view_with_transform(self):
        """Test that make_view applies transform correctly."""

        def downsample(da: sc.DataArray) -> sc.DataArray:
            da = da.fold(dim='dim_0', sizes={'dim_0': 4, 'y_bin': 2})
            da = da.fold(dim='dim_1', sizes={'dim_1': 4, 'x_bin': 2})
            return da

        factory = AreaDetectorLogicalView(
            input_sizes={'dim_0': 8, 'dim_1': 8},
            transform=downsample,
            reduction_dim=['y_bin', 'x_bin'],
        )

        params = _make_default_params()
        workflow = factory.make_view(source_name='detector', params=params)

        # Accumulate a frame
        frame = sc.DataArray(sc.ones(dims=['dim_0', 'dim_1'], shape=[8, 8]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)
        result = workflow.finalize()

        # Output should be downsampled to 4x4
        assert result['cumulative'].shape == (4, 4)
        assert result['current'].shape == (4, 4)


class TestAreaDetectorView:
    def test_accumulate_and_finalize(self):
        """Test basic accumulate and finalize cycle."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        dense_view = DenseDetectorView(logical_view)
        params = _make_default_params()
        workflow = AreaDetectorView(params=params, dense_view=dense_view)

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)

        result = workflow.finalize()

        assert 'cumulative' in result
        assert 'current' in result
        assert result['cumulative'].shape == (4, 4)
        assert 'time' in result['current'].coords

    def test_current_is_delta(self):
        """Test that current result is delta from previous."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        dense_view = DenseDetectorView(logical_view)
        params = _make_default_params()
        workflow = AreaDetectorView(params=params, dense_view=dense_view)

        frame1 = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        frame2 = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]) * 2)

        # First cycle
        workflow.accumulate({'detector': frame1}, start_time=1000, end_time=2000)
        workflow.finalize()

        # Second cycle
        workflow.accumulate({'detector': frame2}, start_time=2000, end_time=3000)
        result2 = workflow.finalize()

        # Current should be only frame2, not frame1+frame2
        assert sc.allclose(result2['current'].data, frame2.data)
        # Cumulative should be frame1+frame2
        expected_cumulative = frame1.data + frame2.data
        assert sc.allclose(result2['cumulative'].data, expected_cumulative)

    def test_clear_resets_state(self):
        """Test that clear resets all state."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        dense_view = DenseDetectorView(logical_view)
        params = _make_default_params()
        workflow = AreaDetectorView(params=params, dense_view=dense_view)

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)
        workflow.finalize()

        workflow.clear()

        # After clear, finalize should fail (no data)
        with pytest.raises(RuntimeError, match="finalize called without"):
            workflow.finalize()

    def test_finalize_without_accumulate_raises(self):
        """Test that finalize raises if no data accumulated."""
        logical_view = _make_identity_logical_view(sizes={'y': 4, 'x': 4})
        dense_view = DenseDetectorView(logical_view)
        params = _make_default_params()
        workflow = AreaDetectorView(params=params, dense_view=dense_view)

        with pytest.raises(RuntimeError, match="finalize called without"):
            workflow.finalize()


# Helper functions


def _make_identity_logical_view(sizes: dict[str, int]):
    """Create a LogicalView with identity transform."""
    from ess.reduce.live import raw

    return raw.LogicalView(
        transform=lambda da: da,
        input_sizes=sizes,
    )


def _make_logical_view(
    sizes: dict[str, int],
    transform=None,
    reduction_dim=None,
):
    """Create a LogicalView with specified transform."""
    from ess.reduce.live import raw

    return raw.LogicalView(
        transform=transform if transform else lambda da: da,
        input_sizes=sizes,
        reduction_dim=reduction_dim,
    )


def _make_default_params() -> DetectorViewParams:
    """Create default DetectorViewParams for testing."""
    from ess.livedata import parameter_models
    from ess.livedata.config import models

    return DetectorViewParams(
        toa_range=parameter_models.TOARange(enabled=False),
        toa_edges=parameter_models.TOAEdges(),
        pixel_weighting=models.PixelWeighting(enabled=False),
    )
