# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.reduce.live import raw

from ess.livedata.handlers.area_detector_view import AreaDetectorView


class TestAreaDetectorViewFactory:
    def test_view_factory_returns_callable(self):
        factory = AreaDetectorView.view_factory()
        workflow = factory('detector')
        assert isinstance(workflow, AreaDetectorView)

    def test_view_factory_with_transform(self):
        def downsample(da: sc.DataArray) -> sc.DataArray:
            da = da.fold(dim='dim_0', sizes={'dim_0': 4, 'y_bin': 2})
            da = da.fold(dim='dim_1', sizes={'dim_1': 4, 'x_bin': 2})
            return da

        factory = AreaDetectorView.view_factory(
            transform=downsample,
            reduction_dim=['y_bin', 'x_bin'],
        )
        workflow = factory('detector')

        frame = sc.DataArray(sc.ones(dims=['dim_0', 'dim_1'], shape=[8, 8]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)
        result = workflow.finalize()

        assert result['cumulative'].shape == (4, 4)
        assert result['current'].shape == (4, 4)


class TestAreaDetectorView:
    def test_accumulate_and_finalize(self):
        logical_view = raw.LogicalView(
            transform=lambda da: da,
            input_sizes={'y': 4, 'x': 4},
        )
        workflow = AreaDetectorView(logical_view)

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)

        result = workflow.finalize()

        assert 'cumulative' in result
        assert 'current' in result
        assert result['cumulative'].shape == (4, 4)
        assert 'time' in result['current'].coords

    def test_current_is_delta(self):
        logical_view = raw.LogicalView(
            transform=lambda da: da,
            input_sizes={'y': 4, 'x': 4},
        )
        workflow = AreaDetectorView(logical_view)

        frame1 = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        frame2 = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]) * 2)

        workflow.accumulate({'detector': frame1}, start_time=1000, end_time=2000)
        workflow.finalize()

        workflow.accumulate({'detector': frame2}, start_time=2000, end_time=3000)
        result2 = workflow.finalize()

        assert sc.allclose(result2['current'].data, frame2.data)
        expected_cumulative = frame1.data + frame2.data
        assert sc.allclose(result2['cumulative'].data, expected_cumulative)

    def test_clear_resets_state(self):
        logical_view = raw.LogicalView(
            transform=lambda da: da,
            input_sizes={'y': 4, 'x': 4},
        )
        workflow = AreaDetectorView(logical_view)

        frame = sc.DataArray(sc.ones(dims=['y', 'x'], shape=[4, 4]))
        workflow.accumulate({'detector': frame}, start_time=1000, end_time=2000)
        workflow.finalize()

        workflow.clear()

        with pytest.raises(RuntimeError, match="finalize called without"):
            workflow.finalize()

    def test_finalize_without_accumulate_raises(self):
        logical_view = raw.LogicalView(
            transform=lambda da: da,
            input_sizes={'y': 4, 'x': 4},
        )
        workflow = AreaDetectorView(logical_view)

        with pytest.raises(RuntimeError, match="finalize called without"):
            workflow.finalize()
