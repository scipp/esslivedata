# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from unittest.mock import Mock

import scipp as sc

from ess.livedata.dashboard.extractors import (
    FullHistoryExtractor,
    LatestValueExtractor,
    WindowAggregatingExtractor,
)
from ess.livedata.dashboard.plot_params import (
    WindowAggregation,
    WindowMode,
    WindowParams,
)
from ess.livedata.dashboard.plotting_controller import create_extractors_from_params


class TestCreateExtractorsFromParams:
    """Tests for create_extractors_from_params factory function."""

    def test_fallback_to_latest_value_when_no_params(self):
        """Test fallback to LatestValueExtractor when no window params provided."""
        keys = ['key1', 'key2']

        extractors = create_extractors_from_params(keys=keys, window=None, spec=None)

        assert len(extractors) == 2
        assert all(isinstance(ext, LatestValueExtractor) for ext in extractors.values())
        assert set(extractors.keys()) == {'key1', 'key2'}

    def test_create_latest_value_extractors_with_window_mode_latest(self):
        """Test creation of LatestValueExtractor when window mode is 'latest'."""
        keys = ['key1']
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 1
        assert isinstance(extractors['key1'], LatestValueExtractor)

    def test_create_window_aggregating_extractors_with_window_mode_window(self):
        """Test creation of WindowAggregatingExtractor when window mode is 'window'."""
        keys = ['key1', 'key2']
        window = WindowParams(
            mode=WindowMode.window,
            window_duration_seconds=5.0,
            aggregation=WindowAggregation.nansum,
        )

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 2
        assert all(
            isinstance(ext, WindowAggregatingExtractor) for ext in extractors.values()
        )

        # Verify behavior through public interface
        extractor = extractors['key1']
        assert extractor.get_required_timespan() == 5.0

    def test_spec_required_extractor_overrides_window_params(self):
        """Test that plotter spec's required extractor overrides window params."""
        keys = ['key1', 'key2']
        window = WindowParams(mode=WindowMode.latest)

        # Create mock spec with required extractor
        spec = Mock()
        spec.data_requirements.required_extractor = FullHistoryExtractor

        extractors = create_extractors_from_params(keys=keys, window=window, spec=spec)

        # Should use FullHistoryExtractor despite window params
        assert len(extractors) == 2
        assert all(isinstance(ext, FullHistoryExtractor) for ext in extractors.values())

    def test_spec_with_no_required_extractor_uses_window_params(self):
        """Test that window params are used when spec has no required extractor."""
        keys = ['key1']
        window = WindowParams(mode=WindowMode.window, window_duration_seconds=3.0)

        # Create mock spec without required extractor
        spec = Mock()
        spec.data_requirements.required_extractor = None

        extractors = create_extractors_from_params(keys=keys, window=window, spec=spec)

        assert isinstance(extractors['key1'], WindowAggregatingExtractor)
        assert extractors['key1'].get_required_timespan() == 3.0

    def test_creates_extractors_for_all_keys(self):
        """Test that extractors are created for all provided keys."""
        keys = ['result1', 'result2', 'result3']
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert len(extractors) == 3
        assert set(extractors.keys()) == {'result1', 'result2', 'result3'}
        assert all(isinstance(ext, LatestValueExtractor) for ext in extractors.values())

    def test_empty_keys_returns_empty_dict(self):
        """Test that empty keys list returns empty extractors dict."""
        keys = []
        window = WindowParams(mode=WindowMode.latest)

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        assert extractors == {}

    def test_window_aggregation_parameters_passed_correctly(self):
        """Test that window aggregation parameters result in correct behavior."""
        keys = ['key1']
        window = WindowParams(
            mode=WindowMode.window,
            window_duration_seconds=10.5,
            aggregation=WindowAggregation.mean,
        )

        extractors = create_extractors_from_params(keys=keys, window=window, spec=None)

        extractor = extractors['key1']
        assert isinstance(extractor, WindowAggregatingExtractor)
        # Verify timespan through public interface
        assert extractor.get_required_timespan() == 10.5

        # Verify aggregation behavior by extracting data
        data = sc.DataArray(
            sc.array(dims=['time', 'x'], values=[[2, 4], [4, 6]], unit='m'),
            coords={
                'time': sc.array(dims=['time'], values=[0.0, 1.0], unit='s'),
                'x': sc.arange('x', 2, unit='m'),
            },
        )
        result = extractor.extract(data)
        # Mean of [2, 4] and [4, 6] = [3, 5], verifying mean aggregation was used
        assert sc.allclose(
            result.data, sc.array(dims=['x'], values=[3.0, 5.0], unit='m')
        )


class TestRateNormalizationParams:
    """Tests for RateNormalizationParams on windowed param classes."""

    def test_plot_params_1d_has_rate_field_default_false(self):
        from ess.livedata.dashboard.plot_params import PlotParams1d

        params = PlotParams1d()
        assert params.rate.normalize_to_rate is False

    def test_plot_params_2d_has_rate_field_default_false(self):
        from ess.livedata.dashboard.plot_params import PlotParams2d

        params = PlotParams2d()
        assert params.rate.normalize_to_rate is False

    def test_plot_params_3d_has_rate_field_default_false(self):
        from ess.livedata.dashboard.plot_params import PlotParams3d

        params = PlotParams3d()
        assert params.rate.normalize_to_rate is False

    def test_plot_params_bars_has_rate_field_default_false(self):
        from ess.livedata.dashboard.plot_params import PlotParamsBars

        params = PlotParamsBars()
        assert params.rate.normalize_to_rate is False

    def test_rate_field_can_be_set_to_true(self):
        from ess.livedata.dashboard.plot_params import (
            PlotParams1d,
            RateNormalizationParams,
        )

        params = PlotParams1d(
            rate=RateNormalizationParams(normalize_to_rate=True),
        )
        assert params.rate.normalize_to_rate is True

    def test_display_params_do_not_have_rate_field(self):
        """PlotDisplayParams1d/2d (used by correlation histograms) should not
        have a rate field, to avoid conflict with their own normalization."""
        from ess.livedata.dashboard.plot_params import (
            PlotDisplayParams1d,
            PlotDisplayParams2d,
        )

        assert not hasattr(PlotDisplayParams1d(), 'rate')
        assert not hasattr(PlotDisplayParams2d(), 'rate')
