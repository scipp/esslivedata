# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for histogram slice providers."""

import scipp as sc

from ess.livedata.handlers.detector_view.providers import (
    histogram_slice_readback,
    parse_histogram_slice,
)
from ess.livedata.handlers.detector_view.types import HistogramSliceRequest


class TestParseHistogramSlice:
    """Tests for parse_histogram_slice provider."""

    def test_none_request_returns_none(self):
        """StreamProcessor initializes context keys to None."""
        result = parse_histogram_slice(None)
        assert result is None

    def test_empty_request_returns_none(self):
        request = HistogramSliceRequest(sc.DataArray(data=sc.zeros(sizes={'bound': 0})))
        result = parse_histogram_slice(request)
        assert result is None

    def test_parses_range_with_unit(self):
        low = sc.scalar(1000.0, unit='ns')
        high = sc.scalar(5000.0, unit='ns')
        request = HistogramSliceRequest(
            sc.DataArray(data=sc.concat([low, high], dim='bound'))
        )
        result = parse_histogram_slice(request)
        assert result is not None
        assert sc.identical(result[0], low)
        assert sc.identical(result[1], high)

    def test_parses_range_without_unit(self):
        low = sc.scalar(10.0)
        high = sc.scalar(50.0)
        request = HistogramSliceRequest(
            sc.DataArray(data=sc.concat([low, high], dim='bound'))
        )
        result = parse_histogram_slice(request)
        assert result is not None
        assert sc.identical(result[0], low)
        assert sc.identical(result[1], high)

    def test_round_trip_with_range_publisher_format(self):
        """Verify parse works with the format RangePublisher produces."""
        low, high, unit = 1000.0, 5000.0, 'ns'
        data = sc.concat(
            [sc.scalar(low, unit=unit), sc.scalar(high, unit=unit)], dim='bound'
        )
        request = HistogramSliceRequest(sc.DataArray(data=data))
        result = parse_histogram_slice(request)
        assert result is not None
        assert result[0].value == low
        assert result[1].value == high
        assert str(result[0].unit) == unit


class TestHistogramSliceReadback:
    """Tests for histogram_slice_readback provider."""

    def _bins(self, unit='ms'):
        return sc.linspace('tof', 0.0, 100.0, 10, unit=unit)

    def test_echoes_request(self):
        data = sc.DataArray(
            data=sc.concat(
                [sc.scalar(1.0, unit='ns'), sc.scalar(2.0, unit='ns')], dim='bound'
            )
        )
        request = HistogramSliceRequest(data)
        readback = histogram_slice_readback(request, self._bins())
        assert isinstance(readback, sc.DataArray)
        assert sc.identical(readback, request)

    def test_none_request_returns_empty_with_unit(self):
        """StreamProcessor initializes context keys to None."""
        readback = histogram_slice_readback(None, self._bins(unit='ms'))
        assert readback.sizes == {'bound': 0}
        assert readback.data.unit == 'ms'

    def test_echoes_empty_request(self):
        request = HistogramSliceRequest(sc.DataArray(data=sc.zeros(sizes={'bound': 0})))
        readback = histogram_slice_readback(request, self._bins())
        assert readback.sizes == {'bound': 0}
