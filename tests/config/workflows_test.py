# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.config.workflows import LatestValueAccumulator


class TestLatestValueAccumulator:
    @pytest.fixture
    def accumulator(self):
        return LatestValueAccumulator()

    def test_is_empty_initially(self, accumulator):
        assert accumulator.is_empty is True

    def test_push_makes_not_empty(self, accumulator):
        value = sc.scalar(1.0, unit='counts')
        accumulator.push(value)
        assert accumulator.is_empty is False

    def test_value_returns_latest(self, accumulator):
        value1 = sc.scalar(1.0, unit='counts')
        value2 = sc.scalar(2.0, unit='counts')

        accumulator.push(value1)
        accumulator.push(value2)

        # Should return the latest pushed value
        assert_identical(accumulator.value, value2)

    def test_value_raises_when_empty(self, accumulator):
        with pytest.raises(ValueError, match="Cannot get value from empty"):
            _ = accumulator.value

    def test_clear_resets_to_empty(self, accumulator):
        value = sc.scalar(1.0, unit='counts')
        accumulator.push(value)
        assert accumulator.is_empty is False

        accumulator.clear()

        assert accumulator.is_empty is True

    def test_keeps_only_latest_value_not_history(self, accumulator):
        """Verify accumulator does not sum or aggregate values."""
        value1 = sc.scalar(5.0, unit='counts')
        value2 = sc.scalar(3.0, unit='counts')

        accumulator.push(value1)
        accumulator.push(value2)

        # Should be exactly value2, not value1 + value2
        result = accumulator.value
        assert result.value == 3.0
        assert result.unit == 'counts'

    def test_works_with_dataarray(self, accumulator):
        """Test with DataArray including coordinates."""
        da1 = sc.DataArray(
            sc.scalar(10.0, unit='counts'),
            coords={'time': sc.scalar(1000, unit='ns')},
        )
        da2 = sc.DataArray(
            sc.scalar(20.0, unit='counts'),
            coords={'time': sc.scalar(2000, unit='ns')},
        )

        accumulator.push(da1)
        accumulator.push(da2)

        result = accumulator.value
        assert result.data.value == 20.0
        assert result.coords['time'].value == 2000
