# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the chopperless wavelength lookup-table workflow."""

from __future__ import annotations

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.handlers.wavelength_lut_workflow import (
    WavelengthLutWorkflow,
    create_chopperless_wavelength_lut_workflow,
)
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
)
from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00


def _params(*, neutrons: int = 50_000) -> WavelengthLutParams:
    p = WavelengthLutParams()
    p.simulation.num_simulated_neutrons = neutrons
    return p


def _trigger() -> dict[str, sc.DataArray]:
    return {CHOPPER_CASCADE_SOURCE: sc.DataArray(sc.scalar(1))}


class TestWavelengthLutWorkflow:
    def test_finalize_without_trigger_raises(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        with pytest.raises(RuntimeError):
            wf.finalize()

    def test_chopperless_produces_table_with_expected_dims(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        out = wf.finalize()
        assert set(out) == {WAVELENGTH_LUT_OUTPUT}
        table = out[WAVELENGTH_LUT_OUTPUT]
        assert table.dims == ('distance', 'event_time_offset')
        assert table.unit == sc.units.angstrom
        # Some finite wavelength values are expected; not all NaN.
        assert np.isfinite(table.values).any()

    def test_provenance_coords_attached(self):
        params = _params()
        wf = create_chopperless_wavelength_lut_workflow(params=params)
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]

        for name in (
            'pulse_period',
            'pulse_stride',
            'distance_resolution',
            'time_resolution',
        ):
            assert name in table.coords, name
        assert_identical(table.coords['pulse_period'], params.pulse_period.get())
        assert_identical(
            table.coords['distance_resolution'],
            params.distance_resolution.get(),
        )
        assert_identical(table.coords['time_resolution'], params.time_resolution.get())
        assert int(table.coords['pulse_stride'].value) == params.simulation.pulse_stride

    def test_finalize_caches_result(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        first = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        second = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert first is second  # cached, no recomputation

    def test_clear_resets_state(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        _ = wf.finalize()
        wf.clear()
        with pytest.raises(RuntimeError):
            wf.finalize()

    def test_accumulate_without_trigger_key_is_noop(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        wf.accumulate(
            {'unrelated': sc.DataArray(sc.scalar(0))}, start_time=0, end_time=1
        )
        with pytest.raises(RuntimeError):
            wf.finalize()

    def test_subclass_marker(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        assert isinstance(wf, WavelengthLutWorkflow)


class TestDa00RoundTrip:
    def test_round_trip_preserves_dims_shape_and_provenance_coords(self):
        wf = create_chopperless_wavelength_lut_workflow(params=_params())
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        original = wf.finalize()[WAVELENGTH_LUT_OUTPUT]

        restored = da00_to_scipp(scipp_to_da00(original))

        assert restored.dims == original.dims
        assert restored.shape == original.shape
        assert set(restored.coords) == set(original.coords)
        # NaNs are expected outside the lookup region; compare finite mask + values.
        mask_orig = np.isnan(original.values)
        mask_rest = np.isnan(restored.values)
        np.testing.assert_array_equal(mask_orig, mask_rest)
        np.testing.assert_allclose(
            original.values[~mask_orig], restored.values[~mask_rest]
        )
        for name in (
            'pulse_period',
            'pulse_stride',
            'distance_resolution',
            'time_resolution',
        ):
            assert_identical(restored.coords[name], original.coords[name])
