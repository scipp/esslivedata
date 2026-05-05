# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the wavelength lookup-table workflow."""

from __future__ import annotations

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.livedata.handlers.wavelength_lut_workflow import (
    HardcodedChopperGeometry,
    create_wavelength_lut_workflow,
)
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
    phase_setpoint_input,
    speed_setpoint_input,
)
from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00


def _params(*, neutrons: int = 50_000) -> WavelengthLutParams:
    p = WavelengthLutParams()
    p.simulation.num_simulated_neutrons = neutrons
    return p


def _trigger() -> dict[str, sc.DataArray]:
    return {CHOPPER_CASCADE_SOURCE: sc.DataArray(sc.scalar(1))}


@pytest.fixture
def lut() -> sc.DataArray:
    wf = create_wavelength_lut_workflow(params=_params())
    wf.accumulate(_trigger(), start_time=0, end_time=1)
    return wf.finalize()[WAVELENGTH_LUT_OUTPUT]


class TestWavelengthLutWorkflow:
    def test_chopperless_produces_table_with_expected_dims(
        self, lut: sc.DataArray
    ) -> None:
        assert lut.dims == ('distance', 'event_time_offset')
        assert lut.unit == sc.units.angstrom
        # Some finite wavelength values are expected; not all NaN.
        assert np.isfinite(lut.values).any()

    def test_provenance_coords_attached(self) -> None:
        params = _params()
        wf = create_wavelength_lut_workflow(params=params)
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]

        for name in (
            'pulse_period',
            'pulse_stride',
            'distance_resolution',
            'time_resolution',
        ):
            assert name in table.coords, name
        assert_identical(table.coords['pulse_period'], params.pulse.get_period())
        assert_identical(
            table.coords['distance_resolution'],
            params.distance_resolution.get(),
        )
        assert_identical(table.coords['time_resolution'], params.time_resolution.get())
        assert int(table.coords['pulse_stride'].value) == params.pulse.stride

    def test_clear_then_retrigger_produces_fresh_table(self) -> None:
        wf = create_wavelength_lut_workflow(params=_params())
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        first = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        wf.clear()
        wf.accumulate(_trigger(), start_time=2, end_time=3)
        second = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert first is not second
        assert first.dims == second.dims
        assert first.unit == second.unit


def _setpoint_log(value: float, *, unit: str) -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['time'], values=[value], unit=unit),
        coords={
            'time': sc.array(dims=['time'], values=[0], unit='ns', dtype='datetime64'),
        },
    )


def _geometry(z_metres: float) -> HardcodedChopperGeometry:
    return HardcodedChopperGeometry(
        axle_position=sc.vector([0.0, 0.0, z_metres], unit='m'),
        beam_position=sc.scalar(0.0, unit='deg'),
        slit_begin=sc.array(dims=['slit'], values=[0.0], unit='deg'),
        slit_end=sc.array(dims=['slit'], values=[90.0], unit='deg'),
        radius=sc.scalar(0.35, unit='m'),
    )


class TestMultiChopperWorkflow:
    def test_two_choppers_locked_produces_table(self) -> None:
        names = ['chopper1', 'chopper2']
        wf = create_wavelength_lut_workflow(
            params=_params(),
            chopper_names=names,
            chopper_geometry={
                'chopper1': _geometry(z_metres=10.0),
                'chopper2': _geometry(z_metres=15.0),
            },
        )
        aux = {
            speed_setpoint_input('chopper1'): _setpoint_log(14.0, unit='Hz'),
            phase_setpoint_input('chopper1'): _setpoint_log(0.0, unit='deg'),
            speed_setpoint_input('chopper2'): _setpoint_log(14.0, unit='Hz'),
            phase_setpoint_input('chopper2'): _setpoint_log(45.0, unit='deg'),
        }
        wf.accumulate(aux | _trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert table.dims == ('distance', 'event_time_offset')
        assert table.unit == sc.units.angstrom

    def test_partial_chopper_data_uses_empty_cascade(self) -> None:
        # Only chopper1 setpoints arrive; chopper2 is missing. The workflow
        # should not raise — it computes against the (still-empty) cached
        # cascade, equivalent to a chopperless run.
        wf = create_wavelength_lut_workflow(
            params=_params(),
            chopper_names=['chopper1', 'chopper2'],
            chopper_geometry={
                'chopper1': _geometry(z_metres=10.0),
                'chopper2': _geometry(z_metres=15.0),
            },
        )
        aux = {
            speed_setpoint_input('chopper1'): _setpoint_log(14.0, unit='Hz'),
            phase_setpoint_input('chopper1'): _setpoint_log(0.0, unit='deg'),
        }
        wf.accumulate(aux | _trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert table.dims == ('distance', 'event_time_offset')

    def test_missing_geometry_raises(self) -> None:
        with pytest.raises(ValueError, match='Missing hardcoded geometry'):
            create_wavelength_lut_workflow(
                params=_params(),
                chopper_names=['chopper1'],
                chopper_geometry={},
            )


class TestDa00RoundTrip:
    def test_round_trip_preserves_dims_shape_and_provenance_coords(
        self, lut: sc.DataArray
    ) -> None:
        restored = da00_to_scipp(scipp_to_da00(lut))

        assert restored.dims == lut.dims
        assert restored.shape == lut.shape
        assert set(restored.coords) == set(lut.coords)
        # NaNs are expected outside the lookup region; compare finite mask + values.
        mask_orig = np.isnan(lut.values)
        mask_rest = np.isnan(restored.values)
        np.testing.assert_array_equal(mask_orig, mask_rest)
        np.testing.assert_allclose(lut.values[~mask_orig], restored.values[~mask_rest])
        for name in (
            'pulse_period',
            'pulse_stride',
            'distance_resolution',
            'time_resolution',
        ):
            assert_identical(restored.coords[name], lut.coords[name])
