# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the wavelength lookup-table workflow."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.livedata.handlers.wavelength_lut_workflow import (
    create_wavelength_lut_workflow,
)
from ess.livedata.handlers.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_LUT_OUTPUT,
    WavelengthLutParams,
    delay_setpoint_input,
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


def _write_chopper_nexus(
    path: Path, names: list[str], *, source_z: float = -25.0
) -> None:
    """Build a minimal NeXus file with NXdisk_chopper groups + NXsource.

    Static fields (slit_edges, radius, axle position) are populated; streamed
    quantities (rotation_speed, delay, ...) are length-0 NXlog placeholders,
    matching what ``make_geometry_nexus.py`` produces for real instruments.
    """
    with snx.File(path, 'w') as root:
        entry = root.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)

        source = instrument.create_class('source', snx.NXsource)
        source.create_field('depends_on', sc.scalar('transformations/t1'))
        src_tr = source.create_class('transformations', snx.NXtransformations)
        src_t1 = src_tr.create_field('t1', sc.scalar(source_z, unit='m'))
        src_t1.attrs['depends_on'] = '.'
        src_t1.attrs['transformation_type'] = 'translation'
        src_t1.attrs['vector'] = sc.vector([0.0, 0.0, 1.0]).value

        for i, name in enumerate(names):
            chop = instrument.create_class(name, snx.NXdisk_chopper)
            chop.create_field('depends_on', sc.scalar('transformations/t1'))
            transformations = chop.create_class(
                'transformations', snx.NXtransformations
            )
            t1 = transformations.create_field(
                't1', sc.scalar(-15.0 + 2.0 * i, unit='m')
            )
            t1.attrs['depends_on'] = '.'
            t1.attrs['transformation_type'] = 'translation'
            t1.attrs['vector'] = sc.vector([0.0, 0.0, 1.0]).value
            chop.create_field(
                'slit_edges',
                sc.array(dims=['dim_0'], values=[0.0, 90.0], unit='deg'),
            )
            chop.create_field('radius', sc.scalar(0.35, unit='m'))

    # Add length-0 NXlog placeholders, mirroring what `make_geometry_nexus.py`
    # writes when copying live NXlogs from coda_*.hdf source files.
    with h5py.File(path, 'a') as f:
        for name in names:
            chopper_path = f'/entry/instrument/{name}'
            for log_name, dtype, unit in [
                ('rotation_speed', 'float64', 'Hz'),
                ('rotation_speed_setpoint', 'float64', 'Hz'),
                ('delay', 'float64', 'ns'),
                ('top_dead_center', 'int64', 'ns'),
            ]:
                grp = f[chopper_path].create_group(log_name)
                grp.attrs['NX_class'] = 'NXlog'
                t = grp.create_dataset('time', shape=(0,), dtype='int64')
                t.attrs['units'] = 'ns'
                t.attrs['start'] = '1970-01-01T00:00:00Z'
                if log_name != 'top_dead_center':
                    v = grp.create_dataset('value', shape=(0,), dtype=dtype)
                    v.attrs['units'] = unit


@pytest.fixture
def two_chopper_geometry(tmp_path: Path) -> Path:
    path = tmp_path / 'two_choppers.nxs'
    _write_chopper_nexus(path, ['chopper1', 'chopper2'])
    return path


class TestMultiChopperWorkflow:
    def test_two_choppers_locked_produces_table(
        self, two_chopper_geometry: Path
    ) -> None:
        names = ['chopper1', 'chopper2']
        wf = create_wavelength_lut_workflow(
            params=_params(),
            chopper_names=names,
            nexus_filename=str(two_chopper_geometry),
        )
        aux = {
            speed_setpoint_input('chopper1'): _setpoint_log(14.0, unit='Hz'),
            delay_setpoint_input('chopper1'): _setpoint_log(0.0, unit='ns'),
            speed_setpoint_input('chopper2'): _setpoint_log(14.0, unit='Hz'),
            delay_setpoint_input('chopper2'): _setpoint_log(1_000_000.0, unit='ns'),
        }
        wf.accumulate(aux | _trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert table.dims == ('distance', 'event_time_offset')
        assert table.unit == sc.units.angstrom

    def test_partial_chopper_data_uses_empty_cascade(
        self, two_chopper_geometry: Path
    ) -> None:
        # Only chopper1 setpoints arrive; chopper2 is missing. The workflow
        # should not raise — it computes against the (still-empty) cached
        # cascade, equivalent to a chopperless run.
        wf = create_wavelength_lut_workflow(
            params=_params(),
            chopper_names=['chopper1', 'chopper2'],
            nexus_filename=str(two_chopper_geometry),
        )
        aux = {
            speed_setpoint_input('chopper1'): _setpoint_log(14.0, unit='Hz'),
            delay_setpoint_input('chopper1'): _setpoint_log(0.0, unit='ns'),
        }
        wf.accumulate(aux | _trigger(), start_time=0, end_time=1)
        table = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert table.dims == ('distance', 'event_time_offset')

    def test_missing_chopper_in_artifact_raises(self, tmp_path: Path) -> None:
        path = tmp_path / 'one_chopper.nxs'
        _write_chopper_nexus(path, ['chopper1'])
        with pytest.raises(ValueError, match=r'missing.*chopper2'):
            create_wavelength_lut_workflow(
                params=_params(),
                chopper_names=['chopper1', 'chopper2'],
                nexus_filename=str(path),
            )

    def test_missing_filename_for_chopper_equipped_raises(self) -> None:
        with pytest.raises(ValueError, match='nexus_filename is required'):
            create_wavelength_lut_workflow(
                params=_params(),
                chopper_names=['chopper1'],
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
