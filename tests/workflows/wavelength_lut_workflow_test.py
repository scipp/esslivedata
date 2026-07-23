# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the wavelength lookup-table workflow (no-chopper and chopper)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pydantic
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.livedata.config.chopper import delay_setpoint_stream, speed_setpoint_stream
from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00
from ess.livedata.workflows.wavelength_lut_workflow import (
    create_wavelength_lut_workflow,
    make_chopper_setpoint_keys,
)
from ess.livedata.workflows.wavelength_lut_workflow_specs import (
    CHOPPER_CASCADE_SOURCE,
    WAVELENGTH_BANDS_OUTPUT,
    WAVELENGTH_LUT_OUTPUT,
    CascadeBands,
    Pulse,
    SourceOffset,
    WavelengthLutParams,
)


def _params() -> WavelengthLutParams:
    return WavelengthLutParams()


def _trigger() -> dict[str, sc.DataArray]:
    return {CHOPPER_CASCADE_SOURCE: sc.DataArray(sc.scalar(1))}


def _write_translation(group, z: float) -> None:
    """Place ``group`` at ``z`` along the beam via a single translation chain."""
    group.create_field('depends_on', sc.scalar('transformations/t1'))
    transformations = group.create_class('transformations', snx.NXtransformations)
    t1 = transformations.create_field('t1', sc.scalar(z, unit='m'))
    t1.attrs['depends_on'] = '.'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['vector'] = sc.vector([0.0, 0.0, 1.0]).value


def _write_chopper_nexus(
    path: Path,
    names: list[str],
    *,
    source_z: float = -25.0,
    moderator_z: float | None = None,
):
    """Build a minimal NeXus file with NXdisk_chopper groups + a neutron source.

    Static fields (slit_edges, radius, axle position) are populated; streamed
    quantities are length-0 NXlog placeholders, matching what
    ``make_geometry_nexus.py`` writes for real instruments. An empty ``names``
    yields a source-only file: the no-chopper geometry an instrument without
    choppers supplies.

    By default the neutron source is a single ``NXsource`` at ``source_z``. When
    ``moderator_z`` is given the file follows BIFROST's convention instead: the
    ``NXsource`` is parked at the origin (accelerator metadata, position
    irrelevant) and the real source is an ``NXmoderator`` at ``moderator_z``.
    """
    with snx.File(path, 'w') as root:
        entry = root.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)

        source = instrument.create_class('source', snx.NXsource)
        _write_translation(source, 0.0 if moderator_z is not None else source_z)
        if moderator_z is not None:
            moderator = instrument.create_class('moderator', snx.NXmoderator)
            _write_translation(moderator, moderator_z)

        for i, name in enumerate(names):
            chop = instrument.create_class(name, snx.NXdisk_chopper)
            _write_translation(chop, -15.0 + 2.0 * i)
            chop.create_field(
                'slit_edges', sc.array(dims=['dim_0'], values=[0.0, 90.0], unit='deg')
            )
            chop.create_field('radius', sc.scalar(0.35, unit='m'))

    with h5py.File(path, 'a') as f:
        for name in names:
            for log_name, dtype, unit in [
                ('rotation_speed', 'float64', 'Hz'),
                ('rotation_speed_setpoint', 'float64', 'Hz'),
                ('delay', 'float64', 'ns'),
                ('top_dead_center', 'int64', 'ns'),
            ]:
                grp = f[f'/entry/instrument/{name}'].create_group(log_name)
                grp.attrs['NX_class'] = 'NXlog'
                t = grp.create_dataset('time', shape=(0,), dtype='int64')
                t.attrs['units'] = 'ns'
                t.attrs['start'] = '1970-01-01T00:00:00Z'
                if log_name != 'top_dead_center':
                    v = grp.create_dataset('value', shape=(0,), dtype=dtype)
                    v.attrs['units'] = unit


def _nxlog(value: float, unit: str | None) -> sc.DataArray:
    """A cumulative NXlog timeseries, as ``set_context`` delivers it."""
    t = sc.epoch(unit='ns') + sc.arange('time', 3, unit='ns')
    return sc.DataArray(
        sc.full(value=value, sizes={'time': 3}, unit=unit), coords={'time': t}
    )


@pytest.fixture
def no_chopper_geometry(tmp_path: Path) -> Path:
    path = tmp_path / 'no_choppers.nxs'
    _write_chopper_nexus(path, [])
    return path


def _build_no_chopper_workflow(geom: Path):
    """Build the LUT workflow against a chopper-free geometry file."""
    wf = create_wavelength_lut_workflow(
        params=_params(), setpoint_keys={}, nexus_filename=str(geom)
    )
    wf.build()
    return wf


@pytest.fixture
def lut(no_chopper_geometry: Path) -> sc.DataArray:
    wf = _build_no_chopper_workflow(no_chopper_geometry)
    wf.accumulate(_trigger(), start_time=0, end_time=1)
    return wf.finalize()[WAVELENGTH_LUT_OUTPUT]


class TestCascadeBands:
    def test_empty_default_yields_no_distances(self) -> None:
        assert CascadeBands().get_distances().sizes == {'distance': 0}

    def test_parses_comma_separated_values(self) -> None:
        var = CascadeBands(distances='6.2, 9.8, 13.0').get_distances()
        assert var.values.tolist() == [6.2, 9.8, 13.0]
        assert var.unit == sc.Unit('m')

    def test_whitespace_only_is_empty(self) -> None:
        assert CascadeBands(distances='   ').get_distances().sizes == {'distance': 0}

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CascadeBands(distances='6.2, foo')


class TestSourceOffset:
    def test_default_is_zero(self) -> None:
        assert_identical(
            SourceOffset().get(), sc.scalar(0.0, unit='m') * sc.vector([0, 0, 1.0])
        )

    def test_offset_is_beam_aligned(self) -> None:
        vec = SourceOffset(offset=2.5).get()
        assert vec.fields.x.value == 0.0
        assert vec.fields.y.value == 0.0
        assert vec.fields.z.value == 2.5
        assert vec.unit == sc.Unit('m')

    def test_negative_offset(self) -> None:
        assert SourceOffset(offset=-3.0).get().fields.z.value == -3.0


class TestNoChopperWorkflow:
    def test_produces_table_with_expected_dims(self, lut: sc.DataArray) -> None:
        assert lut.dims == ('distance', 'event_time_offset')
        assert lut.unit == sc.units.angstrom
        # Some finite wavelength values are expected; not all NaN.
        assert np.isfinite(lut.values).any()

    def test_provenance_coords_attached(self, no_chopper_geometry: Path) -> None:
        params = _params()
        wf = create_wavelength_lut_workflow(
            params=params, setpoint_keys={}, nexus_filename=str(no_chopper_geometry)
        )
        wf.build()
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

    def test_clear_then_retrigger_produces_fresh_table(
        self, no_chopper_geometry: Path
    ) -> None:
        wf = _build_no_chopper_workflow(no_chopper_geometry)
        wf.accumulate(_trigger(), start_time=0, end_time=1)
        first = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        wf.clear()
        wf.accumulate(_trigger(), start_time=2, end_time=3)
        second = wf.finalize()[WAVELENGTH_LUT_OUTPUT]
        assert first is not second
        assert first.dims == second.dims
        assert first.unit == second.unit


def _run_chopper_workflow(
    geom: Path,
    names: list[str],
    setpoints: dict[str, tuple[float, float]],
    params: WavelengthLutParams | None = None,
) -> dict[str, sc.DataArray]:
    """Build the chopper LUT workflow, feed setpoint context + trigger, finalize.

    ``setpoints`` maps chopper name to ``(speed_hz, delay_ns)``. Choppers absent
    from ``setpoints`` get no context value (simulating a not-yet-locked chopper).
    Returns the full ``finalize`` output (LUT and cascade bands).
    """
    keys = {name: make_chopper_setpoint_keys(name) for name in names}
    wf = create_wavelength_lut_workflow(
        params=params or _params(), setpoint_keys=keys, nexus_filename=str(geom)
    )
    context_keys = {}
    for name in names:
        context_keys[speed_setpoint_stream(name)] = keys[name].speed
        context_keys[delay_setpoint_stream(name)] = keys[name].delay
    wf.build(context_keys=context_keys)

    data = dict(_trigger())
    for name, (speed, delay) in setpoints.items():
        data[speed_setpoint_stream(name)] = _nxlog(speed, 'Hz')
        data[delay_setpoint_stream(name)] = _nxlog(delay, 'ns')
    wf.accumulate(data, start_time=0, end_time=1)
    return wf.finalize()


def _run_chopper_lut(
    geom: Path,
    names: list[str],
    setpoints: dict[str, tuple[float, float]],
    params: WavelengthLutParams | None = None,
) -> sc.DataArray:
    """Build the chopper LUT workflow and return just the lookup-table output."""
    return _run_chopper_workflow(geom, names, setpoints, params)[WAVELENGTH_LUT_OUTPUT]


@pytest.fixture
def two_chopper_geometry(tmp_path: Path) -> Path:
    path = tmp_path / 'two_choppers.nxs'
    _write_chopper_nexus(path, ['chopper1', 'chopper2'])
    return path


class TestMultiChopperWorkflow:
    def test_all_choppers_locked_produces_table(
        self, two_chopper_geometry: Path
    ) -> None:
        names = ['chopper1', 'chopper2']
        # Frequencies are negative: ESS choppers rotate anti-clockwise. A positive
        # frequency drives the analytical chopper-cascade into a degenerate frame.
        table = _run_chopper_lut(
            two_chopper_geometry,
            names,
            {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 1_000_000.0)},
        )
        assert table.dims == ('distance', 'event_time_offset')
        assert table.unit == sc.units.angstrom
        assert np.isfinite(table.values).any()

    def test_delay_setpoint_changes_geometry(self, two_chopper_geometry: Path) -> None:
        names = ['chopper1', 'chopper2']
        a = _run_chopper_lut(
            two_chopper_geometry,
            names,
            {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 0.0)},
        )
        b = _run_chopper_lut(
            two_chopper_geometry,
            names,
            {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 2_000_000.0)},
        )
        # A different delay setpoint yields a different table.
        assert not np.array_equal(np.nan_to_num(a.values), np.nan_to_num(b.values))

    def test_auto_stride_guessed_from_slow_chopper(
        self, two_chopper_geometry: Path
    ) -> None:
        # auto_stride is the default. A chopper at half the pulse frequency
        # (7 Hz vs 14 Hz) implies pulse-skipping with stride 2; the workflow's
        # guess provider derives it and the provenance coord reflects it,
        # despite params.pulse.stride staying at its default of 1.
        params = _params()
        assert params.pulse.auto_stride is True
        assert params.pulse.stride == 1
        table = _run_chopper_lut(
            two_chopper_geometry,
            ['chopper1', 'chopper2'],
            {'chopper1': (-7.0, 0.0), 'chopper2': (-14.0, 0.0)},
            params=params,
        )
        assert int(table.coords['pulse_stride'].value) == 2

    def test_manual_stride_overrides_choppers(self, two_chopper_geometry: Path) -> None:
        # With auto-detection disabled the supplied stride is used verbatim,
        # ignoring what the chopper frequencies would imply.
        params = WavelengthLutParams(pulse=Pulse(auto_stride=False, stride=1))
        table = _run_chopper_lut(
            two_chopper_geometry,
            ['chopper1', 'chopper2'],
            {'chopper1': (-7.0, 0.0), 'chopper2': (-14.0, 0.0)},
            params=params,
        )
        assert int(table.coords['pulse_stride'].value) == 1

    def test_distance_measured_from_moderator_not_accelerator_source(
        self, tmp_path: Path
    ) -> None:
        # BIFROST models the moderator as NXmoderator and reserves NXsource for
        # accelerator metadata at the origin. essreduce keys the cascade
        # reference off the unique NXsource, so measuring from it would place
        # the upstream choppers behind the source (negative distance) and the
        # cascade would reject them. The workflow resolves the source from the
        # NXmoderator instead, so the table computes.
        path = tmp_path / 'moderator_choppers.nxs'
        _write_chopper_nexus(path, ['chopper1', 'chopper2'], moderator_z=-25.0)
        table = _run_chopper_lut(
            path,
            ['chopper1', 'chopper2'],
            {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 0.0)},
        )
        assert table.dims == ('distance', 'event_time_offset')
        assert np.isfinite(table.values).any()

    def test_missing_chopper_in_artifact_raises(self, tmp_path: Path) -> None:
        # A configured chopper absent from the geometry artifact is not validated
        # at factory time; it surfaces as a KeyError when the table is computed.
        path = tmp_path / 'one_chopper.nxs'
        _write_chopper_nexus(path, ['chopper1'])
        with pytest.raises(KeyError, match='chopper2'):
            _run_chopper_lut(
                path,
                ['chopper1', 'chopper2'],
                {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 0.0)},
            )


class TestSourceOffsetWorkflow:
    def test_offset_shifts_chopper_distances_from_moderator(
        self, tmp_path: Path
    ) -> None:
        # Moderator at -25 m, choppers at -15 and -13 m: distances 10 and 12 m
        # from the source. A +2 m offset moves the source downstream to -23 m,
        # shrinking every chopper's distance-from-source by 2 m.
        path = tmp_path / 'moderator_choppers.nxs'
        _write_chopper_nexus(path, ['chopper1', 'chopper2'], moderator_z=-25.0)
        names = ['chopper1', 'chopper2']
        setpoints = {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 0.0)}

        base = _run_chopper_workflow(path, names, setpoints)[WAVELENGTH_BANDS_OUTPUT]
        shifted = _run_chopper_workflow(
            path,
            names,
            setpoints,
            params=WavelengthLutParams(source=SourceOffset(offset=2.0)),
        )[WAVELENGTH_BANDS_OUTPUT]

        # Source row stays at distance 0; chopper rows move closer by 2 m.
        base_d = base.coords['distance']
        shifted_d = shifted.coords['distance']
        assert_identical(base_d.max() - sc.scalar(2.0, unit='m'), shifted_d.max())

    def test_offset_applies_without_moderator(self, two_chopper_geometry: Path) -> None:
        # NXsource-only file (no moderator): the offset still shifts the
        # reference, exercising the NXsource branch. The table differs from the
        # unshifted one and remains finite.
        names = ['chopper1', 'chopper2']
        setpoints = {'chopper1': (-14.0, 0.0), 'chopper2': (-14.0, 0.0)}
        base = _run_chopper_lut(two_chopper_geometry, names, setpoints)
        shifted = _run_chopper_lut(
            two_chopper_geometry,
            names,
            setpoints,
            params=WavelengthLutParams(source=SourceOffset(offset=1.5)),
        )
        assert np.isfinite(shifted.values).any()
        assert not np.array_equal(
            np.nan_to_num(base.values), np.nan_to_num(shifted.values)
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
