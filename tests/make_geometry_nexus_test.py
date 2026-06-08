# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest

from ess.livedata.scripts.make_geometry_nexus import write_minimal_geometry


@pytest.fixture
def basic_nexus_file(tmp_path):
    """NeXus file with a detector whose depends_on chain is self-contained."""
    path = tmp_path / 'input.nxs'
    with h5py.File(path, 'w') as f:
        f.attrs['default'] = 'entry'
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        det = inst.create_group('detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset(
            'depends_on',
            data='/entry/instrument/detector_0/transformations/offset',
        )
        det.create_dataset('detector_number', data=np.arange(10))
        det.create_dataset('x_pixel_offset', data=np.zeros(10))
        det.create_dataset('y_pixel_offset', data=np.zeros(10))
        tr = det.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        ds = tr.create_dataset('offset', data=1.5)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

    return path


def test_copies_detector_with_local_transformations(basic_nexus_file, tmp_path):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(basic_nexus_file, output)

    with h5py.File(output, 'r') as f:
        assert 'entry/instrument/detector_0' in f
        assert 'entry/instrument/detector_0/detector_number' in f
        assert 'entry/instrument/detector_0/transformations/offset' in f
        ds = f['entry/instrument/detector_0/transformations/offset']
        assert ds[()] == pytest.approx(1.5)
        assert ds.attrs['depends_on'] == '.'


def test_copies_event_only_monitor_without_depends_on(tmp_path):
    """A backup monitor carrying only event data has no ``depends_on``; the
    geometry artifact must copy it without crashing on the missing chain."""
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'
        mon = inst.create_group('backup_monitor')
        mon.attrs['NX_class'] = 'NXmonitor'
        mon.create_group('events')

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        mon = f['entry/instrument/backup_monitor']
        assert mon.attrs['NX_class'] == 'NXmonitor'
        assert 'depends_on' not in mon


def test_copies_moderator_depends_on(tmp_path):
    """The neutron source may be modelled as ``NXmoderator`` (BIFROST's
    convention) rather than ``NXsource``. Its ``depends_on`` chain must be
    copied so the source position resolves in the artifact; dropping it leaves
    the moderator at the origin and the chopper cascade measures flight
    distances from the wrong point.
    """
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'
        mod = inst.create_group('source')
        mod.attrs['NX_class'] = 'NXmoderator'
        mod.create_dataset(
            'depends_on',
            data='/entry/instrument/source/transformations/distance',
        )
        tr = mod.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        ds = tr.create_dataset('distance', data=-160.0)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        mod = f['entry/instrument/source']
        assert mod.attrs['NX_class'] == 'NXmoderator'
        assert mod['depends_on'][()].decode() == (
            '/entry/instrument/source/transformations/distance'
        )
        ds = f['entry/instrument/source/transformations/distance']
        assert ds[()] == pytest.approx(-160.0)
        assert ds.attrs['depends_on'] == '.'


@pytest.fixture
def nexus_with_nxlog_chain(tmp_path):
    """NeXus file where a detector's depends_on chain passes through an NXlog
    inside an NXpositioner (streaming motor position as transformation node)."""
    path = tmp_path / 'input.nxs'
    with h5py.File(path, 'w') as f:
        f.attrs['default'] = 'entry'
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        # NXpositioner with an NXlog value acting as a transformation
        carriage = inst.create_group('carriage')
        carriage.attrs['NX_class'] = 'NXpositioner'
        value_log = carriage.create_group('value')
        value_log.attrs['NX_class'] = 'NXlog'
        value_log.attrs['transformation_type'] = 'translation'
        value_log.attrs['vector'] = np.array([0.0, 0.0, 1.0])
        value_log.attrs['depends_on'] = (
            '/entry/instrument/carriage/transformations/zero_offset'
        )
        value_log.create_dataset('time', data=np.array([], dtype='uint64'))
        value_log.create_dataset('value', data=np.array([], dtype='float64'))
        avg = value_log.create_dataset('average_value', data=3.5)
        avg.attrs['units'] = 'mm'

        tr = carriage.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        ds = tr.create_dataset('zero_offset', data=5.0)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

        # Detector whose depends_on references the NXlog
        det = inst.create_group('detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset('depends_on', data='/entry/instrument/carriage/value')
        det.create_dataset('detector_number', data=np.arange(5))
        det.create_dataset('x_pixel_offset', data=np.zeros(5))
        det.create_dataset('y_pixel_offset', data=np.zeros(5))

    return path


def test_copies_nxlog_group_as_is_for_depends_on_chain(
    nexus_with_nxlog_chain, tmp_path
):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(nexus_with_nxlog_chain, output)

    with h5py.File(output, 'r') as f:
        # NXlog group should be copied as a group, not converted to a dataset
        value = f['entry/instrument/carriage/value']
        assert isinstance(value, h5py.Group)
        assert value.attrs['NX_class'] == 'NXlog'
        assert value.attrs['transformation_type'] == 'translation'
        np.testing.assert_array_equal(value.attrs['vector'], [0.0, 0.0, 1.0])

        # Chain continues to the static transformation
        dep = value.attrs['depends_on']
        if isinstance(dep, bytes):
            dep = dep.decode()
        assert 'zero_offset' in dep
        zero = f[dep]
        assert zero[()] == pytest.approx(5.0)


def test_copies_nxlog_children(nexus_with_nxlog_chain, tmp_path):
    """NXlog children (time, value, average_value) are preserved."""
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(nexus_with_nxlog_chain, output)

    with h5py.File(output, 'r') as f:
        value_log = f['entry/instrument/carriage/value']
        assert 'time' in value_log
        assert 'value' in value_log
        assert 'average_value' in value_log


@pytest.fixture
def nexus_with_disk_chopper(tmp_path):
    """NeXus file with an NXdisk_chopper carrying static geometry fields,
    length-0 NXlog placeholders for streamed values, and a transformations
    group reached via depends_on."""
    path = tmp_path / 'input.nxs'
    with h5py.File(path, 'w') as f:
        f.attrs['default'] = 'entry'
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        chop = inst.create_group('chopper1')
        chop.attrs['NX_class'] = 'NXdisk_chopper'
        chop.create_dataset(
            'depends_on',
            data='/entry/instrument/chopper1/transformations/location',
        )
        # Static geometry fields
        slits = chop.create_dataset('slits', data=2)
        chop.create_dataset(
            'slit_edges', data=np.array([0.0, 90.0, 180.0, 270.0])
        ).attrs['units'] = 'deg'
        chop.create_dataset('slit_height', data=0.1).attrs['units'] = 'm'
        chop.create_dataset('radius', data=0.35).attrs['units'] = 'm'
        slits.attrs['units'] = ''

        # Length-0 NXlog placeholders for streamed values. ``delay`` is
        # dynamic — operators retune it during a run.
        for log_name in ('rotation_speed', 'phase', 'delay'):
            log = chop.create_group(log_name)
            log.attrs['NX_class'] = 'NXlog'
            log.create_dataset('time', data=np.array([], dtype='uint64'))
            log.create_dataset('value', data=np.array([], dtype='float64'))

        tr = chop.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        ds = tr.create_dataset('location', data=15.0)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

    return path


def test_copies_disk_chopper_static_fields(nexus_with_disk_chopper, tmp_path):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(nexus_with_disk_chopper, output)

    with h5py.File(output, 'r') as f:
        chop = f['entry/instrument/chopper1']
        assert chop.attrs['NX_class'] == 'NXdisk_chopper'
        assert chop['slits'][()] == 2
        np.testing.assert_array_equal(chop['slit_edges'][:], [0.0, 90.0, 180.0, 270.0])
        assert chop['radius'][()] == pytest.approx(0.35)


def test_copies_disk_chopper_nxlog_placeholders(nexus_with_disk_chopper, tmp_path):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(nexus_with_disk_chopper, output)

    with h5py.File(output, 'r') as f:
        # ``delay`` is dynamic in production (operators tune it live), hence
        # an NXlog rather than a static dataset.
        for log_name in ('rotation_speed', 'phase', 'delay'):
            log = f[f'entry/instrument/chopper1/{log_name}']
            assert isinstance(log, h5py.Group)
            assert log.attrs['NX_class'] == 'NXlog'
            assert log['time'].shape == (0,)
            assert log['value'].shape == (0,)


def test_copies_disk_chopper_transformations(nexus_with_disk_chopper, tmp_path):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(nexus_with_disk_chopper, output)

    with h5py.File(output, 'r') as f:
        tr = f['entry/instrument/chopper1/transformations/location']
        assert tr[()] == pytest.approx(15.0)
        assert tr.attrs['transformation_type'] == 'translation'


def test_nested_nxtransformations_inside_nxlog_does_not_conflict(tmp_path):
    """Case where an outer NXtransformations contains an NXlog whose own
    children include another NXtransformations (motor stage with chained
    transforms). The outer branch's recursive NXlog copy must not conflict
    with the visitor's later descent into the inner group.
    """
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        outer = inst.create_group('stage/transformations')
        outer.attrs['NX_class'] = 'NXtransformations'
        rx = outer.create_group('rx')
        rx.attrs['NX_class'] = 'NXlog'
        rx.create_dataset('time', data=np.arange(100, dtype='int64'))
        rx.create_dataset(
            'value', data=np.linspace(0.0, 1.0, 100, dtype='float64')
        ).attrs['units'] = 'deg'
        # Nested NXtransformations inside the NXlog.
        inner = rx.create_group('transformations')
        inner.attrs['NX_class'] = 'NXtransformations'
        ds = inner.create_dataset('rotation', data=0.0)
        ds.attrs['transformation_type'] = 'rotation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'deg'
        ds.attrs['depends_on'] = '.'

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        # Outer NXtransformations preserved.
        assert f['entry/instrument/stage/transformations'].attrs['NX_class'] == (
            'NXtransformations'
        )
        # NXlog samples trimmed; dtype/units/attributes preserved.
        rx = f['entry/instrument/stage/transformations/rx']
        assert rx.attrs['NX_class'] == 'NXlog'
        assert rx['time'].shape == (0,)
        assert rx['value'].shape == (0,)
        assert rx['value'].dtype == np.float64
        assert rx['value'].attrs['units'] == 'deg'
        # Inner NXtransformations preserved with its dataset.
        inner_path = (
            'entry/instrument/stage/transformations/rx/transformations/rotation'
        )
        inner_ds = f[inner_path]
        assert inner_ds[()] == pytest.approx(0.0)
        assert inner_ds.attrs['transformation_type'] == 'rotation'


def test_nxlog_inside_nxtransformations_is_trimmed(tmp_path):
    """An NXtransformations group may contain a streamed transformation
    (NXlog with rotation/translation values). Its samples must be trimmed
    even though the parent group is the dedicated NXtransformations branch.
    """
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'
        det = inst.create_group('detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset(
            'depends_on', data='/entry/instrument/detector_0/transformations/rotation'
        )
        det.create_dataset('detector_number', data=np.arange(5))
        det.create_dataset('x_pixel_offset', data=np.zeros(5))
        det.create_dataset('y_pixel_offset', data=np.zeros(5))

        tr = det.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        log = tr.create_group('rotation')
        log.attrs['NX_class'] = 'NXlog'
        log.attrs['transformation_type'] = 'rotation'
        log.attrs['vector'] = np.array([0.0, 1.0, 0.0])
        log.attrs['depends_on'] = '.'
        log.create_dataset('time', data=np.arange(2000, dtype='int64'))
        log.create_dataset(
            'value', data=np.linspace(0.0, 90.0, 2000, dtype='float64')
        ).attrs['units'] = 'deg'

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        log = f['entry/instrument/detector_0/transformations/rotation']
        assert log.attrs['NX_class'] == 'NXlog'
        assert log.attrs['transformation_type'] == 'rotation'
        assert log['time'].shape == (0,)
        assert log['value'].shape == (0,)
        assert log['value'].attrs['units'] == 'deg'


def test_motor_log_reached_via_depends_on_is_trimmed(tmp_path):
    """NXpositioner whose value NXlog (reached via depends_on) carries real
    samples — production motor readback. The geometry artifact must trim
    these to length-0 while keeping the transformation attributes the
    depends_on chain needs.
    """
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        carriage = inst.create_group('carriage')
        carriage.attrs['NX_class'] = 'NXpositioner'
        value_log = carriage.create_group('value')
        value_log.attrs['NX_class'] = 'NXlog'
        value_log.attrs['transformation_type'] = 'translation'
        value_log.attrs['vector'] = np.array([0.0, 0.0, 1.0])
        value_log.attrs['depends_on'] = '.'
        # 5000 motor readbacks recorded during the run.
        value_log.create_dataset('time', data=np.arange(5000, dtype='int64')).attrs[
            'units'
        ] = 'ns'
        value_log.create_dataset(
            'value', data=np.linspace(0.0, 100.0, 5000, dtype='float64')
        ).attrs['units'] = 'mm'

        det = inst.create_group('detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset('depends_on', data='/entry/instrument/carriage/value')
        det.create_dataset('detector_number', data=np.arange(5))
        det.create_dataset('x_pixel_offset', data=np.zeros(5))
        det.create_dataset('y_pixel_offset', data=np.zeros(5))

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        log = f['entry/instrument/carriage/value']
        assert log.attrs['NX_class'] == 'NXlog'
        # Transformation attributes preserved — depends_on chain still resolves.
        assert log.attrs['transformation_type'] == 'translation'
        np.testing.assert_array_equal(log.attrs['vector'], [0.0, 0.0, 1.0])
        # Samples dropped, schema preserved.
        assert log['time'].shape == (0,)
        assert log['time'].dtype == np.int64
        assert log['value'].shape == (0,)
        assert log['value'].dtype == np.float64
        assert log['value'].attrs['units'] == 'mm'


def test_disk_chopper_log_data_is_trimmed(tmp_path):
    """NXdisk_chopper whose NXlog children carry real samples (as a
    production file would) — the geometry artifact must trim these to
    length-0 placeholders while preserving dtype, units, and attributes.
    """
    src = tmp_path / 'input.nxs'
    with h5py.File(src, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        chop = inst.create_group('chopper1')
        chop.attrs['NX_class'] = 'NXdisk_chopper'
        chop.create_dataset('radius', data=0.35).attrs['units'] = 'm'

        # Populated log: 1000 samples of phase readback.
        phase = chop.create_group('phase')
        phase.attrs['NX_class'] = 'NXlog'
        phase.create_dataset('time', data=np.arange(1000, dtype='int64')).attrs[
            'units'
        ] = 'ns'
        phase.create_dataset(
            'value', data=np.linspace(0.0, 359.0, 1000, dtype='float32')
        ).attrs['units'] = 'deg'
        # A scalar child that should survive verbatim.
        phase.create_dataset('average_value', data=180.0).attrs['units'] = 'deg'

    output = tmp_path / 'output.nxs'
    write_minimal_geometry(src, output)

    with h5py.File(output, 'r') as f:
        phase = f['entry/instrument/chopper1/phase']
        assert phase.attrs['NX_class'] == 'NXlog'
        assert phase['time'].shape == (0,)
        assert phase['time'].dtype == np.int64
        assert phase['time'].attrs['units'] == 'ns'
        assert phase['value'].shape == (0,)
        assert phase['value'].dtype == np.float32
        assert phase['value'].attrs['units'] == 'deg'
        assert phase['average_value'][()] == pytest.approx(180.0)
        assert phase['average_value'].attrs['units'] == 'deg'
