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
