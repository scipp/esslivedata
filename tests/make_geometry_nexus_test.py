# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import h5py
import numpy as np
import pytest

from ess.livedata.scripts.make_geometry_nexus import write_minimal_geometry


@pytest.fixture
def input_file(tmp_path):
    """Create an input NeXus file with typical instrument structure."""
    path = tmp_path / 'input.nxs'
    with h5py.File(path, 'w') as f:
        f.attrs['default'] = 'entry'
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        # Detector with local transformations
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

        # Monitor
        mon = inst.create_group('monitor_1')
        mon.attrs['NX_class'] = 'NXmonitor'
        mon.create_dataset(
            'depends_on',
            data='/entry/instrument/monitor_1/transformations/pos',
        )
        mon_tr = mon.create_group('transformations')
        mon_tr.attrs['NX_class'] = 'NXtransformations'
        ds = mon_tr.create_dataset('pos', data=2.0)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

        # Source
        source = inst.create_group('source')
        source.attrs['NX_class'] = 'NXsource'
        source.create_dataset('depends_on', data='.')

    return path


def test_copies_detector_and_monitor_geometry(input_file, tmp_path):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(input_file, output)

    with h5py.File(output, 'r') as f:
        assert 'entry/instrument/detector_0' in f
        assert 'entry/instrument/detector_0/detector_number' in f
        assert 'entry/instrument/detector_0/transformations/offset' in f
        assert 'entry/instrument/monitor_1' in f
        assert 'entry/instrument/monitor_1/transformations/pos' in f
        assert 'entry/instrument/source' in f


@pytest.fixture
def input_file_with_positioner_chain(tmp_path):
    """NeXus file where a detector depends_on references an NXlog inside
    an NXpositioner (streaming motor position acting as transformation)."""
    path = tmp_path / 'input.nxs'
    with h5py.File(path, 'w') as f:
        f.attrs['default'] = 'entry'
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        # Positioner with NXlog value that acts as a transformation
        carriage = inst.create_group('carriage')
        carriage.attrs['NX_class'] = 'NXpositioner'
        value_log = carriage.create_group('value')
        value_log.attrs['NX_class'] = 'NXlog'
        value_log.attrs['transformation_type'] = 'translation'
        value_log.attrs['vector'] = np.array([0.0, 0.0, 1.0])
        value_log.attrs['depends_on'] = (
            '/entry/instrument/carriage/transformations/zero_offset'
        )
        # Streaming data (would be stripped)
        value_log.create_dataset('time', data=np.array([], dtype='uint64'))
        value_log.create_dataset('value', data=np.array([], dtype='float64'))
        avg = value_log.create_dataset('average_value', data=3.5)
        avg.attrs['units'] = 'mm'

        # Static transformation at end of chain
        tr = carriage.create_group('transformations')
        tr.attrs['NX_class'] = 'NXtransformations'
        ds = tr.create_dataset('zero_offset', data=5.0)
        ds.attrs['transformation_type'] = 'translation'
        ds.attrs['vector'] = [0.0, 0.0, 1.0]
        ds.attrs['units'] = 'm'
        ds.attrs['depends_on'] = '.'

        # Detector whose depends_on references the positioner NXlog
        det = inst.create_group('detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset('depends_on', data='/entry/instrument/carriage/value')
        det.create_dataset('detector_number', data=np.arange(5))
        det.create_dataset('x_pixel_offset', data=np.zeros(5))
        det.create_dataset('y_pixel_offset', data=np.zeros(5))

    return path


def test_resolves_nxlog_transformation_in_depends_on_chain(
    input_file_with_positioner_chain, tmp_path
):
    output = tmp_path / 'output.nxs'
    write_minimal_geometry(input_file_with_positioner_chain, output)

    with h5py.File(output, 'r') as f:
        # The NXlog should be converted to a static transformation dataset
        value_ds = f['entry/instrument/carriage/value']
        assert isinstance(value_ds, h5py.Dataset)
        assert value_ds[()] == pytest.approx(3.5)
        assert value_ds.attrs['units'] == 'mm'
        assert value_ds.attrs['transformation_type'] == 'translation'
        np.testing.assert_array_equal(value_ds.attrs['vector'], [0.0, 0.0, 1.0])
        # Chain continues to the static transformation
        dep = value_ds.attrs['depends_on']
        if isinstance(dep, bytes):
            dep = dep.decode()
        assert 'zero_offset' in dep
        zero = f[dep]
        assert zero[()] == pytest.approx(5.0)
