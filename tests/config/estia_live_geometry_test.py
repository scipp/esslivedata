# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Check the new CODA geometry against the live reduction's actual providers."""

import shutil

import h5py
import pytest
import scipp as sc
from ess.estia import EstiaWorkflow
from ess.reduce.nexus.types import (
    Filename,
    NeXusTransformationChain,
    Position,
    ProtonCharge,
    SampleRun,
)
from ess.reduce.unwrap.types import DetectorLtotal
from ess.reflectometry.types import (
    DetectorRotation,
    RawSampleRotation,
    SampleRotation,
)
from scippnexus import NXdetector, NXsample, NXsource

from ess.livedata.config.instruments.estia.live_geometry import (
    insert_live_geometry,
    load_live_geometry,
    reference_detector_ltotal,
)
from ess.livedata.config.instruments.estia.specs import instrument
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename
from ess.livedata.workflows.lut_ranges import (
    component_ltotal_range,
    padded_ltotal_range,
)

pytestmark = pytest.mark.slow


def _log(value, unit):
    return sc.DataArray(
        sc.array(dims=['time'], values=[-1.0, value], unit=unit),
        coords={'time': sc.datetimes(dims=['time'], values=[0, 1], unit='ns')},
    )


@pytest.fixture
def geometry():
    return load_live_geometry(get_nexus_geometry_filename('estia'), instrument)


@pytest.fixture
def workflow(geometry):
    wf = EstiaWorkflow()
    wf[Filename[SampleRun]] = get_nexus_geometry_filename('estia')
    insert_live_geometry(wf, geometry)
    for stream, key in geometry.context_keys.items():
        wf[key] = key(values=_log(0.0, instrument.streams[stream].units))
    return wf


def test_source_and_monitor_distances_come_from_new_file(workflow):
    source = workflow.compute(Position[NXsource, SampleRun])
    assert sc.isclose(sc.norm(source).to(unit='m'), sc.scalar(35.0, unit='m'))
    start, stop = component_ltotal_range(
        get_nexus_geometry_filename('estia'), 'beam_monitor', is_monitor=True
    )
    assert sc.isclose((start + stop) / 2, sc.scalar(10.68, unit='m'))
    assert callable(instrument.declared_ltotal['multiblade_detector'])


@pytest.mark.parametrize('angle', [0.0, 10.0, 90.0, 180.0])
def test_rotating_arm_stays_in_geometry_derived_range(workflow, geometry, angle):
    key = geometry.context_keys['detector_arm_rotation']
    workflow[key] = key(values=_log(angle, 'deg'))
    assert sc.identical(
        workflow.compute(DetectorRotation[SampleRun]), sc.scalar(angle, unit='deg')
    )
    start, stop = padded_ltotal_range(
        reference_detector_ltotal(str(get_nexus_geometry_filename('estia')))
    )
    paths = workflow.compute(DetectorLtotal[SampleRun]).to(unit='m')
    assert paths.min() >= start
    assert paths.max() <= stop


@pytest.mark.parametrize(('axis', 'index'), [('tx', 0), ('ty', 1), ('tz', 2)])
def test_hexapod_readback_moves_sample_position(workflow, geometry, axis, index):
    key = geometry.context_keys[f'hexapod_{axis}']
    workflow[key] = key(values=_log(2.0, 'mm'))
    expected = [0.0, 0.0, 0.0]
    expected[index] = 2.0
    assert sc.allclose(
        workflow.compute(Position[NXsample, SampleRun]).to(unit='mm'),
        sc.vector(expected, unit='mm'),
    )


def test_sample_rotation_uses_latest_readback(workflow, geometry):
    key = geometry.context_keys['sample_rotation']
    workflow[key] = key(values=_log(0.7, 'deg'))
    assert sc.identical(
        workflow.compute(RawSampleRotation[SampleRun]), sc.scalar(0.7, unit='deg')
    )
    assert sc.isclose(
        workflow.compute(SampleRotation[SampleRun]), sc.scalar(0.7, unit='deg')
    )


def test_shared_lateral_readback_patches_detector_and_sample(workflow, geometry):
    key = geometry.context_keys['sample_lateral_adjustment']
    workflow[key] = key(values=_log(4.0, 'mm'))
    for component in (NXsample, NXdetector):
        (path,) = [
            path for path, bound_key in geometry.bindings[component] if key == bound_key
        ]
        chain = workflow.compute(NeXusTransformationChain[component, SampleRun])
        assert sc.identical(
            chain.transformations[path].value, sc.scalar(4.0, unit='mm')
        )


def test_charge_keeps_measured_units_and_times(workflow, geometry):
    charge = _log(1.5, 'uC')
    key = geometry.context_keys['source/pulse_charge']
    workflow[key] = key(values=charge)
    assert sc.identical(workflow.compute(ProtonCharge[SampleRun]), charge)


def test_missing_motor_metadata_does_not_silently_assume_zero(tmp_path):
    filename = tmp_path / 'missing-stream.nxs'
    shutil.copyfile(get_nexus_geometry_filename('estia'), filename)
    with h5py.File(filename, 'a') as f:
        del f['entry/instrument/hexapod/transformations/rz'].attrs['source']
    with pytest.raises(ValueError, match=r'Missing stream metadata.*hexapod'):
        load_live_geometry(filename, instrument)
