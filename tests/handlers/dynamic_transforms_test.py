# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the centralised dynamic-transform machinery."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    Filename,
    NeXusName,
    NeXusTransformationChain,
    SampleRun,
)
from scippnexus import NXdetector

from ess.livedata.config import Instrument
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.handlers.dynamic_transforms import (
    DynamicTransformBinding,
    TransformValueLog,
    compose_aux_sources,
)

# --- Test fixtures: minimal artifact builders ---


def _write_chain(
    f: h5py.File, parent: str, name: str, depends_on: str, value: float
) -> None:
    """Write a transformation Dataset with a depends_on attribute."""
    ds = f[parent].create_dataset(name, data=value)
    ds.attrs['depends_on'] = depends_on
    ds.attrs['transformation_type'] = 'translation'
    ds.attrs['vector'] = [0.0, 0.0, 1.0]
    ds.attrs['units'] = 'm'


def _write_nxlog(
    f: h5py.File,
    parent: str,
    name: str,
    *,
    depends_on: str,
    samples: int = 0,
    units: str = 'mm',
) -> None:
    """Write an NXlog group placeholder mirroring the production schema.

    Includes the ``average_value`` / ``minimum_value`` / ``maximum_value``
    scalar children so scippnexus loads the group as a DataArray (with
    those scalar coords) rather than falling back to a bare Variable —
    matches the shape produced by ``make_geometry_nexus.py``.
    """
    g = f[parent].create_group(name)
    g.attrs['NX_class'] = 'NXlog'
    g.attrs['depends_on'] = depends_on
    g.attrs['transformation_type'] = 'translation'
    g.attrs['vector'] = [1.0, 0.0, 0.0]
    val = g.create_dataset(
        'value', shape=(samples,), maxshape=(None,), dtype=np.float64
    )
    val.attrs['units'] = units
    t = g.create_dataset('time', shape=(samples,), maxshape=(None,), dtype=np.int64)
    t.attrs['units'] = 'ns'
    t.attrs['start'] = '1970-01-01T00:00:00'
    for stat in ('average_value', 'minimum_value', 'maximum_value'):
        ds = g.create_dataset(stat, data=0.0)
        ds.attrs['units'] = units


def _make_artifact(
    path, *, m4_via_carriage: bool = False, m4_dynamic_only: bool = False
) -> str:
    """Build a minimal LOKI-shaped artifact for tests."""
    fn = str(path / 'geom.nxs')
    with h5py.File(fn, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        inst = entry.create_group('instrument')
        inst.attrs['NX_class'] = 'NXinstrument'

        # Carriage NXlog (the dynamic placeholder).
        carriage = inst.create_group('detector_carriage')
        carriage.attrs['NX_class'] = 'NXpositioner'
        carriage.create_group('transformations')
        _write_chain(
            f,
            'entry/instrument/detector_carriage/transformations',
            'detector_carriage_zero',
            depends_on='.',
            value=5.098,
        )
        _write_nxlog(
            f,
            'entry/instrument/detector_carriage',
            'value',
            depends_on=(
                '/entry/instrument/detector_carriage/transformations/'
                'detector_carriage_zero'
            ),
            samples=0,
        )

        # loki_detector_0 depends on carriage NXlog.
        det = inst.create_group('loki_detector_0')
        det.attrs['NX_class'] = 'NXdetector'
        det.create_dataset(
            'depends_on', data='/entry/instrument/detector_carriage/value'
        )
        # detector_number sentinel so essreduce loaders don't choke.
        det.create_dataset('detector_number', data=np.array([1, 2, 3]))

        # Static detector for the no-dynamic case.
        static_det = inst.create_group('loki_detector_static')
        static_det.attrs['NX_class'] = 'NXdetector'
        static_det.create_group('transformations')
        _write_chain(
            f,
            'entry/instrument/loki_detector_static/transformations',
            'fixed',
            depends_on='.',
            value=1.0,
        )
        static_det.create_dataset(
            'depends_on',
            data='/entry/instrument/loki_detector_static/transformations/fixed',
        )
        static_det.create_dataset('detector_number', data=np.array([10, 11]))
    return fn


class _CarriageLog(TransformValueLog):
    pass


class _OtherLog(TransformValueLog):
    pass


# --- Instrument.apply_dynamic_transforms ---


def _make_workflow_loading(fn: str, source_name: str) -> sciline.Pipeline:
    """A minimal Sciline pipeline mimicking essreduce's NeXus loading shape."""
    from ess.reduce.nexus.workflow import GenericNeXusWorkflow

    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = fn
    wf[NeXusName[NXdetector]] = source_name
    return wf


def _make_instrument(
    bindings: list[DynamicTransformBinding],
) -> Instrument:
    inst = Instrument(name='_test', dynamic_transforms=bindings)
    return inst


def test_apply_no_op_when_chain_has_no_dynamic_nxlog(tmp_path) -> None:
    fn = _make_artifact(tmp_path)
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/entry/instrument/detector_carriage/value',
                stream_name='detector_carriage',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'loki_detector_0'}),
            ),
        ]
    )
    source_name = 'loki_detector_static'
    wf = _make_workflow_loading(fn, source_name)
    context_keys = inst.apply_dynamic_transforms(wf, {source_name: NXdetector})
    assert context_keys == {}


def test_apply_patches_chain_for_matching_component(tmp_path) -> None:
    fn = _make_artifact(tmp_path)
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/entry/instrument/detector_carriage/value',
                stream_name='detector_carriage',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'loki_detector_0'}),
            ),
        ]
    )
    source_name = 'loki_detector_0'
    wf = _make_workflow_loading(fn, source_name)
    context_keys = inst.apply_dynamic_transforms(wf, {source_name: NXdetector})
    assert context_keys == {'detector_carriage': _CarriageLog}


def test_apply_no_samples_yet_raises_at_compute(tmp_path) -> None:
    """End-to-end: pipeline configured, but log container is None at compute
    time -> patched provider raises with a registry-aware message."""
    fn = _make_artifact(tmp_path)
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/entry/instrument/detector_carriage/value',
                stream_name='detector_carriage',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'loki_detector_0'}),
            ),
        ]
    )
    source_name = 'loki_detector_0'
    wf = _make_workflow_loading(fn, source_name)
    inst.apply_dynamic_transforms(wf, {source_name: NXdetector})
    wf[_CarriageLog] = _CarriageLog(values=None)
    with pytest.raises(ValueError, match='No samples yet'):
        wf.compute(NeXusTransformationChain[NXdetector, SampleRun])


def test_apply_uses_latest_sample(tmp_path) -> None:
    fn = _make_artifact(tmp_path)
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/entry/instrument/detector_carriage/value',
                stream_name='detector_carriage',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'loki_detector_0'}),
            ),
        ]
    )
    source_name = 'loki_detector_0'
    wf = _make_workflow_loading(fn, source_name)
    inst.apply_dynamic_transforms(wf, {source_name: NXdetector})
    log = sc.DataArray(
        sc.array(dims=['time'], values=[1.0, 2.0, 7.5], unit='mm'),
        coords={
            'time': sc.array(
                dims=['time'], values=[0, 1, 2], unit='ns', dtype='datetime64'
            )
        },
    )
    wf[_CarriageLog] = _CarriageLog(values=log)
    chain = wf.compute(NeXusTransformationChain[NXdetector, SampleRun])
    patched_value = chain.transformations[
        '/entry/instrument/detector_carriage/value'
    ].value
    assert sc.identical(patched_value, sc.scalar(7.5, unit='mm'))


# --- aux source composition ---


def _job_id(source_name: str) -> JobId:
    import uuid

    return JobId(source_name=source_name, job_number=uuid.uuid4())


def test_compose_aux_inputs_filtered_by_consumers() -> None:
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/a',
                stream_name='stream_a',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'src_a'}),
            ),
            DynamicTransformBinding(
                nxlog_path='/b',
                stream_name='stream_b',
                log_key=_OtherLog,
                dependent_sources=frozenset({'src_b'}),
            ),
        ]
    )
    aux_a = compose_aux_sources(inst, ['src_a'], None)
    assert aux_a is not None
    assert set(aux_a.inputs) == {'stream_a'}

    aux_ab = compose_aux_sources(inst, ['src_a', 'src_b'], None)
    assert aux_ab is not None
    assert set(aux_ab.inputs) == {'stream_a', 'stream_b'}

    assert compose_aux_sources(inst, ['src_unknown'], None) is None


def test_compose_render_filtered_by_source_name() -> None:
    inst = _make_instrument(
        [
            DynamicTransformBinding(
                nxlog_path='/a',
                stream_name='stream_a',
                log_key=_CarriageLog,
                dependent_sources=frozenset({'src_a', 'src_shared'}),
            ),
            DynamicTransformBinding(
                nxlog_path='/b',
                stream_name='stream_b',
                log_key=_OtherLog,
                dependent_sources=frozenset({'src_b'}),
            ),
        ]
    )
    aux = compose_aux_sources(inst, ['src_a', 'src_shared', 'src_b'], None)
    assert aux is not None
    assert aux.render(_job_id('src_a')) == {'stream_a': 'stream_a'}
    assert aux.render(_job_id('src_shared')) == {'stream_a': 'stream_a'}
    assert aux.render(_job_id('src_b')) == {'stream_b': 'stream_b'}
    assert aux.render(_job_id('src_other')) == {}
