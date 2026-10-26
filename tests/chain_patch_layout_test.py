# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Chain patches across the file-writer layouts of a live transformation.

- **In-chain.** The f144 stream's own NXlog is the ``depends_on`` chain entry.
- **Linked.** The chain entry is a separate NXlog in ``NXtransformations``,
  without ``topic``/``source``, whose ``value``/``time`` are hard links to the
  stream's NXlog under an ``NXpositioner``. One stream can drive such an entry
  on several components.

Each test starts from a minimal source file and runs the production steps:
``streams_parsed`` codegen, binding resolution on an :class:`Instrument`, the
geometry artifact, and the chain patch computed by a real Sciline pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusData,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.nexus.workflow import get_transformation_chain
from scippnexus import NXdetector
from scippnexus.field import DependsOn
from scippnexus.nxtransformations import TransformationChain, parse_depends_on_chain

from ess.livedata.config import name_streams
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.stream import F144Stream
from ess.livedata.config.value_log import ValueLog
from ess.livedata.nexus_helpers import (
    extract_stream_info,
    filter_f144_streams,
    generate_streams_parsed_module,
)
from ess.livedata.scripts.make_geometry_nexus import write_minimal_geometry
from ess.livedata.workflows.dynamic_transforms import wire_dynamic_transforms

MOTOR = '/entry/instrument/carriage'


def _entry_path(component: str) -> str:
    return f'/entry/instrument/{component}/transformations/carriage'


class _CarriageLog(ValueLog):
    pass


def _write_log(group: h5py.Group, values: list[float]) -> None:
    group.attrs['NX_class'] = 'NXlog'
    group.create_dataset('value', data=np.array(values)).attrs['units'] = 'deg'
    group.create_dataset('time', data=np.arange(len(values), dtype='int64')).attrs[
        'units'
    ] = 'ns'


def _mark_f144(group: h5py.Group, source: str) -> None:
    group.attrs.update(topic='motion', source=source, writer_module='f144')


def _add_chain_entry(f: h5py.File, component: str) -> h5py.Group:
    """Create ``component`` with a one-entry ``depends_on`` chain, return the entry."""
    det = f.create_group(f'entry/instrument/{component}')
    det.attrs['NX_class'] = 'NXdetector'
    transformations = det.create_group('transformations')
    transformations.attrs['NX_class'] = 'NXtransformations'
    entry = transformations.create_group('carriage')
    entry.attrs.update(
        transformation_type='rotation', vector=[0.0, 0.0, 1.0], depends_on='.'
    )
    det['depends_on'] = _entry_path(component)
    return entry


def _write_in_chain(path: Path) -> None:
    """``det_a``'s chain entry is the stream NXlog itself."""
    with h5py.File(path, 'w') as f:
        entry = _add_chain_entry(f, 'det_a')
        _write_log(entry, [1.0, 2.0])
        _mark_f144(entry, 'CAR:Mtr.RBV')


def _write_linked(path: Path, consumers: tuple[str, ...]) -> None:
    """Each consumer's chain entry hard-links the RBV NXlog of a motor device."""
    with h5py.File(path, 'w') as f:
        motor = f.create_group(MOTOR)
        motor.attrs['NX_class'] = 'NXpositioner'
        rbv = motor.create_group('value')
        _write_log(rbv, [1.0, 2.0])
        _mark_f144(rbv, 'CAR:Mtr.RBV')
        target = motor.create_group('target_value')
        _write_log(target, [2.0])
        _mark_f144(target, 'CAR:Mtr.VAL')
        for component in consumers:
            entry = _add_chain_entry(f, component)
            entry.attrs['NX_class'] = 'NXlog'
            entry['value'] = rbv['value']
            entry['time'] = rbv['time']


@dataclass(frozen=True)
class _Layout:
    source: Path
    stream_path: str
    """``nexus_path`` the binding's stream (or its device RBV) is parsed under."""
    rename_path: str
    """Path :func:`name_streams` names ``'carriage'``: the stream or its device."""
    consumers: tuple[str, ...]


@pytest.fixture
def in_chain(tmp_path: Path) -> _Layout:
    source = tmp_path / 'in_chain.nxs'
    _write_in_chain(source)
    path = _entry_path('det_a')
    return _Layout(source, path, path, ('det_a',))


@pytest.fixture
def linked(tmp_path: Path) -> _Layout:
    source = tmp_path / 'linked.nxs'
    _write_linked(source, ('det_a',))
    return _Layout(source, f'{MOTOR}/value', MOTOR, ('det_a',))


@pytest.fixture
def linked_twice(tmp_path: Path) -> _Layout:
    source = tmp_path / 'linked_twice.nxs'
    _write_linked(source, ('det_a', 'det_b'))
    return _Layout(source, f'{MOTOR}/value', MOTOR, ('det_a', 'det_b'))


def _parsed_streams(source: Path) -> dict[str, F144Stream]:
    infos = filter_f144_streams(extract_stream_info(source))
    namespace: dict[str, Any] = {}
    exec(generate_streams_parsed_module(infos), namespace)  # noqa: S102
    return namespace['PARSED_STREAMS']


def _instrument(layout: _Layout) -> Instrument:
    streams = name_streams(
        _parsed_streams(layout.source), rename={layout.rename_path: 'carriage'}
    )
    instrument = Instrument(name='test', streams=streams)
    instrument.add_context_binding(
        stream_name='carriage',
        workflow_key=_CarriageLog,
        dependent_sources=layout.consumers,
    )
    return instrument


def _resolved_paths(layout: _Layout) -> tuple[str, ...]:
    [binding] = _instrument(layout).chain_patch_bindings
    return binding.transform_paths


def _load_chain(artifact: Path, component: str) -> TransformationChain:
    parent = f'/entry/instrument/{component}'
    with snx.File(artifact, 'r') as f:
        comp = f[parent]
        depends_on = comp['depends_on'][()]
        if not isinstance(depends_on, DependsOn):
            depends_on = DependsOn(parent=parent, value=depends_on)
        return parse_depends_on_chain(comp, depends_on)


def _base_chain(
    component: NeXusComponent[NXdetector, SampleRun],
) -> NeXusTransformationChain[NXdetector, SampleRun]:
    """Unpatched chain; the wired chain-patch provider replaces it on ``insert``."""
    return get_transformation_chain(component)


@dataclass
class _Workflow:
    """Minimal ``SupportsDynamicTransforms`` over a real pipeline."""

    dynamic_keys: dict[str, Any]
    base_pipeline: sciline.Pipeline


def _patched_chain(
    layout: _Layout, artifact: Path, component: str, latest: float
) -> TransformationChain:
    pipeline = sciline.Pipeline((_base_chain,))
    pipeline[NeXusComponent[NXdetector, SampleRun]] = sc.DataGroup(
        {'depends_on': _load_chain(artifact, component)}
    )
    pipeline[_CarriageLog] = _CarriageLog(
        values=sc.DataArray(
            sc.array(dims=['time'], values=[latest], unit='deg'),
            coords={'time': sc.array(dims=['time'], values=[5], unit='ns')},
        )
    )
    workflow = _Workflow({component: NeXusData[NXdetector, SampleRun]}, pipeline)
    wire_dynamic_transforms(workflow, _instrument(layout).chain_patch_bindings)
    return pipeline.compute(NeXusTransformationChain[NXdetector, SampleRun])


class TestGeneratedStreamTable:
    def test_in_chain_stream_records_no_transform_paths(
        self, in_chain: _Layout
    ) -> None:
        stream = _parsed_streams(in_chain.source)[in_chain.stream_path]
        assert stream.transform_paths == ()

    def test_linked_stream_records_its_chain_entry(self, linked: _Layout) -> None:
        parsed = _parsed_streams(linked.source)
        assert parsed[linked.stream_path].transform_paths == (_entry_path('det_a'),)
        # The device's VAL substream shares no dataset with the chain entry.
        assert parsed[f'{MOTOR}/target_value'].transform_paths == ()

    def test_stream_linked_twice_records_both_chain_entries(
        self, linked_twice: _Layout
    ) -> None:
        stream = _parsed_streams(linked_twice.source)[linked_twice.stream_path]
        assert stream.transform_paths == (_entry_path('det_a'), _entry_path('det_b'))


class TestResolvedPaths:
    def test_in_chain_resolves_to_stream_path(self, in_chain: _Layout) -> None:
        assert _resolved_paths(in_chain) == (_entry_path('det_a'),)

    def test_linked_device_resolves_to_chain_entry(self, linked: _Layout) -> None:
        assert _resolved_paths(linked) == (_entry_path('det_a'),)

    def test_stream_linked_twice_resolves_to_both_entries(
        self, linked_twice: _Layout
    ) -> None:
        assert _resolved_paths(linked_twice) == (
            _entry_path('det_a'),
            _entry_path('det_b'),
        )


class TestPatchedChain:
    @pytest.mark.parametrize('layout_name', ['in_chain', 'linked', 'linked_twice'])
    def test_patches_chain_entry_of_every_consumer(
        self, layout_name: str, request: pytest.FixtureRequest, tmp_path: Path
    ) -> None:
        layout: _Layout = request.getfixturevalue(layout_name)
        artifact = tmp_path / 'geometry.nxs'
        write_minimal_geometry(layout.source, artifact)

        for component in layout.consumers:
            chain = _patched_chain(layout, artifact, component, latest=42.0)
            entry = chain.transformations[_entry_path(component)]
            assert sc.identical(entry.value, sc.scalar(42.0, unit='deg'))
