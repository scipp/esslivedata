# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Map geometry NXlog metadata to live ESTIA motor inputs.

Explicit providers support sample chains and motor aliases that automatic
NeXusData-based chain patching cannot resolve.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import h5py
import sciline
import scipp as sc
from ess.reduce.nexus.types import (
    Filename,
    NeXusName,
    NeXusTransformationChain,
    Position,
    SampleRun,
)
from ess.reduce.unwrap import GenericUnwrapWorkflow, LookupTable
from ess.reduce.unwrap.types import DetectorLtotal
from ess.reflectometry.types import DetectorRotation, RawSampleRotation
from scippnexus import NXdetector, NXsample, NXsource

from ess.livedata.config import Device, F144Stream, Instrument
from ess.livedata.config.value_log import ValueLog
from ess.livedata.workflows.dynamic_transforms import build_patched_chain_provider
from ess.livedata.workflows.lut_blocks import select_block
from ess.livedata.workflows.lut_context import DetectorLutContext, detector_lookup_table
from ess.livedata.workflows.wavelength_lut_workflow import neutron_source_position


class DetectorArmRotationLog(ValueLog):
    """Live detector angle used for both placement and scattering coordinates."""


class SampleRotationLog(ValueLog):
    """Live sample-stage rotation."""


@cache
def _motor_log_type(stream_name: str) -> type[ValueLog]:
    # Sciline keys must keep their identity if the geometry is loaded again.
    return type(stream_name.replace('/', '_') + 'Log', (ValueLog,), {})


@dataclass
class LiveGeometry:
    context_keys: dict[str, type[ValueLog]]
    bindings: dict[type, list[tuple[str, type[ValueLog]]]]
    source_position: sc.Variable


def load_live_geometry(filename: str | Path, instrument: Instrument) -> LiveGeometry:
    """Resolve each component's linked transforms to canonical device streams."""
    sources_to_names = {}
    for name, stream in instrument.streams.items():
        if isinstance(stream, F144Stream) and stream.nexus_path is not None:
            sources_to_names[stream.topic, stream.source] = name
    for name, stream in instrument.streams.items():
        if isinstance(stream, Device):
            rbv = instrument.streams[stream.value]
            sources_to_names[rbv.topic, rbv.source] = name

    keys: dict[str, type[ValueLog]] = {
        'detector_arm_rotation': DetectorArmRotationLog,
        'sample_rotation': SampleRotationLog,
    }
    wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = filename
    (detector,) = instrument.detector_names
    wf[NeXusName[NXdetector]] = detector
    bindings = {}
    with h5py.File(filename) as f:
        for component in (NXsample, NXdetector):
            chain = wf.compute(NeXusTransformationChain[component, SampleRun])
            bindings[component] = []
            for path, transform in chain.transformations.items():
                if not isinstance(transform.value, sc.DataArray):
                    continue
                attrs = f[path].attrs
                if 'source' not in attrs or 'topic' not in attrs:
                    raise ValueError(
                        f'Missing stream metadata for live transform {path}'
                    )
                name = sources_to_names[attrs['topic'], attrs['source']]
                if name not in keys:
                    keys[name] = _motor_log_type(name)
                bindings[component].append((path, keys[name]))
    if not all(bindings.values()):
        raise ValueError('ESTIA geometry has no linked live transformation logs')
    return LiveGeometry(
        context_keys=keys,
        bindings=bindings,
        source_position=neutron_source_position(str(filename)),
    )


def reference_detector_ltotal(filename: str) -> sc.Variable:
    """Compute reference flight paths with live axes at their zero positions.

    The LUT factory pads these paths; the reduction checks coverage at live positions.
    """
    wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = filename
    wf[NeXusName[NXdetector]] = 'multiblade_detector'
    wf[Position[NXsource, SampleRun]] = neutron_source_position(filename)
    for component in (NXsample, NXdetector):
        chain_key = NeXusTransformationChain[component, SampleRun]
        chain = deepcopy(wf.compute(chain_key))
        for transform in chain.transformations.values():
            if isinstance(transform.value, sc.DataArray):
                transform.value = sc.scalar(0.0, unit=transform.value.unit)
        wf[chain_key] = chain
    return wf.compute(DetectorLtotal[SampleRun]).to(unit='m')


def sample_rotation(log: SampleRotationLog) -> RawSampleRotation[SampleRun]:
    return RawSampleRotation[SampleRun](log.values['time', -1].data)


def detector_rotation(log: DetectorArmRotationLog) -> DetectorRotation[SampleRun]:
    return DetectorRotation[SampleRun](log.values['time', -1].data)


def covered_detector_lookup_table(
    wire: DetectorLutContext, ltotal: DetectorLtotal[SampleRun]
) -> LookupTable[SampleRun, NXdetector]:
    """Require full coverage when sample motion changes the reference paths."""
    block = select_block(wire, ltotal)
    distances = block.coords['distance']
    if (
        ltotal.min().to(unit=distances.unit) < distances[0]
        or ltotal.max().to(unit=distances.unit) > distances[-1]
    ):
        raise ValueError(
            'ESTIA sample motion puts pixels outside the reference lookup-table '
            'range. Confirm sample travel bounds and extend the LUT coverage.'
        )
    return detector_lookup_table(wire, ltotal)


def insert_live_geometry(workflow: sciline.Pipeline, geometry: LiveGeometry) -> None:
    """Use the latest readbacks in both placement chains and angle providers."""
    for component, bindings in geometry.bindings.items():
        workflow.insert(build_patched_chain_provider(component, bindings))
    workflow[Position[NXsource, SampleRun]] = geometry.source_position
    workflow.insert(sample_rotation)
    workflow.insert(detector_rotation)
    workflow.insert(covered_detector_lookup_table)
