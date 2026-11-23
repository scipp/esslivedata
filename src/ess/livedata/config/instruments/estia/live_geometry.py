# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry-artifact inputs of ESTIA's reflectometry reduction.

The artifact's transform chains are filled from the motor streams by the
workflow build. This module adds what the build cannot derive: the reference
flight path the lookup table is placed by, a coverage check against it, and the
two angles essestia reads as plain inputs.
"""

from copy import deepcopy
from typing import NewType

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

from ess.livedata.workflows.dynamic_transforms import placeholders
from ess.livedata.workflows.lut_blocks import select_block
from ess.livedata.workflows.lut_context import DetectorLutContext, detector_lookup_table
from ess.livedata.workflows.wavelength_lut_workflow import neutron_source_position

#: Readback histories of the two angles essestia reads as scalars.
SampleRotationLog = NewType('SampleRotationLog', sc.DataArray)
DetectorArmRotationLog = NewType('DetectorArmRotationLog', sc.DataArray)


def sample_rotation(log: SampleRotationLog) -> RawSampleRotation[SampleRun]:
    return RawSampleRotation[SampleRun](log['time', -1].data)


def detector_rotation(log: DetectorArmRotationLog) -> DetectorRotation[SampleRun]:
    return DetectorRotation[SampleRun](log['time', -1].data)


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
        for path in placeholders(chain):
            transform = chain.transformations[path]
            transform.value = sc.scalar(0.0, unit=transform.value.unit)
        wf[chain_key] = chain
    return wf.compute(DetectorLtotal[SampleRun]).to(unit='m')


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


def insert_live_geometry(workflow: sciline.Pipeline, filename: str) -> None:
    """Place the moderator, read the angles, and require lookup-table coverage."""
    workflow[Position[NXsource, SampleRun]] = neutron_source_position(filename)
    workflow.insert(sample_rotation)
    workflow.insert(detector_rotation)
    workflow.insert(covered_detector_lookup_table)
