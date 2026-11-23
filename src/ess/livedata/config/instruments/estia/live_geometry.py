# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry-artifact inputs of ESTIA's reflectometry reduction.

The artifact's transform chains are filled from the motor streams by the
workflow build. This module adds what the build cannot derive: the moderator
position and the two angles essestia reads as plain inputs.
"""

from typing import NewType

import sciline
import scipp as sc
from ess.reduce.nexus.types import Position, SampleRun
from ess.reflectometry.types import DetectorRotation, RawSampleRotation
from scippnexus import NXsource

from ess.livedata.workflows.lut_context import detector_lookup_table
from ess.livedata.workflows.wavelength_lut_workflow import neutron_source_position

#: Readback histories of the two angles essestia reads as scalars.
SampleRotationLog = NewType('SampleRotationLog', sc.DataArray)
DetectorArmRotationLog = NewType('DetectorArmRotationLog', sc.DataArray)


def sample_rotation(log: SampleRotationLog) -> RawSampleRotation[SampleRun]:
    return RawSampleRotation[SampleRun](log['time', -1].data)


def detector_rotation(log: DetectorArmRotationLog) -> DetectorRotation[SampleRun]:
    return DetectorRotation[SampleRun](log['time', -1].data)


def insert_live_geometry(workflow: sciline.Pipeline, filename: str) -> None:
    """Place the moderator and read the angles and lookup table from context."""
    workflow[Position[NXsource, SampleRun]] = neutron_source_position(filename)
    workflow.insert(sample_rotation)
    workflow.insert(detector_rotation)
    workflow.insert(detector_lookup_table)
