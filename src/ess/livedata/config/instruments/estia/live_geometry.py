# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry-artifact inputs of ESTIA's reflectometry reduction.

The artifact's transform chains are filled from the motor streams by the
workflow build. This module adds what the build cannot derive: the two angles
essestia reads as plain inputs, and the streamed lookup table.
"""

from typing import NewType

import sciline
import scipp as sc
from ess.reduce.nexus.types import SampleRun
from ess.reflectometry.types import DetectorRotation, RawSampleRotation

from ess.livedata.workflows.lut_context import detector_lookup_table

#: Readback histories of the two angles essestia reads as scalars.
SampleRotationLog = NewType('SampleRotationLog', sc.DataArray)
DetectorArmRotationLog = NewType('DetectorArmRotationLog', sc.DataArray)


def sample_rotation(log: SampleRotationLog) -> RawSampleRotation[SampleRun]:
    return RawSampleRotation[SampleRun](log['time', -1].data)


def detector_rotation(log: DetectorArmRotationLog) -> DetectorRotation[SampleRun]:
    return DetectorRotation[SampleRun](log['time', -1].data)


def insert_live_geometry(workflow: sciline.Pipeline) -> None:
    """Read the angles and the lookup table from context."""
    workflow.insert(sample_rotation)
    workflow.insert(detector_rotation)
    workflow.insert(detector_lookup_table)
