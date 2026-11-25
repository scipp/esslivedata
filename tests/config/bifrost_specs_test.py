# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scippnexus as snx
from ess.bifrost import BifrostWorkflow
from ess.reduce.nexus.types import Filename, SampleRun
from ess.reduce.nexus.types import TransformationTimeFilter as TimeFilter
from ess.reduce.unwrap.types import DetectorLtotal

from ess.livedata.config.instruments.bifrost.specs import instrument
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


@pytest.mark.parametrize('angle', [-45.0, 0.0, 30.0, 90.0])
def test_declared_detector_ltotal_is_l1_at_any_sample_angle(angle: float) -> None:
    """The Q-maps look the table up at L1, which the goniometers and the sample
    rotation leave unchanged: all three rotate about the sample."""

    def rotated(transform: sc.DataArray) -> sc.Variable:
        return sc.scalar(angle, unit=transform.unit)

    workflow = BifrostWorkflow()
    workflow[Filename[SampleRun]] = str(get_nexus_geometry_filename('bifrost'))
    workflow[TimeFilter[snx.NXsample, SampleRun]] = rotated
    l1 = workflow.compute(DetectorLtotal[SampleRun]).to(unit='m')

    declared = instrument.declared_ltotal['unified_detector']
    assert declared.value == pytest.approx(l1.value, abs=1e-3)
