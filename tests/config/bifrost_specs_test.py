# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scippnexus as snx
from ess.bifrost import BifrostWorkflow
from ess.bifrost.single_crystal import BifrostBraggPeakMonitorWorkflow
from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    GravityVector,
    NeutronSourcePosition,
    Position,
    SampleRun,
)
from ess.reduce.nexus.types import TransformationTimeFilter as TimeFilter
from ess.reduce.unwrap.to_wavelength import (
    detector_ltotal_from_straight_line_approximation,
)
from ess.reduce.unwrap.types import DetectorLtotal
from ess.spectroscopy.types import ElasticMonitor

from ess.livedata.config.instruments.bifrost.specs import instrument
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename

ANGLES = [-90.0, -45.0, 0.0, 30.0, 90.0]


def _at(angle: float):
    def rotated(transform: sc.DataArray) -> sc.Variable:
        return sc.scalar(angle, unit=transform.unit)

    return rotated


@pytest.fixture
def declared() -> sc.Variable:
    return instrument.declared_ltotal['unified_detector'].to(unit='m')


@pytest.mark.parametrize('angle', ANGLES)
def test_declared_detector_ltotal_starts_at_l1_at_any_sample_angle(
    declared: sc.Variable, angle: float
) -> None:
    """The detector Q-maps look the table up at L1, which the goniometers and the
    sample rotation leave unchanged: all three rotate about the sample."""
    workflow = BifrostWorkflow()
    workflow[Filename[SampleRun]] = str(get_nexus_geometry_filename('bifrost'))
    workflow[TimeFilter[snx.NXsample, SampleRun]] = _at(angle)
    l1 = workflow.compute(DetectorLtotal[SampleRun]).to(unit='m')

    assert declared.min().value == pytest.approx(l1.value, abs=1e-3)


@pytest.mark.parametrize('angle', ANGLES)
def test_declared_detector_ltotal_ends_at_elastic_monitor_at_any_tank_angle(
    declared: sc.Variable, angle: float
) -> None:
    """The elastic monitor Q-map looks the table up at the monitor's straight-line
    path, which the tank leaves unchanged: it rotates about the sample.

    Upstream computes that path from the events grouped by rotation, so the
    geometry is computed here and passed to the same function.
    """
    workflow = BifrostBraggPeakMonitorWorkflow()
    workflow[Filename[SampleRun]] = str(get_nexus_geometry_filename('bifrost'))
    workflow[TimeFilter[ElasticMonitor, SampleRun]] = _at(angle)
    workflow[TimeFilter[snx.NXsample, SampleRun]] = _at(0.0)
    geometry = workflow.compute(
        (
            EmptyDetector[SampleRun],
            NeutronSourcePosition[SampleRun],
            Position[snx.NXsample, SampleRun],
            GravityVector,
        )
    )
    ltotal = detector_ltotal_from_straight_line_approximation(
        geometry[EmptyDetector[SampleRun]],
        source_position=geometry[NeutronSourcePosition[SampleRun]],
        sample_position=geometry[Position[snx.NXsample, SampleRun]],
        gravity=geometry[GravityVector],
    ).to(unit='m')

    assert declared.max().value == pytest.approx(ltotal.value, abs=1e-3)
