# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic
import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import Filename, NeXusName, SampleRun
from ess.reduce.nexus.types import TransformationTimeFilter as TimeFilter
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.types import DetectorLtotal

from ess.livedata import parameter_models
from ess.livedata.config.instruments.estia.specs import (
    BeamDivergenceLimits,
    EstiaReflectometryReductionOutputs,
    EstiaReflectometryReductionParams,
    IndexLimits,
    instrument,
)
from ess.livedata.preprocessors.detector_data import get_nexus_geometry_filename


def test_estia_reflectometry_reduction_default_limits_match_previous_literals():
    params = EstiaReflectometryReductionParams()

    assert sc.identical(params.y_index_limits.get_limits()[0], sc.scalar(0))
    assert sc.identical(params.y_index_limits.get_limits()[1], sc.scalar(63))
    assert sc.identical(params.z_index_limits.get_limits()[0], sc.scalar(0))
    assert sc.identical(params.z_index_limits.get_limits()[1], sc.scalar(1535))
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[0],
        sc.scalar(-0.75, unit='deg'),
    )
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[1],
        sc.scalar(0.75, unit='deg'),
    )


def test_estia_reflectometry_reduction_custom_limits_convert_to_scipp_scalars():
    params = EstiaReflectometryReductionParams(
        y_index_limits={'start': 2, 'stop': 5},
        z_index_limits=IndexLimits(start=10, stop=20),
        beam_divergence_limits={'start': -0.1, 'stop': 0.2, 'unit': 'rad'},
    )

    assert sc.identical(params.y_index_limits.get_limits()[0], sc.scalar(2))
    assert sc.identical(params.y_index_limits.get_limits()[1], sc.scalar(5))
    assert sc.identical(params.z_index_limits.get_limits()[0], sc.scalar(10))
    assert sc.identical(params.z_index_limits.get_limits()[1], sc.scalar(20))
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[0],
        sc.scalar(-0.1, unit='rad'),
    )
    assert sc.identical(
        params.beam_divergence_limits.get_limits()[1],
        sc.scalar(0.2, unit='rad'),
    )


def test_estia_reflectometry_reduction_reviewed_defaults_and_titles():
    params = EstiaReflectometryReductionParams()

    assert isinstance(params.theta_edges, parameter_models.ThetaEdges)
    assert params.q_edges.num_bins == 400
    assert EstiaReflectometryReductionOutputs.model_fields['i_of_wavelength'].title == (
        'I(λ)'
    )
    assert (
        EstiaReflectometryReductionOutputs.model_fields['i_of_theta_wavelength'].title
        == 'I(θ, λ)'
    )


def test_estia_index_limits_reject_stop_less_than_start():
    with pytest.raises(pydantic.ValidationError, match='greater than or equal'):
        IndexLimits(start=2, stop=1)


def test_estia_beam_divergence_limits_reject_stop_less_than_start():
    with pytest.raises(pydantic.ValidationError, match='greater than start'):
        BeamDivergenceLimits(start=1.0, stop=0.0)


@pytest.mark.parametrize('arm_angle', [-45.0, 0.0, 30.0, 90.0])
def test_reference_detector_ltotal_matches_the_artifact_at_any_arm_angle(
    arm_angle: float,
) -> None:
    """The detector arm rotates about the sample, so the reference flight path is
    the artifact's own at every angle. The sample and the other live axes are
    held at zero."""

    def at_rest(transform: sc.DataArray) -> sc.Variable:
        angle = sc.scalar(arm_angle, unit='deg')
        zero = sc.scalar(0.0, unit=transform.unit)
        return angle if transform.unit == 'deg' else zero

    workflow = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])
    workflow[Filename[SampleRun]] = str(get_nexus_geometry_filename('estia'))
    workflow[NeXusName[snx.NXdetector]] = 'multiblade_detector'
    workflow[TimeFilter[snx.NXdetector, SampleRun]] = at_rest
    workflow[TimeFilter[snx.NXsample, SampleRun]] = at_rest
    ltotal = workflow.compute(DetectorLtotal[SampleRun]).to(unit='m')
    reference = instrument.declared_ltotal['multiblade_detector'](
        str(get_nexus_geometry_filename('estia'))
    )

    assert reference.min().value == pytest.approx(ltotal.min().value, abs=1e-3)
    assert reference.max().value == pytest.approx(ltotal.max().value, abs=1e-3)
