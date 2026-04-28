# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
ESTIA instrument spec registration.
"""

import pydantic
import scipp as sc

from ess.livedata import parameter_models
from ess.livedata.config import Instrument, instrument_registry
from ess.livedata.config.workflow_spec import WorkflowOutputsBase
from ess.livedata.handlers.detector_view_specs import SpectrumViewSpec
from ess.livedata.handlers.monitor_workflow_specs import (
    TOAOnlyMonitorDataParams,
    register_monitor_workflow_specs,
)

from .._ess import GENERIC_CBM_DESCRIPTION_NOTE, GENERIC_CBM_MONITORS
from .views import get_multiblade_view

detector_names = ['multiblade_detector']


class ThetaEdges(parameter_models.EdgesModel):
    """Model for theta bin edges."""

    unit: parameter_models.AngleUnit = pydantic.Field(
        default=parameter_models.AngleUnit.DEGREE,
        description="Unit of the theta bin edges.",
    )

    def get_edges(self) -> sc.Variable:
        """Get the edges as a scipp variable."""
        return parameter_models.make_edges(
            model=self, dim='theta', unit=self.unit.value
        )


class EstiaLiveDiagnosticsParams(pydantic.BaseModel):
    """Parameters for lightweight ESTIA live reflectometry diagnostics."""

    wavelength_edges: parameter_models.WavelengthEdges = pydantic.Field(
        title='Wavelength bins',
        description='Define the bin edges for intensity histograms in wavelength.',
        default=parameter_models.WavelengthEdges(
            start=3.5,
            stop=12.0,
            num_bins=120,
            scale=parameter_models.Scale.LOG,
            unit=parameter_models.WavelengthUnit.ANGSTROM,
        ),
    )
    q_edges: parameter_models.QEdges = pydantic.Field(
        title='Q bins',
        description='Define the bin edges for intensity histograms in Q.',
        default=parameter_models.QEdges(
            start=0.01,
            stop=0.5,
            num_bins=401,
            scale=parameter_models.Scale.LOG,
            unit=parameter_models.QUnit.INVERSE_ANGSTROM,
        ),
    )
    theta_edges: ThetaEdges = pydantic.Field(
        title='Theta bins',
        description='Define the bin edges for intensity histograms in theta.',
        default=ThetaEdges(
            start=-1.0,
            stop=2.0,
            num_bins=120,
            unit=parameter_models.AngleUnit.DEGREE,
        ),
    )


def _make_wavelength_template() -> sc.DataArray:
    """Create an empty 1D template for I(wavelength)."""
    return sc.DataArray(
        sc.zeros(dims=['wavelength'], shape=[0], unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 0, unit='angstrom')},
    )


def _make_q_template() -> sc.DataArray:
    """Create an empty 1D template for I(Q)."""
    return sc.DataArray(
        sc.zeros(dims=['Q'], shape=[0], unit='counts'),
        coords={'Q': sc.arange('Q', 0, unit='1/angstrom')},
    )


def _make_theta_wavelength_template() -> sc.DataArray:
    """Create an empty 2D template for I(theta, wavelength)."""
    return sc.DataArray(
        sc.zeros(dims=['theta', 'wavelength'], shape=[0, 0], unit='counts'),
        coords={
            'theta': sc.arange('theta', 0, unit='deg'),
            'wavelength': sc.arange('wavelength', 0, unit='angstrom'),
        },
    )


class EstiaLiveDiagnosticsOutputs(WorkflowOutputsBase):
    """Outputs for the ESTIA live diagnostics workflow."""

    i_of_wavelength: sc.DataArray = pydantic.Field(
        default_factory=_make_wavelength_template,
        title='I(wavelength)',
        description='Detector intensity histogrammed by wavelength.',
    )
    i_of_q: sc.DataArray = pydantic.Field(
        default_factory=_make_q_template,
        title='I(Q)',
        description='Detector intensity histogrammed by momentum transfer Q.',
    )
    i_of_theta_wavelength: sc.DataArray = pydantic.Field(
        default_factory=_make_theta_wavelength_template,
        title='I(theta, wavelength)',
        description='Detector intensity histogrammed by theta and wavelength.',
    )


instrument = Instrument(
    name='estia',
    detector_names=detector_names,
    monitors=list(GENERIC_CBM_MONITORS),
    f144_attribute_registry={},
)

instrument_registry.register(instrument)

monitor_handle = register_monitor_workflow_specs(
    instrument,
    instrument.monitors,
    params=TOAOnlyMonitorDataParams,
    extra_description=GENERIC_CBM_DESCRIPTION_NOTE,
)


def _estia_spectrum_transform(histogram: sc.DataArray) -> sc.DataArray:
    """Sum over the ``strip`` axis (constant scattering angle)."""
    return histogram.sum('strip')


instrument.add_logical_view(
    name='estia_multiblade_detector_view',
    title='Multiblade Detector',
    description='Counts folded into strip, blade, and wire dimensions',
    source_names=['multiblade_detector'],
    transform=get_multiblade_view,
    roi_support=False,
    output_ndim=3,
    spectrum_view=SpectrumViewSpec(
        transform=_estia_spectrum_transform,
        output_dims=['blade', 'wire'],
        extra_description='Summed across strips, yielding per-blade, per-wire spectra.',
    ),
)

live_diagnostics_handle = instrument.register_spec(
    name='live_diagnostics',
    version=1,
    title='Live diagnostics',
    description=(
        'Raw ESTIA reflectometry diagnostics: detector intensity binned by '
        'wavelength, Q, and theta versus wavelength.'
    ),
    source_names=detector_names,
    outputs=EstiaLiveDiagnosticsOutputs,
    params=EstiaLiveDiagnosticsParams,
)
