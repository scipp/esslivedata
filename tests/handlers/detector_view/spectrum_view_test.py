# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Integration tests for the unified spectrum-view output."""

import scipp as sc
from ess.reduce.nexus.types import RawDetector, SampleRun

from ess.livedata.core.timestamp import Timestamp
from ess.livedata.handlers.detector_view.data_source import DetectorNumberSource
from ess.livedata.handlers.detector_view.factory import DetectorViewFactory
from ess.livedata.handlers.detector_view.types import LogicalViewConfig
from ess.livedata.handlers.detector_view_specs import (
    SpectrumViewRebin,
    SpectrumViewSpec,
    make_detector_view_params,
)

from .utils import make_fake_detector_number, make_fake_nexus_detector_data


def _sum_y_transform(histogram: sc.DataArray, rebin: int) -> sc.DataArray:
    """Sum the ``y`` dim; the ``rebin`` argument is intentionally ignored."""
    return histogram.sum('y')


def _rebin_x_transform(histogram: sc.DataArray, rebin: int) -> sc.DataArray:
    """Group ``rebin`` adjacent bins along ``x`` (reshape + sum)."""
    return histogram.fold('x', sizes={'x': -1, 'subpixel': rebin}).sum('subpixel')


def _make_factory_with_spectrum(
    spec: SpectrumViewSpec, *, y_size: int = 4, x_size: int = 4
) -> DetectorViewFactory:
    detector_number = make_fake_detector_number(y_size, x_size)

    def logical_transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
        return da.fold(dim='detector_number', sizes={'y': y_size, 'x': x_size})

    return DetectorViewFactory(
        data_source=DetectorNumberSource(detector_number),
        view_config=LogicalViewConfig(transform=logical_transform, spectrum_view=spec),
    )


class TestSpectrumViewIntegration:
    def test_spectrum_view_sums_over_declared_dim(self):
        spec = SpectrumViewSpec(
            transform=_sum_y_transform,
            output_dims=['x', 'time_of_arrival'],
        )
        factory = _make_factory_with_spectrum(spec)
        params = make_detector_view_params(spectrum_view=spec)()
        workflow = factory.make_workflow('detector', params=params)

        events = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=5)
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events)},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        result = workflow.finalize()

        assert 'spectrum_view' in result
        spectrum = result['spectrum_view']
        assert spectrum.dims == ('x', 'time_of_arrival')
        cumulative = result['cumulative']
        # Cumulative is summed over time_of_arrival => total counts match total events.
        assert sc.isclose(spectrum.sum().data, cumulative.sum().data).value

    def test_spectrum_view_rebin_factor_applied(self):
        spec = SpectrumViewSpec(
            transform=_rebin_x_transform,
            output_dims=['y', 'x', 'time_of_arrival'],
        )
        factory = _make_factory_with_spectrum(spec, y_size=4, x_size=4)
        Params = make_detector_view_params(spectrum_view=spec)
        params = Params(spectrum_rebin=SpectrumViewRebin(factor=2))
        workflow = factory.make_workflow('detector', params=params)

        events = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=5)
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events)},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        result = workflow.finalize()

        spectrum = result['spectrum_view']
        assert spectrum.dims == ('y', 'x', 'time_of_arrival')
        # With factor=2 grouping along x (size 4), output x size is 2.
        assert spectrum.sizes['x'] == 2
        assert spectrum.sizes['y'] == 4

    def test_no_spectrum_view_when_spec_absent(self):
        """Without a SpectrumViewSpec, spectrum_view is not in the outputs."""
        detector_number = make_fake_detector_number(4, 4)

        def logical_transform(da: sc.DataArray, source_name: str) -> sc.DataArray:
            return da.fold(dim='detector_number', sizes={'y': 4, 'x': 4})

        factory = DetectorViewFactory(
            data_source=DetectorNumberSource(detector_number),
            view_config=LogicalViewConfig(transform=logical_transform),
        )
        params = make_detector_view_params()()
        workflow = factory.make_workflow('detector', params=params)

        events = make_fake_nexus_detector_data(y_size=4, x_size=4, n_events_per_pixel=2)
        workflow.accumulate(
            {'detector': RawDetector[SampleRun](events)},
            start_time=Timestamp.from_ns(1000),
            end_time=Timestamp.from_ns(2000),
        )
        result = workflow.finalize()
        assert 'spectrum_view' not in result
