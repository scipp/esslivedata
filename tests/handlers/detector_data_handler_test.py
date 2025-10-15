# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid

import pytest
import scipp as sc

from ess.livedata import StreamKind
from ess.livedata.config import instrument_registry
from ess.livedata.config.instrument import Instrument
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.handler import StreamId
from ess.livedata.handlers.accumulators import LatestValue
from ess.livedata.handlers.detector_data_handler import (
    DetectorHandlerFactory,
    DetectorROIAuxSources,
    get_nexus_geometry_filename,
)


def get_instrument(instrument_name: str) -> Instrument:
    _ = get_config(instrument_name)  # Load the module to register the instrument
    return instrument_registry[instrument_name]


@pytest.mark.parametrize('instrument', ['dream', 'loki'])
def test_get_nexus_filename_returns_file_for_given_date(instrument: str) -> None:
    filename = get_nexus_geometry_filename(
        instrument, date=sc.datetime('2025-01-02T00:00:00')
    )
    assert str(filename).endswith(f'geometry-{instrument}-2025-01-01.nxs')


def test_get_nexus_filename_uses_current_date_by_default() -> None:
    auto = get_nexus_geometry_filename('dream')
    explicit = get_nexus_geometry_filename('dream', date=sc.datetime('now'))
    assert auto == explicit


def test_get_nexus_filename_raises_if_instrument_unknown() -> None:
    with pytest.raises(ValueError, match='No geometry files found for instrument'):
        get_nexus_geometry_filename('abcde', date=sc.datetime('2025-01-01T00:00:00'))


def test_get_nexus_filename_raises_if_datetime_out_of_range() -> None:
    with pytest.raises(ValueError, match='No geometry file found for given date'):
        get_nexus_geometry_filename('dream', date=sc.datetime('2020-01-01T00:00:00'))


@pytest.mark.parametrize('instrument_name', available_instruments())
def test_factory_can_create_preprocessor(instrument_name: str) -> None:
    instrument = get_instrument(instrument_name)
    factory = DetectorHandlerFactory(instrument=instrument)
    for name in instrument.detector_names:
        _ = factory.make_preprocessor(
            StreamId(kind=StreamKind.DETECTOR_EVENTS, name=name)
        )


def test_factory_creates_latest_value_accumulator_for_roi_messages() -> None:
    """Test that DetectorHandlerFactory creates a LatestValue accumulator for ROI."""
    instrument = get_instrument('dummy')
    factory = DetectorHandlerFactory(instrument=instrument)

    # Create a stream ID for an ROI message
    roi_stream_id = StreamId(
        kind=StreamKind.LIVEDATA_ROI, name='test-job-123/roi_rectangle_0'
    )

    preprocessor = factory.make_preprocessor(roi_stream_id)

    # Should return a LatestValue accumulator
    assert preprocessor is not None
    assert isinstance(preprocessor, LatestValue)


def test_factory_returns_none_for_unknown_stream_kinds() -> None:
    """Test that DetectorHandlerFactory returns None for unknown stream kinds."""
    instrument = get_instrument('dummy')
    factory = DetectorHandlerFactory(instrument=instrument)

    # Try various stream kinds that should not be handled
    unknown_stream_id = StreamId(kind=StreamKind.LOG, name='some_log')
    preprocessor = factory.make_preprocessor(unknown_stream_id)
    assert preprocessor is None

    config_stream_id = StreamId(kind=StreamKind.LIVEDATA_CONFIG, name='config')
    preprocessor = factory.make_preprocessor(config_stream_id)
    assert preprocessor is None


class TestDetectorROIAuxSources:
    """Tests for DetectorROIAuxSources auxiliary source model."""

    def test_default_roi_shape_is_rectangle(self) -> None:
        """Test that the default ROI shape is rectangle."""
        aux_sources = DetectorROIAuxSources()
        assert aux_sources.roi == 'rectangle'

    def test_can_select_rectangle_roi(self) -> None:
        """Test that rectangle ROI shape can be selected."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        assert aux_sources.roi == 'rectangle'

    def test_validator_rejects_polygon_roi(self) -> None:
        """Test that polygon ROI shape is rejected by validator."""
        with pytest.raises(ValueError, match="Currently only 'rectangle' ROI shape"):
            DetectorROIAuxSources(roi='polygon')

    def test_validator_rejects_ellipse_roi(self) -> None:
        """Test that ellipse ROI shape is rejected by validator."""
        with pytest.raises(ValueError, match="Currently only 'rectangle' ROI shape"):
            DetectorROIAuxSources(roi='ellipse')

    def test_render_prefixes_stream_name_with_job_number_and_roi(self) -> None:
        """Test that render() prefixes stream name with job_number and roi_ prefix."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=123))

        rendered = aux_sources.render(job_id)

        # Should prefix with job number and roi_ prefix
        expected_stream = f"{job_id.job_number}/roi_rectangle"
        assert rendered == {'roi': expected_stream}

    def test_render_creates_unique_streams_for_different_jobs(self) -> None:
        """Test that different jobs get unique ROI stream names."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        job_id_1 = JobId(source_name='detector1', job_number=uuid.UUID(int=111))
        job_id_2 = JobId(source_name='detector1', job_number=uuid.UUID(int=222))

        rendered_1 = aux_sources.render(job_id_1)
        rendered_2 = aux_sources.render(job_id_2)

        # Each job should get its own unique stream name
        assert rendered_1['roi'] != rendered_2['roi']
        assert rendered_1['roi'] == f"{job_id_1.job_number}/roi_rectangle"
        assert rendered_2['roi'] == f"{job_id_2.job_number}/roi_rectangle"

    def test_render_field_name_is_roi(self) -> None:
        """Test that the field name in rendered dict is 'roi'."""
        aux_sources = DetectorROIAuxSources()
        job_id = JobId(source_name='detector1', job_number=uuid.UUID(int=789))

        rendered = aux_sources.render(job_id)

        # Field name should be 'roi' (what the workflow expects)
        assert 'roi' in rendered
        assert len(rendered) == 1

    def test_model_dump_returns_roi_shape(self) -> None:
        """Test that model_dump returns the selected ROI shape."""
        aux_sources = DetectorROIAuxSources(roi='rectangle')
        dumped = aux_sources.model_dump(mode='json')
        assert dumped == {'roi': 'rectangle'}
