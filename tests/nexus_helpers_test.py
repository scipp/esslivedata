# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import tempfile
from pathlib import Path

import h5py
import pytest

from ess.livedata.nexus_helpers import (
    StreamInfo,
    _decode_attr,
    extract_stream_info,
    filter_f144_streams,
    generate_f144_log_streams_code,
    suggest_internal_name,
)


class TestDecodeAttr:
    def test_decodes_bytes_to_string(self) -> None:
        assert _decode_attr(b'hello') == 'hello'

    def test_decodes_utf8_bytes(self) -> None:
        assert _decode_attr(b'caf\xc3\xa9') == 'café'

    def test_converts_string_to_string(self) -> None:
        assert _decode_attr('hello') == 'hello'

    def test_converts_int_to_string(self) -> None:
        assert _decode_attr(42) == '42'

    def test_converts_float_to_string(self) -> None:
        assert _decode_attr(3.14) == '3.14'


class TestExtractStreamInfo:
    @pytest.fixture
    def in_memory_file(self):
        """Create an in-memory HDF5 file for testing."""
        # Use core driver with backing_store=False for pure in-memory operation
        f = h5py.File('test.h5', 'w', driver='core', backing_store=False)
        yield f
        f.close()

    def test_empty_file_returns_empty_list(self, in_memory_file) -> None:
        # Extract from the in-memory file object directly
        result = extract_stream_info(in_memory_file)
        assert result == []

    def test_file_without_streaming_groups_returns_empty_list(
        self, in_memory_file
    ) -> None:
        # Create groups without streaming attributes
        in_memory_file.create_group('entry')
        in_memory_file['entry'].create_group('instrument')
        in_memory_file['entry/instrument'].attrs['NX_class'] = 'NXinstrument'

        result = extract_stream_info(in_memory_file)
        assert result == []

    def test_extracts_single_streaming_group(self, in_memory_file) -> None:
        # Create a streaming data group
        group = in_memory_file.create_group('entry/detector/events')
        group.attrs['topic'] = 'detector_events'
        group.attrs['source'] = 'detector_1'
        group.attrs['NX_class'] = 'NXevent_data'
        group.attrs['writer_module'] = 'ev44'

        # Add parent NX_class
        in_memory_file['entry/detector'].attrs['NX_class'] = 'NXdetector'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].group_path == 'entry/detector/events'
        assert result[0].topic == 'detector_events'
        assert result[0].source == 'detector_1'
        assert result[0].nx_class == 'NXevent_data'
        assert result[0].parent_nx_class == 'NXdetector'
        assert result[0].writer_module == 'ev44'

    def test_extracts_multiple_streaming_groups(self, in_memory_file) -> None:
        # Create first streaming group
        group1 = in_memory_file.create_group('entry/detector/events')
        group1.attrs['topic'] = 'detector_events'
        group1.attrs['source'] = 'detector_1'
        group1.attrs['NX_class'] = 'NXevent_data'
        group1.attrs['writer_module'] = 'ev44'
        in_memory_file['entry/detector'].attrs['NX_class'] = 'NXdetector'

        # Create second streaming group
        group2 = in_memory_file.create_group('entry/monitor/data')
        group2.attrs['topic'] = 'monitor_data'
        group2.attrs['source'] = 'monitor_1'
        group2.attrs['NX_class'] = 'NXlog'
        group2.attrs['writer_module'] = 'f144'
        in_memory_file['entry/monitor'].attrs['NX_class'] = 'NXmonitor'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 2
        paths = {info.group_path for info in result}
        assert paths == {'entry/detector/events', 'entry/monitor/data'}

    def test_ignores_datasets(self, in_memory_file) -> None:
        # Create a dataset with streaming-like attributes (should be ignored)
        dataset = in_memory_file.create_dataset('entry/data', data=[1, 2, 3])
        dataset.attrs['topic'] = 'should_be_ignored'
        dataset.attrs['source'] = 'dataset_source'

        # Create a valid streaming group
        group = in_memory_file.create_group('entry/detector/events')
        group.attrs['topic'] = 'detector_events'
        group.attrs['source'] = 'detector_1'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].group_path == 'entry/detector/events'

    def test_requires_both_topic_and_source(self, in_memory_file) -> None:
        # Group with only topic
        group1 = in_memory_file.create_group('entry/detector/events')
        group1.attrs['topic'] = 'detector_events'

        # Group with only source
        group2 = in_memory_file.create_group('entry/monitor/data')
        group2.attrs['source'] = 'monitor_1'

        # Group with both
        group3 = in_memory_file.create_group('entry/chopper/data')
        group3.attrs['topic'] = 'chopper_topic'
        group3.attrs['source'] = 'chopper_1'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].group_path == 'entry/chopper/data'

    def test_handles_missing_optional_attributes(self, in_memory_file) -> None:
        # Create group with only required attributes
        group = in_memory_file.create_group('entry/detector/events')
        group.attrs['topic'] = 'detector_events'
        group.attrs['source'] = 'detector_1'
        # No NX_class, writer_module, or parent NX_class

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].nx_class == 'N/A'
        assert result[0].parent_nx_class == 'N/A'
        assert result[0].writer_module == 'N/A'

    def test_handles_bytes_attributes(self, in_memory_file) -> None:
        # HDF5 often stores strings as bytes
        group = in_memory_file.create_group('entry/detector/events')
        group.attrs['topic'] = b'detector_events'
        group.attrs['source'] = b'detector_1'
        group.attrs['NX_class'] = b'NXevent_data'
        group.attrs['writer_module'] = b'ev44'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].topic == 'detector_events'
        assert result[0].source == 'detector_1'
        assert result[0].nx_class == 'NXevent_data'
        assert result[0].writer_module == 'ev44'

    def test_handles_parent_without_nx_class(self, in_memory_file) -> None:
        # Create group with streaming attrs but parent has no NX_class
        group = in_memory_file.create_group('entry/detector/events')
        group.attrs['topic'] = 'detector_events'
        group.attrs['source'] = 'detector_1'
        # Parent exists but has no NX_class attribute

        result = extract_stream_info(in_memory_file)

        assert len(result) == 1
        assert result[0].parent_nx_class == 'N/A'

    def test_handles_nested_structure(self, in_memory_file) -> None:
        # Create deeply nested streaming groups
        group1 = in_memory_file.create_group('entry/instrument/detector_1/events')
        group1.attrs['topic'] = 'det1_events'
        group1.attrs['source'] = 'detector_1'
        in_memory_file['entry/instrument/detector_1'].attrs['NX_class'] = 'NXdetector'

        group2 = in_memory_file.create_group('entry/instrument/chopper/rotation_speed')
        group2.attrs['topic'] = 'chopper_rotation'
        group2.attrs['source'] = 'chopper_1'
        in_memory_file['entry/instrument/chopper'].attrs['NX_class'] = 'NXdisk_chopper'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 2
        parents = {info.parent_nx_class for info in result}
        assert parents == {'NXdetector', 'NXdisk_chopper'}

    def test_realistic_nexus_structure(self, in_memory_file) -> None:
        # Simulate a realistic NeXus file structure
        entry = in_memory_file.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'

        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'

        # Detector with event data
        detector = instrument.create_group('detector_1')
        detector.attrs['NX_class'] = 'NXdetector'

        events = detector.create_group('events')
        events.attrs['topic'] = 'DREAM_detectors'
        events.attrs['source'] = 'DREAM_detector_1'
        events.attrs['NX_class'] = 'NXevent_data'
        events.attrs['writer_module'] = 'ev44'

        # Monitor with log data
        monitor = instrument.create_group('monitor_1')
        monitor.attrs['NX_class'] = 'NXmonitor'

        monitor_data = monitor.create_group('data')
        monitor_data.attrs['topic'] = 'DREAM_monitors'
        monitor_data.attrs['source'] = 'DREAM_monitor_1'
        monitor_data.attrs['NX_class'] = 'NXlog'
        monitor_data.attrs['writer_module'] = 'f144'

        # Chopper with timestamp log
        chopper = instrument.create_group('chopper_1')
        chopper.attrs['NX_class'] = 'NXdisk_chopper'

        rotation = chopper.create_group('rotation_speed')
        rotation.attrs['topic'] = 'DREAM_choppers'
        rotation.attrs['source'] = 'DREAM_chopper_1_rotation'
        rotation.attrs['NX_class'] = 'NXlog'
        rotation.attrs['writer_module'] = 'tdct'

        result = extract_stream_info(in_memory_file)

        assert len(result) == 3

        # Check detector events
        det_info = next(
            info
            for info in result
            if info.group_path == 'entry/instrument/detector_1/events'
        )
        assert det_info.topic == 'DREAM_detectors'
        assert det_info.source == 'DREAM_detector_1'
        assert det_info.nx_class == 'NXevent_data'
        assert det_info.parent_nx_class == 'NXdetector'
        assert det_info.writer_module == 'ev44'

        # Check monitor data
        mon_info = next(
            info
            for info in result
            if info.group_path == 'entry/instrument/monitor_1/data'
        )
        assert mon_info.topic == 'DREAM_monitors'
        assert mon_info.nx_class == 'NXlog'
        assert mon_info.parent_nx_class == 'NXmonitor'
        assert mon_info.writer_module == 'f144'

        # Check chopper
        chop_info = next(
            info
            for info in result
            if info.group_path == 'entry/instrument/chopper_1/rotation_speed'
        )
        assert chop_info.topic == 'DREAM_choppers'
        assert chop_info.nx_class == 'NXlog'
        assert chop_info.parent_nx_class == 'NXdisk_chopper'
        assert chop_info.writer_module == 'tdct'

    def test_extract_from_file_path_str(self) -> None:
        # Test that function works with file path as string
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create file with streaming data
            with h5py.File(tmp_path, 'w') as f:
                group = f.create_group('entry/detector/events')
                group.attrs['topic'] = 'detector_events'
                group.attrs['source'] = 'detector_1'
                group.attrs['NX_class'] = 'NXevent_data'

            # Extract using string path
            result = extract_stream_info(tmp_path)

            assert len(result) == 1
            assert result[0].group_path == 'entry/detector/events'
            assert result[0].topic == 'detector_events'
            assert result[0].source == 'detector_1'
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_extract_from_file_path_pathlib(self) -> None:
        # Test that function works with Path object
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Create file with streaming data
            with h5py.File(tmp_path, 'w') as f:
                group = f.create_group('entry/monitor/data')
                group.attrs['topic'] = 'monitor_data'
                group.attrs['source'] = 'monitor_1'

            # Extract using Path object
            result = extract_stream_info(tmp_path)

            assert len(result) == 1
            assert result[0].group_path == 'entry/monitor/data'
            assert result[0].topic == 'monitor_data'
        finally:
            tmp_path.unlink(missing_ok=True)


class TestStreamInfo:
    def test_dataclass_construction(self) -> None:
        info = StreamInfo(
            group_path='entry/detector/events',
            topic='detector_events',
            source='detector_1',
            nx_class='NXevent_data',
            parent_nx_class='NXdetector',
            writer_module='ev44',
        )

        assert info.group_path == 'entry/detector/events'
        assert info.topic == 'detector_events'
        assert info.source == 'detector_1'
        assert info.nx_class == 'NXevent_data'
        assert info.parent_nx_class == 'NXdetector'
        assert info.writer_module == 'ev44'

    def test_dataclass_equality(self) -> None:
        info1 = StreamInfo(
            group_path='entry/detector/events',
            topic='detector_events',
            source='detector_1',
            nx_class='NXevent_data',
            parent_nx_class='NXdetector',
            writer_module='ev44',
        )
        info2 = StreamInfo(
            group_path='entry/detector/events',
            topic='detector_events',
            source='detector_1',
            nx_class='NXevent_data',
            parent_nx_class='NXdetector',
            writer_module='ev44',
        )

        assert info1 == info2

    def test_dataclass_inequality(self) -> None:
        info1 = StreamInfo(
            group_path='entry/detector/events',
            topic='detector_events',
            source='detector_1',
            nx_class='NXevent_data',
            parent_nx_class='NXdetector',
            writer_module='ev44',
        )
        info2 = StreamInfo(
            group_path='entry/detector/events',
            topic='different_topic',  # Different
            source='detector_1',
            nx_class='NXevent_data',
            parent_nx_class='NXdetector',
            writer_module='ev44',
        )

        assert info1 != info2

    def test_units_default_is_empty_string(self) -> None:
        info = StreamInfo(
            group_path='entry/detector/events',
            topic='detector_events',
            source='detector_1',
            nx_class='NXlog',
            parent_nx_class='NXdetector',
            writer_module='f144',
        )
        assert info.units == ''

    def test_units_can_be_set(self) -> None:
        info = StreamInfo(
            group_path='entry/motor/value',
            topic='motion',
            source='motor_1',
            nx_class='NXlog',
            parent_nx_class='NXpositioner',
            writer_module='f144',
            units='degrees',
        )
        assert info.units == 'degrees'


class TestSuggestInternalName:
    def test_extracts_name_before_value(self) -> None:
        info = StreamInfo(
            group_path='entry/instrument/rotation_stage/value',
            topic='motion',
            source='motor_1',
            nx_class='NXlog',
            parent_nx_class='NXpositioner',
            writer_module='f144',
        )
        assert suggest_internal_name(info) == 'rotation_stage'

    def test_extracts_name_before_idle_flag(self) -> None:
        info = StreamInfo(
            group_path='entry/instrument/rotation_stage/idle_flag',
            topic='motion',
            source='motor_1',
            nx_class='NXlog',
            parent_nx_class='NXpositioner',
            writer_module='f144',
        )
        assert suggest_internal_name(info) == 'rotation_stage'

    def test_removes_r0_suffix(self) -> None:
        info = StreamInfo(
            group_path='entry/instrument/detector_tank_angle_r0/value',
            topic='motion',
            source='motor_1',
            nx_class='NXlog',
            parent_nx_class='NXpositioner',
            writer_module='f144',
        )
        assert suggest_internal_name(info) == 'detector_tank_angle'

    def test_removes_t0_suffix(self) -> None:
        info = StreamInfo(
            group_path='entry/instrument/sample_stage_t0/value',
            topic='motion',
            source='motor_1',
            nx_class='NXlog',
            parent_nx_class='NXpositioner',
            writer_module='f144',
        )
        assert suggest_internal_name(info) == 'sample_stage'


class TestFilterF144Streams:
    @pytest.fixture
    def mixed_streams(self) -> list[StreamInfo]:
        return [
            StreamInfo(
                group_path='entry/detector/events',
                topic='detector',
                source='det_1',
                nx_class='NXevent_data',
                parent_nx_class='NXdetector',
                writer_module='ev44',
            ),
            StreamInfo(
                group_path='entry/motor/value',
                topic='motion',
                source='motor_1',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='degrees',
            ),
            StreamInfo(
                group_path='entry/chopper/rotation',
                topic='choppers',
                source='chopper_1',
                nx_class='NXlog',
                parent_nx_class='NXdisk_chopper',
                writer_module='f144',
                units='Hz',
            ),
        ]

    def test_filters_to_f144_only(self, mixed_streams) -> None:
        result = filter_f144_streams(mixed_streams)
        assert len(result) == 2
        assert all(info.writer_module == 'f144' for info in result)

    def test_filters_by_topic(self, mixed_streams) -> None:
        result = filter_f144_streams(mixed_streams, topic_filter='motion')
        assert len(result) == 1
        assert result[0].source == 'motor_1'

    def test_excludes_by_pattern(self, mixed_streams) -> None:
        result = filter_f144_streams(mixed_streams, exclude_patterns=['chopper'])
        assert len(result) == 1
        assert result[0].source == 'motor_1'


class TestGenerateF144LogStreamsCode:
    def test_generates_valid_python_dict(self) -> None:
        infos = [
            StreamInfo(
                group_path='entry/motor/value',
                topic='motion',
                source='MOTOR:PV:RBV',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='degrees',
            ),
        ]
        code = generate_f144_log_streams_code(infos, topic='motion')
        assert "f144_log_streams: dict[str, dict[str, str]] = {" in code
        assert (
            "'motor': {'source': 'MOTOR:PV:RBV', 'units': 'degrees', 'topic': 'motion'}"
            in code
        )

    def test_uses_dimensionless_for_empty_units(self) -> None:
        infos = [
            StreamInfo(
                group_path='entry/switch/value',
                topic='motion',
                source='SWITCH:PV',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='',
            ),
        ]
        code = generate_f144_log_streams_code(infos, topic='motion')
        assert "'units': 'dimensionless', 'topic': 'motion'" in code

    def test_prefers_value_over_idle_flag(self) -> None:
        infos = [
            StreamInfo(
                group_path='entry/motor/idle_flag',
                topic='motion',
                source='MOTOR:DMOV',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='',
            ),
            StreamInfo(
                group_path='entry/motor/value',
                topic='motion',
                source='MOTOR:RBV',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='degrees',
            ),
        ]
        code = generate_f144_log_streams_code(infos, topic='motion')
        # Should use 'value' entry, not 'idle_flag'
        assert 'MOTOR:RBV' in code
        assert 'MOTOR:DMOV' not in code

    def test_custom_variable_name(self) -> None:
        infos = [
            StreamInfo(
                group_path='entry/motor/value',
                topic='motion',
                source='MOTOR:RBV',
                nx_class='NXlog',
                parent_nx_class='NXpositioner',
                writer_module='f144',
                units='deg',
            ),
        ]
        code = generate_f144_log_streams_code(
            infos, topic='motion', variable_name='my_streams'
        )
        assert 'my_streams: dict[str, dict[str, str]] = {' in code
