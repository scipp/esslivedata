# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Domain converters for transforming between domain models and schema formats.

This module contains domain converters that transform between application domain types
(e.g., DetectorEvents, LogData) and wire format schema types (e.g., ev44, f144).
These converters are Kafka-independent and focus purely on data transformation logic.
"""

import scipp as sc
from streaming_data_types import (
    dataarray_da00,
    eventdata_ev44,
    logdata_f144,
)
from streaming_data_types.status_x5f2 import StatusMessage as StatusX5f2

from ..handlers.accumulators import DetectorEvents, LogData, MonitorEvents
from ..handlers.config_handler import ConfigUpdate
from ..kafka.message_adapter import RawConfigItem
from ..kafka.scipp_da00_compat import da00_to_scipp, scipp_to_da00
from ..kafka.x5f2_compat import StatusMessage
from .job import JobStatus
from .message import (
    DomainConverter,
    Message,
    SchemaMessage,
    SchemaSerializer,
    StreamKind,
)


class ScippDa00Converter(DomainConverter[dataarray_da00.Variable, sc.DataArray]):
    """Domain converter between scipp DataArray and da00 format."""

    def to_schema(
        self, msg: Message[sc.DataArray]
    ) -> SchemaMessage[list[dataarray_da00.Variable]]:
        return SchemaMessage(
            stream=msg.stream,
            data=scipp_to_da00(msg.value),
            timestamp=msg.timestamp,
            key=None,
        )

    def to_domain(
        self, schema_msg: SchemaMessage[list[dataarray_da00.Variable]]
    ) -> Message[sc.DataArray]:
        return Message(
            stream=schema_msg.stream,
            value=da00_to_scipp(schema_msg.data),
            timestamp=schema_msg.timestamp,
        )


class MonitorEventsEv44Converter(
    DomainConverter[eventdata_ev44.EventData, MonitorEvents]
):
    """Domain converter between MonitorEvents and ev44 EventData format."""

    def to_domain(
        self, schema_msg: SchemaMessage[eventdata_ev44.EventData]
    ) -> Message[MonitorEvents]:
        return Message(
            stream=schema_msg.stream,
            value=MonitorEvents.from_ev44(schema_msg.data),
            timestamp=schema_msg.timestamp,
        )

    def to_schema(
        self, msg: Message[MonitorEvents]
    ) -> SchemaMessage[eventdata_ev44.EventData]:
        # Create minimal ev44 message from MonitorEvents
        ev44 = eventdata_ev44.EventData(
            source_name=msg.stream.name,
            message_id=0,
            reference_time=[msg.timestamp],
            reference_time_index=[0],
            time_of_arrival=msg.value.time_of_arrival,
            pixel_id=[],
        )
        return SchemaMessage(
            stream=msg.stream, data=ev44, timestamp=msg.timestamp, key=None
        )


class DetectorEventsEv44Converter(
    DomainConverter[eventdata_ev44.EventData, DetectorEvents]
):
    """Domain converter between DetectorEvents and ev44 EventData format."""

    def to_domain(
        self, schema_msg: SchemaMessage[eventdata_ev44.EventData]
    ) -> Message[DetectorEvents]:
        return Message(
            stream=schema_msg.stream,
            value=DetectorEvents.from_ev44(schema_msg.data),
            timestamp=schema_msg.timestamp,
        )

    def to_schema(
        self, msg: Message[DetectorEvents]
    ) -> SchemaMessage[eventdata_ev44.EventData]:
        # Create minimal ev44 message from DetectorEvents
        ev44 = eventdata_ev44.EventData(
            source_name=msg.stream.name,
            message_id=0,
            reference_time=[msg.timestamp],
            reference_time_index=[0],
            time_of_flight=msg.value.time_of_arrival,
            pixel_id=msg.value.pixel_id,
        )
        return SchemaMessage(
            stream=msg.stream, data=ev44, timestamp=msg.timestamp, key=None
        )


class LogDataF144Converter(DomainConverter[logdata_f144.ExtractedLogData, LogData]):
    """Domain converter between LogData and f144 format."""

    def to_domain(
        self, schema_msg: SchemaMessage[logdata_f144.ExtractedLogData]
    ) -> Message[LogData]:
        return Message(
            stream=schema_msg.stream,
            value=LogData.from_f144(schema_msg.data),
            timestamp=schema_msg.timestamp,
        )

    def to_schema(
        self, msg: Message[LogData]
    ) -> SchemaMessage[logdata_f144.ExtractedLogData]:
        # Create ExtractedLogData from LogData
        log_data = logdata_f144.ExtractedLogData(
            source_name=msg.stream.name,
            timestamp_unix_ns=msg.value.time,
            value=msg.value.value,
        )
        return SchemaMessage(
            stream=msg.stream, data=log_data, timestamp=msg.timestamp, key=None
        )


class StatusX5f2ToJobStatusConverter(DomainConverter[StatusX5f2, JobStatus]):
    """Domain converter between StatusX5f2 (wire format) and JobStatus (domain)."""

    def to_domain(self, schema_msg: SchemaMessage[StatusX5f2]) -> Message[JobStatus]:
        # Use StatusMessage Pydantic model as parsing helper
        status_message = StatusMessage(**schema_msg.data._asdict())
        return Message(
            stream=schema_msg.stream,
            value=status_message.to_job_status(),
            timestamp=schema_msg.timestamp,
        )

    def to_schema(self, msg: Message[JobStatus]) -> SchemaMessage[StatusX5f2]:
        # Use StatusMessage Pydantic model to build and serialize
        status_message = StatusMessage.from_job_status(msg.value)
        # model_dump with mode='json' applies field_serializer methods
        status_x5f2 = StatusX5f2(**status_message.model_dump(mode='json'))
        return SchemaMessage(
            stream=msg.stream, data=status_x5f2, timestamp=msg.timestamp, key=None
        )


class ConfigUpdateConverter(DomainConverter[RawConfigItem, ConfigUpdate]):
    """Domain converter between ConfigUpdate and RawConfigItem."""

    def to_domain(
        self, schema_msg: SchemaMessage[RawConfigItem]
    ) -> Message[ConfigUpdate]:
        return Message(
            stream=schema_msg.stream,
            value=ConfigUpdate.from_raw(schema_msg.data),
            timestamp=schema_msg.timestamp,
        )

    def to_schema(self, msg: Message[ConfigUpdate]) -> SchemaMessage[RawConfigItem]:
        # Encode config_key and value to bytes
        key_bytes = str(msg.value.config_key).encode('utf-8')
        value_bytes = msg.value.value.model_dump_json().encode('utf-8')

        return SchemaMessage(
            stream=msg.stream,
            data=RawConfigItem(key=key_bytes, value=value_bytes),
            timestamp=msg.timestamp,
            key=key_bytes,  # Key for Kafka compaction
        )


class CompoundDomainConverter:
    """
    Domain converter that dispatches to registered converters by StreamKind.

    This allows converting multiple domain types to their corresponding schema formats.
    """

    def __init__(self):
        self._converters: dict[StreamKind, DomainConverter] = {}

    def register(
        self, stream_kind: StreamKind, converter: DomainConverter
    ) -> 'CompoundDomainConverter':
        """
        Register a converter for a specific StreamKind.

        Parameters
        ----------
        stream_kind:
            The stream kind to register the converter for.
        converter:
            The domain converter to use for this stream kind.

        Returns
        -------
        :
            Self, for method chaining.
        """
        self._converters[stream_kind] = converter
        return self

    def to_schema(self, msg: Message) -> SchemaMessage:
        """
        Convert domain message to schema message.

        Parameters
        ----------
        msg:
            Domain message to convert.

        Returns
        -------
        :
            Schema message.
        """
        if msg.stream.kind not in self._converters:
            raise ValueError(
                f"No converter registered for StreamKind {msg.stream.kind}. "
                f"Registered kinds: {set(self._converters.keys())}"
            )
        converter = self._converters[msg.stream.kind]
        return converter.to_schema(msg)

    def to_domain(self, schema_msg: SchemaMessage) -> Message:
        """
        Convert schema message to domain message.

        Parameters
        ----------
        schema_msg:
            Schema message to convert.

        Returns
        -------
        :
            Domain message.
        """
        if schema_msg.stream.kind not in self._converters:
            raise ValueError(
                f"No converter registered for StreamKind {schema_msg.stream.kind}. "
                f"Registered kinds: {set(self._converters.keys())}"
            )
        converter = self._converters[schema_msg.stream.kind]
        return converter.to_domain(schema_msg)

    @classmethod
    def make_standard(cls) -> 'CompoundDomainConverter':
        """
        Create a CompoundDomainConverter with all standard domain converters.

        This registers converters for common StreamKinds to their schema formats.

        Returns
        -------
        :
            Configured compound converter.
        """
        # Create converter instances (can be shared where domain types are the same)
        monitor_events_converter = MonitorEventsEv44Converter()
        detector_events_converter = DetectorEventsEv44Converter()
        scipp_da00_converter = ScippDa00Converter()
        log_data_converter = LogDataF144Converter()
        status_converter = StatusX5f2ToJobStatusConverter()
        config_converter = ConfigUpdateConverter()

        compound = cls()
        compound.register(StreamKind.MONITOR_EVENTS, monitor_events_converter)
        compound.register(StreamKind.MONITOR_COUNTS, monitor_events_converter)
        compound.register(StreamKind.DETECTOR_EVENTS, detector_events_converter)
        compound.register(StreamKind.LIVEDATA_DATA, scipp_da00_converter)
        compound.register(StreamKind.LIVEDATA_ROI, scipp_da00_converter)
        compound.register(StreamKind.LOG, log_data_converter)
        compound.register(StreamKind.LIVEDATA_STATUS, status_converter)
        compound.register(StreamKind.LIVEDATA_COMMANDS, config_converter)
        compound.register(StreamKind.LIVEDATA_RESPONSES, config_converter)

        return compound


class CompoundSchemaSerializer:
    """
    Schema serializer that dispatches to registered serializers by StreamKind.

    This allows a single sink to handle messages with different schema formats.
    Multiple StreamKinds can share the same serializer instance.
    """

    def __init__(self):
        self._serializers: dict[StreamKind, SchemaSerializer] = {}

    def register(
        self, stream_kind: StreamKind, serializer: SchemaSerializer
    ) -> 'CompoundSchemaSerializer':
        """
        Register a serializer for a specific StreamKind.

        Parameters
        ----------
        stream_kind:
            The stream kind to register the serializer for.
        serializer:
            The schema serializer to use for this stream kind.

        Returns
        -------
        :
            Self, for method chaining.
        """
        self._serializers[stream_kind] = serializer
        return self

    def serialize(self, msg: SchemaMessage) -> bytes:
        """
        Serialize schema message by dispatching to the registered serializer.

        Parameters
        ----------
        msg:
            Schema message to serialize.

        Returns
        -------
        :
            Serialized bytes.
        """
        if msg.stream.kind not in self._serializers:
            raise ValueError(
                f"No serializer registered for StreamKind {msg.stream.kind}. "
                f"Registered kinds: {set(self._serializers.keys())}"
            )
        serializer = self._serializers[msg.stream.kind]
        return serializer.serialize(msg)

    @classmethod
    def make_standard(cls) -> 'CompoundSchemaSerializer':
        """
        Create a CompoundSchemaSerializer with all standard schema serializers.

        This registers serializers for common StreamKinds:
        - MONITOR_EVENTS, DETECTOR_EVENTS, MONITOR_COUNTS: ev44
        - LIVEDATA_DATA, LIVEDATA_ROI: da00
        - LOG: f144
        - LIVEDATA_STATUS: x5f2
        - LIVEDATA_COMMANDS, LIVEDATA_RESPONSES: config

        Returns
        -------
        :
            Configured compound serializer.
        """
        # Import here to avoid circular dependency
        from ..kafka.schema_codecs import (
            ConfigSerializer,
            Da00Serializer,
            Ev44Serializer,
            F144Serializer,
            X5f2Serializer,
        )

        # Create serializer instances (can be shared where formats are the same)
        ev44_serializer = Ev44Serializer()
        da00_serializer = Da00Serializer()
        f144_serializer = F144Serializer()
        x5f2_serializer = X5f2Serializer()
        config_serializer = ConfigSerializer()

        compound = cls()
        compound.register(StreamKind.MONITOR_EVENTS, ev44_serializer)
        compound.register(StreamKind.MONITOR_COUNTS, ev44_serializer)
        compound.register(StreamKind.DETECTOR_EVENTS, ev44_serializer)
        compound.register(StreamKind.LIVEDATA_DATA, da00_serializer)
        compound.register(StreamKind.LIVEDATA_ROI, da00_serializer)
        compound.register(StreamKind.LOG, f144_serializer)
        compound.register(StreamKind.LIVEDATA_STATUS, x5f2_serializer)
        compound.register(StreamKind.LIVEDATA_COMMANDS, config_serializer)
        compound.register(StreamKind.LIVEDATA_RESPONSES, config_serializer)

        return compound
