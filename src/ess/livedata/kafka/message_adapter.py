# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, TypeVar

import scipp as sc
import streaming_data_types
import streaming_data_types.exceptions
import structlog
from streaming_data_types import (
    area_detector_ad00,
    dataarray_da00,
    eventdata_ev44,
    logdata_f144,
)
from streaming_data_types.fbschemas.eventdata_ev44 import Event44Message

from ess.livedata.core.job import JobStatus, ServiceStatus

from ..config.acknowledgement import CommandAcknowledgement
from ..core.message import (
    COMMANDS_STREAM_ID,
    RESPONSES_STREAM_ID,
    STATUS_STREAM_ID,
    Message,
    MessageSource,
    StreamId,
    StreamKind,
)
from ..handlers.accumulators import LogData
from ..handlers.to_nxevent_data import DetectorEvents, MonitorEvents
from .scipp_ad00_compat import ad00_to_scipp
from .scipp_da00_compat import da00_to_scipp
from .stream_mapping import InputStreamKey, StreamLUT
from .x5f2_compat import x5f2_to_status

logger = structlog.get_logger(__name__)

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class KafkaMessage(Protocol):
    """Simplified Kafka message interface for testing purposes."""

    def error(self) -> Any | None:
        pass

    def key(self) -> bytes: ...

    def value(self) -> bytes: ...

    def timestamp(self) -> tuple[int, int]: ...

    def topic(self) -> str: ...


class FakeKafkaMessage(KafkaMessage):
    def __init__(
        self, *, key: bytes = b'', value: bytes, topic: str, timestamp: int = 0
    ):
        self._key = key
        self._value = value
        self._topic = topic
        self._timestamp = timestamp

    def error(self) -> Any | None:
        return None

    def key(self) -> bytes:
        return self._key

    def value(self) -> bytes:
        return self._value

    def timestamp(self) -> tuple[int, int]:
        return (0, self._timestamp)

    def topic(self) -> str:
        return self._topic

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FakeKafkaMessage):
            return False
        return self._value == other._value and self._topic == other._topic


class MessageAdapter(Protocol, Generic[T, U]):
    def adapt(self, message: T) -> U: ...


class KafkaAdapter(MessageAdapter[KafkaMessage, Message[T]]):
    """
    Base class for Kafka adapters.

    This provides a common interface for converting the unique (topic, source_name) to
    the Livedata-internal stream ID. The actual conversion is done by the subclasses.
    This conversion serves as a mechanism to isolate Livedata from irrelevant details of
    the Kafka topics.
    """

    def __init__(self, *, stream_lut: StreamLUT | None = None, stream_kind: StreamKind):
        self._stream_lut = stream_lut
        self._stream_kind = stream_kind

    def get_stream_id(self, topic: str, source_name: str) -> StreamId:
        if self._stream_lut is None:
            # Assume the source name is unique
            return StreamId(kind=self._stream_kind, name=source_name)
        input_key = InputStreamKey(topic=topic, source_name=source_name)
        return StreamId(kind=self._stream_kind, name=self._stream_lut[input_key])


class KafkaToEv44Adapter(KafkaAdapter[eventdata_ev44.EventData]):
    def adapt(self, message: KafkaMessage) -> Message[eventdata_ev44.EventData]:
        ev44 = eventdata_ev44.deserialise_ev44(message.value())
        stream = self.get_stream_id(topic=message.topic(), source_name=ev44.source_name)
        # A fallback, useful in particular for testing so serialized data can be reused.
        if ev44.reference_time.size > 0:
            timestamp = ev44.reference_time[-1]
        else:
            timestamp = message.timestamp()[1]
        return Message(timestamp=timestamp, stream=stream, value=ev44)


class KafkaToDa00Adapter(KafkaAdapter[list[dataarray_da00.Variable]]):
    def adapt(self, message: KafkaMessage) -> Message[list[dataarray_da00.Variable]]:
        da00: dataarray_da00.da00_DataArray_t
        da00 = dataarray_da00.deserialise_da00(message.value())  # type: ignore[reportAssignmentType]
        key = self.get_stream_id(topic=message.topic(), source_name=da00.source_name)
        timestamp = da00.timestamp_ns
        return Message(timestamp=timestamp, stream=key, value=da00.data)


class KafkaToF144Adapter(KafkaAdapter[logdata_f144.ExtractedLogData]):
    def __init__(self, *, stream_lut: StreamLUT | None = None):
        super().__init__(stream_lut=stream_lut, stream_kind=StreamKind.LOG)

    def adapt(self, message: KafkaMessage) -> Message[logdata_f144.ExtractedLogData]:
        log_data = logdata_f144.deserialise_f144(message.value())
        key = self.get_stream_id(
            topic=message.topic(), source_name=log_data.source_name
        )
        timestamp = log_data.timestamp_unix_ns
        return Message(timestamp=timestamp, stream=key, value=log_data)


class F144ToLogDataAdapter(
    MessageAdapter[Message[logdata_f144.ExtractedLogData], Message[LogData]]
):
    def adapt(
        self, message: Message[logdata_f144.ExtractedLogData]
    ) -> Message[LogData]:
        return Message(
            timestamp=message.timestamp,
            stream=message.stream,
            value=LogData.from_f144(message.value),
        )


class Ev44ToMonitorEventsAdapter(
    MessageAdapter[Message[eventdata_ev44.EventData], list[Message[MonitorEvents]]]
):
    def adapt(
        self, message: Message[eventdata_ev44.EventData]
    ) -> list[Message[MonitorEvents]]:
        return [
            Message(
                timestamp=ts if ts is not None else message.timestamp,
                stream=message.stream,
                value=events,
            )
            for ts, events in MonitorEvents.from_ev44(message.value)
        ]


class X5f2ToStatusAdapter(
    MessageAdapter[KafkaMessage, Message[JobStatus | ServiceStatus]]
):
    """
    Adapter for status messages that returns JobStatus or ServiceStatus.

    Discriminates based on the `message_type` field in the x5f2 status_json.
    """

    def adapt(self, message: KafkaMessage) -> Message[JobStatus | ServiceStatus]:
        return Message(
            timestamp=message.timestamp()[1],
            stream=STATUS_STREAM_ID,
            value=x5f2_to_status(message.value()),
        )


class KafkaToMonitorEventsAdapter(KafkaAdapter[MonitorEvents]):
    """
    Directly adapts a Kafka message to MonitorEvents.

    This bypasses an intermediate eventdata_ev44.EventData object, which would require
    decoding unused fields. If we know the ev44 is for a monitor then avoiding this
    yields better performance.
    """

    def __init__(self, stream_lut: StreamLUT):
        super().__init__(stream_lut=stream_lut, stream_kind=StreamKind.MONITOR_EVENTS)

    def adapt(self, message: KafkaMessage) -> list[Message[MonitorEvents]]:
        buffer = message.value()
        eventdata_ev44.check_schema_identifier(buffer, eventdata_ev44.FILE_IDENTIFIER)
        event = Event44Message.Event44Message.GetRootAs(buffer, 0)
        stream = self.get_stream_id(
            topic=message.topic(), source_name=event.SourceName().decode("utf-8")
        )
        reference_time = event.ReferenceTimeAsNumpy()
        reference_time_index = event.ReferenceTimeIndexAsNumpy()
        time_of_arrival = event.TimeOfFlightAsNumpy()

        if reference_time.size == 0:
            # A fallback, useful for testing so serialized data can be reused.
            return [
                Message(
                    timestamp=message.timestamp()[1],
                    stream=stream,
                    value=MonitorEvents(time_of_arrival=time_of_arrival, unit='ns'),
                )
            ]

        results: list[Message[MonitorEvents]] = []
        for i in range(len(reference_time)):
            start = reference_time_index[i]
            end = (
                reference_time_index[i + 1]
                if i + 1 < len(reference_time_index)
                else len(time_of_arrival)
            )
            results.append(
                Message(
                    timestamp=int(reference_time[i]),
                    stream=stream,
                    value=MonitorEvents(
                        time_of_arrival=time_of_arrival[start:end], unit='ns'
                    ),
                )
            )
        return results


class Ev44ToDetectorEventsAdapter(
    MessageAdapter[Message[eventdata_ev44.EventData], list[Message[DetectorEvents]]]
):
    def __init__(self, *, merge_detectors: bool = False):
        """
        Parameters
        ----------
        merge_detectors
            If True, all detectors are merged into a single "unified_detector". This is
            useful for instruments with many detector banks that should be treated as a
            single bank. Note that event_id/detector_number must be unique across all
            detectors.
        """
        self._merge_detectors = merge_detectors

    def adapt(
        self, message: Message[eventdata_ev44.EventData]
    ) -> list[Message[DetectorEvents]]:
        stream = message.stream
        if self._merge_detectors:
            stream = replace(stream, name='unified_detector')
        return [
            Message(
                timestamp=ts if ts is not None else message.timestamp,
                stream=stream,
                value=events,
            )
            for ts, events in DetectorEvents.from_ev44(message.value)
        ]


class Da00ToScippAdapter(
    MessageAdapter[Message[list[dataarray_da00.Variable]], Message[sc.DataArray]]
):
    def adapt(
        self, message: Message[list[dataarray_da00.Variable]]
    ) -> Message[sc.DataArray]:
        return Message(
            timestamp=message.timestamp,
            stream=message.stream,
            value=da00_to_scipp(message.value),
        )


class KafkaToAd00Adapter(KafkaAdapter[area_detector_ad00.ADArray]):
    def adapt(self, message: KafkaMessage) -> Message[area_detector_ad00.ADArray]:
        ad00 = area_detector_ad00.deserialise_ad00(message.value())
        key = self.get_stream_id(topic=message.topic(), source_name=ad00.source_name)
        timestamp = ad00.timestamp_ns
        return Message(timestamp=timestamp, stream=key, value=ad00)


class Ad00ToScippAdapter(
    MessageAdapter[Message[area_detector_ad00.ADArray], Message[sc.DataArray]]
):
    def adapt(
        self, message: Message[area_detector_ad00.ADArray]
    ) -> Message[sc.DataArray]:
        return Message(
            timestamp=message.timestamp,
            stream=message.stream,
            value=ad00_to_scipp(message.value),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RawConfigItem:
    key: bytes
    value: bytes


class CommandsAdapter(MessageAdapter[KafkaMessage, Message[RawConfigItem]]):
    """Adapts Kafka messages from the livedata commands topic."""

    def adapt(self, message: KafkaMessage) -> Message[RawConfigItem]:
        timestamp = message.timestamp()[1]
        # Livedata configuration uses a compacted Kafka topic. The Kafka message key
        # is the encoded string representation of a :py:class:`ConfigKey` object.
        item = RawConfigItem(key=message.key(), value=message.value())
        return Message(stream=COMMANDS_STREAM_ID, timestamp=timestamp, value=item)


class ResponsesAdapter(MessageAdapter[KafkaMessage, Message[CommandAcknowledgement]]):
    """Adapts Kafka messages from the livedata responses topic."""

    def adapt(self, message: KafkaMessage) -> Message[CommandAcknowledgement]:
        timestamp = message.timestamp()[1]
        ack = CommandAcknowledgement.model_validate_json(message.value())
        return Message(stream=RESPONSES_STREAM_ID, timestamp=timestamp, value=ack)


class ChainedAdapter(MessageAdapter[T, V]):
    """
    Chains two adapters together.
    """

    def __init__(self, first: MessageAdapter[T, U], second: MessageAdapter[U, V]):
        self._first = first
        self._second = second

    def adapt(self, message: T) -> V:
        intermediate = self._first.adapt(message)
        return self._second.adapt(intermediate)


class RouteBySchemaAdapter(MessageAdapter[KafkaMessage, T]):
    """
    Routes messages to different adapters based on the schema.
    """

    def __init__(self, routes: dict[str, MessageAdapter[KafkaMessage, T]]):
        self._routes = routes

    def adapt(self, message: KafkaMessage) -> T:
        schema = streaming_data_types.utils.get_schema(message.value())
        if schema is None:
            raise streaming_data_types.exceptions.WrongSchemaException(
                f"Could not determine schema from message (topic={message.topic()}). "
                f"Expected one of: {list(self._routes.keys())}"
            )
        if schema not in self._routes:
            raise streaming_data_types.exceptions.WrongSchemaException(
                f"Unexpected schema '{schema}' from topic '{message.topic()}'. "
                f"Expected one of: {list(self._routes.keys())}"
            )
        return self._routes[schema].adapt(message)


class RouteByTopicAdapter(MessageAdapter[KafkaMessage, T]):
    """
    Routes messages to different adapters based on the topic.
    """

    def __init__(self, routes: dict[str, MessageAdapter[KafkaMessage, T]]):
        self._routes = routes

    @property
    def topics(self) -> list[str]:
        """Returns the list of topics to subscribe to."""
        return list(self._routes.keys())

    def adapt(self, message: KafkaMessage) -> T:
        topic = message.topic()
        if topic not in self._routes:
            raise KeyError(
                f"Received message from unexpected topic '{topic}'. "
                f"Configured topics: {list(self._routes.keys())}"
            )
        return self._routes[topic].adapt(message)


class AdaptingMessageSource(MessageSource[U]):
    """
    Wraps a source of messages and adapts them to a different type.
    """

    def __init__(
        self,
        source: MessageSource[T],
        adapter: MessageAdapter[T, U],
        raise_on_error: bool = False,
    ):
        """
        Parameters
        ----------
        source
            The source of messages to adapt.
        adapter
            The adapter to use.
        raise_on_error
            If True, exceptions during adaptation will be re-raised. If False,
            they will be logged and the message will be skipped.
        """
        self._source = source
        self._adapter = adapter
        self._raise_on_error = raise_on_error

    def get_messages(self) -> Sequence[U]:
        raw_messages = self._source.get_messages()
        adapted = []
        for msg in raw_messages:
            try:
                result = self._adapter.adapt(msg)
            except streaming_data_types.exceptions.WrongSchemaException:
                logger.warning('Message %s has an unknown schema. Skipping.', msg)
                if self._raise_on_error:
                    raise
            except Exception as e:
                logger.exception('Error adapting message %s: %s', msg, e)
                if self._raise_on_error:
                    raise
            else:
                if isinstance(result, list):
                    adapted.extend(result)
                else:
                    adapted.append(result)
        return adapted
