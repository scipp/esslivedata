# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Helper class for testing ESSlivedata services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from streaming_data_types import eventdata_ev44, logdata_f144

from ess.livedata import Service, StreamKind
from ess.livedata.config import models
from ess.livedata.config.env import StreamingEnv
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.core.message_batcher import NaiveMessageBatcher
from ess.livedata.core.orchestrating_processor import OrchestratingProcessor
from ess.livedata.fakes import FakeMessageSink
from ess.livedata.kafka.message_adapter import FakeKafkaMessage, KafkaMessage
from ess.livedata.kafka.sink import UnrollingSinkAdapter
from ess.livedata.kafka.source import KafkaConsumer
from ess.livedata.service_factory import DataServiceBuilder


class FakeConsumer(KafkaConsumer):
    """Fake consumer that can be filled with custom messages for consumption."""

    def __init__(self) -> None:
        self._messages: list[KafkaMessage] = []

    def add_message(self, message: KafkaMessage) -> None:
        self._messages.append(message)

    def consume(self, num_messages: int, timeout: float) -> list[KafkaMessage]:
        _ = timeout  # Ignore timeout for the fake consumer
        result = self._messages[:num_messages]
        self._messages = self._messages[num_messages:]
        return result


@dataclass(kw_only=True)
class LivedataApp:
    """A testable "application" with a fake consumer and sink."""

    service: Service
    consumer: FakeConsumer
    sink: FakeMessageSink
    instrument: str

    def __post_init__(self) -> None:
        self.detector_topic = stream_kind_to_topic(
            instrument=self.instrument, kind=StreamKind.DETECTOR_EVENTS
        )
        self.monitor_topic = stream_kind_to_topic(
            instrument=self.instrument, kind=StreamKind.MONITOR_EVENTS
        )
        self._detector_config = get_config(self.instrument).detector_fakes
        dev_mapping = get_config(self.instrument).stream_mapping[StreamingEnv.DEV]
        self._monitor_source_names = [key.source_name for key in dev_mapping.monitors]
        self._rng = np.random.default_rng(seed=1234)  # Avoid test flakiness
        self._detector_events: bytes | None = None
        self._monitor_events: dict[str, bytes] = {}

    @staticmethod
    def from_service_builder(
        builder: DataServiceBuilder, use_naive_message_batcher: bool = True
    ) -> LivedataApp:
        """
        Create a LivedataApp from a service builder.

        The app is created with a fake sink that allows to inspect the messages sent to
        the sink. The consumer is a fake consumer that can be filled with custom
        messages for consumption.
        """
        sink = FakeMessageSink()
        consumer = FakeConsumer()
        if use_naive_message_batcher:
            builder._processor_cls = partial(
                OrchestratingProcessor, message_batcher=NaiveMessageBatcher()
            )
        service = builder.from_consumer(
            consumer=consumer,
            sink=UnrollingSinkAdapter(sink),
            raise_on_adapter_error=False,
            use_background_source=False,
        )
        return LivedataApp(
            service=service, consumer=consumer, sink=sink, instrument=builder.instrument
        )

    def step(self) -> None:
        """Run one step of the service."""
        self.service.step()

    def publish_config_message(self, key: models.ConfigKey, value: Any) -> None:
        message = FakeKafkaMessage(
            key=str(key).encode('utf-8'),
            value=json.dumps(value).encode('utf-8'),
            topic=stream_kind_to_topic(
                instrument=self.instrument, kind=StreamKind.LIVEDATA_COMMANDS
            ),
            timestamp=0,
        )
        self.consumer.add_message(message)

    def publish_log_message(
        self, *, source_name: str, time: float, value: float
    ) -> None:
        """Publish a log message to the consumer."""
        message = FakeKafkaMessage(
            value=logdata_f144.serialise_f144(
                source_name=source_name,
                value=value,
                timestamp_unix_ns=int(time * 1_000_000_000),
            ),
            topic=stream_kind_to_topic(instrument=self.instrument, kind=StreamKind.LOG),
            timestamp=int(time * 1_000_000_000),
        )
        self.consumer.add_message(message)

    def publish_monitor_events(
        self, *, size: int, time: int, reuse_events: bool = False
    ) -> None:
        """
        Publish monitor events to the consumer.

        If `reuse_events` is True, the same events are reused for each call to this
        method. This is useful for speeding up tests that need to send many event
        messages but do not require different events for each call.
        """
        for monitor_name in self._monitor_source_names:
            if not reuse_events or monitor_name not in self._monitor_events:
                events = self.make_serialized_ev44(
                    name=monitor_name, size=size, with_ids=False
                )
                self._monitor_events[monitor_name] = events
            else:
                events = self._monitor_events[monitor_name]

            monitor_message = FakeKafkaMessage(
                value=events,
                topic=self.monitor_topic,
                timestamp=time * 1_000_000_000,
            )
            self.consumer.add_message(monitor_message)

    def publish_events(
        self, *, size: int, time: int, reuse_events: bool = False
    ) -> None:
        """
        Publish events to the consumer.

        If `reuse_events` is True, the same events are reused for each call to this
        method. This is useful for speeding up tests that need to send many event
        messages but do not require different events for each call.
        """
        if not reuse_events or self._detector_events is None:
            events = self.make_serialized_ev44(
                name=next(iter(self._detector_config)), size=size, with_ids=True
            )
            self._detector_events = events
        else:
            events = self._detector_events
        message = FakeKafkaMessage(
            value=events,
            topic=self.detector_topic,
            timestamp=time * 1_000_000_000,
        )
        self.consumer.add_message(message)

    def publish_data(self, *, topic: str, time: int, data: bytes) -> None:
        """Publish data to the consumer."""
        message = FakeKafkaMessage(
            value=data,
            topic=topic,
            timestamp=time * 1_000_000_000,
        )
        self.consumer.add_message(message)

    def make_serialized_ev44(self, name: str, size: int, with_ids: bool) -> bytes:
        time_of_arrival = self._rng.uniform(0, 70_000_000, size).astype(np.int32)
        if with_ids:
            first, last = self._detector_config[name]
            pixel_id = self._rng.integers(first, last + 1, size, dtype=np.int32)
        else:
            pixel_id = np.zeros(size, dtype=np.int32)
        # Empty reference_time. KafkaToEv44Adapter falls back to message.timestamp().
        return eventdata_ev44.serialise_ev44(
            source_name=name,
            message_id=0,
            reference_time=[],
            reference_time_index=0,
            time_of_flight=time_of_arrival,
            pixel_id=pixel_id,
        )
