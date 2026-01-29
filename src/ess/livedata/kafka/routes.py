# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Self

from ..core.message import StreamKind
from .message_adapter import (
    Ad00ToScippAdapter,
    ChainedAdapter,
    CommandsAdapter,
    Da00ToScippAdapter,
    Ev44ToDetectorEventsAdapter,
    F144ToLogDataAdapter,
    KafkaToAd00Adapter,
    KafkaToDa00Adapter,
    KafkaToEv44Adapter,
    KafkaToF144Adapter,
    KafkaToMonitorEventsAdapter,
    MessageAdapter,
    ResponsesAdapter,
    RouteBySchemaAdapter,
    RouteByTopicAdapter,
    X5f2ToStatusAdapter,
)
from .stream_mapping import StreamMapping


class RoutingAdapterBuilder:
    def __init__(self, *, stream_mapping: StreamMapping):
        self._stream_mapping = stream_mapping
        self._routes: dict[str, MessageAdapter] = {}

    def build(self) -> RouteByTopicAdapter:
        """Builds the routing adapter."""
        return RouteByTopicAdapter(self._routes)

    def with_beam_monitor_route(self) -> Self:
        """Adds the beam monitor route."""
        adapter = RouteBySchemaAdapter(
            routes={
                'ev44': KafkaToMonitorEventsAdapter(
                    stream_lut=self._stream_mapping.monitors
                ),
                'da00': ChainedAdapter(
                    first=KafkaToDa00Adapter(
                        stream_lut=self._stream_mapping.monitors,
                        stream_kind=StreamKind.MONITOR_COUNTS,
                    ),
                    second=Da00ToScippAdapter(),
                ),
            }
        )
        for topic in self._stream_mapping.monitor_topics:
            self._routes[topic] = adapter
        return self

    def with_detector_route(self) -> Self:
        """Adds the detector route for ev44 schema."""
        adapter = ChainedAdapter(
            first=KafkaToEv44Adapter(
                stream_lut=self._stream_mapping.detectors,
                stream_kind=StreamKind.DETECTOR_EVENTS,
            ),
            second=Ev44ToDetectorEventsAdapter(
                merge_detectors=self._stream_mapping.instrument == 'bifrost'
            ),
        )
        for topic in self._stream_mapping.detector_topics:
            self._routes[topic] = adapter
        return self

    def with_area_detector_route(self) -> Self:
        """Adds the area detector route for ad00 schema."""
        adapter = ChainedAdapter(
            first=KafkaToAd00Adapter(
                stream_lut=self._stream_mapping.area_detectors,
                stream_kind=StreamKind.AREA_DETECTOR,
            ),
            second=Ad00ToScippAdapter(),
        )
        for topic in self._stream_mapping.area_detector_topics:
            self._routes[topic] = adapter
        return self

    def with_logdata_route(self) -> Self:
        """Adds the logdata route."""
        adapter = ChainedAdapter(
            first=KafkaToF144Adapter(stream_lut=self._stream_mapping.logs),
            second=F144ToLogDataAdapter(),
        )
        for topic in self._stream_mapping.log_topics:
            self._routes[topic] = adapter
        return self

    def with_livedata_data_route(self) -> Self:
        """Adds the livedata data route."""
        self._routes[self._stream_mapping.livedata_data_topic] = ChainedAdapter(
            first=KafkaToDa00Adapter(stream_kind=StreamKind.LIVEDATA_DATA),
            second=Da00ToScippAdapter(),
        )
        return self

    def with_livedata_commands_route(self) -> Self:
        """Adds the livedata commands route."""
        self._routes[self._stream_mapping.livedata_commands_topic] = CommandsAdapter()
        return self

    def with_livedata_responses_route(self) -> Self:
        """Adds the livedata responses route."""
        self._routes[self._stream_mapping.livedata_responses_topic] = ResponsesAdapter()
        return self

    def with_livedata_roi_route(self) -> Self:
        """Adds the livedata ROI route."""
        self._routes[self._stream_mapping.livedata_roi_topic] = ChainedAdapter(
            first=KafkaToDa00Adapter(stream_kind=StreamKind.LIVEDATA_ROI),
            second=Da00ToScippAdapter(),
        )
        return self

    def with_livedata_status_route(self) -> Self:
        """Adds the livedata status route for job and service heartbeats."""
        self._routes[self._stream_mapping.livedata_status_topic] = X5f2ToStatusAdapter()
        return self
