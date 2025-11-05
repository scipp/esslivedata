# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass

KafkaTopic = str


@dataclass(frozen=True, slots=True, kw_only=True)
class InputStreamKey:
    """Unique identified for an input stream."""

    topic: KafkaTopic
    source_name: str


StreamLUT = dict[InputStreamKey, str]


class StreamMapping:
    """
    Helper for mapping input streams to ESSlivedata-internal stream names.

    This isolates the internals of ESSlivedata from the input stream identifiers,
    which may contain irrelevant information as well as implementation details
    such as split topics.
    """

    def __init__(
        self,
        *,
        instrument: str,
        detectors: StreamLUT,
        monitors: StreamLUT,
        log_topics: set[KafkaTopic] | None = None,
        livedata_commands_topic: str,
        livedata_data_topic: str,
        livedata_roi_topic: str,
        livedata_status_topic: str,
    ) -> None:
        self.instrument = instrument
        self._detectors = detectors
        self._monitors = monitors
        # Currently we simply reuse the source_name as the stream name
        self._logs = None
        self._log_topics = log_topics or set()
        self._livedata_commands_topic = livedata_commands_topic
        self._livedata_data_topic = livedata_data_topic
        self._livedata_roi_topic = livedata_roi_topic
        self._livedata_status_topic = livedata_status_topic

    @property
    def livedata_commands_topic(self) -> KafkaTopic:
        """Returns the livedata commands topic."""
        return self._livedata_commands_topic

    @property
    def livedata_data_topic(self) -> KafkaTopic:
        """Returns the livedata data topic."""
        return self._livedata_data_topic

    @property
    def livedata_roi_topic(self) -> KafkaTopic:
        """Returns the livedata ROI topic."""
        return self._livedata_roi_topic

    @property
    def livedata_status_topic(self) -> KafkaTopic:
        """Returns the livedata status topic."""
        return self._livedata_status_topic

    @property
    def detector_topics(self) -> set[KafkaTopic]:
        """Returns the list of detector topics."""
        return {stream.topic for stream in self.detectors.keys()}

    @property
    def monitor_topics(self) -> set[KafkaTopic]:
        """Returns the list of monitor topics."""
        return {stream.topic for stream in self.monitors.keys()}

    @property
    def log_topics(self) -> set[KafkaTopic]:
        """Returns the list of log topics."""
        return self._log_topics

    @property
    def detectors(self) -> StreamLUT:
        """Returns the mapping for detector data."""
        return self._detectors

    @property
    def monitors(self) -> StreamLUT:
        """Returns the mapping for monitor data."""
        return self._monitors

    @property
    def logs(self) -> StreamLUT | None:
        """Returns the mapping for log data."""
        return self._logs
