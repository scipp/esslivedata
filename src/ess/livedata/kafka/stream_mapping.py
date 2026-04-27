# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass

KafkaTopic = str


@dataclass(frozen=True, slots=True, kw_only=True)
class InputStreamKey:
    """Unique identified for an input stream."""

    topic: KafkaTopic
    source_name: str


StreamLUT = dict[InputStreamKey, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class LivedataTopics:
    """Fixed per-instrument topic names for livedata infrastructure."""

    instrument: str
    livedata_commands: str
    livedata_data: str
    livedata_responses: str
    livedata_roi: str
    livedata_status: str

    @property
    def filewriter(self) -> KafkaTopic:
        """Returns the filewriter topic for run start/stop messages."""
        return f"{self.instrument}_filewriter"


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
        area_detectors: StreamLUT | None = None,
        logs: StreamLUT | None = None,
        livedata_commands_topic: str,
        livedata_data_topic: str,
        livedata_responses_topic: str,
        livedata_roi_topic: str,
        livedata_status_topic: str,
    ) -> None:
        self.instrument = instrument
        self._detectors = detectors
        self._monitors = monitors
        self._area_detectors = area_detectors or {}
        self._logs = logs
        self._topics = LivedataTopics(
            instrument=instrument,
            livedata_commands=livedata_commands_topic,
            livedata_data=livedata_data_topic,
            livedata_responses=livedata_responses_topic,
            livedata_roi=livedata_roi_topic,
            livedata_status=livedata_status_topic,
        )

    @classmethod
    def _from_luts(
        cls,
        *,
        topics: LivedataTopics,
        detectors: StreamLUT,
        monitors: StreamLUT,
        area_detectors: StreamLUT | None = None,
        logs: StreamLUT | None = None,
    ) -> StreamMapping:
        mapping = cls.__new__(cls)
        mapping.instrument = topics.instrument
        mapping._detectors = detectors
        mapping._monitors = monitors
        mapping._area_detectors = area_detectors or {}
        mapping._logs = logs
        mapping._topics = topics
        return mapping

    @property
    def topics(self) -> LivedataTopics:
        """Returns the fixed livedata infrastructure topics."""
        return self._topics

    @property
    def detector_topics(self) -> set[KafkaTopic]:
        """Returns the list of detector topics."""
        return {stream.topic for stream in self.detectors.keys()}

    @property
    def area_detector_topics(self) -> set[KafkaTopic]:
        """Returns the list of area detector topics."""
        return {stream.topic for stream in self.area_detectors.keys()}

    @property
    def monitor_topics(self) -> set[KafkaTopic]:
        """Returns the list of monitor topics."""
        return {stream.topic for stream in self.monitors.keys()}

    @property
    def log_topics(self) -> set[KafkaTopic]:
        """Returns the set of log topics."""
        if self._logs is None:
            return set()
        return {stream.topic for stream in self._logs.keys()}

    @property
    def detectors(self) -> StreamLUT:
        """Returns the mapping for detector data."""
        return self._detectors

    @property
    def monitors(self) -> StreamLUT:
        """Returns the mapping for monitor data."""
        return self._monitors

    @property
    def area_detectors(self) -> StreamLUT:
        """Returns the mapping for area detector data."""
        return self._area_detectors

    @property
    def logs(self) -> StreamLUT | None:
        """Returns the mapping for log data."""
        return self._logs

    @property
    def all_stream_names(self) -> set[str]:
        """Returns the set of all internal stream names across all LUTs."""
        names = set(self._detectors.values())
        names |= set(self._monitors.values())
        names |= set(self._area_detectors.values())
        if self._logs is not None:
            names |= set(self._logs.values())
        return names

    def filtered(self, needed: set[str]) -> StreamMapping:
        """Return copy with only entries whose internal names are in ``needed``."""
        return StreamMapping._from_luts(
            topics=self._topics,
            detectors={k: v for k, v in self._detectors.items() if v in needed},
            monitors={k: v for k, v in self._monitors.items() if v in needed},
            area_detectors={
                k: v for k, v in self._area_detectors.items() if v in needed
            },
            logs={k: v for k, v in self._logs.items() if v in needed}
            if self._logs is not None
            else None,
        )
