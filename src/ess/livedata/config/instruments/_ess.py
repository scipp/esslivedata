# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Helpers for setting up the stream mapping for the ESS instruments.
"""

from __future__ import annotations

from typing import Any

from ess.livedata import StreamKind
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.kafka import InputStreamKey, StreamLUT, StreamMapping


def _make_cbm_monitors(
    instrument: str, monitor_count: int = 10, monitor_names: list[str] | None = None
) -> StreamLUT:
    # Might also be MONITOR_COUNTS, but topic is supposedly the same.
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.MONITOR_EVENTS)
    if monitor_names is None:
        monitor_names = [f'monitor{monitor}' for monitor in range(monitor_count)]
    return {
        InputStreamKey(topic=topic, source_name=f'cbm{monitor}'): name
        for monitor, name in enumerate(monitor_names)
    }


def _make_dev_detectors(*, instrument: str, detectors: list[str]) -> StreamLUT:
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.DETECTOR_EVENTS)
    return {InputStreamKey(topic=topic, source_name=name): name for name in detectors}


def _make_dev_beam_monitors(
    instrument: str, monitor_names: list[str] | None = None
) -> StreamLUT:
    # Might also be MONITOR_COUNTS, but topic is supposedly the same.
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.MONITOR_EVENTS)
    if monitor_names is None:
        monitor_names = [f'monitor{monitor}' for monitor in range(10)]
    return {
        InputStreamKey(topic=topic, source_name=f'monitor{monitor + 1}'): name
        for monitor, name in enumerate(monitor_names)
    }


def make_dev_stream_mapping(
    instrument: str,
    *,
    detector_names: list[str],
    monitor_names: list[str] | None = None,
) -> StreamMapping:
    motion_topic = f'{instrument}_motion'
    log_topics = {motion_topic}
    return StreamMapping(
        instrument=instrument,
        detectors=_make_dev_detectors(instrument=instrument, detectors=detector_names),
        monitors=_make_dev_beam_monitors(instrument, monitor_names=monitor_names),
        log_topics=log_topics,
        livedata_config_topic=stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_CONFIG
        ),
        livedata_data_topic=stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_DATA
        ),
        livedata_roi_topic=stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_ROI
        ),
        livedata_status_topic=stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_STATUS
        ),
    )


def make_common_stream_mapping_inputs(
    instrument: str, *, monitor_names: list[str] | None = None
) -> dict[str, Any]:
    return {
        'instrument': instrument,
        'monitors': _make_cbm_monitors(instrument, monitor_names=monitor_names),
        'log_topics': None,
        'livedata_config_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_CONFIG
        ),
        'livedata_data_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_DATA
        ),
        'livedata_roi_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_ROI
        ),
        'livedata_status_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_STATUS
        ),
    }
