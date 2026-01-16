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
        monitor_names = [f'monitor{i}' for i in range(1, monitor_count + 1)]
    return {
        InputStreamKey(topic=topic, source_name=f'cbm{monitor}'): name
        for monitor, name in enumerate(monitor_names, start=1)
    }


def _make_dev_detectors(*, instrument: str, detectors: list[str]) -> StreamLUT:
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.DETECTOR_EVENTS)
    return {InputStreamKey(topic=topic, source_name=name): name for name in detectors}


def _make_dev_area_detectors(
    *, instrument: str, area_detectors: list[str]
) -> StreamLUT:
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.AREA_DETECTOR)
    return {
        InputStreamKey(topic=topic, source_name=name): name for name in area_detectors
    }


def _make_dev_logs(*, instrument: str, log_names: list[str]) -> StreamLUT:
    """Create log stream mapping for dev mode where source_name equals internal name."""
    topic = f'{instrument}_motion'
    return {InputStreamKey(topic=topic, source_name=name): name for name in log_names}


def _make_dev_beam_monitors(
    instrument: str, monitor_names: list[str] | None = None
) -> StreamLUT:
    # Might also be MONITOR_COUNTS, but topic is supposedly the same.
    topic = stream_kind_to_topic(instrument=instrument, kind=StreamKind.MONITOR_EVENTS)
    if monitor_names is None:
        monitor_names = [f'monitor{i}' for i in range(1, 11)]
    return {
        InputStreamKey(topic=topic, source_name=f'monitor{i}'): name
        for i, name in enumerate(monitor_names, start=1)
    }


def _make_livedata_topics(instrument: str) -> dict[str, str]:
    """Create common livedata topic configuration for an instrument."""
    return {
        'livedata_commands_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_COMMANDS
        ),
        'livedata_data_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_DATA
        ),
        'livedata_responses_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_RESPONSES
        ),
        'livedata_roi_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_ROI
        ),
        'livedata_status_topic': stream_kind_to_topic(
            instrument=instrument, kind=StreamKind.LIVEDATA_STATUS
        ),
    }


def make_dev_stream_mapping(
    instrument: str,
    *,
    detector_names: list[str],
    area_detector_names: list[str] | None = None,
    monitor_names: list[str] | None = None,
    log_names: list[str] | None = None,
) -> StreamMapping:
    area_detectors = (
        _make_dev_area_detectors(
            instrument=instrument, area_detectors=area_detector_names
        )
        if area_detector_names
        else {}
    )
    logs = (
        _make_dev_logs(instrument=instrument, log_names=log_names)
        if log_names
        else None
    )
    return StreamMapping(
        instrument=instrument,
        detectors=_make_dev_detectors(instrument=instrument, detectors=detector_names),
        monitors=_make_dev_beam_monitors(instrument, monitor_names=monitor_names),
        area_detectors=area_detectors,
        logs=logs,
        **_make_livedata_topics(instrument),
    )


def make_common_stream_mapping_inputs(
    instrument: str, *, monitor_names: list[str] | None = None
) -> dict[str, Any]:
    return {
        'instrument': instrument,
        'monitors': _make_cbm_monitors(instrument, monitor_names=monitor_names),
        **_make_livedata_topics(instrument),
    }
