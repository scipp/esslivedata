#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tiny Panel dashboard standing in for NICOS on the derived-device topic.

A development stand-in for NICOS while it has no counterpart: a live web view
with one row per contract device showing the current value (+/- error), the
generation marker, and a flash when it jumps (accumulation reset). It exercises
the three things the projection promises NICOS -- the value is *receivable* on a
dedicated topic, *identifiable* by a stable device name (free of the random
``job_number``), and carries a ``start_time`` coordinate that lets a reset be
told apart from a genuine low reading.

A single background thread drains the ``{instrument}_livedata_projection`` Kafka
topic into the ``Device`` objects; each browser session just renders their
state on a periodic tick. Read-only -- a fresh random consumer group from the
latest offset, disturbing nothing.

The owning ``(workflow, source)`` job must be running for a device to emit, so
start the contracted workflow first (for bifrost, ``monitor_histogram`` via the
``monitor_data`` service). Run (dev Kafka on localhost:9092)::

    python scripts/nicos_device_monitor.py --instrument bifrost
    # then open http://localhost:5010
"""

from __future__ import annotations

import argparse
import datetime
import threading
import time

import confluent_kafka as kafka
import panel as pn
from streaming_data_types import dataarray_da00

from ess.livedata.config import instrument_registry
from ess.livedata.config.device_contract import DeviceContract
from ess.livedata.config.instruments import get_config
from ess.livedata.config.streams import stream_kind_to_topic
from ess.livedata.core.message import StreamKind
from ess.livedata.kafka.scipp_da00_compat import da00_to_scipp

GENERATION_COORD = 'start_time'  # marks the current accumulation generation
RESET_FLASH_S = 4.0  # how long a generation change stays highlighted


class Device:
    """NICOS-side view of one derived device, keyed by stable device name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.value: float | None = None
        self.error: float | None = None
        self.generation: int | None = None
        self.updates = 0
        self.resets = 0
        self.last_reset_at: float | None = None  # monotonic time of last reset

    def update(self, value: float, error: float | None, generation: int | None) -> None:
        if (
            generation is not None
            and self.generation is not None
            and generation != self.generation
        ):
            self.resets += 1
            self.last_reset_at = time.monotonic()
        self.value, self.error, self.generation = value, error, generation
        self.updates += 1

    @property
    def flashing(self) -> bool:
        return (
            self.last_reset_at is not None
            and time.monotonic() - self.last_reset_at < RESET_FLASH_S
        )


def _format_generation(generation_ns: int | None) -> str:
    if generation_ns is None:
        return "&mdash;"
    dt = datetime.datetime.fromtimestamp(generation_ns / 1e9, tz=datetime.timezone.utc)
    return dt.strftime("%H:%M:%S")


def _load_devices(instrument: str) -> dict[str, Device]:
    get_config(instrument)  # registers the instrument in the registry
    contract = DeviceContract.from_instrument(instrument_registry[instrument])
    devices = {entry.device_name: Device(entry.device_name) for entry in contract}
    if not devices:
        raise SystemExit(f"No device contract entries for instrument {instrument!r}.")
    return devices


def _extract(buf: bytes) -> tuple[str, float, float | None, int | None]:
    da00 = dataarray_da00.deserialise_da00(buf)
    da = da00_to_scipp(da00.data)
    error = None if da.variance is None else float(da.variance) ** 0.5
    generation = None
    if GENERATION_COORD in da.coords:
        generation = int(da.coords[GENERATION_COORD].value)
    return da00.source_name, float(da.value), error, generation


def _render(devices: dict[str, Device]) -> str:
    rows = []
    for d in devices.values():
        value = "&mdash;" if d.value is None else f"{d.value:.6g}"
        error = "" if d.error is None else f" &plusmn; {d.error:.3g}"
        bg = "#fff3cd" if d.flashing else ("#f8f9fa" if d.updates else "#fff")
        badge = (
            f" <b style='color:#b8860b'>RESET &times;{d.resets}</b>"
            if d.flashing
            else (
                f" <span style='color:#999'>resets: {d.resets}</span>"
                if d.resets
                else ""
            )
        )
        rows.append(
            f"<tr style='background:{bg}'>"
            f"<td style='padding:4px 10px'><code>{d.name}</code></td>"
            f"<td style='padding:4px 10px;text-align:right'>{value}{error}</td>"
            f"<td style='padding:4px 10px;text-align:center'>"
            f"{_format_generation(d.generation)}{badge}</td>"
            f"<td style='padding:4px 10px;text-align:right;color:#999'>{d.updates}</td>"
            f"</tr>"
        )
    header = (
        "<tr style='border-bottom:2px solid #ccc;text-align:left'>"
        "<th style='padding:4px 10px'>Device</th>"
        "<th style='padding:4px 10px;text-align:right'>Value</th>"
        "<th style='padding:4px 10px;text-align:center'>Generation</th>"
        "<th style='padding:4px 10px;text-align:right'>Updates</th></tr>"
    )
    return (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:13px'>"
        f"{header}{''.join(rows)}</table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instrument', default='bifrost')
    parser.add_argument('--bootstrap-servers', default='localhost:9092')
    parser.add_argument('--port', type=int, default=5010)
    args = parser.parse_args()

    devices = _load_devices(args.instrument)
    topic = stream_kind_to_topic(args.instrument, StreamKind.LIVEDATA_PROJECTION)

    consumer = kafka.Consumer(
        {
            'bootstrap.servers': args.bootstrap_servers,
            'group.id': f'nicos-device-panel-probe-{time.time()}',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False,
        }
    )
    consumer.subscribe([topic])

    def drain() -> None:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            name, value, error, generation = _extract(msg.value())
            if (device := devices.get(name)) is not None:
                device.update(value, error, generation)

    threading.Thread(target=drain, daemon=True).start()

    def app() -> pn.viewable.Viewable:
        table = pn.pane.HTML(_render(devices), sizing_mode='stretch_width')
        pn.state.add_periodic_callback(
            lambda: setattr(table, 'object', _render(devices)), period=500
        )
        return pn.Column(
            f"# NICOS derived devices &mdash; `{args.instrument}`",
            f"Topic `{topic}` &middot; {len(devices)} devices &middot; "
            "values reset when **Generation** changes.",
            table,
        )

    pn.serve(app, port=args.port, show=False, title='NICOS derived devices')


if __name__ == '__main__':
    main()
