# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Merge per-device RBV/VAL/DMOV substreams into a single Device stream.

The synthesizer is a :class:`MessageSource` decorator: it wraps an
already-adapted source (i.e. domain-level :class:`Message` objects) and
emits synthetic :class:`DeviceSample` messages on a new
``StreamKind.DEVICE`` stream per device. Substream messages belonging to a
configured device are suppressed from forwarding; other messages pass
through unchanged.

State per device: last-seen ``(time, value)`` for each of its configured
substreams. Emission policy: union-anchored — on every input event for a
configured substream, emit a sample. Bootstrap: suppress emit until every
configured substream of that device has been observed at least once.
Emit timestamp policy: ``max(rbv_time, val_time, dmov_time)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import structlog

from ..config.stream import Device
from ..core.message import Message, MessageSource, StreamId, StreamKind
from ..core.timestamp import Timestamp
from ..handlers.accumulators import DeviceSample, LogData

logger = structlog.get_logger(__name__)

_Role = Literal['value', 'target', 'settled']


@dataclass(slots=True)
class _DeviceState:
    """Mutable per-device state held by the synthesizer."""

    device_name: str
    has_target: bool
    has_settled: bool
    value: float | None = None
    target: float | None = None
    settled: bool | None = None
    value_time: int | None = None
    target_time: int | None = None
    settled_time: int | None = None

    def bootstrapped(self) -> bool:
        if self.value_time is None:
            return False
        if self.has_target and self.target_time is None:
            return False
        if self.has_settled and self.settled_time is None:
            return False
        return True

    def max_time(self) -> int:
        return max(
            t
            for t in (self.value_time, self.target_time, self.settled_time)
            if t is not None
        )


class DeviceSynthesizer(MessageSource[Message]):
    """Decorator that synthesizes per-device merged streams.

    Parameters
    ----------
    wrapped:
        The already-adapted upstream message source.
    devices:
        Mapping ``device_name -> Device``. Substreams referenced by these
        devices are suppressed from forwarding; a synthesized
        :class:`DeviceSample` is emitted in their place once bootstrapped.
    """

    def __init__(
        self,
        wrapped: MessageSource[Message],
        *,
        devices: Mapping[str, Device],
    ) -> None:
        self._wrapped = wrapped
        # substream_name -> (device_state, role). One substream is owned by
        # exactly one device.
        self._by_substream: dict[str, tuple[_DeviceState, _Role]] = {}
        self._states: dict[str, _DeviceState] = {}
        for name, device in devices.items():
            state = _DeviceState(
                device_name=name,
                has_target=device.target is not None,
                has_settled=device.settled is not None,
            )
            self._states[name] = state
            self._register(state, device.value, 'value')
            if device.target is not None:
                self._register(state, device.target, 'target')
            if device.settled is not None:
                self._register(state, device.settled, 'settled')

    def _register(self, state: _DeviceState, substream: str, role: _Role) -> None:
        if substream in self._by_substream:
            other = self._by_substream[substream][0].device_name
            raise ValueError(
                f"substream {substream!r} configured for both devices "
                f"{other!r} and {state.device_name!r}"
            )
        self._by_substream[substream] = (state, role)

    def get_messages(self) -> Sequence[Message]:
        out: list[Message] = []
        for msg in self._wrapped.get_messages():
            owner = self._by_substream.get(msg.stream.name)
            if owner is None:
                out.append(msg)
                continue
            state, role = owner
            if not isinstance(msg.value, LogData):
                logger.warning(
                    "device_substream_unexpected_payload",
                    device=state.device_name,
                    role=role,
                    substream=msg.stream.name,
                    value_type=type(msg.value).__name__,
                )
                continue
            self._update_state(state, role, msg.value)
            if state.bootstrapped():
                out.append(self._make_sample_message(state))
        return out

    @staticmethod
    def _update_state(state: _DeviceState, role: _Role, log: LogData) -> None:
        if role == 'value':
            state.value = float(log.value)
            state.value_time = int(log.time)
        elif role == 'target':
            state.target = float(log.value)
            state.target_time = int(log.time)
        else:  # settled / DMOV
            state.settled = bool(log.value)
            state.settled_time = int(log.time)

    @staticmethod
    def _make_sample_message(state: _DeviceState) -> Message[DeviceSample]:
        sample_time_ns = state.max_time()
        sample_time = Timestamp.from_ns(sample_time_ns)
        sample = DeviceSample(
            time=sample_time,
            value=state.value,  # type: ignore[arg-type]
            target=state.target,
            settled=state.settled,
        )
        return Message(
            timestamp=sample_time,
            stream=StreamId(kind=StreamKind.DEVICE, name=state.device_name),
            value=sample,
        )
