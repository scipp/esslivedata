# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Merge per-device RBV/VAL/DMOV substreams into a single Device stream.

The synthesizer is a :class:`MessageSource` decorator: it wraps an
already-adapted source (i.e. domain-level :class:`Message` objects) and
emits synthetic :class:`LogData` messages on a new ``StreamKind.DEVICE``
stream per device, carrying ``value`` plus optional ``target`` / ``settled``
fields. Substream messages belonging to a configured device are suppressed
from forwarding; other messages pass through unchanged.

State per device: last-seen ``(time, value)`` for each of its configured
substreams. Emission policy: union-anchored — on every input event for a
configured substream, emit a sample. Bootstrap: suppress emit until every
configured substream of that device has been observed at least once.
Emit timestamp policy: ``max(rbv_time, val_time, dmov_time)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import structlog

from ..config.stream import Device
from ..core.message import Message, MessageSource, StreamId, StreamKind
from ..core.timestamp import Timestamp
from ..handlers.accumulators import LogData

logger = structlog.get_logger(__name__)

_Role = Literal['value', 'target', 'settled']
_V = TypeVar('_V', float, bool)


@dataclass
class _Substream(Generic[_V]):
    """Last-seen value and timestamp for one device substream."""

    value: _V
    time: Timestamp


@dataclass(slots=True)
class _DeviceState:
    """Mutable per-device state held by the synthesizer."""

    device_name: str
    has_target: bool
    has_settled: bool
    value: _Substream[float] | None = None
    target: _Substream[float] | None = None
    settled: _Substream[bool] | None = None

    def push(self, role: _Role, log: LogData) -> Message[LogData] | None:
        """Record a substream event and emit a sample if all substreams seen."""
        time = Timestamp.from_ns(int(log.time))
        if role == 'value':
            self.value = _Substream(value=float(log.value), time=time)
        elif role == 'target':
            self.target = _Substream(value=float(log.value), time=time)
        else:  # settled / DMOV
            self.settled = _Substream(value=bool(log.value), time=time)
        if self.value is None:
            return None
        if self.has_target and self.target is None:
            return None
        if self.has_settled and self.settled is None:
            return None
        sample_time = max(
            s.time for s in (self.value, self.target, self.settled) if s is not None
        )
        return Message(
            timestamp=sample_time,
            stream=StreamId(kind=StreamKind.DEVICE, name=self.device_name),
            value=LogData(
                time=sample_time.to_ns(),
                value=self.value.value,
                target=self.target.value if self.target is not None else None,
                settled=self.settled.value if self.settled is not None else None,
            ),
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
        :class:`LogData` (with ``target`` / ``settled`` populated as
        configured) is emitted in their place once bootstrapped.
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
            if (sample := state.push(role, msg.value)) is not None:
                out.append(sample)
        return out
