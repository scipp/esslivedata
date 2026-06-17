# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Chopper stream conventions, an instrument-level data-model concern.

An instrument that declares ``choppers`` owns the streams those choppers
produce: a clean ``rotation_speed_setpoint`` and a noisy ``delay`` readback per
chopper (real upstream PVs), plus the synthetic ``delay_setpoint`` the
``ChopperSynthesizer`` derives from the readback. The naming and declaration of
those streams live here, not in any workflow; the wavelength-LUT workflow is
the (currently sole) consumer, not the owner.

Which quantities are streamed and how each is treated is provisional — see
:class:`~ess.livedata.kafka.chopper_synthesizer.ChopperSynthesizer` for the
seam and the sites a change touches.
"""

from __future__ import annotations

from collections.abc import Sequence

from .stream import F144Stream, Stream


def speed_setpoint_stream(chopper: str) -> str:
    """Internal stream name of a chopper's clean rotation-speed setpoint f144.

    A real upstream PV (subscribed from Kafka). Consumed by the LUT workflow
    as context and cached by :class:`ChopperSynthesizer` to detect changes.
    """
    return f'{chopper}/rotation_speed_setpoint'


def delay_readback_stream(chopper: str) -> str:
    """Internal stream name of a chopper's noisy delay readback f144.

    A real upstream PV (subscribed from Kafka). Plateau-detected by
    :class:`ChopperSynthesizer`; not consumed by the LUT workflow directly.
    """
    return f'{chopper}/delay'


def delay_setpoint_stream(chopper: str) -> str:
    """Internal stream name of a chopper's synthesized delay setpoint.

    Emitted in-process by :class:`ChopperSynthesizer` once the delay readback
    plateaus. Not a Kafka topic. Consumed by the LUT workflow as context — the
    locked plateau value, not the noisy readback, is what the cascade fires on.
    """
    return f'{chopper}/delay_setpoint'


def declare_chopper_setpoint_streams(
    streams: dict[str, Stream], choppers: Sequence[str]
) -> None:
    """Declare the synthetic ``delay_setpoint`` streams the synthesizer emits.

    The ``ChopperSynthesizer`` (timeseries service) plateau-detects each
    chopper's noisy ``delay`` readback and injects a synthetic
    ``<chopper>/delay_setpoint`` f144. It is not a Kafka topic (no topic/source,
    so it stays out of the StreamLUT) but must be a declared stream so the
    preprocessor accepts it with the readback's unit — the LUT workflow consumes
    it as context and ``DiskChopper.from_nexus`` needs the unit.

    The readback unit must be ``ns``: the synthesizer plateau-detects on raw
    f144 values and ``Instrument.chopper_delay_atol_ns`` is a ns threshold,
    so a differently-unitted readback would silently mis-scale the detector.

    Invoked automatically from :meth:`Instrument.__post_init__` for any
    instrument that declares ``choppers``; mutates ``streams`` in place before
    the f144 stream set is snapshotted for timeseries-spec registration.
    """
    for chopper in choppers:
        readback = streams[delay_readback_stream(chopper)]
        if readback.units != 'ns':
            raise ValueError(
                f"Chopper {chopper!r} delay readback declares units "
                f"{readback.units!r}, expected 'ns': ChopperSynthesizer plateau "
                f"detection and chopper_delay_atol_ns assume nanosecond samples."
            )
        streams[delay_setpoint_stream(chopper)] = F144Stream(units=readback.units)
