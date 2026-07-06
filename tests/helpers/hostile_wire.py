# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Corpus of hostile wire-level payloads for boundary-robustness testing.

The backend trusts values decoded from Kafka payloads (most critically
timestamps, which serve as the batcher's clock). This module provides
serialized payloads that a broken or misconfigured upstream producer could
emit, in two families:

- **Malformed**: payloads that fail to decode or violate the schema's
  structural assumptions (garbage bytes, truncated flatbuffers, absent event
  vectors). These exercise the per-message containment in
  ``AdaptingMessageSource``.
- **Well-formed but insane**: payloads that decode fine but carry data-derived
  values no sane producer would send, e.g. timestamps centuries in the future.
  These exercise (currently: document the absence of) validation at the
  adapter boundary — see #1038 and #1047.

Used by ``tests/kafka/adapter_robustness_test.py`` (adapter-level containment
and timestamp-bound invariants) and
``tests/services/hostile_input_liveness_test.py`` (service-level liveness).
Add new corruption modes here so both layers pick them up.
"""

from __future__ import annotations

import numpy as np
from streaming_data_types import (
    area_detector_ad00,
    dataarray_da00,
    eventdata_ev44,
    logdata_f144,
)
from streaming_data_types.fbschemas.eventdata_ev44 import Event44Message

# 2200-01-01 in nanoseconds since epoch: decodes fine, but is beyond any
# plausible wall clock or test-data timestamp. A single message carrying this
# as its data-derived timestamp is the reproduced trigger for the batcher
# wedge in #1038 (finding 1).
FAR_FUTURE_NS = 7_258_118_400 * 1_000_000_000

# 2026-01-01: a fixed, deterministic "realistic" base time for good messages.
REALISTIC_EPOCH_NS = 1_767_225_600 * 1_000_000_000


def ev44_events(
    source_name: str,
    *,
    reference_time_ns: int | None,
    n_events: int = 10,
    seed: int = 0,
) -> bytes:
    """Well-formed ev44 event data with an explicit data-derived timestamp.

    Parameters
    ----------
    source_name:
        Source name stored in the payload.
    reference_time_ns:
        Value for ``reference_time[-1]``, which the adapters use as the
        message timestamp. ``None`` leaves the vector empty so adapters fall
        back to the Kafka message timestamp.
    n_events:
        Number of events in the payload.
    seed:
        Seed for the event time-of-arrival values.

    Returns
    -------
    :
        Serialized ev44 payload.
    """
    rng = np.random.default_rng(seed)
    time_of_arrival = rng.uniform(0, 70_000_000, n_events).astype(np.int32)
    reference_time = [] if reference_time_ns is None else [reference_time_ns]
    return eventdata_ev44.serialise_ev44(
        source_name=source_name,
        message_id=0,
        reference_time=reference_time,
        reference_time_index=0,
        time_of_flight=time_of_arrival,
        pixel_id=np.zeros(n_events, dtype=np.int32),
    )


def ev44_without_event_vectors(source_name: str) -> bytes:
    """ev44 payload whose event vectors are absent (not just empty).

    ``serialise_ev44`` always writes the vectors, but the flatbuffer schema
    does not require them, so other producers can omit them. The
    flatbuffers-python accessors then return scalar ``0`` instead of an empty
    array, which trips code expecting ``.size``/``len()`` — see #1038
    (finding 2).
    """
    import flatbuffers

    builder = flatbuffers.Builder(64)
    builder.ForceDefaults(True)
    source = builder.CreateString(source_name)
    Event44Message.Event44MessageStart(builder)
    Event44Message.Event44MessageAddMessageId(builder, 0)
    Event44Message.Event44MessageAddSourceName(builder, source)
    data = Event44Message.Event44MessageEnd(builder)
    builder.Finish(data, file_identifier=eventdata_ev44.FILE_IDENTIFIER)
    return bytes(builder.Output())


def f144_log(source_name: str, *, timestamp_ns: int, value: float = 1.0) -> bytes:
    """Well-formed f144 log datum with an explicit data-derived timestamp."""
    return logdata_f144.serialise_f144(
        source_name=source_name, value=value, timestamp_unix_ns=timestamp_ns
    )


def da00_array(
    source_name: str,
    *,
    timestamp_ns: int,
    reference_time_ns: int | None = None,
    reference_time_dtype: type = np.int64,
) -> bytes:
    """Well-formed da00 data array, optionally with a reference_time variable.

    Parameters
    ----------
    source_name:
        Source name stored in the payload.
    timestamp_ns:
        Top-level ``timestamp_ns`` (the fallback timestamp).
    reference_time_ns:
        If given, a ``reference_time`` variable is included; adapters prefer
        its last value over ``timestamp_ns``.
    reference_time_dtype:
        Dtype of the reference_time data. Dtypes other than int64/uint64 are
        rejected by ``_extract_reference_time`` (fallback to ``timestamp_ns``).
    """
    variables = [
        dataarray_da00.Variable(
            name='signal',
            data=np.arange(4, dtype=np.float64),
            axes=['x'],
            unit='counts',
        )
    ]
    if reference_time_ns is not None:
        variables.append(
            dataarray_da00.Variable(
                name='reference_time',
                data=np.array([reference_time_ns], dtype=reference_time_dtype),
                axes=['time'],
                unit='ns',
            )
        )
    return dataarray_da00.serialise_da00(
        source_name=source_name, timestamp_ns=timestamp_ns, data=variables
    )


def ad00_frame(source_name: str, *, timestamp_ns: int) -> bytes:
    """Well-formed ad00 area-detector frame with an explicit timestamp."""
    return area_detector_ad00.serialise_ad00(
        source_name=source_name,
        unique_id=1,
        timestamp_ns=timestamp_ns,
        data=np.zeros((2, 2), dtype=np.uint16),
    )


def garbage_bytes() -> bytes:
    """Deterministic bytes that are not a flatbuffer of any schema."""
    return bytes(range(256)) * 4


def truncated(payload: bytes, keep: int = 12) -> bytes:
    """Truncate a valid payload so the schema id survives but the body breaks.

    Flatbuffer schema identifiers live at bytes 4:8, so ``keep >= 8`` keeps
    schema routing working while decoding fails.
    """
    return payload[:keep]


def malformed_corpus(source_name: str) -> dict[str, bytes]:
    """All malformed payloads, keyed by case id.

    Every entry must be *contained* by the adapter layer: dropped (possibly
    with a logged error) without the exception escaping and without affecting
    the handling of subsequent messages.
    """
    return {
        'garbage': garbage_bytes(),
        'empty': b'',
        'truncated_ev44': truncated(
            ev44_events(source_name, reference_time_ns=REALISTIC_EPOCH_NS)
        ),
        'ev44_without_event_vectors': ev44_without_event_vectors(source_name),
        'wrong_schema_f144': f144_log(source_name, timestamp_ns=REALISTIC_EPOCH_NS),
        'unmapped_source': ev44_events(
            'not_a_known_source', reference_time_ns=REALISTIC_EPOCH_NS
        ),
    }
