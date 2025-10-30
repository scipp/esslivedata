# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unified runner for fake data-producing services."""

import logging
from typing import NoReturn

from ..config import config_names
from ..config.config_loader import load_config
from ..core import IdentityProcessor, Service
from ..core.message import MessageSource, StreamKind
from ..kafka.message_adapter import AdaptingMessageSource, MessageAdapter
from ..transport.utils import extract_http_sinks


def run_fake_service(
    *,
    instrument: str,
    source: MessageSource,
    output_stream_kind: StreamKind,
    service_name: str,
    log_level: int = logging.INFO,
    adapter: MessageAdapter | None = None,
) -> NoReturn:
    """
    Run a fake data-producing service using YAML-based transport configuration.

    This function consolidates the common setup for all fake services, handling:
    - Transport configuration loading
    - Sink creation and configuration
    - HTTP sink lifecycle management
    - Service creation and startup

    Parameters
    ----------
    instrument:
        Instrument name.
    source:
        Message source that generates fake data.
    output_stream_kind:
        The kind of stream this service produces.
    service_name:
        Base name for the service (e.g., 'fake_detector_producer').
    log_level:
        Logging level.
    adapter:
        Optional message adapter to apply to source messages.
    """
    from ..config.transport_config import load_transport_config
    from ..transport.factory import create_sink_from_config

    # Wrap source with adapter if provided
    if adapter is not None:
        source = AdaptingMessageSource(source=source, adapter=adapter)

    # Load transport configuration
    transport_config = load_transport_config(instrument)

    # Validate output stream kind is configured
    configured_kinds = {s.kind for s in transport_config.streams}
    if output_stream_kind not in configured_kinds:
        raise ValueError(
            f"Stream kind {output_stream_kind.value} not found in transport config. "
            f"Available kinds: {[k.value for k in configured_kinds]}"
        )

    # Load Kafka config for strategies that need it
    kafka_downstream_config = load_config(namespace=config_names.kafka_downstream)

    # Create sink from config with explicit output stream kinds
    sink = create_sink_from_config(
        instrument=instrument,
        transport_config=transport_config,
        output_stream_kinds=[output_stream_kind],
        kafka_config=kafka_downstream_config,
    )

    # Extract HTTP sinks that need to be started
    http_sinks = extract_http_sinks(sink)

    # Create processor
    processor = IdentityProcessor(source=source, sink=sink)

    # Start HTTP sinks before service starts
    for http_sink in http_sinks:
        http_sink.start()

    # Create and start service
    service = Service(
        processor=processor,
        name=f'{instrument}_{service_name}',
        log_level=log_level,
    )
    try:
        service.start()
    finally:
        # Stop HTTP sinks on exit
        for http_sink in http_sinks:
            http_sink.stop()
