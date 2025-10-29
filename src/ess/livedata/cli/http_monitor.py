"""Command-line tool to monitor HTTP message endpoints.

This tool polls message endpoints and displays received messages.
Useful for testing and debugging HTTP transport services.
"""
# ruff: noqa: T201  # print is appropriate for CLI output

import argparse
import logging
import sys
import time
from typing import Any

import scipp as sc

from ..http_transport.serialization import (
    DA00MessageSerializer,
    GenericJSONMessageSerializer,
    StatusMessageSerializer,
)
from ..http_transport.source import HTTPMessageSource, MultiHTTPSource

logger = logging.getLogger(__name__)

# Standard endpoints and their serializers
ENDPOINT_SERIALIZERS = {
    '/data': DA00MessageSerializer(),
    '/status': StatusMessageSerializer(),
    '/config': GenericJSONMessageSerializer(),
}


def format_message(msg: Any) -> str:
    """Format a message for display."""
    stream_info = f"{msg.stream.kind.value}:{msg.stream.name}"
    timestamp = msg.timestamp

    # Format the data value
    if isinstance(msg.value, sc.DataArray):
        data = msg.value
        if data.ndim == 0:
            value_str = f"scalar={data.value}"
        else:
            value_str = f"shape={data.shape}, dims={data.dims}"
    else:
        # Handle JobStatus and other types
        value_str = str(msg.value)[:100]  # Truncate long values

    return f"[{timestamp}] {stream_info}: {value_str}"


def main() -> int:
    """Main entry point for the HTTP monitor tool."""
    parser = argparse.ArgumentParser(description="Monitor messages from HTTP endpoints")
    parser.add_argument(
        "url",
        help="Base URL of the service to monitor (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--endpoints",
        nargs='+',
        default=None,
        help="Endpoint paths to monitor (default: all standard endpoints)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP request timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine which endpoints to monitor
    if args.endpoints:
        endpoints = args.endpoints
        # For custom endpoints, use generic JSON serializer
        sources = [
            HTTPMessageSource(
                base_url=args.url,
                serializer=GenericJSONMessageSerializer(),
                endpoint=endpoint,
                timeout=args.timeout,
                logger=logger,
            )
            for endpoint in endpoints
        ]
    else:
        # Monitor all standard endpoints
        endpoints = list(ENDPOINT_SERIALIZERS.keys())
        sources = [
            HTTPMessageSource(
                base_url=args.url,
                serializer=ENDPOINT_SERIALIZERS[endpoint],
                endpoint=endpoint,
                timeout=args.timeout,
                logger=logger,
            )
            for endpoint in endpoints
        ]

    # Combine multiple sources
    if len(sources) == 1:
        source = sources[0]
    else:
        source = MultiHTTPSource(sources)

    endpoints_str = ', '.join(endpoints)
    print(f"Monitoring {args.url} endpoints: {endpoints_str}")
    print(f"Polling every {args.interval}s")
    print("Press Ctrl+C to stop\n")

    message_count = 0
    try:
        while True:
            messages = source.get_messages()

            if messages:
                for msg in messages:
                    print(format_message(msg))
                    message_count += 1
                print(f"  (Total messages received: {message_count})\n")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Total messages received: {message_count}")
        source.close()
        return 0
    except Exception as e:
        logger.exception("Error: %s", e)
        source.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
