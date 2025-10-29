"""Command-line tool to monitor HTTP message endpoints.

This tool polls a message endpoint and displays received messages.
Useful for testing and debugging HTTP transport services.
"""
# ruff: noqa: T201  # print is appropriate for CLI output

import argparse
import logging
import sys
import time
from typing import Any

import scipp as sc

from ..http_transport.serialization import DA00MessageSerializer
from ..http_transport.source import HTTPMessageSource

logger = logging.getLogger(__name__)


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
        value_str = str(msg.value)

    return f"[{timestamp}] {stream_info}: {value_str}"


def main() -> int:
    """Main entry point for the HTTP monitor tool."""
    parser = argparse.ArgumentParser(
        description="Monitor messages from an HTTP endpoint"
    )
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
        "--endpoint",
        default="/messages",
        help="Endpoint path (default: /messages)",
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

    # Create HTTP source
    serializer = DA00MessageSerializer()
    source = HTTPMessageSource(
        base_url=args.url,
        serializer=serializer,
        endpoint=args.endpoint,
        timeout=args.timeout,
        logger=logger,
    )

    print(f"Monitoring {args.url}{args.endpoint} (polling every {args.interval}s)")
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
        return 0
    except Exception as e:
        logger.exception("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
