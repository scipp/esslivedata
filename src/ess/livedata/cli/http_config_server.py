"""HTTP server for publishing configuration messages.

This tool runs an HTTP server that serves configuration messages to services.
Services poll this server to receive config updates. Config messages can be
loaded from a file or published interactively.

This is the "reverse" of http_monitor.py - instead of consuming messages,
it serves messages that others can consume.
"""
# ruff: noqa: T201, S104  # print is appropriate for CLI, 0.0.0.0 is intentional

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

from ..config.models import ConfigKey
from ..core.message import CONFIG_STREAM_ID, Message
from ..http_transport.serialization import GenericJSONMessageSerializer
from ..http_transport.service import HTTPMultiEndpointSink

logger = logging.getLogger(__name__)


def create_config_message(config_key_str: str, config_value: dict[str, Any]) -> Message:
    """
    Create a config message from key string and value dict.

    Parameters
    ----------
    config_key_str:
        Config key in format 'source_name/service_name/key' (use * for wildcards)
    config_value:
        Configuration value as dictionary

    Returns
    -------
    :
        Message containing config data as dict (JSON-serializable)
    """
    config_key = ConfigKey.from_string(config_key_str)

    # For JSON serialization over HTTP, use a simple dict format
    # The consumer will reconstruct RawConfigItem from this
    config_dict = {
        'key': str(config_key),  # String representation
        'value': config_value,  # Already a dict
    }

    return Message(stream=CONFIG_STREAM_ID, value=config_dict)


def load_config_from_file(config_file: Path) -> tuple[str, dict[str, Any]]:
    """
    Load config from JSON file.

    Expected format:
    {
        "key": "source_name/service_name/config_key",
        "value": { ... config value ... }
    }

    Parameters
    ----------
    config_file:
        Path to JSON file

    Returns
    -------
    :
        Tuple of (config_key, config_value)
    """
    with open(config_file) as f:
        data = json.load(f)

    if "key" not in data or "value" not in data:
        raise ValueError("Config file must have 'key' and 'value' fields")

    return data["key"], data["value"]


def run_server(
    *,
    host: str,
    port: int,
    config_file: Path | None = None,
    config_key: str | None = None,
    config_value: dict[str, Any] | None = None,
) -> NoReturn:
    """
    Run HTTP config server.

    Parameters
    ----------
    host:
        Host to bind to
    port:
        Port to bind to
    config_file:
        Optional path to config file to load on startup
    config_key:
        Optional config key string (if not using file)
    config_value:
        Optional config value dict (if not using file)
    """
    print(f"Starting HTTP config server on http://{host}:{port}")
    print(f"Services can poll http://{host}:{port}/config for config updates\n")

    # Create HTTP multi-endpoint sink
    serializer = GenericJSONMessageSerializer()
    sink = HTTPMultiEndpointSink(
        data_serializer=serializer,
        status_serializer=serializer,
        config_serializer=serializer,
        host=host,
        port=port,
        max_queue_size=100,
    )

    try:
        sink.start()
        print("✓ Server started successfully\n")

        # Load initial config if provided
        if config_file is not None:
            print(f"Loading config from {config_file}...")
            key, value = load_config_from_file(config_file)
            msg = create_config_message(key, value)
            sink.publish_messages([msg])
            print(f"✓ Published config: {key}\n")
        elif config_key is not None and config_value is not None:
            print("Publishing config from command line...")
            msg = create_config_message(config_key, config_value)
            sink.publish_messages([msg])
            print(f"✓ Published config: {config_key}\n")

        print("Server is running. Press Ctrl+C to stop.")
        print("(Config messages are now available at the /config endpoint)\n")

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping server...")
    finally:
        sink.stop()
        print("Server stopped.")
        sys.exit(0)


def main() -> NoReturn:
    """Main entry point for the HTTP config server."""
    parser = argparse.ArgumentParser(
        description="Run HTTP server to publish config messages",
        epilog="""
Examples:
  # Start server and load config from file
  python -m ess.livedata.cli.http_config_server --port 9000 \\
    --config-file workflow_config.json

  # Start server with inline config
  python -m ess.livedata.cli.http_config_server --port 9000 \\
    --key "monitor1/*/workflow_config" \\
    --value '{"identifier": {...}, "params": {}}'

  # Just start server (publish config later programmatically)
  python -m ess.livedata.cli.http_config_server --port 9000

Config file format (JSON):
{
  "key": "source_name/service_name/config_key",
  "value": {
    "identifier": {
      "instrument": "dummy",
      "namespace": "monitor_data",
      "name": "toa_histogram",
      "version": 1
    },
    "params": {}
  }
}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9000,
        help="Port to bind to (default: 9000)",
    )
    parser.add_argument(
        "--config-file",
        "-f",
        type=Path,
        help="Path to JSON file containing config to publish on startup",
    )
    parser.add_argument(
        "--key",
        "-k",
        help="Config key (use with --value, format: source_name/service_name/key)",
    )
    parser.add_argument(
        "--value",
        "-v",
        help="Config value as JSON string (use with --key)",
    )
    parser.add_argument(
        "--verbose",
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

    # Validate config input
    if args.config_file and (args.key or args.value):
        print("Error: Cannot specify both --config-file and --key/--value")
        sys.exit(1)

    if (args.key and not args.value) or (args.value and not args.key):
        print("Error: Must specify both --key and --value together")
        sys.exit(1)

    # Parse inline value if provided
    config_value = None
    if args.value:
        try:
            config_value = json.loads(args.value)
        except json.JSONDecodeError as e:
            print(f"Error parsing --value JSON: {e}")
            sys.exit(1)

    run_server(
        host=args.host,
        port=args.port,
        config_file=args.config_file,
        config_key=args.key,
        config_value=config_value,
    )


if __name__ == "__main__":
    main()
