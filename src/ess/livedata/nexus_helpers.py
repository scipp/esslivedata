"""Utilities for extracting metadata from NeXus files.

This module provides functionality to extract Kafka streaming metadata from NeXus/HDF5
files, including topic, source, and writer_module attributes that indicate how data
was streamed during acquisition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass
class StreamInfo:
    """Information about a streaming data node in a NeXus file.

    Attributes
    ----------
    group_path:
        Full HDF5 path to the group (e.g., 'entry/instrument/detector/data').
    topic:
        Kafka topic name the data was streamed from.
    source:
        Source identifier for the data stream (e.g., EPICS PV name, detector ID).
    nx_class:
        NeXus class of the group (e.g., 'NXevent_data', 'NXlog').
    parent_nx_class:
        NeXus class of the parent group (e.g., 'NXdetector', 'NXdisk_chopper').
    writer_module:
        FileWriter module used to write this data (e.g., 'ev44', 'f144', 'tdct').
    units:
        Units string from the 'units' attribute, if present.
    """

    group_path: str
    topic: str
    source: str
    nx_class: str
    parent_nx_class: str
    writer_module: str
    units: str = ''


def _decode_attr(value) -> str:
    """Decode HDF5 attribute value to string."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def extract_stream_info(file_path: str | Path | h5py.File) -> list[StreamInfo]:
    """Extract streaming information from a NeXus file.

    Searches for all groups that have both 'topic' and 'source' attributes,
    indicating they contain streamed data. Collects metadata about these groups
    including their NeXus classes and writer modules.

    Parameters
    ----------
    file_path:
        Path to the NeXus/HDF5 file, or an open h5py.File object.

    Returns
    -------
    :
        List of StreamInfo objects, one for each streaming data group found.
        Returns empty list if no streaming groups are found.
    """
    stream_infos = []

    def _collect_stream_nodes(name: str, node) -> None:
        """Visit function to collect nodes with streaming attributes."""
        # Skip datasets early - only groups have streaming info
        if isinstance(node, h5py.Dataset):
            return

        attrs = node.attrs
        if 'source' not in attrs or 'topic' not in attrs:
            return

        # Extract node attributes
        source = _decode_attr(attrs['source'])
        topic = _decode_attr(attrs['topic'])
        nx_class = _decode_attr(attrs.get('NX_class', 'N/A'))
        writer_module = _decode_attr(attrs.get('writer_module', 'N/A'))

        # Get parent NX_class directly from parent node
        parent_nx_class = 'N/A'
        if node.parent is not None and 'NX_class' in node.parent.attrs:
            parent_nx_class = _decode_attr(node.parent.attrs['NX_class'])

        # Try to get units from 'value' dataset within the group (common for f144)
        units = ''
        if 'value' in node and isinstance(node['value'], h5py.Dataset):
            if 'units' in node['value'].attrs:
                units = _decode_attr(node['value'].attrs['units'])

        stream_infos.append(
            StreamInfo(
                group_path=name,
                topic=topic,
                source=source,
                nx_class=nx_class,
                parent_nx_class=parent_nx_class,
                writer_module=writer_module,
                units=units,
            )
        )

    # Single pass: collect all streaming nodes with parent info
    if isinstance(file_path, h5py.File):
        file_path.visititems(_collect_stream_nodes)
    else:
        with h5py.File(file_path, 'r') as f:
            f.visititems(_collect_stream_nodes)

    return stream_infos


def suggest_internal_name(info: StreamInfo) -> str:
    """Suggest an internal name based on the group path.

    Uses the parent group name (last path component before 'value', 'idle_flag', etc.)
    as the basis for the internal name.
    """
    parts = info.group_path.split('/')
    # For paths like '.../rotation_stage/value', use 'rotation_stage'
    for i, part in enumerate(parts):
        if part in ('value', 'idle_flag', 'target_value'):
            if i > 0:
                return parts[i - 1]
    # Fallback: use last non-value component
    return parts[-2] if len(parts) >= 2 else parts[-1]


def filter_f144_streams(
    infos: list[StreamInfo],
    *,
    topic_filter: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[StreamInfo]:
    """Filter stream infos to only f144 writer module entries.

    Parameters
    ----------
    infos:
        List of StreamInfo objects to filter.
    topic_filter:
        If provided, only include streams from this topic.
    exclude_patterns:
        List of regex patterns to exclude from results (matched against group_path).
    """
    exclude_patterns = exclude_patterns or []
    exclude_regexes = [re.compile(p) for p in exclude_patterns]

    result = []
    for info in infos:
        if info.writer_module != 'f144':
            continue
        if topic_filter and info.topic != topic_filter:
            continue
        if any(regex.search(info.group_path) for regex in exclude_regexes):
            continue
        result.append(info)
    return result


def generate_f144_log_streams_code(
    infos: list[StreamInfo],
    *,
    topic: str,
    variable_name: str = 'f144_log_streams',
) -> str:
    """Generate Python code for f144_log_streams dictionary.

    The generated dictionary maps internal names to source, units, and topic info,
    which can be used to derive both f144_attribute_registry and StreamLUT.

    Parameters
    ----------
    infos:
        List of StreamInfo objects (should be pre-filtered to f144 only).
    topic:
        The Kafka topic these streams come from.
    variable_name:
        Name for the generated variable.
    """
    # Group by suggested internal name, preferring 'value' entries
    by_name: dict[str, StreamInfo] = {}
    for info in infos:
        if info.topic != topic:
            continue
        name = suggest_internal_name(info)
        # Prefer 'value' entries over 'idle_flag' or 'target_value'
        if name not in by_name or info.group_path.endswith('/value'):
            by_name[name] = info

    lines = [
        "# Generated from NeXus file - review and adjust names as needed",
        f"# Topic: {topic}",
        f"{variable_name}: dict[str, dict[str, str]] = {{",
    ]

    for name in sorted(by_name.keys()):
        info = by_name[name]
        units = info.units or 'dimensionless'
        entry = (
            f"    '{name}': {{'source': '{info.source}', 'units': '{units}', "
            f"'topic': '{topic}'}}"
        )
        lines.append(f"{entry},")

    lines.append("}")
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Extract streaming metadata from NeXus files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all streaming groups
  python -m ess.livedata.nexus_helpers file.hdf

  # List only f144 streams
  python -m ess.livedata.nexus_helpers file.hdf --f144

  # Generate code for motion topic
  python -m ess.livedata.nexus_helpers file.hdf --generate --topic bifrost_motion

  # Filter by topic and exclude choppers
  python -m ess.livedata.nexus_helpers file.hdf --f144 --topic bifrost_motion \\
      --exclude "Chopper"
""",
    )
    parser.add_argument('nexus_file', help='Path to NeXus/HDF5 file')
    parser.add_argument(
        '--f144', action='store_true', help='Only show f144 (log data) streams'
    )
    parser.add_argument('--topic', help='Filter by Kafka topic')
    parser.add_argument(
        '--exclude',
        action='append',
        default=[],
        help='Regex pattern to exclude (can be used multiple times)',
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate Python code for f144_log_streams dict',
    )
    parser.add_argument(
        '--var-name',
        default='f144_log_streams',
        help='Variable name for generated code (default: f144_log_streams)',
    )

    args = parser.parse_args()

    infos = extract_stream_info(args.nexus_file)

    if args.f144 or args.generate:
        infos = filter_f144_streams(
            infos, topic_filter=args.topic, exclude_patterns=args.exclude
        )

    if args.generate:
        if not args.topic:
            sys.stderr.write("Error: --generate requires --topic\n")
            sys.exit(1)
        code = generate_f144_log_streams_code(
            infos, topic=args.topic, variable_name=args.var_name
        )
        sys.stdout.write(code + '\n')
    else:
        sys.stdout.write(f"Found {len(infos)} streaming data groups\n\n")
        for info in infos:
            sys.stdout.write(f"{info}\n")
