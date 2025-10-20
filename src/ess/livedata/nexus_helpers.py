"""Utilities for extracting metadata from NeXus files.

This module provides functionality to extract Kafka streaming metadata from NeXus/HDF5
files, including topic, source, and writer_module attributes that indicate how data
was streamed during acquisition.
"""

from __future__ import annotations

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
    """

    group_path: str
    topic: str
    source: str
    nx_class: str
    parent_nx_class: str
    writer_module: str


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

        stream_infos.append(
            StreamInfo(
                group_path=name,
                topic=topic,
                source=source,
                nx_class=nx_class,
                parent_nx_class=parent_nx_class,
                writer_module=writer_module,
            )
        )

    # Single pass: collect all streaming nodes with parent info
    if isinstance(file_path, h5py.File):
        file_path.visititems(_collect_stream_nodes)
    else:
        with h5py.File(file_path, 'r') as f:
            f.visititems(_collect_stream_nodes)

    return stream_infos


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: python -m ess.livedata.nexus_helpers <nexus_file.hdf>\n"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    infos = extract_stream_info(file_path)

    sys.stdout.write(f"Found {len(infos)} streaming data groups\n\n")
    for info in infos:
        sys.stdout.write(f"{info}\n")
