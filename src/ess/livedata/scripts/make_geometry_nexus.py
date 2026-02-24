# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Script to create a copy of a NeXus file with only geometry information."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def _copy_attributes(src: h5py.Group, dst: h5py.Group) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_detector_fields(
    src_group: h5py.Group, dst_group: h5py.Group, use_pixel_shape: bool
) -> None:
    compression_fields: list[str] = [
        'detector_number',
        'x_pixel_offset',
        'y_pixel_offset',
        'z_pixel_offset',
    ]
    for field in compression_fields:
        if field in src_group:
            # Using compression makes a 10x size difference (tested on DREAM)
            data: np.ndarray = src_group[field][:]
            dst_group.create_dataset(
                field,
                data=data,
                compression='gzip',
                compression_opts=1,
                shuffle=True,
            )
            _copy_attributes(src_group[field], dst_group[field])
    src_group.copy('depends_on', dst_group)
    if use_pixel_shape and 'pixel_shape' in src_group:
        src_group.copy('pixel_shape', dst_group)


def _copy_monitor_fields(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    src_group.copy('depends_on', dst_group)


def _read_depends_on(value: bytes | str) -> str | None:
    if isinstance(value, bytes):
        value = value.decode()
    return None if value == '.' else value.lstrip('/')


def _collect_depends_on_targets(f: h5py.File) -> set[str]:
    """Collect all absolute paths referenced by ``depends_on`` in the file."""
    targets: set[str] = set()

    def _visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and name.endswith('depends_on'):
            if (path := _read_depends_on(obj[()])) is not None:
                targets.add(path)
        if 'depends_on' in obj.attrs:
            val = obj.attrs['depends_on']
            if (path := _read_depends_on(val)) is not None:
                if not path.startswith('/') and '/' in name:
                    parent = name.rsplit('/', 1)[0]
                    path = f'{parent}/{path}'
                targets.add(path.lstrip('/'))

    f.visititems(_visitor)
    return targets


def _ensure_parent_groups(fin: h5py.File, fout: h5py.File, path: str) -> None:
    """Create parent groups in *fout*, copying attributes from *fin*."""
    parts = path.split('/')
    current = ''
    for part in parts[:-1]:
        if not part:
            continue
        current = f'{current}/{part}'
        if current not in fout:
            fout.create_group(current)
            _copy_attributes(fin[current], fout[current])


def _resolve_depends_on_chains(fin: h5py.File, fout: h5py.File) -> None:
    """Copy any ``depends_on`` targets that are missing from the output file.

    After copying geometry components, ``depends_on`` chains may reference
    nodes that were not copied — for example an NXlog group inside an
    NXpositioner that acts as a transformation node.  This function follows
    all chains and copies missing nodes (groups and datasets) as-is.
    """
    resolved: set[str] = set()
    while True:
        unresolved = _collect_depends_on_targets(fout) - resolved
        unresolved = {t for t in unresolved if t not in fout}
        if not unresolved:
            break
        for path in unresolved:
            resolved.add(path)
            if path not in fin:
                continue
            _ensure_parent_groups(fin, fout, path)
            parent_path = path.rsplit('/', 1)[0] if '/' in path else ''
            fin.copy(path, fout[parent_path or '/'])


def write_minimal_geometry(
    input_filename: Path, output_filename: Path, use_pixel_shape: bool = True
) -> None:
    """Create minimal geometry file with only detector positions and transformations."""
    with h5py.File(input_filename, 'r') as fin, h5py.File(output_filename, 'w') as fout:

        def ensure_parent_groups(name: str) -> None:
            parts: list[str] = name.split('/')
            current_path: str = ''
            for part in parts[:-1]:  # Skip the last part (current group name)
                if not part:
                    continue
                current_path = f"{current_path}/{part}"
                if current_path not in fout:
                    src_group: h5py.Group = fin[current_path]
                    dst_group: h5py.Group = fout.create_group(current_path)
                    _copy_attributes(src_group, dst_group)

        def visit_and_copy(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                if 'NX_class' in obj.attrs:
                    nx_class: str | bytes = obj.attrs['NX_class']
                    if isinstance(nx_class, bytes):
                        nx_class = nx_class.decode()

                    if nx_class == 'NXdetector':
                        ensure_parent_groups(name)
                        dst_group: h5py.Group = fout.create_group(name)
                        _copy_attributes(obj, dst_group)
                        _copy_detector_fields(
                            obj, dst_group, use_pixel_shape=use_pixel_shape
                        )
                    elif nx_class == 'NXmonitor':
                        ensure_parent_groups(name)
                        dst_group: h5py.Group = fout.create_group(name)
                        _copy_attributes(obj, dst_group)
                        _copy_monitor_fields(obj, dst_group)
                    elif nx_class in ('NXsource', 'NXsample'):
                        ensure_parent_groups(name)
                        dst_group: h5py.Group = fout.create_group(name)
                        _copy_attributes(obj, dst_group)
                        obj.copy('depends_on', dst_group)
                    elif nx_class == 'NXtransformations':
                        ensure_parent_groups(name)
                        dst_group: h5py.Group = fout.create_group(name)
                        _copy_attributes(obj, dst_group)
                        # Copy all contents of the transformations group
                        for key in obj:
                            obj.copy(key, dst_group)

        # Copy root attributes
        _copy_attributes(fin, fout)
        fin.visititems(visit_and_copy)
        _resolve_depends_on_chains(fin, fout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Create a copy of a NeXus file, stripping non-geometry data such '
            'as event data.'
        )
    )
    parser.add_argument('input', type=Path, help='Input NeXus file')
    parser.add_argument('output', type=Path, help='Output NeXus file')
    parser.add_argument(
        '--no-pixel-shape',
        action='store_false',
        dest='use_shape',
        help='Do not keep pixel shape (which can be large) if present in input file',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it exists',
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f'Input file {args.input} does not exist', file=sys.stderr)  # noqa: T201
        return 1

    if args.output.exists() and not args.force:
        print(f'Output file {args.output} already exists', file=sys.stderr)  # noqa: T201
        return 1

    write_minimal_geometry(args.input, args.output, use_pixel_shape=args.use_shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
