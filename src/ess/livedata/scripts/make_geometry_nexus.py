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


def _nx_class(obj: h5py.Group | h5py.Dataset) -> str:
    nx_class = obj.attrs.get('NX_class', b'')
    if isinstance(nx_class, bytes):
        nx_class = nx_class.decode()
    return nx_class


def _copy_nxlog_placeholder(src: h5py.Group, dst: h5py.Group) -> None:
    """Copy an NXlog group, trimming length-N datasets to length 0.

    Production NeXus files may contain logs with thousands of samples; the
    geometry artifact wants only the schema (dtype, units, attributes) so
    downstream consumers see the field shape without historical data.
    Scalar datasets are kept verbatim; nested groups (including any further
    NXlogs) are dispatched through :func:`_copy_child`.
    """
    _copy_attributes(src, dst)
    for child_name, child in src.items():
        if isinstance(child, h5py.Dataset) and child.ndim >= 1:
            ds = dst.create_dataset(
                child_name,
                shape=(0, *child.shape[1:]),
                dtype=child.dtype,
            )
            _copy_attributes(child, ds)
        else:
            _copy_child(src, child_name, dst)


def _copy_child(src: h5py.Group, key: str, dst: h5py.Group) -> None:
    """Copy ``src[key]`` into ``dst``, trimming any NXlog descendants.

    Datasets are copied verbatim. NXlog groups become length-0 placeholders.
    Other groups are recreated and recursed into so that NXlogs anywhere
    below get trimmed. No-op if ``key`` already exists in ``dst``.
    """
    if key in dst:
        return
    child = src[key]
    if isinstance(child, h5py.Dataset):
        src.copy(key, dst)
        return
    if _nx_class(child) == 'NXlog':
        _copy_nxlog_placeholder(child, dst.create_group(key))
        return
    sub_dst = dst.create_group(key)
    _copy_attributes(child, sub_dst)
    for sub_key in child:
        _copy_child(child, sub_key, sub_dst)


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


def _get_or_create_dst(
    fin: h5py.File, fout: h5py.File, name: str, src: h5py.Group
) -> h5py.Group:
    """Return ``fout[name]``, creating it (and parents) from ``src`` if absent."""
    _ensure_parent_groups(fin, fout, name)
    if name in fout:
        return fout[name]
    dst = fout.create_group(name)
    _copy_attributes(src, dst)
    return dst


def _resolve_depends_on_chains(fin: h5py.File, fout: h5py.File) -> None:
    """Copy any ``depends_on`` targets that are missing from the output file.

    After copying geometry components, ``depends_on`` chains may reference
    nodes that were not copied — for example an NXlog group inside an
    NXpositioner that acts as a transformation node. NXlog groups are
    copied as length-0 placeholders so historical motor samples in the
    source do not bloat the geometry artifact; everything else is copied
    as-is.
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
            leaf = path.rsplit('/', 1)[-1]
            _copy_child(fin[parent_path or '/'], leaf, fout[parent_path or '/'])


_HANDLED_NX_CLASSES = (
    'NXdetector',
    'NXmonitor',
    'NXsource',
    'NXsample',
    'NXtransformations',
    'NXdisk_chopper',
)


def write_minimal_geometry(
    input_filename: Path, output_filename: Path, use_pixel_shape: bool = True
) -> None:
    """Create minimal geometry file with only detector positions and transformations."""
    with h5py.File(input_filename, 'r') as fin, h5py.File(output_filename, 'w') as fout:

        def visit_and_copy(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Group) or 'NX_class' not in obj.attrs:
                return
            nx_class = _nx_class(obj)
            if nx_class not in _HANDLED_NX_CLASSES:
                return
            dst = _get_or_create_dst(fin, fout, name, obj)
            if nx_class == 'NXdetector':
                _copy_detector_fields(obj, dst, use_pixel_shape=use_pixel_shape)
            elif nx_class == 'NXmonitor':
                _copy_monitor_fields(obj, dst)
            elif nx_class in ('NXsource', 'NXsample'):
                obj.copy('depends_on', dst)
            else:  # NXtransformations or NXdisk_chopper
                for key in obj:
                    _copy_child(obj, key, dst)

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
        print(f'Input file {args.input} does not exist', file=sys.stderr)
        return 1

    if args.output.exists() and not args.force:
        print(f'Output file {args.output} already exists', file=sys.stderr)
        return 1

    write_minimal_geometry(args.input, args.output, use_pixel_shape=args.use_shape)
    return 0


if __name__ == '__main__':
    sys.exit(main())
