# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Upload a geometry file to the GitHub Release.

This is the final step in the geometry file workflow:

1. Create the geometry file from a full NeXus file::

       ess-livedata-make-geometry-nexus input.nxs geometry-<instrument>-<date>.nxs

2. Test that the geometry file works (load it, verify detector views and
   workflows produce correct results).

3. Upload the verified file::

       python -m ess.livedata.handlers.upload_geometry geometry-<instrument>-<date>.nxs

4. Add the printed registry entry to ``_registry`` in
   ``src/ess/livedata/handlers/detector_data_handler.py`` and commit.

The script validates that the file was properly stripped (no NXevent_data groups)
and that the filename follows the expected naming convention.
"""

import hashlib
import pathlib
import re
import subprocess

_RELEASE_TAG = 'geometry-v0'
_FILENAME_PATTERN = re.compile(r'^geometry-[a-z]+-(\d{4}-\d{2}-\d{2})\.nxs$')
_FILENAME_PATTERN_NO_SHAPE = re.compile(
    r'^geometry-[a-z]+-no-shape-(\d{4}-\d{2}-\d{2})\.nxs$'
)


def _compute_md5(path: pathlib.Path) -> str:
    h = hashlib.md5()  # noqa: S324
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _validate_filename(name: str) -> None:
    if not (_FILENAME_PATTERN.match(name) or _FILENAME_PATTERN_NO_SHAPE.match(name)):
        raise SystemExit(
            f"Filename '{name}' does not match expected pattern "
            "'geometry-<instrument>-<YYYY-MM-DD>.nxs' or "
            "'geometry-<instrument>-no-shape-<YYYY-MM-DD>.nxs'"
        )


def _validate_no_event_data(path: pathlib.Path) -> None:
    """Check that the file does not contain NXevent_data groups."""
    import h5py

    event_data_groups: list[str] = []

    def _visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            nx_class = obj.attrs.get('NX_class', b'')
            if isinstance(nx_class, bytes):
                nx_class = nx_class.decode()
            if nx_class == 'NXevent_data':
                event_data_groups.append(name)

    with h5py.File(path, 'r') as f:
        f.visititems(_visitor)

    if event_data_groups:
        raise SystemExit(
            "File contains NXevent_data groups and was likely not stripped:\n"
            + "\n".join(f"  - {g}" for g in event_data_groups)
            + "\n\nUse 'ess-livedata-make-geometry-nexus' to create a stripped file."
        )


def _upload(path: pathlib.Path) -> None:
    result = subprocess.run(  # noqa: S603
        ['gh', 'release', 'upload', _RELEASE_TAG, str(path), '--clobber'],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"Upload failed: {result.stderr.strip()}")


def main(filepath: str) -> None:
    path = pathlib.Path(filepath)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    _validate_filename(path.name)
    _validate_no_event_data(path)
    md5 = _compute_md5(path)

    print(f"Uploading {path.name} to release '{_RELEASE_TAG}'...")  # noqa: T201
    _upload(path)

    print(  # noqa: T201
        f"\nDone. Add this line to _registry in"
        f" src/ess/livedata/handlers/detector_data_handler.py:\n\n"
        f"    '{path.name}': 'md5:{md5}',"
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Upload a geometry file to the GitHub Release.',
        epilog=(
            'Workflow:\n'
            '  1. ess-livedata-make-geometry-nexus input.nxs'
            ' geometry-<instrument>-<date>.nxs\n'
            '  2. Test the file (load it, verify detector views and workflows)\n'
            '  3. python -m ess.livedata.handlers.upload_geometry'
            ' geometry-<instrument>-<date>.nxs\n'
            '  4. Add the printed registry entry to _registry in\n'
            '     src/ess/livedata/handlers/detector_data_handler.py and commit'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('file', help='Geometry .nxs file to upload')
    args = parser.parse_args()
    main(args.file)
