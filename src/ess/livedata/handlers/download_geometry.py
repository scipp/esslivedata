# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Download all geometry files to LIVEDATA_DATA_DIR.

Intended for use during Docker image builds to ensure geometry files are
available at runtime without network access.
"""

import os
import pathlib
import urllib.request

import structlog

from .detector_data_handler import _GEOMETRY_RELEASE_URL, _registry

_logger = structlog.get_logger()


def download_all(target_dir: pathlib.Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in _registry:
        dest = target_dir / filename
        if dest.exists():
            continue
        url = _GEOMETRY_RELEASE_URL + filename
        _logger.info("Downloading", filename=filename)
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    _logger.info("Geometry files downloaded", target_dir=target_dir)


if __name__ == '__main__':
    data_dir = os.environ.get('LIVEDATA_DATA_DIR')
    if data_dir is None:
        raise SystemExit("LIVEDATA_DATA_DIR environment variable is not set")
    download_all(pathlib.Path(data_dir))
