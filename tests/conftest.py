# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest


@pytest.fixture(autouse=True, scope="session")
def _patch_essdiffraction_lut_hash():
    """Workaround for hash mismatch in essdiffraction 26.3.0 data registry.

    The file DREAM-high-flux-tof-lut-5m-80m-bc240.h5 on the server has a
    different MD5 hash than what is recorded in the essdiffraction 26.3.0
    pooch registry. This prevents pooch from caching the file.
    Remove this fixture once essdiffraction ships a version with the
    corrected hash.
    """
    try:
        from ess.dream import data

        registry = data._registry._registry.registry
        key = 'DREAM-high-flux-tof-lut-5m-80m-bc240.h5'
        if key in registry:
            registry[key] = 'md5:e97ffd491bd11bdceec28036802a00ad'
    except ImportError:
        pass
