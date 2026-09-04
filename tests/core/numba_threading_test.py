# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.core.numba_threading import (
    THREAD_SAFE_LAYERS,
    check_threading_layer,
    verify_numba_threading_layer,
)


@pytest.mark.parametrize('layer', sorted(THREAD_SAFE_LAYERS))
def test_check_threading_layer_accepts_thread_safe_layers(layer: str) -> None:
    check_threading_layer(layer)


@pytest.mark.parametrize('layer', ['workqueue', 'unknown'])
def test_check_threading_layer_rejects_unsafe_layers(layer: str) -> None:
    with pytest.raises(RuntimeError, match=layer):
        check_threading_layer(layer)


def test_installed_numba_selects_a_thread_safe_layer() -> None:
    assert verify_numba_threading_layer() in THREAD_SAFE_LAYERS
