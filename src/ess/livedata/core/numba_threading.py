# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Verification of Numba's threading layer at service startup.

``ess.reduce.unwrap`` interpolates the wavelength lookup table with a
Numba-compiled kernel (``@njit(parallel=True)``), 30-100x faster than the SciPy
fallback it silently drops to when Numba is unavailable. Jobs run concurrently
in the JobManager's thread pool (``--job-threads``, default 5), so that kernel
is entered from several Python threads at once.

Numba's ``workqueue`` threading layer is not safe under that access pattern and
crashes the process (scipp/ess#705). It is chosen only when neither OpenMP nor
TBB can be loaded -- on Linux ``libgomp.so.1`` gives us ``omp``. Rather than
discover this as a segfault under load, the service refuses to start.
"""

from __future__ import annotations

#: Threading layers that tolerate concurrent entry from multiple Python threads.
THREAD_SAFE_LAYERS = frozenset({'omp', 'tbb'})


def check_threading_layer(layer: str) -> None:
    """Raise unless `layer` is safe to enter from several threads at once."""
    if layer not in THREAD_SAFE_LAYERS:
        raise RuntimeError(
            f"Numba selected the {layer!r} threading layer, which is not safe "
            f"for the concurrent workflow execution this service performs and "
            f"would crash it under load. Install a runtime providing one of "
            f"{sorted(THREAD_SAFE_LAYERS)}: on Linux the system OpenMP library "
            f"(libgomp1 / libgomp), otherwise 'tbb' or 'llvm-openmp'."
        )


def verify_numba_threading_layer() -> str:
    """Force Numba's threading-layer selection and check it is thread-safe.

    Returns
    -------
    :
        Name of the selected threading layer.
    """
    from numba import get_num_threads, threading_layer

    # The layer is picked lazily on first use; this call forces the choice so
    # that it can be inspected before any workflow runs.
    get_num_threads()
    layer = threading_layer()
    check_threading_layer(layer)
    return layer
