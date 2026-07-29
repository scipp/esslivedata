# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Shared geometry-change signal driving reset-on-move for detector and monitor views.

A rigid component move is detected by stamping the component's resolved placement
onto the accumulated histogram as a scalar coord and letting the cumulative
accumulator reset when it changes (see
:func:`~.accumulators.make_no_copy_accumulator_pair`). The signal must be a 0-dim
coord that survives ``hist``/``sum``/slicing, so it is the ``NeXusTransformation``
itself rather than ``position``: a per-pixel ``position`` is an array, not a scalar,
and is dropped when the histogram collapses the pixel dimension.
"""

from __future__ import annotations

import scipp as sc
from ess.reduce.nexus.types import NeXusTransformation

_TRANSFORM_DTYPES = (
    sc.DType.translation3,
    sc.DType.affine_transform3,
    sc.DType.linear_transform3,
    sc.DType.rotation3,
)


def geometry_signal(transform: NeXusTransformation) -> sc.Variable | None:
    """Reduce a resolved NeXus transformation to a scalar reset signal.

    The transformation is recomputed from the (possibly carriage-patched)
    transformation chain each cycle, so a component move changes it.

    A component with no ``depends_on`` chain resolves to an identity sentinel
    (a plain scalar, not a spatial transform): there is no geometry to move, so
    return ``None`` and stamp no coord.

    Parameters
    ----------
    transform:
        The component's resolved ``NeXusTransformation``. ``.value`` is the
        resolved scalar transform (time-dependence is rejected upstream).

    Returns
    -------
    :
        The 0-dim placement transform, or ``None`` for an identity sentinel.
    """
    value = transform.value
    if value.dtype not in _TRANSFORM_DTYPES:
        return None
    return value
