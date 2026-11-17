# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Find the f144-driven transforms on the ``depends_on`` chains a workflow reads.

The geometry artifact represents a motor-driven chain entry as a length-0 NXlog
placeholder. A workflow reading a chain through such a placeholder needs the
latest sample of the motor's f144 stream, delivered through essreduce's
``TransformationOverrides``, or essreduce's ``to_transformation`` fails.

Both halves of that statement are read rather than declared: the graph says
which chains the targets read, and the chains say which of their entries are
placeholders. Whether a job reads a chain can depend on its params, so this runs
per job (see ``StreamProcessorWorkflow.build``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import get_args, get_origin

import sciline
import sciline.typing
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusMultiComponent,
    NeXusMultiTransformationChain,
    NeXusTransformationChain,
    TransformationOverrides,
)


def placeholders(chain: snx.TransformationChain) -> set[str]:
    """Paths on a chain whose value is a length-0 NXlog."""
    return {
        path
        for path, transform in chain.transformations.items()
        if isinstance(transform.value, sc.DataArray)
        and transform.value.sizes.get('time') == 0
    }


def find_live_transforms(
    pipeline: sciline.Pipeline, required: Iterable[sciline.typing.Key]
) -> dict[sciline.typing.Key, set[str]]:
    """Map each overrides key the targets read to the placeholders it must fill.

    Parameters
    ----------
    pipeline:
        The workflow's pipeline. The components whose chains are read are loaded
        once and stored in it, so that building the ``StreamProcessor`` from this
        pipeline does not load them again.
    required:
        Every key the workflow's targets depend on.

    Returns
    -------
    :
        ``{TransformationOverrides[Component, Run]: paths}``, without the
        overrides keys whose chains have no placeholder.
    """
    required = set(required)
    live: dict[sciline.typing.Key, set[str]] = {}
    for key in required:
        if get_origin(key) is not TransformationOverrides:
            continue
        component, run = get_args(key)
        chains = []
        if NeXusTransformationChain[component, run] in required:
            source = NeXusComponent[component, run]
            pipeline[source] = group = pipeline.compute(source)
            chains.append(group['depends_on'])
        if NeXusMultiTransformationChain[component, run] in required:
            source = NeXusMultiComponent[component, run]
            pipeline[source] = groups = pipeline.compute(source)
            chains.extend(group['depends_on'] for group in groups.values())
        if paths := {path for chain in chains for path in placeholders(chain)}:
            live[key] = paths
    return live
