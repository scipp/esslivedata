# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
DREAM logical detector view definitions.

This module defines transform functions and registers them with a LogicalViewRegistry.
The registry is used by specs.py (lightweight spec registration) and factories.py
(heavy factory attachment) to ensure transforms are always correctly paired with
their spec metadata.
"""

import scipp as sc

from ess.livedata.handlers.logical_view_registry import LogicalViewRegistry

# Bank sizes for mantle detector logical views
_bank_sizes = {
    'mantle_detector': {
        'wire': 32,
        'module': 5,
        'segment': 6,
        'strip': 256,
        'counter': 2,
    },
}


def _get_mantle_front_layer(da: sc.DataArray) -> sc.DataArray:
    """Transform function to extract mantle front layer."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .transpose(('wire', 'module', 'segment', 'counter', 'strip'))['wire', 0]
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
    )


def _get_wire_view(da: sc.DataArray) -> sc.DataArray:
    """Transform function to extract wire view."""
    return (
        da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector'])
        .sum('strip')
        .flatten(('module', 'segment', 'counter'), to='mod/seg/cntr')
        # Transpose so that wire is the "x" dimension for more natural plotting.
        .transpose()
    )


def _get_strip_view(da: sc.DataArray) -> sc.DataArray:
    """Transform function to extract strip view (sum over all but strip)."""
    return da.fold(dim=da.dim, sizes=_bank_sizes['mantle_detector']).sum(
        ('wire', 'module', 'segment', 'counter')
    )


# Create registry and add all mantle logical views
mantle_views = LogicalViewRegistry()

mantle_views.add(
    name='mantle_front_layer',
    title='Mantle front layer',
    description='All voxels of the front layer of the mantle detector.',
    source_names=['mantle_detector'],
    transform=_get_mantle_front_layer,
)

mantle_views.add(
    name='mantle_wire_view',
    title='Mantle wire view',
    description='Sum over strips to show counts per wire in the mantle detector.',
    source_names=['mantle_detector'],
    transform=_get_wire_view,
)

mantle_views.add(
    name='mantle_strip_view',
    title='Mantle strip view',
    description='Sum over all dimensions except strip to show counts per strip.',
    source_names=['mantle_detector'],
    transform=_get_strip_view,
)
