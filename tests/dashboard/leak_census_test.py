# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the dashboard leak census.

Uses real HoloViews objects (no mocks): the census walks the live object graph,
so a faithful test must construct genuine customized elements and retain them.
"""

from __future__ import annotations

import gc

import holoviews as hv
import pytest

from ess.livedata.dashboard.leak_census import collect_leak_census

hv.extension('bokeh')


class _Holder:
    """Stand-in for a per-session component bundle that retains plot objects."""

    def __init__(self, pipe: hv.streams.Pipe, dmap: hv.DynamicMap) -> None:
        self.pipe = pipe
        self.dmap = dmap


def _make_retained_plots(n: int) -> list[_Holder]:
    """Create ``n`` customized Curves, each retained by a Pipe in a DynamicMap."""
    holders = []
    for i in range(n):
        element = hv.Curve([(0, i), (1, i + 1)]).opts(line_width=2)
        pipe = hv.streams.Pipe(data=element)
        dmap = hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)
        holders.append(_Holder(pipe, dmap))
    gc.collect()
    return holders


def test_census_returns_expected_fields() -> None:
    census = collect_leak_census()
    assert 'census_error' not in census
    for key in ('store_n', 'live_with_id', 'leaked_by_class', 'live_types', 'owners'):
        assert key in census


def test_retained_customized_elements_show_as_live_with_id() -> None:
    before = collect_leak_census()['live_with_id']
    holders = _make_retained_plots(50)
    after = collect_leak_census()
    try:
        # Genuine retention: the customized elements are alive AND keyed in the
        # store, so live_with_id tracks store growth (the "retention" branch of
        # the discriminator, as opposed to orphaned ids).
        assert after['live_with_id'] >= before + 50
        assert 'Curve' in after['leaked_by_class']
        # The holder types accumulate 1:1 and surface in the live-type census.
        assert 'Pipe:' in after['live_types']
        assert 'DynamicMap:' in after['live_types']
    finally:
        del holders
        gc.collect()


def test_released_elements_are_reclaimed() -> None:
    holders = _make_retained_plots(50)
    with_leak = collect_leak_census()['live_with_id']
    del holders
    gc.collect()
    after = collect_leak_census()['live_with_id']
    # Dropping the holders releases the elements; the store self-cleans via the
    # weakref finalizer, so live_with_id falls back.
    assert after < with_leak


@pytest.mark.parametrize('count', [10, 100])
def test_owners_names_the_proximate_holder(count: int) -> None:
    holders = _make_retained_plots(count)
    try:
        owners = collect_leak_census()['owners']
        # The referrer walk reaches the Pipe that holds each sampled element.
        assert 'Pipe' in owners
    finally:
        del holders
        gc.collect()
