# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for the dynamic detector geometry providers."""

from __future__ import annotations

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import NeXusComponent, SampleRun

from ess.livedata.handlers.detector_view.types import (
    TransformName,
    TransformValue,
    TransformValueLog,
)
from ess.livedata.handlers.detector_view.workflow import (
    get_transformation_chain_with_value,
    transform_value_from_log,
)


def _make_chain() -> snx.TransformationChain:
    """Build a minimal TransformationChain with two named transformation entries."""
    chain = snx.TransformationChain(parent='/det', value='transformations/carriage')
    chain.transformations = sc.DataGroup()

    # Each entry is a snx.Transformation with a mutable .value attribute.
    # We bypass full construction and use a tiny stub that mimics the
    # interface relied upon by get_transformation_chain_with_value.
    class _Transform:
        def __init__(self, v: sc.Variable) -> None:
            self.value = v

    chain.transformations['carriage'] = _Transform(sc.scalar(1.0, unit='mm'))
    chain.transformations['other'] = _Transform(sc.scalar(7.0, unit='mm'))
    return chain


def _make_log(values: list[float], unit: str = 'mm') -> sc.DataArray:
    times = sc.array(
        dims=['time'], values=list(range(len(values))), unit='ns', dtype='int64'
    )
    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit=unit),
        coords={'time': sc.epoch(unit='ns') + times},
    )


class TestTransformValueFromLog:
    def test_returns_noop_when_log_is_none(self):
        tv = transform_value_from_log(
            None,  # type: ignore[arg-type]
            TransformName('detector_carriage'),
        )
        assert tv.name == ''

    def test_returns_noop_when_name_is_empty(self):
        log = TransformValueLog(_make_log([1.0]))
        tv = transform_value_from_log(log, TransformName(''))
        assert tv.name == ''

    def test_picks_latest_value(self):
        log = TransformValueLog(_make_log([1.0, 2.0, 7.5]))
        tv = transform_value_from_log(log, TransformName('detector_carriage'))
        assert tv.name == 'detector_carriage'
        assert tv.value.value == 7.5
        assert tv.value.unit == sc.Unit('mm')
        # Scalar (no time dim) so to_transformation's filter branch is bypassed.
        assert tv.value.sizes == {}

    def test_units_propagate(self):
        log = TransformValueLog(_make_log([4.2], unit='m'))
        tv = transform_value_from_log(log, TransformName('detector_carriage'))
        assert tv.value.unit == sc.Unit('m')


class TestTransformValueDataclass:
    def test_noop_constructible(self):
        tv = TransformValue(name='', value=sc.scalar(0.0))
        assert tv.name == ''


class TestChainInjection:
    def _component(self, chain: snx.TransformationChain) -> NeXusComponent:
        # NeXusComponent is a sciline.Scope wrapper around sc.DataGroup; for
        # the purpose of get_transformation_chain_with_value we only need
        # something that supports __getitem__('depends_on').
        return NeXusComponent[snx.NXdetector, SampleRun](
            sc.DataGroup({'depends_on': chain})
        )

    def test_noop_value_returns_unchanged(self):
        chain = _make_chain()
        comp = self._component(chain)
        out = get_transformation_chain_with_value(
            comp, TransformValue(name='', value=sc.scalar(0.0))
        )
        assert out.transformations['carriage'].value.value == 1.0
        assert out.transformations['other'].value.value == 7.0

    def test_replaces_only_named_entry(self):
        chain = _make_chain()
        comp = self._component(chain)
        new_value = sc.scalar(42.0, unit='mm')
        out = get_transformation_chain_with_value(
            comp,
            TransformValue(name='carriage', value=new_value),
        )
        assert out.transformations['carriage'].value.value == 42.0
        assert out.transformations['other'].value.value == 7.0
        # Original chain is untouched (deepcopy in provider).
        assert chain.transformations['carriage'].value.value == 1.0
