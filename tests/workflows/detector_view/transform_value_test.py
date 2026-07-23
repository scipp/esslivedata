# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for the fused dynamic-transform chain-patch provider."""

from __future__ import annotations

import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import NeXusComponent, SampleRun

from ess.livedata.config.value_log import ValueLog
from ess.livedata.workflows.dynamic_transforms import build_patched_chain_provider


class _CarriageLog(ValueLog):
    """Per-binding key for the carriage NXlog (test-local)."""


class _OtherLog(ValueLog):
    """Per-binding key for a second NXlog (test-local)."""


def _make_chain() -> snx.TransformationChain:
    chain = snx.TransformationChain(parent='/det', value='transformations/carriage')
    chain.transformations = sc.DataGroup()

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


def _component(chain: snx.TransformationChain) -> NeXusComponent:
    return NeXusComponent[snx.NXdetector, SampleRun](
        sc.DataGroup({'depends_on': chain})
    )


class TestSingleBinding:
    def _provider(self):
        return build_patched_chain_provider(
            snx.NXdetector, [('carriage', _CarriageLog)]
        )

    def test_patches_named_entry(self):
        comp = _component(_make_chain())
        log = _CarriageLog(values=_make_log([1.0, 2.0, 7.5]))
        out = self._provider()(comp, log)
        assert out.transformations['carriage'].value.value == 7.5
        # Untouched entries preserve their original value.
        assert out.transformations['other'].value.value == 7.0

    def test_raises_when_path_not_in_chain(self):
        provider = build_patched_chain_provider(
            snx.NXdetector, [('missing', _CarriageLog)]
        )
        comp = _component(_make_chain())
        log = _CarriageLog(values=_make_log([1.0]))
        with pytest.raises(KeyError, match='missing'):
            provider(comp, log)

    def test_does_not_mutate_original_chain(self):
        chain = _make_chain()
        comp = _component(chain)
        log = _CarriageLog(values=_make_log([42.0]))
        self._provider()(comp, log)
        # Deepcopy in the provider keeps the cached NeXusComponent clean.
        assert chain.transformations['carriage'].value.value == 1.0


class TestMultipleBindings:
    """Item (4) from motion-context-wiring.md: multiple dynamic transforms on
    one chain. With per-binding ValueLog subclasses each binding has a distinct
    Sciline key, so the fused provider can patch them all in one pass."""

    def test_two_bindings_patch_independently(self):
        provider = build_patched_chain_provider(
            snx.NXdetector,
            [('carriage', _CarriageLog), ('other', _OtherLog)],
        )
        comp = _component(_make_chain())
        out = provider(
            comp,
            _CarriageLog(values=_make_log([10.0])),
            _OtherLog(values=_make_log([20.0])),
        )
        assert out.transformations['carriage'].value.value == 10.0
        assert out.transformations['other'].value.value == 20.0


class TestSynthesisedProvider:
    """Sciline introspects providers via inspect.getfullargspec, so a
    synthesised provider's argspec must report named typed positional params
    rather than ``*args``."""

    def test_argspec_lists_named_params(self):
        import inspect

        provider = build_patched_chain_provider(
            snx.NXdetector,
            [('carriage', _CarriageLog), ('other', _OtherLog)],
        )
        spec = inspect.getfullargspec(provider)
        assert spec.args == ['component', 'log_0', 'log_1']
        assert spec.varargs is None
        assert spec.annotations['log_0'] is _CarriageLog
        assert spec.annotations['log_1'] is _OtherLog
