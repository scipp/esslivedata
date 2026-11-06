# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unit tests for the fused dynamic-transform overrides provider."""

from __future__ import annotations

import inspect

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import SampleRun, TransformationOverrides

from ess.livedata.config.value_log import ValueLog
from ess.livedata.workflows.dynamic_transforms import (
    build_transformation_overrides_provider,
)


class _CarriageLog(ValueLog):
    """Per-binding key for the carriage NXlog (test-local)."""


class _OtherLog(ValueLog):
    """Per-binding key for a second NXlog (test-local)."""


def _make_log(values: list[float], unit: str = 'mm') -> sc.DataArray:
    times = sc.array(
        dims=['time'], values=list(range(len(values))), unit='ns', dtype='int64'
    )
    return sc.DataArray(
        sc.array(dims=['time'], values=values, unit=unit),
        coords={'time': sc.epoch(unit='ns') + times},
    )


class TestProvider:
    def test_maps_path_to_latest_sample(self):
        provider = build_transformation_overrides_provider(
            snx.NXdetector, [('carriage', _CarriageLog)]
        )
        out = provider(_CarriageLog(values=_make_log([1.0, 2.0, 7.5])))
        assert sc.identical(out['carriage'], sc.scalar(7.5, unit='mm'))

    def test_two_bindings_map_independently(self):
        provider = build_transformation_overrides_provider(
            snx.NXdetector,
            [('carriage', _CarriageLog), ('other', _OtherLog)],
        )
        out = provider(
            _CarriageLog(values=_make_log([10.0])),
            _OtherLog(values=_make_log([20.0])),
        )
        assert sc.identical(out['carriage'], sc.scalar(10.0, unit='mm'))
        assert sc.identical(out['other'], sc.scalar(20.0, unit='mm'))


class TestSynthesisedProvider:
    """Sciline introspects providers via inspect.getfullargspec, so a
    synthesised provider's argspec must report named typed positional params
    rather than ``*args``."""

    def test_argspec_lists_named_params(self):
        provider = build_transformation_overrides_provider(
            snx.NXdetector,
            [('carriage', _CarriageLog), ('other', _OtherLog)],
        )
        spec = inspect.getfullargspec(provider)
        assert spec.args == ['log_0', 'log_1']
        assert spec.varargs is None
        assert spec.annotations['log_0'] is _CarriageLog
        assert spec.annotations['log_1'] is _OtherLog
        assert (
            spec.annotations['return']
            is TransformationOverrides[snx.NXdetector, SampleRun]
        )
