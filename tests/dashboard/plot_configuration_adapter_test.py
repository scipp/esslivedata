# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for PlotConfigurationAdapter — focused on params_factory wiring."""

from __future__ import annotations

from typing import Literal

import pydantic

from ess.livedata.dashboard.plot_configuration_adapter import (
    PlotConfigurationAdapter,
)
from ess.livedata.dashboard.plotter_registry import (
    DataRequirements,
    PlotterSpec,
)


class _StaticParams(pydantic.BaseModel):
    pass


def _make_spec() -> PlotterSpec:
    return PlotterSpec(
        name='dummy',
        title='Dummy',
        description='',
        params=_StaticParams,
        data_requirements=DataRequirements(min_dims=0, max_dims=10),
    )


def _adapter(
    *, params_factory=None, output_template_dims=None
) -> PlotConfigurationAdapter:
    return PlotConfigurationAdapter(
        plot_spec=_make_spec(),
        source_names=['s'],
        success_callback=lambda *_: None,
        output_template_dims=output_template_dims,
        params_factory=params_factory,
    )


def test_returns_static_params_when_no_factory() -> None:
    adapter = _adapter(output_template_dims=('a', 'b'))
    assert adapter.model_class() is _StaticParams


def test_returns_static_params_when_factory_set_but_no_dims() -> None:
    def factory(dims):  # would raise if called with None
        raise AssertionError('factory should not be called without dims')

    adapter = _adapter(params_factory=factory, output_template_dims=None)
    assert adapter.model_class() is _StaticParams


def test_calls_factory_with_dims_when_both_provided() -> None:
    captured: list[tuple[str, ...]] = []

    def factory(dims):
        captured.append(dims)

        class _Dyn(_StaticParams):
            pick: Literal[*dims] = dims[0]  # type: ignore[valid-type]

        return _Dyn

    adapter = _adapter(params_factory=factory, output_template_dims=('x', 'y', 'z'))
    Dyn = adapter.model_class()
    assert captured == [('x', 'y', 'z')]
    assert Dyn is not _StaticParams
    assert issubclass(Dyn, _StaticParams)
    assert Dyn.model_fields['pick'].annotation == Literal['x', 'y', 'z']
