# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Fake plotters and presenters for dashboard widget tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import holoviews as hv
import pydantic

from ess.livedata.dashboard.plot_params import LegendPosition
from ess.livedata.dashboard.plots import PresenterBase, TimeBounds
from ess.livedata.dashboard.range_hook import Axis


class EchoPresenter(PresenterBase):
    """Presents piped data as-is, so the DynamicMap renders whatever was computed."""

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        return hv.DynamicMap(lambda data: data, streams=[pipe], cache_size=1)


class CurvePresenter(PresenterBase):
    """Presents a fixed empty curve regardless of piped data."""

    def present(self, pipe: hv.streams.Pipe) -> hv.DynamicMap:
        return hv.DynamicMap(lambda data: hv.Curve([]), streams=[pipe])


class FakePlotter:
    """Minimal plotter covering the surface SessionLayer and cell composition use.

    Parameters
    ----------
    name:
        Optional label for telling instances apart in tests.
    cached_state:
        Initial cached state; ``compute`` replaces it.
    time_bounds:
        Value returned by the ``time_bounds`` property.
    presenter_cls:
        Presenter class instantiated by ``create_presenter``.
    """

    AUTOSCALE_AXES: ClassVar[frozenset[Axis]] = frozenset({'x', 'y'})

    def __init__(
        self,
        *,
        name: str | None = None,
        cached_state: Any = None,
        time_bounds: TimeBounds | None = None,
        presenter_cls: type[PresenterBase] = EchoPresenter,
    ) -> None:
        self.name = name
        self._cached_state = cached_state
        self._time_bounds = time_bounds
        self._presenter_cls = presenter_cls
        self._presenters: list[PresenterBase] = []

    @property
    def is_overlayable(self) -> bool:
        # Mirror a real plotter: a Layout cannot share a figure with siblings.
        return not isinstance(self._cached_state, hv.Layout)

    @property
    def autoscale_axes(self) -> frozenset[Axis]:
        return self.AUTOSCALE_AXES

    @property
    def legend_position(self) -> LegendPosition | None:
        # Mirror a plotter drawing no legend: the cell takes its placement from
        # whichever layer does.
        return None

    @property
    def time_bounds(self) -> TimeBounds | None:
        return self._time_bounds

    @time_bounds.setter
    def time_bounds(self, value: TimeBounds | None) -> None:
        # Settable so tests can age a stream without wall-clock sleeps.
        self._time_bounds = value

    def compute(self, data: Any) -> None:
        self._cached_state = data
        self.mark_presenters_dirty()

    def get_cached_state(self) -> Any:
        return self._cached_state

    def has_cached_state(self) -> bool:
        return self._cached_state is not None

    def create_presenter(self, *, owner: Any = None) -> PresenterBase:
        presenter = self._presenter_cls(self, owner=owner)
        self._presenters.append(presenter)
        return presenter

    def mark_presenters_dirty(self) -> None:
        for p in self._presenters:
            p._mark_dirty()

    def iter_range_targets(self) -> Iterator[tuple[Any, Any]]:
        return iter(())


class EmptyParams(pydantic.BaseModel):
    """Plot params placeholder for configs whose params are irrelevant."""


class ViewerToken:
    """Weakref-able stand-in for another session's viewer interest token."""
