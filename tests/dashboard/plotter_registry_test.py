# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for the plotter registry and cross-cutting plotter contracts."""

import pytest

from ess.livedata.dashboard.plotter_registry import PlotterCategory, plotter_registry


def _data_plotter_names() -> list[str]:
    """Get names of all DATA-category plotters from the registry."""
    return [
        name
        for name, entry in plotter_registry.items()
        if entry.spec.category == PlotterCategory.DATA
    ]


class TestPlotterComputeSignature:
    """Verify all registered plotters accept the kwargs passed by PlotOrchestrator."""

    @pytest.mark.parametrize("plotter_name", _data_plotter_names())
    def test_compute_accepts_source_title_kwarg(self, plotter_name):
        """PlotOrchestrator passes source_title= to all plotters.

        Each plotter's compute() must accept this keyword argument
        (either explicitly or via **kwargs) without raising TypeError.
        """
        entry = plotter_registry[plotter_name]
        default_params = entry.spec.params()
        plotter = entry.factory(default_params)

        try:
            plotter.compute({}, source_title=lambda x: x)
        except TypeError as e:
            if "source_title" in str(e):
                pytest.fail(
                    f"Plotter '{plotter_name}' does not accept 'source_title' kwarg. "
                    f"Add **kwargs to its compute() signature."
                )
        except Exception:  # noqa: S110
            pass  # Other errors (empty data, missing keys, etc.) are expected
