# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Registry for logical detector views.

This module provides a registry pattern that couples transform functions with their
spec metadata, making it easy to add new logical detector views without mismatching
handles and transforms.

The two-phase registration pattern (specs.py then factories.py) is preserved because:
1. specs.py should stay lightweight for frontend use
2. factories.py imports heavy dependencies (ess.reduce, etc.)

Usage
-----
In the instrument's views.py (or transforms.py), define transforms and register them:

.. code-block:: python

    from ess.livedata.handlers.logical_view_registry import LogicalViewRegistry
    import scipp as sc

    # Create registry instance for this instrument
    mantle_views = LogicalViewRegistry()

    def _get_wire_view(da: sc.DataArray) -> sc.DataArray:
        return da.fold(...).sum('strip')...

    mantle_views.add(
        name='mantle_wire_view',
        title='Mantle wire view',
        description='Sum over strips to show counts per wire.',
        source_names=['mantle_detector'],
        transform=_get_wire_view,
    )

In specs.py (lightweight):

.. code-block:: python

    from .views import mantle_views
    mantle_views.register_specs(instrument)

In factories.py (heavy imports):

.. code-block:: python

    from .views import mantle_views
    mantle_views.attach_factories(instrument)
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import scipp as sc

from ..config.instrument import Instrument
from .workflow_factory import SpecHandle


@dataclass
class LogicalViewConfig:
    """Configuration for a single logical detector view."""

    name: str
    title: str
    description: str
    source_names: list[str]
    transform: Callable[[sc.DataArray], sc.DataArray]
    roi_support: bool = True
    output_ndim: int | None = None


@dataclass
class LogicalViewRegistry:
    """
    Registry coupling transform functions with their spec metadata.

    This eliminates the error-prone pattern of separately defining handles in specs.py
    and transforms in factories.py, where they can easily get out of sync.
    """

    _configs: list[LogicalViewConfig] = field(default_factory=list)
    _handles: dict[str, SpecHandle] = field(default_factory=dict)

    def add(
        self,
        *,
        name: str,
        title: str,
        description: str,
        source_names: list[str],
        transform: Callable[[sc.DataArray], sc.DataArray],
        roi_support: bool = True,
        output_ndim: int | None = None,
    ) -> None:
        """
        Add a logical view configuration to the registry.

        Parameters
        ----------
        name:
            Unique name for the spec within the detector_data namespace.
        title:
            Human-readable title for the view.
        description:
            Description of the view.
        source_names:
            List of detector source names this view applies to.
        transform:
            Function that transforms raw detector data to the view output.
        roi_support:
            Whether ROI selection is supported for this view.
        output_ndim:
            Number of dimensions for spatial outputs.
        """
        self._configs.append(
            LogicalViewConfig(
                name=name,
                title=title,
                description=description,
                source_names=list(source_names),
                transform=transform,
                roi_support=roi_support,
                output_ndim=output_ndim,
            )
        )

    def register_specs(self, instrument: Instrument) -> dict[str, SpecHandle]:
        """
        Register all view specs with the instrument.

        This is the lightweight phase - it does not import heavy dependencies.
        Call this from specs.py.

        Parameters
        ----------
        instrument:
            Instrument to register specs with.

        Returns
        -------
        :
            Dict mapping view names to their spec handles.
        """
        from .detector_view_specs import register_logical_detector_view_spec

        self._handles.clear()
        for config in self._configs:
            handle = register_logical_detector_view_spec(
                instrument=instrument,
                name=config.name,
                title=config.title,
                description=config.description,
                source_names=config.source_names,
                roi_support=config.roi_support,
                output_ndim=config.output_ndim,
            )
            self._handles[config.name] = handle
        return dict(self._handles)

    def attach_factories(self, instrument: Instrument) -> None:
        """
        Attach factory implementations to all registered specs.

        This is the heavy phase - it imports DetectorLogicalView and creates
        factory functions. Call this from factories.py.

        Parameters
        ----------
        instrument:
            Instrument to use for creating views. Must be the same instrument
            used in register_specs().

        Raises
        ------
        RuntimeError
            If register_specs() was not called first.
        """
        if not self._handles:
            raise RuntimeError(
                "No handles registered. Call register_specs() "
                "before attach_factories()."
            )

        # Heavy import - only happens in factories.py
        from .detector_data_handler import DetectorLogicalView

        for config in self._configs:
            handle = self._handles[config.name]
            view = DetectorLogicalView(
                instrument=instrument, transform=config.transform
            )
            handle.attach_factory()(view.make_view)

    def __len__(self) -> int:
        return len(self._configs)

    def __iter__(self):
        return iter(self._configs)
