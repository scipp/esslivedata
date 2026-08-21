# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Per-component flight-path ranges for the wavelength lookup table.

A lookup table is indexed by ``distance``; a lookup outside its range yields
``NaN`` silently. Each component therefore gets a table covering exactly its own
``Ltotal`` span, and that span must be expressed in the same ``Ltotal`` the
consumer uses at lookup time.

Rather than re-deriving that definition here, the range is computed by asking
essreduce for the very providers the consumer runs --
``DetectorLtotal`` (scattering geometry, source to sample to pixel) and
``MonitorLtotal`` (a straight line from the source). Reusing them is what makes
the range and the lookup agree by construction rather than by review.

Padding is applied on top and is deliberately generous: widening a range adds
distance rows at fixed resolution, which costs recompute in the LUT job and
nothing else, while a range that is too narrow blanks the component.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    Filename,
    NeXusComponent,
    NeXusName,
    NeXusTransformationChain,
    SampleRun,
)
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.types import DetectorLtotal, MonitorLtotal

from ..config.stream import MotionEnvelope

#: Fraction of a component's own span added at each end, plus a floor, so that a
#: component whose pixels all sit at one distance (every monitor, and a detector
#: bank modelled as a point) still gets a table with usable width rather than a
#: degenerate one.
_RELATIVE_PAD = 0.01
_MINIMUM_PAD = sc.scalar(0.1, unit='m')


class LtotalRangeError(ValueError):
    """Raised when a component's flight-path range cannot be derived."""


def _park_live_transforms(
    workflow, component_class: type, motion: Mapping[str, MotionEnvelope]
) -> sc.Variable:
    """Resolve a component's live transforms to their nominal values.

    The artifact stores an f144-driven transform as an empty NXlog, so a
    component riding one has no position until something supplies a value.
    Each such transform is parked at its declared nominal, and the travel of
    every axis the component rides is summed into the padding its range needs.

    Which components ride which axis is derived here rather than declared: a
    component is affected precisely when the axis appears in its ``depends_on``
    chain.
    """
    chain_key = NeXusTransformationChain[component_class, SampleRun]
    chain = workflow.compute(chain_key)
    live = {
        path: transform
        for path, transform in chain.transformations.items()
        if isinstance(value := getattr(transform, 'value', None), sc.DataArray)
        and 'time' in value.dims
    }
    if not live:
        return sc.scalar(0.0, unit='m')

    chain = copy.deepcopy(chain)
    travel = sc.scalar(0.0, unit='m')
    for path in live:
        if (envelope := motion.get(path)) is None:
            raise LtotalRangeError(
                f"transform {path!r} is driven by a live stream and the artifact "
                "carries no value for it; declare a MotionEnvelope for this axis"
            )
        transform = chain.transformations[path]
        transform.value = sc.DataArray(
            sc.array(
                dims=['time'],
                values=[envelope.nominal.to(unit=transform.value.unit).value],
                unit=transform.value.unit,
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0], unit='ns', dtype='int64')
            },
        )
        travel = travel + envelope.travel.to(unit='m')

    component_key = NeXusComponent[component_class, SampleRun]
    component = copy.copy(workflow.compute(component_key))
    component['depends_on'] = chain
    workflow[component_key] = component
    return travel


def _ltotal(
    nexus_filename: str,
    component: str,
    *,
    is_monitor: bool,
    motion: Mapping[str, MotionEnvelope],
) -> tuple[sc.Variable, sc.Variable]:
    """Compute a component's ``Ltotal`` and its motion padding."""
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[snx.NXmonitor]
    )
    workflow[Filename[SampleRun]] = nexus_filename
    component_class = snx.NXmonitor if is_monitor else snx.NXdetector
    workflow[NeXusName[component_class]] = component
    travel = _park_live_transforms(workflow, component_class, motion)
    ltotal = (
        workflow.compute(MonitorLtotal[SampleRun, snx.NXmonitor])
        if is_monitor
        else workflow.compute(DetectorLtotal[SampleRun])
    )
    return ltotal, travel


def component_ltotal_range(
    nexus_filename: str,
    component: str,
    *,
    is_monitor: bool,
    motion: Mapping[str, MotionEnvelope] | None = None,
) -> tuple[sc.Variable, sc.Variable]:
    """Derive the flight-path range covered by one component's lookup table.

    Parameters
    ----------
    nexus_filename:
        Geometry artifact to read the component's position from.
    component:
        NeXus name of the detector or monitor.
    is_monitor:
        Select the straight-line ``Ltotal`` used for monitors over the
        scattering-geometry one used for detectors.
    motion:
        Declared envelopes for the instrument's moving axes, keyed by NeXus
        transform path. Only the axes this component actually rides are applied.

    Returns
    -------
    :
        ``(start, stop)`` in metres, padded, and extended downstream by the
        travel of every axis the component rides.

    Raises
    ------
    LtotalRangeError:
        If the component's position cannot be resolved. Either the component is
        absent from the artifact, or it rides a live axis with no declared
        :class:`MotionEnvelope`.
    """
    try:
        ltotal, travel = _ltotal(
            nexus_filename, component, is_monitor=is_monitor, motion=motion or {}
        )
    except LtotalRangeError as exc:
        raise LtotalRangeError(f"Cannot place {component!r}: {exc}") from exc
    except Exception as exc:
        raise LtotalRangeError(
            f"Cannot derive the flight-path range of {component!r} from "
            f"{nexus_filename}: {exc}"
        ) from exc
    ltotal = ltotal.to(unit='m')
    start, stop = ltotal.min(), ltotal.max()
    pad = sc.max(sc.concat([(stop - start) * _RELATIVE_PAD, _MINIMUM_PAD], 'pad'))
    return start - pad, stop + pad + travel


def component_ltotal_ranges(
    nexus_filename: str,
    *,
    detectors: Iterable[str],
    monitors: Iterable[str],
    motion: Mapping[str, MotionEnvelope] | None = None,
) -> dict[str, tuple[sc.Variable, sc.Variable]]:
    """Derive the flight-path range of every component, keyed by component name.

    Parameters
    ----------
    nexus_filename:
        Geometry artifact to read component positions from.
    detectors:
        Detector names, given the scattering-geometry ``Ltotal``.
    monitors:
        Monitor names, given the straight-line ``Ltotal``.
    motion:
        Declared envelopes for the instrument's moving axes; see
        :func:`component_ltotal_range`.
    """
    return {
        name: component_ltotal_range(
            nexus_filename, name, is_monitor=is_monitor, motion=motion
        )
        for names, is_monitor in ((detectors, False), (monitors, True))
        for name in names
    }
