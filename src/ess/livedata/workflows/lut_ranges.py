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
import itertools
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

from ..config.stream import AxisRange

#: Fraction of a component's own span added at each end, plus a floor, so that a
#: component whose pixels all sit at one distance (every monitor, and a detector
#: bank modelled as a point) still gets a table with usable width rather than a
#: degenerate one.
_RELATIVE_PAD = 0.01
_MINIMUM_PAD = sc.scalar(0.1, unit='m')


class LtotalRangeError(ValueError):
    """Raised when a component's flight-path range cannot be derived."""


def _park(
    chain: snx.TransformationChain, values: Mapping[str, sc.Variable]
) -> snx.TransformationChain:
    """Return a copy of the chain with each named transform held at one value."""
    chain = copy.deepcopy(chain)
    for path, value in values.items():
        transform = chain.transformations[path]
        transform.value = sc.DataArray(
            sc.array(
                dims=['time'],
                values=[value.to(unit=transform.value.unit).value],
                unit=transform.value.unit,
            ),
            coords={
                'time': sc.array(dims=['time'], values=[0], unit='ns', dtype='int64')
            },
        )
    return chain


def _axis_bounds(
    chain: snx.TransformationChain, axis_ranges: Mapping[str, AxisRange]
) -> dict[str, tuple[sc.Variable, sc.Variable]]:
    """The declared bounds of every live axis the component rides.

    The artifact stores an f144-driven transform as an empty NXlog, so a
    component riding one has no position until something supplies a value.
    Which components ride which axis is derived here rather than declared: a
    component is affected precisely when the axis appears in its ``depends_on``
    chain.

    A live *rotation* is refused outright: ``Ltotal`` is not monotonic in an
    angle, so evaluating the bounds would silently under-cover the swing rather
    than bracket it. Refusing costs the component its table, which is the
    handled path; accepting would cost it a table that is wrong where it
    matters. An axis nobody declared leaves the component unplaceable too.
    """
    bounds = {}
    for path, transform in chain.transformations.items():
        value = getattr(transform, 'value', None)
        if not (isinstance(value, sc.DataArray) and 'time' in value.dims):
            continue
        if transform.transformation_type == 'rotation':
            raise LtotalRangeError(
                f"transform {path!r} is a live rotation axis; deriving a range "
                "from its bounds would under-cover the swing"
            )
        if (axis := axis_ranges.get(path)) is None:
            raise LtotalRangeError(
                f"transform {path!r} is driven by a live stream and the artifact "
                "carries no value for it; declare an AxisRange for this axis"
            )
        bounds[path] = (axis.lower, axis.upper)
    return bounds


def _ltotal(
    nexus_filename: str,
    component: str,
    *,
    is_monitor: bool,
    axis_ranges: Mapping[str, AxisRange],
) -> sc.Variable:
    """Compute a component's ``Ltotal`` over the full extent of its motion.

    The geometry is evaluated at every corner of the box the declared axis
    bounds span, and the results concatenated. Pixel positions are affine in a
    translation's value, so ``Ltotal`` is convex and its maximum is attained at
    a corner; the minimum can in principle lie inside the box, which the
    padding in :func:`component_ltotal_range` absorbs.
    """
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[snx.NXmonitor]
    )
    workflow[Filename[SampleRun]] = nexus_filename
    component_class = snx.NXmonitor if is_monitor else snx.NXdetector
    workflow[NeXusName[component_class]] = component
    ltotal_key = (
        MonitorLtotal[SampleRun, snx.NXmonitor]
        if is_monitor
        else DetectorLtotal[SampleRun]
    )

    chain_key = NeXusTransformationChain[component_class, SampleRun]
    chain = workflow.compute(chain_key)
    bounds = _axis_bounds(chain, axis_ranges)
    if not bounds:
        return workflow.compute(ltotal_key).to(unit='m')

    component_key = NeXusComponent[component_class, SampleRun]
    unplaced = workflow.compute(component_key)
    corners = []
    for values in itertools.product(*bounds.values()):
        placed = copy.copy(unplaced)
        placed['depends_on'] = _park(chain, dict(zip(bounds, values, strict=True)))
        workflow[component_key] = placed
        corners.append(workflow.compute(ltotal_key).to(unit='m').flatten(to='corner'))
    return sc.concat(corners, 'corner')


def component_ltotal_range(
    nexus_filename: str,
    component: str,
    *,
    is_monitor: bool,
    axis_ranges: Mapping[str, AxisRange] | None = None,
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
    axis_ranges:
        Declared value ranges of the instrument's moving axes, keyed by NeXus
        transform path. Only the axes this component actually rides are applied.

    Returns
    -------
    :
        ``(start, stop)`` in metres, padded, and spanning the full extent of
        every axis the component rides.

    Raises
    ------
    LtotalRangeError:
        If the component's position cannot be resolved. Either the component is
        absent from the artifact, or it rides a live axis that has no declared
        :class:`AxisRange` or is a rotation.
    """
    try:
        ltotal = _ltotal(
            nexus_filename,
            component,
            is_monitor=is_monitor,
            axis_ranges=axis_ranges or {},
        )
    except LtotalRangeError as exc:
        raise LtotalRangeError(f"Cannot place {component!r}: {exc}") from exc
    except Exception as exc:
        raise LtotalRangeError(
            f"Cannot derive the flight-path range of {component!r} from "
            f"{nexus_filename}: {exc}"
        ) from exc
    start, stop = ltotal.min(), ltotal.max()
    pad = sc.max(sc.concat([(stop - start) * _RELATIVE_PAD, _MINIMUM_PAD], 'pad'))
    return start - pad, stop + pad


def component_ltotal_ranges(
    nexus_filename: str,
    *,
    detectors: Iterable[str],
    monitors: Iterable[str],
    axis_ranges: Mapping[str, AxisRange] | None = None,
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
    axis_ranges:
        Declared value ranges of the instrument's moving axes; see
        :func:`component_ltotal_range`.
    """
    return {
        name: component_ltotal_range(
            nexus_filename, name, is_monitor=is_monitor, axis_ranges=axis_ranges
        )
        for names, is_monitor in ((detectors, False), (monitors, True))
        for name in names
    }
