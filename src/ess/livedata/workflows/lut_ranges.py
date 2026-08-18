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

from collections.abc import Iterable

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import Filename, NeXusName, SampleRun
from ess.reduce.unwrap import GenericUnwrapWorkflow
from ess.reduce.unwrap.types import DetectorLtotal, MonitorLtotal

#: Fraction of a component's own span added at each end, plus a floor, so that a
#: component whose pixels all sit at one distance (every monitor, and a detector
#: bank modelled as a point) still gets a table with usable width rather than a
#: degenerate one.
_RELATIVE_PAD = 0.01
_MINIMUM_PAD = sc.scalar(0.1, unit='m')


class LtotalRangeError(ValueError):
    """Raised when a component's flight-path range cannot be derived."""


def _ltotal(nexus_filename: str, component: str, *, is_monitor: bool) -> sc.Variable:
    """Compute a component's ``Ltotal`` from the geometry artifact."""
    workflow = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[snx.NXmonitor]
    )
    workflow[Filename[SampleRun]] = nexus_filename
    if is_monitor:
        workflow[NeXusName[snx.NXmonitor]] = component
        return workflow.compute(MonitorLtotal[SampleRun, snx.NXmonitor])
    workflow[NeXusName[snx.NXdetector]] = component
    return workflow.compute(DetectorLtotal[SampleRun])


def component_ltotal_range(
    nexus_filename: str,
    component: str,
    *,
    is_monitor: bool,
    travel: sc.Variable | None = None,
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
    travel:
        Travel envelope of the axis this component rides, added downstream of
        the nominal position. ``None`` for a static component.

    Returns
    -------
    :
        ``(start, stop)`` in metres, padded.

    Raises
    ------
    LtotalRangeError:
        If the component's position cannot be resolved from the artifact. The
        common cause is a ``depends_on`` chain with a live (f144-driven)
        transform, whose nominal value the artifact does not carry.
    """
    try:
        ltotal = _ltotal(nexus_filename, component, is_monitor=is_monitor).to(unit='m')
    except Exception as exc:
        raise LtotalRangeError(
            f"Cannot derive the flight-path range of {component!r} from "
            f"{nexus_filename}: {exc}"
        ) from exc
    start, stop = ltotal.min(), ltotal.max()
    pad = sc.max(sc.concat([(stop - start) * _RELATIVE_PAD, _MINIMUM_PAD], 'pad'))
    stop = stop + pad
    if travel is not None:
        stop = stop + travel.to(unit='m')
    return start - pad, stop


def component_ltotal_ranges(
    nexus_filename: str,
    *,
    detectors: Iterable[str],
    monitors: Iterable[str],
    travel: dict[str, sc.Variable] | None = None,
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
    travel:
        Per-component travel envelopes; see :func:`component_ltotal_range`.
    """
    travel = travel or {}
    return {
        name: component_ltotal_range(
            nexus_filename,
            name,
            is_monitor=is_monitor,
            travel=travel.get(name),
        )
        for names, is_monitor in ((detectors, False), (monitors, True))
        for name in names
    }
