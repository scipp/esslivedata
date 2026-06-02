# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Heavyweight leak census for the dashboard custom-options store growth.

The gradual dashboard lag is driven by the process-global HoloViews
custom-options store (``Store._custom_options['bokeh']``) growing without bound:
its size is logged cheaply every minute as ``hv_custom_options`` (see
``dashboard_services._collect_diagnostics``). That number tells us the store
*grows* but not *why*. This module answers the why in a single pass over the
live object graph, emitted as ``dashboard_leak_census`` at a slow cadence.

The decisive field is the pair ``store_n`` vs ``live_with_id``:

* ``live_with_id`` ≈ ``store_n`` — the customized HoloViews elements are still
  alive and *held* by something; the leak is genuine object retention. Chase
  ``owners`` to find the retainer.
* ``live_with_id`` ≪ ``store_n`` — the elements are dead but their ids were
  never popped from the store; the weakref cleanup is not firing. A different
  bug entirely, and ``owners`` would be a dead end.

The pass also censuses live object counts by suspect type and library bucket;
whichever count climbs monotonically over hours localizes the leak (e.g. to our
own ``session_layer``/presenter objects, to leaked ``DynamicMap``/``Pipe``
instances, or to retained Bokeh models). Finally it walks the GC referrer graph
up from a few leaked elements to name the long-lived owners holding them alive.

Everything here runs in the freeze-watchdog daemon thread, holds the GIL for the
duration of the scan (~1-2 s), and is gated to a slow cadence. It must never
raise into the caller; :func:`collect_leak_census` returns partial results on
any error.
"""

from __future__ import annotations

import gc
from collections import Counter

# Suspect types whose live instance count we track over time. A monotonic climb
# in any of these points at the retainer. Matched by class name only (cheap).
_SUSPECT_NAMES = frozenset(
    {
        # HoloViews stream/space/options machinery
        'Pipe',
        'Buffer',
        'DynamicMap',
        'OptionTree',
        # our per-session rendering objects
        'SessionComponents',
        'SessionLayer',
        'DefaultPresenter',
        'StaticPresenter',
        'PassthroughPresenter',
        'LayerStateMachine',
        'DataSubscriber',
    }
)

# When walking referrers, these are the "named owners" we stop at: an instance
# from our own code, anything in HoloViews stream/space machinery, or any Bokeh
# model. Containers (list/dict/...) and frames are walked through, not reported.
_HOLDER_NAMES = frozenset({'Pipe', 'Buffer', 'DynamicMap', 'OptionTree'})
_SKIP_TYPE_NAMES = frozenset(
    {'frame', 'traceback', 'function', 'cell', 'method', 'builtin_function_or_method'}
)


def _rss_mb() -> float:
    """Resident set size in MiB, or 0.0 if unavailable."""
    try:
        with open('/proc/self/statm') as f:
            resident_pages = int(f.read().split()[1])
        import os

        return resident_pages * os.sysconf('SC_PAGE_SIZE') / (1024 * 1024)
    except Exception:
        return 0.0


def _is_named_owner(rtype: type, module: str) -> bool:
    return (
        module.startswith('ess.livedata')
        or module.startswith('bokeh')
        or rtype.__name__ in _HOLDER_NAMES
    )


def _named_owners(
    obj: object, *, max_depth: int, node_budget: int, max_owners: int
) -> dict[str, int]:
    """Walk the referrer graph up from ``obj``; return ``{owner_label: depth}``.

    Breadth-first over ``gc.get_referrers``, skipping frames and our own scan
    containers. Records every named owner (our classes / Bokeh models /
    HoloViews holders) with the depth it was first seen, and keeps traversing
    *through* it toward the roots — so the full ownership ladder is reported
    (e.g. ``Curve -> Pipe@4 -> DynamicMap@5 -> SessionComponents@6``), not just
    the nearest holder. Bounded by ``max_depth``, ``node_budget`` total visited
    nodes, and ``max_owners`` distinct finds.
    """
    found: dict[str, int] = {}
    seen: set[int] = {id(obj)}
    frontier: list[object] = [obj]
    # Our own scan containers refer to the objects we inspect; never report them.
    internal = {id(found), id(seen), id(frontier)}
    for depth in range(1, max_depth + 1):
        nxt: list[object] = []
        internal.add(id(nxt))
        for node in frontier:
            for ref in gc.get_referrers(node):
                rid = id(ref)
                if rid in seen or rid in internal:
                    continue
                seen.add(rid)
                rtype = type(ref)
                rname = rtype.__name__
                if rname in _SKIP_TYPE_NAMES:
                    continue
                module = getattr(rtype, '__module__', '')
                if not isinstance(module, str):
                    module = ''
                if _is_named_owner(rtype, module):
                    found.setdefault(f'{rname}@{depth}', depth)
                    if len(found) >= max_owners:
                        return found
                # Traverse upward regardless, to reach the long-lived roots that
                # hold the proximate holders (Pipe/DynamicMap) alive.
                nxt.append(ref)
                if len(seen) >= node_budget:
                    return found
        frontier = nxt
        if not frontier:
            break
    return found


def _fmt_counter(counter: Counter, top: int) -> str:
    """Compact, greppable ``a:3,b:1`` rendering of the most common entries."""
    return ','.join(f'{name}:{count}' for name, count in counter.most_common(top))


def collect_leak_census(
    *,
    sample: int = 3,
    max_depth: int = 8,
    node_budget: int = 600,
    max_owners: int = 8,
) -> dict[str, object]:
    """Single-pass census of the live object graph for the store-growth leak.

    Parameters
    ----------
    sample:
        Number of leaked customized elements to trace referrers for.
    max_depth:
        Maximum referrer-graph depth when searching for named owners.
    node_budget:
        Maximum nodes visited per element during the referrer walk.
    max_owners:
        Maximum distinct named owners reported per element.

    Returns
    -------
    :
        Flat mapping of compact, greppable diagnostics. Partial on error.
    """
    try:
        from holoviews.core.dimension import LabelledData
        from holoviews.core.options import Store

        store = Store._custom_options.get('bokeh', {})
        store_ids = set(store.keys())

        leaked_by_class: Counter = Counter()
        live_types: Counter = Counter()
        bokeh_n = hv_n = dashboard_n = total_n = 0
        samples: list[object] = []

        objects = gc.get_objects()
        for obj in objects:
            total_n += 1
            otype = type(obj)
            # Some C extension types expose ``__module__``/``__name__`` as a
            # descriptor rather than a str; coerce defensively (this loop walks
            # every live object, including exotic ones).
            module = otype.__module__
            if not isinstance(module, str):
                continue
            name = otype.__name__
            if not isinstance(name, str):
                continue
            if name in _SUSPECT_NAMES:
                live_types[name] += 1
            if module.startswith('bokeh'):
                bokeh_n += 1
            elif module.startswith('holoviews'):
                hv_n += 1
                # isinstance is C-level (no __getattr__ side effects, unlike a
                # duck-typed getattr that would trip HoloViews lazy-module
                # proxies); a real LabelledData has a plain integer ``id``.
                if isinstance(obj, LabelledData):
                    oid = obj.id
                    if type(oid) is int and oid in store_ids:
                        leaked_by_class[name] += 1
                        if len(samples) < sample:
                            samples.append(obj)
            elif module.startswith('ess.livedata'):
                dashboard_n += 1
        del objects

        owners = [
            _named_owners(
                obj,
                max_depth=max_depth,
                node_budget=node_budget,
                max_owners=max_owners,
            )
            for obj in samples
        ]
        owners_str = ' ; '.join(
            f'{type(obj).__name__}<-{",".join(found) or "?"}'
            for obj, found in zip(samples, owners, strict=True)
        )
        del samples

        counts = gc.get_count()
        return {
            'store_n': len(store_ids),
            'live_with_id': sum(leaked_by_class.values()),
            'leaked_by_class': _fmt_counter(leaked_by_class, top=8),
            'live_types': _fmt_counter(live_types, top=len(_SUSPECT_NAMES)),
            'owners': owners_str,
            'bokeh_objects': bokeh_n,
            'hv_objects': hv_n,
            'dashboard_objects': dashboard_n,
            'total_objects': total_n,
            'gc_garbage': len(gc.garbage),
            'gc_count': f'{counts[0]},{counts[1]},{counts[2]}',
            'id_min': min(store_ids) if store_ids else 0,
            'id_max': max(store_ids) if store_ids else 0,
            'rss_mb': round(_rss_mb(), 1),
        }
    except Exception as exc:  # never raise into the watchdog loop
        return {'census_error': f'{type(exc).__name__}: {exc}'}
