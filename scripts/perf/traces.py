"""Where the dashboard page's browser main thread spent its time, from Chrome traces.

Each trace file opens in the Performance panel of Chrome DevTools. Paint and
composite times are meaningless under software rendering (headless Chromium,
CI runners, the devcontainer); scripting, style and layout are not affected.
"""

from __future__ import annotations

import bisect
import gzip
import json
from pathlib import Path

CATEGORIES = [
    'toplevel',
    'devtools.timeline',
    'disabled-by-default-devtools.timeline',
    'v8.execute',
    'disabled-by-default-v8.cpu_profiler',
    'blink.user_timing',
]
LONG_TASK_US = 50_000

# Trace event name -> category. An unlisted event inherits the category of the
# nearest listed ancestor, as in the DevTools summary.
EVENT_CATEGORY = {
    'RunTask': 'other',
    'ThreadControllerImpl::RunTask': 'other',
    **dict.fromkeys(
        (
            'FunctionCall',
            'EvaluateScript',
            'TimerFire',
            'EventDispatch',
            'FireAnimationFrame',
            'FireIdleCallback',
            'RunMicrotasks',
            'v8.run',
            'v8.callFunction',
            'v8.compile',
            'v8.compileModule',
            'V8.CompileCode',
            'V8.Execute',
        ),
        'scripting',
    ),
    **dict.fromkeys(
        ('UpdateLayoutTree', 'RecalculateStyles', 'ParseAuthorStyleSheet'), 'style'
    ),
    'Layout': 'layout',
    **dict.fromkeys(('PrePaint', 'Paint', 'PaintImage', 'Decode Image'), 'paint'),
    **dict.fromkeys(
        ('UpdateLayerTree', 'Layerize', 'CompositeLayers', 'Commit'), 'composite'
    ),
}
GC_PREFIXES = ('MinorGC', 'MajorGC', 'V8.GC', 'BlinkGC', 'CppGC')
GROUPS = ('scripting', 'style', 'layout', 'paint', 'composite', 'gc', 'other')
# Events that recalculate style or lay out; counted separately when JavaScript
# forced them synchronously (e.g. by ``getComputedStyle``) rather than the
# browser running them before a frame.
STYLE_EVENTS = ('UpdateLayoutTree', 'RecalculateStyles')
LAYOUT_EVENTS = ('Layout',)


def _event_category(name: str) -> str | None:
    if name.startswith(GC_PREFIXES):
        return 'gc'
    return EVENT_CATEGORY.get(name)


def load(path: Path) -> list[dict]:
    data = json.loads(gzip.decompress(path.read_bytes()))
    return data['traceEvents'] if isinstance(data, dict) else data


def main_thread(events: list[dict]) -> tuple[int, int]:
    """(pid, tid) of the renderer main thread that ran the most task time."""
    mains = {
        (e['pid'], e['tid'])
        for e in events
        if e.get('ph') == 'M'
        and e.get('name') == 'thread_name'
        and e['args'].get('name') == 'CrRendererMain'
    }
    busy = dict.fromkeys(mains, 0.0)
    for e in events:
        key = (e.get('pid'), e.get('tid'))
        if key in busy and e.get('ph') == 'X' and e['name'] == 'RunTask':
            busy[key] += e.get('dur', 0)
    return max(busy, key=busy.get)


def breakdown(events: list[dict], thread: tuple[int, int]) -> dict:
    """Self time per category on one thread, event counts, and the long tasks.

    Returns a dict with:

    - ``us``: ``{'all' | 'long' | 'forced': {category: microseconds}}``: all
      tasks, tasks over 50 ms, and style/layout run synchronously from
      JavaScript;
    - ``counts``: number of style and layout events, and how many of them were
      forced from JavaScript;
    - ``long_tasks``: (start, end) of every task over 50 ms.
    """
    spans = sorted(
        (
            (e['ts'], e['ts'] + e.get('dur', 0), e['name'])
            for e in events
            if (e.get('pid'), e.get('tid')) == thread and e.get('ph') == 'X'
        ),
        key=lambda s: (s[0], -s[1]),
    )
    us = {k: dict.fromkeys(GROUPS, 0.0) for k in ('all', 'long', 'forced')}
    counts = dict.fromkeys(('style', 'layout', 'forced_style', 'forced_layout'), 0)
    long_tasks: list[tuple[float, float]] = []
    # Stack entries: [end, category, child time, is long task, under scripting].
    stack: list[list] = []
    starts: list[float] = []

    def close(entry: list, start: float) -> None:
        end, category, children, long_task, under_js = entry
        self_us = end - start - children
        us['all'][category] += self_us
        if long_task:
            us['long'][category] += self_us
        if under_js and category in ('style', 'layout'):
            us['forced'][category] += self_us

    for start, end, name in spans:
        while stack and stack[-1][0] <= start:
            close(stack.pop(), starts.pop())
        parent = stack[-1] if stack else None
        if parent is None and end - start <= 0:
            continue
        category = _event_category(name) or (parent[1] if parent else 'other')
        if parent is not None:
            end = min(end, parent[0])
            parent[2] += end - start
            long_task = parent[3]
            under_js = parent[4] or parent[1] == 'scripting'
        else:
            long_task = end - start > LONG_TASK_US
            if long_task:
                long_tasks.append((start, end))
            under_js = False
        for kind, names in (('style', STYLE_EVENTS), ('layout', LAYOUT_EVENTS)):
            if name in names:
                counts[kind] += 1
                counts[f'forced_{kind}'] += under_js
        stack.append([end, category, 0.0, long_task, under_js])
        starts.append(start)
    while stack:
        close(stack.pop(), starts.pop())
    return {'us': us, 'counts': counts, 'long_tasks': long_tasks}


def _frame_key(frame: dict) -> str:
    name = frame.get('functionName') or '(anonymous)'
    url = frame.get('url', '').rsplit('/', 1)[-1].split('?')[0]
    if not url:
        return name
    return f"{name} {url}:{frame['lineNumber'] + 1}:{frame['columnNumber'] + 1}"


def function_times(
    events: list[dict],
    thread: tuple[int, int],
    spans: list[tuple[float, float]] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Self and inclusive sampled time (us) per JS function.

    Only samples inside ``spans`` (e.g. the long tasks) count; ``None`` counts
    every sample.
    """
    profiles: dict[str, dict] = {}
    for e in events:
        if e.get('pid') != thread[0] or e.get('ph') != 'P':
            continue
        data = e.get('args', {}).get('data', {})
        prof = profiles.setdefault(
            e['id'], {'tid': None, 'start': None, 'nodes': {}, 'samples': []}
        )
        if e['name'] == 'Profile':
            prof['tid'], prof['start'] = e['tid'], data['startTime']
        elif e['name'] == 'ProfileChunk':
            cpu = data.get('cpuProfile', {})
            for node in cpu.get('nodes', []):
                prof['nodes'][node['id']] = node
            prof['samples'].extend(
                zip(cpu.get('samples', []), data.get('timeDeltas', []), strict=False)
            )
    prof = next(
        (p for p in profiles.values() if p['tid'] == thread[1]),
        max(profiles.values(), key=lambda p: len(p['samples']), default=None),
    )
    self_us: dict[str, float] = {}
    incl_us: dict[str, float] = {}
    if prof is None or prof['start'] is None:
        return self_us, incl_us
    spans = spans if spans is not None else [(float('-inf'), float('inf'))]
    span_starts = [s for s, _ in spans]
    times, ts = [], prof['start']
    for _, delta in prof['samples']:
        ts += delta
        times.append(ts)
    for i, (node_id, _) in enumerate(prof['samples'][:-1]):
        ts, weight = times[i], times[i + 1] - times[i]
        k = bisect.bisect_right(span_starts, ts) - 1
        if k < 0 or ts >= spans[k][1]:
            continue
        node = prof['nodes'].get(node_id)
        if node is None:
            continue
        key = _frame_key(node['callFrame'])
        self_us[key] = self_us.get(key, 0.0) + weight
        seen = set()
        while node is not None:
            key = _frame_key(node['callFrame'])
            if key not in seen:
                seen.add(key)
                incl_us[key] = incl_us.get(key, 0.0) + weight
            node = prof['nodes'].get(node.get('parent'))
    return self_us, incl_us


def summarize(path: Path) -> dict:
    """Main-thread milliseconds per category and style/layout event counts."""
    events = load(path)
    result = breakdown(events, main_thread(events))
    out = {f'ms_{g}': v / 1e3 for g, v in result['us']['all'].items()}
    out['ms_forced'] = (
        result['us']['forced']['style'] + result['us']['forced']['layout']
    ) / 1e3
    out['ms_long_tasks'] = sum(e - s for s, e in result['long_tasks']) / 1e3
    return out | result['counts']


def print_analysis(directory: Path, top: int = 25) -> None:
    """Print the main-thread breakdown of every trace in ``directory``."""
    for path in sorted(directory.glob('*.json.gz')):
        events = load(path)
        thread = main_thread(events)
        result = breakdown(events, thread)
        print(f'\n{path.name}: browser main thread, ms')
        print(f"{'':18s}" + ''.join(f'{g:>11s}' for g in (*GROUPS, 'total')))
        for part, label in (
            ('all', 'all tasks'),
            ('long', 'tasks over 50 ms'),
            ('forced', 'forced by JS'),
        ):
            row = [result['us'][part][g] / 1e3 for g in GROUPS]
            print(f'{label:18s}' + ''.join(f'{v:11.0f}' for v in (*row, sum(row))))
        print('events:', result['counts'])
        self_us, _ = function_times(events, thread)
        print(f'JS functions by self time (top {top}), ms:')
        for key, value in sorted(self_us.items(), key=lambda kv: -kv[1])[:top]:
            print(f'{value / 1e3:8.1f}  {key}')
