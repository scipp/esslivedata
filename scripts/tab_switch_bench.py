#!/usr/bin/env python
"""Tab-switch benchmark of the reduction dashboard on a fake LOKI backend (#1401).

Starts the dashboard of a given checkout with ``--transport fake`` and every LOKI
workflow running on all sources, opens one browser session on an 11-cell grid
(9 bank images, a timeseries and a bar chart of the bank totals), and switches
between the grid and the static tabs. For every switch it records:

- Server, on the IOLoop: the tab-switch handler (which blocks every session),
  split into building the PATCH-DOC message, the data flush into the revealed
  cells and Bokeh's model-graph recompute; plus the slowest response of a static
  file polled every 20 ms during the switch window.
- Browser: main-thread busy time (CDP ``TaskDuration``) and long tasks during
  the window. Run headed on real hardware; headless Chromium draws in software.

The script is self-contained and does not need to live in the checkout it
measures, so copy it out before switching branches::

    git show origin/1401-tab-switch-bench:scripts/tab_switch_bench.py > ~/tsb.py

Per branch under test (``[test]`` is what registers the LOKI workflows)::

    git switch <branch> && pip install -e ".[test,dashboard]"
    python ~/tsb.py run --repo . --out <branch>.json

then ``python ~/tsb.py compare main.json <branch>.json``.

``run --trace DIR`` also records a Chrome trace (with the V8 sampling profiler)
of the baseline window and of every grid reveal into ``DIR``, and prints where
the browser main thread spent its time: per category (scripting, style, layout,
paint, ...), and the JavaScript functions with the most time inside tasks over
50 ms. On real hardware the 1 Hz data updates stay below 50 ms, so these tasks
are the reveal itself; compare with the baseline window to check. ``python
~/tsb.py trace DIR`` repeats that analysis, and each file in
``DIR`` opens in the Performance panel of Chrome DevTools. Tracing slows the
browser down, so do not compare the numbers of a traced run with untraced ones.

The driver needs
``pip install playwright && playwright install chromium`` in the environment
that runs this script (or ``PLAYWRIGHT_CHROMIUM_EXECUTABLE`` pointing at a
Chromium); ``--python`` selects a different environment for the dashboard.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

GRID = 'Banks'
SEQUENCE = ('Workflows', GRID, 'System Status', GRID, 'Manage Plots', GRID)

# Written to a temporary directory on the dashboard's PYTHONPATH. Active only in
# the dashboard process (TABBENCH=1). Patching happens synchronously at startup:
# a session built before the patch would keep the original bound methods.
SITECUSTOMIZE = r'''
import json, os, time
if os.environ.get('TABBENCH') == '1':
    import importlib
    pc, ct = time.perf_counter, time.thread_time
    _stack = []
    _env_sent = [False]

    def _emit(**ev):
        print('[tb] ' + json.dumps(ev), flush=True)

    def _outer(cls, name, kind, extra=None):
        orig = getattr(cls, name)

        def wrapper(self, *a, **k):
            if not _env_sent[0]:
                import bokeh
                from bokeh.core.has_props import HasProps
                _env_sent[0] = True
                _emit(kind='env', bokeh=bokeh.__version__,
                      query_impl=HasProps.query_properties_with_values.__module__)
            if _stack:
                return orig(self, *a, **k)
            acc = {}
            _stack.append(acc)
            t0, c0 = pc(), ct()
            try:
                return orig(self, *a, **k)
            finally:
                _stack.pop()
                ev = dict(kind=kind, t=time.time(), ms=(pc() - t0) * 1e3,
                          cpu=(ct() - c0) * 1e3)
                ev.update({k_: v * 1e3 for k_, v in acc.items()})
                if extra:
                    ev.update(extra(self, a, k))
                _emit(**ev)

        setattr(cls, name, wrapper)

    def _inner(cls, name, label):
        orig = getattr(cls, name)
        depth = [0]

        def wrapper(*a, **k):
            if not _stack or depth[0]:
                return orig(*a, **k)
            depth[0] += 1
            t0 = pc()
            try:
                return orig(*a, **k)
            finally:
                depth[0] -= 1
                _stack[-1][label] = _stack[-1].get(label, 0.0) + pc() - t0

        setattr(cls, name, wrapper)

    pgt = importlib.import_module('ess.livedata.dashboard.widgets.plot_grid_tabs')
    su = importlib.import_module('ess.livedata.dashboard.session_updater')
    import panel.layout.tabs as pt
    from bokeh.document.models import DocumentModelManager
    from bokeh.protocol import Protocol

    _outer(pgt._BatchedTabs, '_update_active', 'switch',
           lambda self, a, k: {'active': self.active})
    _outer(su.SessionUpdater, '_tick', 'tick',
           lambda self, a, k: {'full': k.get('full')})
    _inner(pt.Tabs, '_get_objects', 'build')
    _inner(DocumentModelManager, 'recompute', 'recompute')
    _inner(Protocol, 'create', 'patch_doc')
    _inner(pgt.PlotGridTabs, '_poll_for_plot_updates', 'flush')
'''

GEN_CONFIG = r'''
import sys
from pathlib import Path
import yaml
from ess.livedata.config import instrument_registry
from ess.livedata.config.instruments import get_config

out = Path(sys.argv[1]) / 'loki'
out.mkdir(parents=True, exist_ok=True)
get_config('loki')
workflows = instrument_registry['loki'].workflow_factory
configs = {
    str(wid): {
        'jobs': {s: {'aux_source_names': {}, 'params': {}} for s in spec.source_names}
    }
    for wid, spec in workflows.items()
}
(out / 'workflow_configs.yaml').write_text(yaml.safe_dump(configs))

banks = [f'loki_detector_{i}' for i in range(9)]

def layer(view, plot, sources):
    return {'data_sources': {'primary': {'workflow_id': 'loki/detector_xy_projection/1',
                                         'source_names': sources, 'view_name': view}},
            'plot_name': plot, 'params': {}}

cells = [
    {'geometry': {'row': i // 4, 'col': i % 4, 'row_span': 1, 'col_span': 1},
     'layers': [layer('current', 'image', [bank])]}
    for i, bank in enumerate(banks)
]
cells.append({'geometry': {'row': 2, 'col': 1, 'row_span': 1, 'col_span': 1},
              'layers': [layer('counts_total', 'timeseries', banks)]})
cells.append({'geometry': {'row': 2, 'col': 2, 'row_span': 1, 'col_span': 1},
              'layers': [layer('counts_total', 'bars', banks)]})
grid = {'title': 'Banks', 'nrows': 3, 'ncols': 4, 'cells': cells}
plots = {'plot_grids': {'grids': [grid]}}
(out / 'plot_configs.yaml').write_text(yaml.safe_dump(plots))
'''

LONGTASK_OBSERVER = '''
window.__longtasks = [];
new PerformanceObserver((list) => {
  for (const e of list.getEntries()) window.__longtasks.push([e.startTime, e.duration]);
}).observe({type: 'longtask', buffered: true});
'''

TRACE_CATEGORIES = [
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
TRACE_GROUPS = ('scripting', 'style', 'layout', 'paint', 'composite', 'gc', 'other')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_until_ready(url: str, proc: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f'Dashboard exited early:\n{log.read_text()[-3000:]}')
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise SystemExit(f'Dashboard not ready after 180 s:\n{log.read_text()[-3000:]}')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ['git', '-C', str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def run(args: argparse.Namespace) -> None:
    from playwright.sync_api import sync_playwright

    repo = args.repo.resolve()
    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='tab-switch-bench-'))
    (tmp / 'site').mkdir()
    (tmp / 'site' / 'sitecustomize.py').write_text(SITECUSTOMIZE)
    pythonpath = os.pathsep.join([str(tmp / 'site'), str(repo / 'src')])
    subprocess.run(  # noqa: S603
        [args.python, '-c', GEN_CONFIG, str(tmp / 'cfg')],
        env={**os.environ, 'PYTHONPATH': str(repo / 'src')},
        check=True,
    )

    port = _free_port()
    url = f'http://localhost:{port}'
    log = tmp / 'dashboard.log'
    proc = subprocess.Popen(  # noqa: S603
        [
            args.python,
            '-u',
            '-m',
            'ess.livedata.dashboard.reduction',
            '--instrument',
            'loki',
            '--transport',
            'fake',
            '--port',
            str(port),
            '--config-dir',
            str(tmp / 'cfg'),
            '--auto-start',
            '--no-fetch-announcements',
        ],
        cwd=repo,
        env={**os.environ, 'TABBENCH': '1', 'PYTHONPATH': pythonpath},
        stdout=log.open('w'),
        stderr=subprocess.STDOUT,
    )
    probe: list[tuple[float, float]] = []
    stop = threading.Event()

    def poll_static(path: str) -> None:
        while not stop.is_set():
            t0, p0 = time.time(), time.perf_counter()
            try:
                urllib.request.urlopen(url + path, timeout=30).read()  # noqa: S310
            except OSError:
                pass
            probe.append((t0, time.perf_counter() - p0))
            time.sleep(0.02)

    marks = []
    try:
        _wait_until_ready(url, proc, log)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=args.headless,
                channel=args.channel,
                executable_path=os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'),
            )
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            page.add_init_script(LONGTASK_OBSERVER)
            page.goto(url)
            page.locator('.bk-tab', has_text=GRID).first.wait_for(timeout=120_000)
            cdp = page.context.new_cdp_session(page)
            cdp.send('Performance.enable')

            def busy_s() -> float:
                metrics = cdp.send('Performance.getMetrics')['metrics']
                return next(m['value'] for m in metrics if m['name'] == 'TaskDuration')

            def window(label: str, action, trace: str | None = None) -> None:
                if trace and args.trace:
                    browser.start_tracing(page=page, categories=TRACE_CATEGORIES)
                b0, t0 = busy_s(), time.time()
                p0 = page.evaluate('performance.now()')
                action()
                time.sleep(args.window)
                tasks = [
                    d
                    for s, d in page.evaluate('window.__longtasks')
                    if p0 <= s < p0 + args.window * 1e3
                ]
                marks.append(
                    {
                        'label': label,
                        't': t0,
                        't_end': time.time(),
                        'browser_busy_ms': (busy_s() - b0) * 1e3,
                        'longtask_ms': sum(tasks),
                        'longtask_max_ms': max(tasks, default=0),
                    }
                )
                if trace and args.trace:
                    data = browser.stop_tracing()
                    (args.trace / f'{trace}.json.gz').write_bytes(gzip.compress(data))

            def click(name: str) -> None:
                page.locator('.bk-tab', has_text=name).first.click()

            window(f'Workflows -> {GRID} (first)', lambda: click(GRID), 'reveal-first')
            time.sleep(args.settle)
            asset = min(
                (
                    n
                    for n, size in page.evaluate(
                        "performance.getEntriesByType('resource')"
                        ".map(e => [e.name, e.transferSize])"
                    )
                    if '/static/' in n and size > 0
                ),
                key=len,
            ).split(str(port), 1)[1]
            threading.Thread(target=poll_static, args=(asset,), daemon=True).start()
            window(f'baseline ({GRID})', lambda: None, 'baseline')
            current = GRID
            reveals = 0
            for _ in range(args.reps):
                for name in SEQUENCE:
                    trace = None
                    if name == GRID:
                        reveals += 1
                        trace = f'reveal-{reveals:02d}'
                    window(f'{current} -> {name}', lambda name=name: click(name), trace)
                    current = name
            browser.close()
    finally:
        stop.set()
        proc.terminate()
        proc.wait(timeout=30)

    events = [
        json.loads(line[5:])
        for line in log.read_text().splitlines()
        if line.startswith('[tb] ')
    ]
    env = next((e for e in events if e['kind'] == 'env'), {})
    result = {
        'repo': str(repo),
        'branch': _git(repo, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'commit': _git(repo, 'rev-parse', '--short', 'HEAD'),
        'bokeh': env.get('bokeh'),
        'query_impl': env.get('query_impl'),
        'headless': args.headless,
        'traced': args.trace is not None,
        'marks': marks,
        'events': events,
        'probe': probe,
    }
    args.out.write_text(json.dumps(result))
    print(f'wrote {args.out} (server log: {log})')
    summarize(result, args.out.name)
    if args.trace:
        analyze_traces(args.trace)


def _rows(result: dict) -> dict[str, dict[str, float]]:
    """Median per transition of the per-window metrics."""
    per_label: dict[str, list[dict[str, float]]] = {}
    for m in result['marks']:
        evs = [
            e
            for e in result['events']
            if e['kind'] in ('switch', 'tick') and m['t'] <= e['t'] < m['t_end']
        ]
        switch = [e for e in evs if e['kind'] == 'switch']
        ticks = [e for e in evs if e['kind'] == 'tick']
        lat = [d for t, d in result['probe'] if m['t'] <= t < m['t_end']]
        per_label.setdefault(m['label'], []).append(
            {
                'block': sum(e['ms'] for e in switch),
                'patch_doc': sum(e.get('patch_doc', 0) for e in switch),
                'flush': sum(e.get('flush', 0) for e in switch),
                'recompute': sum(e.get('recompute', 0) for e in switch),
                'ticks': sum(e['ms'] for e in ticks),
                'probe_max': max(lat, default=0) * 1e3,
                'browser': m['browser_busy_ms'],
                'longtasks': m['longtask_ms'],
                'longest': m['longtask_max_ms'],
            }
        )
    return {
        label: {k: statistics.median(r[k] for r in rs) for k in rs[0]} | {'n': len(rs)}
        for label, rs in per_label.items()
    }


COLUMNS = (
    ('block', 'server block'),
    ('patch_doc', 'PATCH-DOC'),
    ('flush', 'flush'),
    ('recompute', 'recompute'),
    ('ticks', 'ticks'),
    ('probe_max', 'probe max'),
    ('browser', 'browser busy'),
    ('longtasks', 'long tasks'),
    ('longest', 'longest'),
)


def summarize(result: dict, name: str) -> None:
    print(
        f"\n{name}: {result['branch']} @ {result['commit']}, bokeh {result['bokeh']}, "
        f"serialization by {result['query_impl']}"
        f"{' (headless)' if result['headless'] else ''}"
        f"{' (traced)' if result.get('traced') else ''}"
    )
    print('medians in ms over each window; n = repetitions')
    print(f"{'transition':34s}{'n':>3s}" + ''.join(f'{h:>14s}' for _, h in COLUMNS))
    for label, row in _rows(result).items():
        print(
            f"{label:34s}{row['n']:>3d}"
            + ''.join(f'{row[k]:14.0f}' for k, _ in COLUMNS)
        )


def compare(args: argparse.Namespace) -> None:
    for path in args.results:
        summarize(json.loads(path.read_text()), path.name)


def _event_category(name: str) -> str | None:
    if name.startswith(GC_PREFIXES):
        return 'gc'
    return EVENT_CATEGORY.get(name)


def _load_trace(path: Path) -> list[dict]:
    data = json.loads(gzip.decompress(path.read_bytes()))
    return data['traceEvents'] if isinstance(data, dict) else data


def _main_thread(events: list[dict]) -> tuple[int, int]:
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


def _thread_breakdown(
    events: list[dict], thread: tuple[int, int]
) -> tuple[dict[str, dict[str, float]], list[tuple[float, float]]]:
    """Self time per category on one thread, and the long tasks.

    Returns ``{'all' | 'long' | 'forced': {category: us}}``: all tasks, tasks over
    50 ms, and style/layout run synchronously from JavaScript (e.g. by
    ``getComputedStyle``). The second item lists the long tasks as (start, end).
    """
    spans = sorted(
        (
            (e['ts'], e['ts'] + e.get('dur', 0), e['name'])
            for e in events
            if (e.get('pid'), e.get('tid')) == thread and e.get('ph') == 'X'
        ),
        key=lambda s: (s[0], -s[1]),
    )
    totals = {k: dict.fromkeys(TRACE_GROUPS, 0.0) for k in ('all', 'long', 'forced')}
    long_tasks: list[tuple[float, float]] = []
    # Stack entries: [end, category, child time, is long task, under scripting].
    stack: list[list] = []

    def close(entry: list, start: float) -> None:
        end, category, children, long_task, under_js = entry
        self_us = end - start - children
        totals['all'][category] += self_us
        if long_task:
            totals['long'][category] += self_us
        if under_js and category in ('style', 'layout'):
            totals['forced'][category] += self_us

    starts: list[float] = []
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
        stack.append([end, category, 0.0, long_task, under_js])
        starts.append(start)
    while stack:
        close(stack.pop(), starts.pop())
    return totals, long_tasks


def _frame_key(frame: dict) -> str:
    name = frame.get('functionName') or '(anonymous)'
    url = frame.get('url', '').rsplit('/', 1)[-1].split('?')[0]
    if not url:
        return name
    return f"{name} {url}:{frame['lineNumber'] + 1}:{frame['columnNumber'] + 1}"


def _profile_times(
    events: list[dict], thread: tuple[int, int], long_tasks: list[tuple[float, float]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Self and inclusive sampled time (us) per JS function inside long tasks."""
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
    starts = [s for s, _ in long_tasks]
    times, ts = [], prof['start']
    for _, delta in prof['samples']:
        ts += delta
        times.append(ts)
    for i, (node_id, _) in enumerate(prof['samples'][:-1]):
        ts, weight = times[i], times[i + 1] - times[i]
        k = bisect.bisect_right(starts, ts) - 1
        if k < 0 or ts >= long_tasks[k][1]:
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


def analyze_traces(directory: Path) -> None:
    """Print the main-thread breakdown of the traces written by ``run --trace``."""
    kinds = {'baseline': [], 'reveal-first': [], 'reveal': []}
    for path in sorted(directory.glob('*.json.gz')):
        stem = path.name.removesuffix('.json.gz')
        kinds['reveal' if stem.removeprefix('reveal-').isdigit() else stem].append(path)
    functions: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    print(f'\n{directory}: browser main thread, ms per window (mean)')
    header = f"{'':22s}" + ''.join(f'{g:>11s}' for g in (*TRACE_GROUPS, 'total'))
    for kind, paths in kinds.items():
        if not paths:
            continue
        sums = {k: dict.fromkeys(TRACE_GROUPS, 0.0) for k in ('all', 'long', 'forced')}
        self_sum: dict[str, float] = {}
        incl_sum: dict[str, float] = {}
        for path in paths:
            events = _load_trace(path)
            thread = _main_thread(events)
            totals, long_tasks = _thread_breakdown(events, thread)
            for part, groups in totals.items():
                for g, us in groups.items():
                    sums[part][g] += us
            s, i = _profile_times(events, thread, long_tasks)
            for src, dst in ((s, self_sum), (i, incl_sum)):
                for key, us in src.items():
                    dst[key] = dst.get(key, 0.0) + us
        n = len(paths)
        functions[kind] = (
            {k: v / n for k, v in self_sum.items()},
            {k: v / n for k, v in incl_sum.items()},
        )
        print(f'\n{kind} (n={n})')
        print(header)
        for part, label in (
            ('all', 'all tasks'),
            ('long', 'tasks over 50 ms'),
            ('forced', 'forced by JS'),
        ):
            row = [sums[part][g] / n / 1e3 for g in TRACE_GROUPS]
            print(f'{label:22s}' + ''.join(f'{v:11.0f}' for v in (*row, sum(row))))
    for kind in ('reveal', 'reveal-first', 'baseline'):
        if kind not in functions:
            continue
        self_ms, incl_ms = functions[kind]
        for title, table in (('self', self_ms), ('inclusive', incl_ms)):
            print(f'\n{kind}: JS functions inside tasks over 50 ms, {title} ms')
            for key, us in sorted(table.items(), key=lambda kv: -kv[1])[:25]:
                print(f'{us / 1e3:8.1f}  {key}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(required=True)
    p_run = sub.add_parser('run', help='Run the benchmark against one checkout')
    p_run.add_argument('--repo', type=Path, required=True, help='Checkout to measure')
    p_run.add_argument('--out', type=Path, required=True, help='Result JSON')
    p_run.add_argument(
        '--python', default=sys.executable, help='Python of the dashboard environment'
    )
    p_run.add_argument('--reps', type=int, default=6, help='Cycles of SEQUENCE')
    p_run.add_argument('--window', type=float, default=8.0, help='Seconds per switch')
    p_run.add_argument(
        '--settle', type=float, default=25.0, help='Seconds before measuring'
    )
    p_run.add_argument('--headless', action='store_true')
    p_run.add_argument(
        '--channel', default=None, help="Playwright browser channel, e.g. 'chrome'"
    )
    p_run.add_argument(
        '--trace', type=Path, default=None, help='Directory for Chrome traces'
    )
    p_run.set_defaults(func=run)
    p_cmp = sub.add_parser('compare', help='Print the summaries of result files')
    p_cmp.add_argument('results', type=Path, nargs='+')
    p_cmp.set_defaults(func=compare)
    p_trace = sub.add_parser('trace', help='Analyze the traces of a run --trace')
    p_trace.add_argument('dir', type=Path)
    p_trace.set_defaults(func=lambda args: analyze_traces(args.dir))
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
