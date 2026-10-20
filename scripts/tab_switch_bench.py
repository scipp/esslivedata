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

then ``python ~/tsb.py compare main.json <branch>.json``. The driver needs
``pip install playwright && playwright install chromium`` in the environment
that runs this script (or ``PLAYWRIGHT_CHROMIUM_EXECUTABLE`` pointing at a
Chromium); ``--python`` selects a different environment for the dashboard.
"""

from __future__ import annotations

import argparse
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

            def window(label: str, action) -> None:
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

            def click(name: str) -> None:
                page.locator('.bk-tab', has_text=name).first.click()

            window(f'Workflows -> {GRID} (first)', lambda: click(GRID))
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
            window(f'baseline ({GRID})', lambda: None)
            current = GRID
            for _ in range(args.reps):
                for name in SEQUENCE:
                    window(f'{current} -> {name}', lambda name=name: click(name))
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
        'marks': marks,
        'events': events,
        'probe': probe,
    }
    args.out.write_text(json.dumps(result))
    print(f'wrote {args.out} (server log: {log})')
    summarize(result, args.out.name)


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
    p_run.set_defaults(func=run)
    p_cmp = sub.add_parser('compare', help='Print the summaries of result files')
    p_cmp.add_argument('results', type=Path, nargs='+')
    p_cmp.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
