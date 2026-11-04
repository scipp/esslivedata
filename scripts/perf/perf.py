#!/usr/bin/env python
"""Performance harness for the reduction dashboard (#1419).

Starts the dashboard of a checkout on the fake backend with the LOKI grid fixture
(9 bank images, a timeseries and a bar chart; ``tests/dashboard/ui_config_fixtures/
loki``), drives it with Playwright and records what the browser and the server do.

Most comparisons rely on counts, not times: calls that read layout or computed
style, forced style recalculations and layouts, ``PATCH-DOC`` messages, and
document events by attribute. On the fake backend (fixed 1 Hz) these repeat
within a few percent from run to run, while the regressions they are meant to
catch change them by 2x or more. Times are reported, but noisy on shared
machines and meaningless for paint under software rendering.

Scenarios, all in one dashboard process per run:

- ``steady``: one session on the grid, no interaction (untraced windows, then
  one traced window);
- ``reveal``: leave the grid for the Workflows tab and come back;
- ``sessions``: more sessions open the grid, the measured one is watched and
  then re-reveals it. Every session must still show all its plots.

Subcommands::

    # one environment; run with that environment's Python
    python scripts/perf/perf.py run --repo . --label main --out run.json

    # alternate several environments (any Python works; it starts ``run``)
    python scripts/perf/perf.py ab --out perf/results.json \\
        --env base=venv-base/bin/python:base --env cand=venv-cand/bin/python:cand

    python scripts/perf/perf.py report perf/results.json   # markdown summary
    python scripts/perf/perf.py trace perf/traces          # trace breakdown

Each environment needs the checkout's test requirements and ``playwright``
(the same version in all environments, so they use the same Chromium). Set
``PLAYWRIGHT_CHROMIUM_EXECUTABLE`` to use another Chromium binary. The fixture
comes from this harness's checkout, so every environment runs the same scenario.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import traces
from probes import CANVAS_COUNT_JS, COUNTERS_JS, SITECUSTOMIZE

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent.parent / 'tests' / 'dashboard' / 'ui_config_fixtures' / 'loki'
GRID = 'Banks'
VIEWPORT = {'width': 1600, 'height': 1000}
PACKAGES = ('bokeh', 'panel', 'holoviews', 'param')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_until_ready(url: str, proc: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'Dashboard exited early:\n{log.read_text()[-3000:]}')
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f'Dashboard not ready after 180 s:\n{log.read_text()[-3000:]}')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ['git', '-C', str(repo), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = 'missing'
    return out


def run(args: argparse.Namespace) -> int:
    """Measure one environment: the Python running this, the code in ``--repo``."""
    repo = args.repo.resolve()
    result: dict = {
        'label': args.label,
        'round': args.round,
        'repo': str(repo),
        'commit': _git(repo, 'rev-parse', '--short', 'HEAD'),
        'versions': _versions(),
        'windows': [],
        'canvas_checks': [],
        'page_errors': [],
    }
    tmp = Path(tempfile.mkdtemp(prefix='perf-'))
    log = tmp / 'dashboard.log'
    try:
        _run_scenarios(args, repo, tmp, log, result)
    except Exception as exc:
        # A failed start (e.g. a version gate) is a result, not a harness crash.
        result['error'] = f'{type(exc).__name__}: {exc}'[-3000:]
    if log.exists():
        text = log.read_text()
        before_teardown = text[: result.get('log_offset', len(text))]
        result['server_tracebacks'] = before_teardown.count(
            'Traceback (most recent call last)'
        )
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(log, args.log_dir / f'{args.label}-r{args.round}.log')
        events = [
            json.loads(line[7:]) for line in text.splitlines() if line[:7] == '[perf] '
        ]
        for w in result['windows']:
            w['server'] = _server_window(events, result.get('probe', []), w)
    result.pop('probe', None)
    args.out.write_text(json.dumps(result))
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if 'error' in result else 0


def _run_scenarios(
    args: argparse.Namespace, repo: Path, tmp: Path, log: Path, result: dict
) -> None:
    from playwright.sync_api import Page, sync_playwright

    (tmp / 'site').mkdir()
    (tmp / 'site' / 'sitecustomize.py').write_text(SITECUSTOMIZE)
    shutil.copytree(FIXTURE, tmp / 'cfg' / 'loki')
    port = _free_port()
    url = f'http://localhost:{port}'
    t_start = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            '-u',
            '-m',
            'ess.livedata.dashboard.reduction',
            *('--instrument', 'loki', '--transport', 'fake', '--port', str(port)),
            *('--config-dir', str(tmp / 'cfg')),
            '--auto-start',
            '--no-fetch-announcements',
        ],
        cwd=repo,
        env={
            **os.environ,
            'PERF_PROBES': '1',
            'PYTHONPATH': os.pathsep.join([str(tmp / 'site'), str(repo / 'src')]),
        },
        stdout=log.open('w'),
        stderr=subprocess.STDOUT,
    )
    stop = threading.Event()
    probe: list[tuple[float, float]] = []
    result['probe'] = probe
    try:
        _wait_until_ready(url, proc, log)
        result['startup_s'] = time.monotonic() - t_start
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=not args.headed,
                executable_path=os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'),
            )
            result['chromium'] = browser.version

            def open_page(errors_label: str) -> Page:
                page = browser.new_context(viewport=VIEWPORT).new_page()
                page.add_init_script(COUNTERS_JS)
                page.on(
                    'pageerror',
                    lambda e: result['page_errors'].append(
                        f'{errors_label}: {str(e)[:300]}'
                    ),
                )
                return page

            def show_grid(page: Page) -> int:
                page.locator('.bk-tab', has_text=GRID).first.click()
                return _wait_for_canvases(page)

            page = open_page('measured')
            t0 = time.monotonic()
            page.goto(url)
            page.locator('.bk-tab', has_text=GRID).first.wait_for(timeout=120_000)
            result['first_load_s'] = time.monotonic() - t0
            reference = show_grid(page)
            result['canvases'] = reference
            time.sleep(args.settle)
            asset = min(
                (
                    n
                    for n, size in page.evaluate(
                        "performance.getEntriesByType('resource')"
                        '.map(e => [e.name, e.transferSize])'
                    )
                    if '/static/' in n and size > 0
                ),
                key=len,
            )
            threading.Thread(
                target=_poll_static, args=(asset, probe, stop), daemon=True
            ).start()
            cdp = page.context.new_cdp_session(page)
            cdp.send('Performance.enable')

            def window(scenario: str, action: Callable[[], None], traced: bool) -> None:
                w = _window(args, result, browser, page, cdp, scenario, action, traced)
                result['windows'].append(w)

            def check_canvases(where: str, target: Page) -> None:
                got = target.evaluate(CANVAS_COUNT_JS)
                result['canvas_checks'].append(
                    {'where': where, 'expected': reference, 'got': got}
                )

            def reveal() -> None:
                page.locator('.bk-tab', has_text=GRID).first.click()

            for _ in range(args.steady):
                window('steady', lambda: None, traced=False)
            if args.trace_dir:
                window('steady', lambda: None, traced=True)
            for i in range(args.reveals):
                page.locator('.bk-tab', has_text='Workflows').first.click()
                time.sleep(4)
                window('reveal', reveal, traced=bool(args.trace_dir) and i == 0)
                check_canvases(f'measured session, reveal {i + 1}', page)
            if args.sessions > 1:
                others = [
                    open_page(f'session {i + 2}') for i in range(args.sessions - 1)
                ]
                for i, other in enumerate(others):
                    other.goto(url)
                    other.locator('.bk-tab', has_text=GRID).first.wait_for(
                        timeout=120_000
                    )
                    other.locator('.bk-tab', has_text=GRID).first.click()
                    try:
                        _wait_for_canvases(other, reference)
                    except TimeoutError:
                        pass  # recorded by the check below
                    check_canvases(f'session {i + 2}, first reveal', other)
                time.sleep(args.settle / 2)
                window('sessions', lambda: None, traced=False)
                page.locator('.bk-tab', has_text='Workflows').first.click()
                time.sleep(4)
                window('sessions-reveal', reveal, traced=False)
                check_canvases('measured session, reveal among sessions', page)
                for i, other in enumerate(others):
                    check_canvases(f'session {i + 2}, after the re-reveal', other)
            # Closing the browser drops the sessions' sockets, which the server
            # logs with tracebacks; only what came before counts.
            result['log_offset'] = log.stat().st_size
            browser.close()
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_for_canvases(page, expected: int | None = None) -> int:
    """Wait until the grid shows ``expected`` canvases (or a stable nonzero count)."""
    deadline = time.monotonic() + 60
    last, stable = -1, 0
    while time.monotonic() < deadline:
        n = page.evaluate(CANVAS_COUNT_JS)
        if expected is not None and n == expected:
            return n
        stable = stable + 1 if n == last and n > 0 else 0
        if expected is None and stable >= 8:
            return n
        last = n
        time.sleep(0.25)
    raise TimeoutError(f'grid shows {last} canvases, expected {expected}')


def _poll_static(asset: str, probe: list, stop: threading.Event) -> None:
    """Latency of a small static file every 20 ms: a blocked IOLoop delays it."""
    while not stop.is_set():
        t0, p0 = time.time(), time.perf_counter()
        try:
            urllib.request.urlopen(asset, timeout=30).read()  # noqa: S310
        except OSError:
            pass
        probe.append((t0, time.perf_counter() - p0))
        time.sleep(0.02)


def _window(
    args: argparse.Namespace,
    result: dict,
    browser,
    page,
    cdp,
    scenario: str,
    action: Callable[[], None],
    traced: bool,
) -> dict:
    def busy_s() -> float:
        metrics = cdp.send('Performance.getMetrics')['metrics']
        return next(m['value'] for m in metrics if m['name'] == 'TaskDuration')

    # Clear in place: the init script's closure holds this object.
    page.evaluate('for (const k in window.__perf) delete window.__perf[k]')
    if traced:
        browser.start_tracing(page=page, categories=traces.CATEGORIES)
    b0, p0, t0 = busy_s(), page.evaluate('performance.now()'), time.time()
    action()
    time.sleep(args.window)
    counts = page.evaluate('({...window.__perf})')
    busy = busy_s() - b0
    t1 = time.time()
    tasks = [
        d
        for s, d in page.evaluate('window.__longtasks')
        if p0 <= s < p0 + args.window * 1e3
    ]
    w = {
        'scenario': scenario,
        'traced': traced,
        't0': t0,
        't1': t1,
        'seconds': t1 - t0,
        'browser': counts
        | {
            'busy_ms': busy * 1e3,
            'longtask_ms': sum(tasks),
            'longtask_max_ms': max(tasks, default=0),
        },
    }
    if traced:
        path = args.trace_dir / f'{args.label}-r{args.round}-{scenario}.json.gz'
        path.write_bytes(gzip.compress(browser.stop_tracing()))
        w['trace'] = {'file': path.name} | traces.summarize(path)
    return w


def _server_window(events: list[dict], probe: list, w: dict) -> dict:
    inside = [e for e in events if w['t0'] <= e['t'] < w['t1']]
    ticks = [e for e in inside if e['kind'] == 'tick']
    switches = [e for e in inside if e['kind'] == 'switch']
    lat = [d for t, d in probe if w['t0'] <= t < w['t1']]
    return {
        'ticks': len(ticks),
        'ticks_off_loop': sum(not e['loop_thread'] for e in ticks),
        'tick_cpu_ms': sum(e['cpu'] for e in ticks),
        'switch_ms': sum(e['ms'] for e in switches),
        'switch_patch_doc_ms': sum(e.get('patch_doc', 0) for e in switches),
        'switch_recompute_ms': sum(e.get('recompute', 0) for e in switches),
        'probe_max_ms': max(lat, default=0) * 1e3,
    }


def ab(args: argparse.Namespace) -> int:
    """Run ``run`` per environment, alternating (A B, B A, A B, ...)."""
    envs = []
    for spec in args.env:
        label, rest = spec.split('=', 1)
        python, repo = rest.split(':', 1)
        envs.append((label, python, repo))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out.parent / 'traces'
    log_dir = args.out.parent / 'logs'
    trace_dir.mkdir(exist_ok=True)
    results = []
    for rnd in range(args.rounds):
        for label, python, repo in envs if rnd % 2 == 0 else envs[::-1]:
            part = args.out.parent / f'{label}-r{rnd}.json'
            cmd = [
                python,
                str(HERE / 'perf.py'),
                'run',
                *('--repo', repo, '--label', label, '--round', str(rnd)),
                *('--out', str(part), '--log-dir', str(log_dir)),
                *('--trace-dir', str(trace_dir)),
                *('--sessions', str(args.sessions), '--window', str(args.window)),
            ]
            t0 = time.monotonic()
            subprocess.run(cmd, check=False, timeout=1200)  # noqa: S603
            if part.exists():
                results.append(json.loads(part.read_text()))
                part.unlink()
            else:
                results.append(
                    {'label': label, 'round': rnd, 'error': 'run wrote no result'}
                )
            status = results[-1].get('error', 'ok').splitlines()[0]
            print(f'{label} round {rnd}: {time.monotonic() - t0:.0f} s, {status}')
            args.out.write_text(json.dumps(results))
    return 0


# --- report -----------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    name: str
    scenario: str
    tier: int
    value: Callable[[dict], float | None]
    # Differences smaller than this are not flagged, whatever the ratio.
    floor: float
    traced: bool = False


def _per_s(key: str, scale: float = 1.0) -> Callable[[dict], float]:
    return lambda w: w['browser'].get(key, 0) / w['seconds'] * scale


def _count(key: str) -> Callable[[dict], float]:
    return lambda w: w['browser'].get(key, 0)


def _server(key: str, per_s: bool = False) -> Callable[[dict], float | None]:
    def value(w: dict) -> float | None:
        if 'server' not in w:
            return None
        return w['server'][key] / (w['seconds'] if per_s else 1)

    return value


def _trace(key: str) -> Callable[[dict], float | None]:
    return lambda w: w['trace'][key] / w['seconds'] if 'trace' in w else None


def _off_loop_share(w: dict) -> float | None:
    server = w.get('server')
    if not server or not server['ticks']:
        return None
    return server['ticks_off_loop'] / server['ticks']


METRICS = [
    Metric('PATCH-DOC messages /s', 'steady', 1, _per_s('msg:PATCH-DOC'), 0.5),
    Metric('document events /s', 'steady', 1, _per_s('doc_events'), 2),
    Metric('WebSocket kB /s', 'steady', 1, _per_s('ws_bytes', 1e-3), 20),
    Metric('getBoundingClientRect /s', 'steady', 1, _per_s('getBoundingClientRect'), 5),
    Metric('getComputedStyle /s', 'steady', 1, _per_s('getComputedStyle'), 50),
    Metric('getPropertyValue /s', 'steady', 1, _per_s('getPropertyValue'), 50),
    Metric('session passes off the loop thread', 'steady', 1, _off_loop_share, 0.05),
    Metric('forced style recalcs /s', 'steady', 1, _trace('forced_style'), 2, True),
    Metric('forced layouts /s', 'steady', 1, _trace('forced_layout'), 2, True),
    Metric(
        'reveal: getBoundingClientRect',
        'reveal',
        1,
        _count('getBoundingClientRect'),
        50,
    ),
    Metric('reveal: getComputedStyle', 'reveal', 1, _count('getComputedStyle'), 500),
    Metric('reveal: document events', 'reveal', 1, _count('doc_events'), 20),
    Metric(
        '3 sessions: PATCH-DOC messages /s', 'sessions', 1, _per_s('msg:PATCH-DOC'), 0.5
    ),
    Metric(
        '3 sessions: getComputedStyle /s', 'sessions', 1, _per_s('getComputedStyle'), 50
    ),
    Metric('browser busy ms/s', 'steady', 2, _per_s('busy_ms'), 10),
    Metric(
        'session pass CPU ms/s', 'steady', 2, _server('tick_cpu_ms', per_s=True), 10
    ),
    Metric('scripting ms/s (traced)', 'steady', 2, _trace('ms_scripting'), 10, True),
    Metric('style ms/s (traced)', 'steady', 2, _trace('ms_style'), 5, True),
    Metric('layout ms/s (traced)', 'steady', 2, _trace('ms_layout'), 5, True),
    Metric('reveal: browser busy ms', 'reveal', 2, _count('busy_ms'), 50),
    Metric('reveal: long tasks ms', 'reveal', 2, _count('longtask_ms'), 50),
    Metric('reveal: server switch ms', 'reveal', 2, _server('switch_ms'), 20),
    Metric(
        'reveal: PATCH-DOC build ms', 'reveal', 2, _server('switch_patch_doc_ms'), 10
    ),
    Metric(
        'reveal: worst static-file latency ms', 'reveal', 2, _server('probe_max_ms'), 20
    ),
    Metric(
        '3 sessions: session pass CPU ms/s',
        'sessions',
        2,
        _server('tick_cpu_ms', per_s=True),
        10,
    ),
    Metric('3 sessions: browser busy ms/s', 'sessions', 2, _per_s('busy_ms'), 10),
]
TIER1_RATIO = 1.3
TIER2_RATIO = 1.25
DATA_RATE_TOLERANCE = 0.2


def _values(runs: list[dict], metric: Metric) -> list[float]:
    out = []
    for r in runs:
        for w in r.get('windows', []):
            # Tracing slows the browser down: traced and untraced windows are
            # separate samples.
            if w['scenario'] != metric.scenario or w['traced'] != metric.traced:
                continue
            v = metric.value(w)
            if v is not None:
                out.append(v)
    return out


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f'{v:.0f}'
    if abs(v) >= 1:
        return f'{v:.1f}'
    return f'{v:.2f}'


def _flag(metric: Metric, base: list[float], cand: list[float]) -> str:
    mb, mc = statistics.median(base), statistics.median(cand)
    if abs(mc - mb) < metric.floor:
        return ''
    ratio = mc / mb if mb else float('inf')
    if metric.tier == 1:
        if ratio > TIER1_RATIO or ratio < 1 / TIER1_RATIO:
            return 'changed'
        return ''
    if min(cand) > max(base) and ratio > TIER2_RATIO:
        return 'slower'
    if max(cand) < min(base) and ratio < 1 / TIER2_RATIO:
        return 'faster'
    return ''


def report(args: argparse.Namespace) -> int:
    results = json.loads(args.results.read_text())
    labels = list(dict.fromkeys(r['label'] for r in results))
    base, cand = labels[0], labels[-1]
    by_label = {lab: [r for r in results if r['label'] == lab] for lab in labels}
    lines = ['## Dashboard performance', '']
    lines += ['| env | rounds | commit | ' + ' | '.join(PACKAGES) + ' | chromium |']
    lines += ['|---' * (len(PACKAGES) + 4) + '|']
    for lab, runs in by_label.items():
        first = next((r for r in runs if 'versions' in r), {})
        versions = first.get('versions', {})
        lines.append(
            f"| {lab} | {len(runs)} | {first.get('commit', '?')} | "
            + ' | '.join(versions.get(p, '?') for p in PACKAGES)
            + f" | {first.get('chromium', '?')} |"
        )
    failures = _tier0(by_label, base, cand)
    lines += ['', '### Tier 0: functional checks', '']
    lines += [f'- FAIL: {f}' for f in failures] or ['- all passed']
    for tier, title in (
        (1, 'Tier 1: counts (flagged beyond 1.3x either way)'),
        (2, 'Tier 2: times (flagged only if the ranges do not overlap)'),
    ):
        lines += ['', f'### {title}', '']
        lines += [
            f'| metric | {base} median | {base} spread | {cand} median '
            f'| {cand} spread | ratio | n | flag |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for m in (m for m in METRICS if m.tier == tier):
            b, c = _values(by_label[base], m), _values(by_label[cand], m)
            if not b or not c:
                continue
            mb, mc = statistics.median(b), statistics.median(c)
            ratio = f'{mc / mb:.2f}' if mb else '-'
            lines.append(
                f'| {m.name} | {_fmt(mb)} | {_spread(b)} | {_fmt(mc)} | {_spread(c)} '
                f'| {ratio} | {len(b)}/{len(c)} | {_flag(m, b, c)} |'
            )
    lines += _event_diff(by_label[base], by_label[cand], base, cand)
    lines += _function_diff(args, by_label[base], by_label[cand], base, cand)
    lines += [
        '',
        'Spread: (max - min) / median over windows. Counts are per window '
        f'(reveals) or per second. Windows of {_window_s(results)} s; traces, '
        'server logs and the raw results are in the run artifact.',
    ]
    print('\n'.join(lines))
    return 1 if failures else 0


def _spread(values: list[float]) -> str:
    med = statistics.median(values)
    if not med or len(values) < 2:
        return '-'
    return f'{(max(values) - min(values)) / abs(med):.0%}'


def _window_s(results: list[dict]) -> str:
    seconds = [w['seconds'] for r in results for w in r.get('windows', [])]
    return f'{statistics.median(seconds):.0f}' if seconds else '?'


def _tier0(by_label: dict[str, list[dict]], base: str, cand: str) -> list[str]:
    failures = []
    for lab, runs in by_label.items():
        for r in runs:
            where = f"{lab} round {r['round']}"
            if 'error' in r:
                failures.append(f"{where}: {r['error'].splitlines()[0][:300]}")
            if r.get('server_tracebacks'):
                failures.append(
                    f"{where}: {r['server_tracebacks']} tracebacks in the server log"
                )
            failures.extend(
                f'{where}: page error {err}'
                for err in sorted(set(r.get('page_errors', [])))[:5]
            )
            failures.extend(
                f"{where}: {c['where']} shows {c['got']} canvases, "
                f"expected {c['expected']}"
                for c in r.get('canvas_checks', [])
                if c['got'] != c['expected']
            )
    rate = next(m for m in METRICS if m.name == 'document events /s')
    b, c = _values(by_label[base], rate), _values(by_label[cand], rate)
    if b and c:
        mb, mc = statistics.median(b), statistics.median(c)
        if abs(mc - mb) > DATA_RATE_TOLERANCE * mb:
            failures.append(
                f'document events /s differ ({_fmt(mb)} against {_fmt(mc)}): the '
                'environments did not see the same data, per-second counts are '
                'not comparable'
            )
    return failures


def _event_diff(
    base_runs: list[dict], cand_runs: list[dict], base: str, cand: str
) -> list[str]:
    """Document events per second by kind and attribute, where they differ."""

    def rates(runs: list[dict]) -> dict[str, float]:
        per_key: dict[str, list[float]] = {}
        windows = [
            w
            for r in runs
            for w in r.get('windows', [])
            if w['scenario'] == 'steady' and not w['traced']
        ]
        keys = {k for w in windows for k in w['browser'] if k.startswith('event:')}
        for k in keys:
            per_key[k] = [w['browser'].get(k, 0) / w['seconds'] for w in windows]
        return {k: statistics.median(v) for k, v in per_key.items()}

    rb, rc = rates(base_runs), rates(cand_runs)
    rows = []
    for k in sorted(set(rb) | set(rc)):
        vb, vc = rb.get(k, 0.0), rc.get(k, 0.0)
        # Some attributes (e.g. the age text of freshness pills) change at a
        # rate that depends on timing; only large differences are listed.
        if abs(vc - vb) >= 0.5 and (not vb or not 0.5 < vc / vb < 2):
            rows.append(f'| {k[6:]} | {_fmt(vb)} | {_fmt(vc)} |')
    if not rows:
        return []
    return [
        '',
        '### Document events /s that differ by 2x or more (steady)',
        '',
        f'| kind:attribute | {base} | {cand} |',
        '|---|---|---|',
        *rows,
    ]


def _function_diff(
    args: argparse.Namespace,
    base_runs: list[dict],
    cand_runs: list[dict],
    base: str,
    cand: str,
) -> list[str]:
    """JS functions whose sampled self time changed most (traced steady windows)."""

    def self_ms(runs: list[dict]) -> dict[str, float]:
        per_fn: dict[str, list[float]] = {}
        files = [
            w['trace']['file']
            for r in runs
            for w in r.get('windows', [])
            if w['scenario'] == 'steady' and 'trace' in w
        ]
        for name in files:
            path = args.results.parent / 'traces' / name
            if not path.exists():
                continue
            events = traces.load(path)
            self_us, _ = traces.function_times(events, traces.main_thread(events))
            for k, v in self_us.items():
                if k not in ('(idle)', '(program)'):
                    per_fn.setdefault(k, []).append(v / 1e3)
        n = len(files)
        return {k: sum(v) / n for k, v in per_fn.items()} if n else {}

    fb, fc = self_ms(base_runs), self_ms(cand_runs)
    if not fb or not fc:
        return []
    deltas = sorted(
        ((fc.get(k, 0.0) - fb.get(k, 0.0), k) for k in set(fb) | set(fc)),
        key=lambda d: -abs(d[0]),
    )[: args.top]
    rows = [
        f'| `{k}` | {_fmt(fb.get(k, 0.0))} | {_fmt(fc.get(k, 0.0))} | {d:+.0f} |'
        for d, k in deltas
    ]
    return [
        '',
        '### JS functions with the largest change in self time '
        '(ms per traced steady window, sampled)',
        '',
        f'| function | {base} | {cand} | change |',
        '|---|---|---|---|',
        *rows,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(required=True)

    p_run = sub.add_parser('run', help='Measure one environment')
    p_run.add_argument('--repo', type=Path, required=True, help='Checkout to measure')
    p_run.add_argument('--out', type=Path, required=True, help='Result JSON')
    p_run.add_argument('--label', default='run')
    p_run.add_argument('--round', type=int, default=0)
    p_run.add_argument('--trace-dir', type=Path, default=None)
    p_run.add_argument('--log-dir', type=Path, default=None)
    p_run.add_argument('--window', type=float, default=8.0, help='Seconds per window')
    p_run.add_argument('--settle', type=float, default=15.0, help='Seconds before')
    p_run.add_argument('--steady', type=int, default=3, help='Untraced steady windows')
    p_run.add_argument('--reveals', type=int, default=2)
    p_run.add_argument('--sessions', type=int, default=3, help='1 skips the scenario')
    p_run.add_argument('--headed', action='store_true')
    p_run.set_defaults(func=run)

    p_ab = sub.add_parser('ab', help='Alternate environments, run by run')
    p_ab.add_argument(
        '--env',
        action='append',
        required=True,
        help='LABEL=PYTHON:CHECKOUT; the first is the baseline, the last the candidate',
    )
    p_ab.add_argument('--out', type=Path, required=True)
    p_ab.add_argument('--rounds', type=int, default=3)
    p_ab.add_argument('--sessions', type=int, default=3)
    p_ab.add_argument('--window', type=float, default=8.0)
    p_ab.set_defaults(func=ab)

    p_report = sub.add_parser('report', help='Markdown comparison of an ab result')
    p_report.add_argument('results', type=Path)
    p_report.add_argument('--top', type=int, default=10, help='JS functions listed')
    p_report.set_defaults(func=report)

    p_trace = sub.add_parser('trace', help='Main-thread breakdown of trace files')
    p_trace.add_argument('dir', type=Path)
    p_trace.set_defaults(func=lambda args: traces.print_analysis(args.dir) or 0)

    args = parser.parse_args()
    if getattr(args, 'trace_dir', None):
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
