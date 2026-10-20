"""Instrumentation injected into the dashboard process and into the browser page.

Both are plain source text: the server part becomes a ``sitecustomize`` module on
the dashboard's ``PYTHONPATH``, the browser part an init script that runs before
the page's own scripts. Neither imports anything from the harness, so the
harness can measure any checkout in any environment.
"""

# Active only in the dashboard process (PERF_PROBES=1). Patching happens at
# interpreter startup: a session built before the patch would keep the original
# bound methods. Bokeh >= 3.10 runs session callbacks on worker threads, hence
# the per-thread timer stack, CPU time from ``thread_time`` (the IOLoop thread's
# CPU no longer contains the session work), and one locked write per event line.
SITECUSTOMIZE = r'''
import json, os, sys, threading, time
if os.environ.get('PERF_PROBES') == '1':
    import importlib
    pc, ct = time.perf_counter, time.thread_time
    _tls = threading.local()
    _emit_lock = threading.Lock()

    def _stack():
        return _tls.__dict__.setdefault('stack', [])

    def _depth():
        return _tls.__dict__.setdefault('depth', {})

    def _emit(**ev):
        line = '[perf] ' + json.dumps(ev) + '\n'
        with _emit_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def _outer(cls, name, kind):
        """Time the outermost call of ``name`` on this thread, with inner timers."""
        orig = getattr(cls, name)

        def wrapper(self, *a, **k):
            stack = _stack()
            if stack:
                return orig(self, *a, **k)
            acc = {}
            stack.append(acc)
            t0, c0 = pc(), ct()
            try:
                return orig(self, *a, **k)
            finally:
                stack.pop()
                thread = threading.current_thread()
                _emit(kind=kind, t=time.time(), ms=(pc() - t0) * 1e3,
                      cpu=(ct() - c0) * 1e3,
                      loop_thread=thread is threading.main_thread(),
                      **{k_: v * 1e3 for k_, v in acc.items()})

        setattr(cls, name, wrapper)

    def _inner(cls, name, label):
        """Add the time spent in ``name`` to the enclosing outer timer."""
        orig = getattr(cls, name)

        def wrapper(*a, **k):
            stack, depth = _stack(), _depth()
            if not stack or depth.get(label):
                return orig(*a, **k)
            depth[label] = 1
            t0 = pc()
            try:
                return orig(*a, **k)
            finally:
                depth[label] = 0
                stack[-1][label] = stack[-1].get(label, 0.0) + pc() - t0

        setattr(cls, name, wrapper)

    pgt = importlib.import_module('ess.livedata.dashboard.widgets.plot_grid_tabs')
    su = importlib.import_module('ess.livedata.dashboard.session_updater')
    import panel.layout.tabs as pt
    from bokeh.document.models import DocumentModelManager
    from bokeh.protocol import Protocol

    _outer(pgt._BatchedTabs, '_update_active', 'switch')
    _outer(su.SessionUpdater, '_tick', 'tick')
    _inner(pt.Tabs, '_get_objects', 'build')
    _inner(DocumentModelManager, 'recompute', 'recompute')
    _inner(Protocol, 'create', 'patch_doc')
    _inner(pgt.PlotGridTabs, '_poll_for_plot_updates', 'flush')
'''

# Counts calls that read layout or computed style, and taps the session's
# WebSocket. A call is cheap unless style or layout is dirty, so these counts go
# together with the forced recalculations counted in a trace. ``window.__perf``
# is reset by the harness at the start of every window.
COUNTERS_JS = r'''
(() => {
  const c = window.__perf = {};
  const bump = (key, n = 1) => { c[key] = (c[key] || 0) + n; };
  const wrap = (obj, name, key) => {
    const f = obj[name];
    obj[name] = function (...a) { bump(key); return f.apply(this, a); };
  };
  wrap(Element.prototype, 'getBoundingClientRect', 'getBoundingClientRect');
  wrap(window, 'getComputedStyle', 'getComputedStyle');
  wrap(CSSStyleDeclaration.prototype, 'getPropertyValue', 'getPropertyValue');
  const WS = window.WebSocket;
  window.WebSocket = class extends WS {
    constructor(...a) {
      super(...a);
      this.addEventListener('message', (e) => {
        if (typeof e.data !== 'string') {
          bump('ws_bytes', e.data.byteLength ?? e.data.size ?? 0);
          return;
        }
        bump('ws_bytes', e.data.length);
        const m = /"msgtype":\s*"([A-Z-]+)"/.exec(e.data);
        if (m) {
          bump('msg:' + m[1]);
        } else if (e.data.startsWith('{"events"')) {
          for (const ev of JSON.parse(e.data).events) {
            bump('doc_events');
            bump('event:' + ev.kind + (ev.attr ? ':' + ev.attr : ''));
          }
        }
      });
    }
  };
  window.__longtasks = [];
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      window.__longtasks.push([e.startTime, e.duration]);
    }
  }).observe({type: 'longtask', buffered: true});
})();
'''

# Canvases across all shadow roots: one or more per rendered figure. A cell whose
# figure failed to build shows its frame but no canvas.
CANVAS_COUNT_JS = '''() => {
  let n = 0;
  const walk = (root) => {
    n += root.querySelectorAll('canvas').length;
    root.querySelectorAll('*').forEach(e => e.shadowRoot && walk(e.shadowRoot));
  };
  walk(document);
  return n;
}'''
