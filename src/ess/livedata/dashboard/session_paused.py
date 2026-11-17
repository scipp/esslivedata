# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Static page a reaped session's browser moves to.

A tab hidden for longer than its grace stops sending heartbeats (see
``HeartbeatWidget.hidden_grace_ms``), so the stale-session reaper drops its
session although its websocket is still open. Such a tab must not stay on the
dashboard page afterwards: its widgets would keep accepting input and act on
shared state while the session never updates again. Instead the tab leaves for
this page, which has no Bokeh session, closes the websocket, and costs the
server nothing while the tab stays hidden. Once the tab is visible again, the
page reloads the dashboard URL the tab was on, which starts a fresh session.

The dashboard URL travels in ``sessionStorage`` (per tab, same origin), not in
the query string, so this page cannot be used as an open redirect.
"""

from __future__ import annotations

from tornado.web import RequestHandler

#: Path of the paused page, relative to the dashboard app (served at the root).
SESSION_PAUSED_PATH = 'session-paused'
#: ``sessionStorage`` key holding the dashboard URL to return to.
RESUME_URL_KEY = 'livedata-resume-url'

_PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESSlivedata - session paused</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 3em auto; max-width: 40em;
         padding: 0 16px; line-height: 1.5; }}
</style>
</head>
<body>
<p>This dashboard session was closed while the tab was in the background.
It reloads when the tab is shown.</p>
<p><a id="resume" href=".">Reload the dashboard</a></p>
<script>
  function resumeUrl() {{
    try {{
      const url = new URL(sessionStorage.getItem('{RESUME_URL_KEY}'));
      if (url.origin === window.location.origin) return url.href;
    }} catch (e) {{}}
    return new URL('.', window.location.href).href;
  }}
  function resume() {{
    if (document.visibilityState === 'visible') window.location.replace(resumeUrl());
  }}
  document.getElementById('resume').href = resumeUrl();
  document.addEventListener('visibilitychange', resume);
  resume();
</script>
</body>
</html>
"""


class SessionPausedHandler(RequestHandler):
    """Serves the paused page; see the module docstring."""

    def get(self) -> None:
        self.set_header('Content-Type', 'text/html; charset=utf-8')
        self.set_header('Cache-Control', 'no-store')
        self.write(_PAGE)
