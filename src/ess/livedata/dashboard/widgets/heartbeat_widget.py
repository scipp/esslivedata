# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Browser heartbeat widget using ReactiveHTML.

This widget runs JavaScript in the browser that periodically increments a counter.
The counter value syncs to Python, allowing detection of browser disconnection.
When the browser disconnects (tab close, refresh, network issue), the counter
stops updating and the session can be marked as stale.
"""

from __future__ import annotations

from typing import ClassVar

import param
from panel.reactive import ReactiveHTML

from ..session_paused import RESUME_URL_KEY, SESSION_PAUSED_PATH

# Seconds a browser tab may stay hidden before it stops sending heartbeats. The
# reaper then drops the session after its stale timeout, so a hidden tab costs
# the server nothing after grace plus timeout, in every browser. Reaping is not
# free for the user: the tab reloads when shown again, losing open dialogs and
# unsaved forms. Four minutes plus the 60 s stale timeout matches what Chrome's
# timer throttling used to do to hidden tabs, and leaves room for switching to
# another application (a window fully covered by another may count as hidden).
#
# Four minutes is also the largest grace Chrome honours. After five minutes
# hidden it fires the heartbeat timer about once a minute, which is right at
# the stale timeout, so with a larger grace a hidden Chrome tab is reaped
# erratically from then on instead of at grace plus timeout.
DEFAULT_HIDDEN_TAB_GRACE = 240.0


class HeartbeatWidget(ReactiveHTML):
    """
    Hidden widget that sends periodic heartbeats from the browser.

    The browser-side JavaScript increments the `counter` parameter every
    `interval_ms` milliseconds. Python code can watch for changes to detect
    browser liveness. If the counter stops changing, the browser has
    disconnected.

    This uses ReactiveHTML's `_scripts['render']` which runs when the
    component loads - no manual triggering required.

    Parameters
    ----------
    interval_ms:
        Interval between heartbeats in milliseconds. Default is 5000 (5 seconds).
    counter:
        The heartbeat counter, incremented by JavaScript. Watch this parameter
        to detect browser liveness.
    hidden_grace_ms:
        How long the tab may stay hidden before the heartbeat stops, so that
        the server reaps the session. Browsers throttle background timers
        differently: Chrome fires them about once a minute after five minutes
        hidden, Firefox does not throttle a page holding an open WebSocket at
        all. Stopping on a fixed grace makes a hidden tab cost the same in
        every browser. Visibility comes from the ``visibilitychange`` event,
        which no browser throttles; the heartbeat resumes when the tab is
        shown again, unless the session was reaped meanwhile.
    reaped:
        Set by the server once it has reaped the session. The browser then
        leaves for the paused page (see :mod:`..session_paused`), remembering
        its URL so the page can reload it.
    """

    interval_ms = param.Integer(
        default=5000, bounds=(1000, 60000), doc="Heartbeat interval in milliseconds"
    )
    counter = param.Integer(
        default=0,
        doc="Heartbeat counter incremented by browser JS",
    )
    hidden_grace_ms = param.Integer(
        default=int(DEFAULT_HIDDEN_TAB_GRACE * 1000),
        bounds=(0, None),
        doc="Hidden time before heartbeats stop",
    )
    reaped = param.Boolean(
        default=False,
        doc="Set by the server on reap; the browser then leaves the page",
    )

    # Invisible container - the div exists but takes no space
    _template = """<div id="heartbeat" style="display:none;"></div>"""

    # The render script runs when the component loads.
    # It sets up a periodic interval that increments data.counter, skipped once
    # the tab has been hidden for longer than the grace.
    # Note: Use -= 1 or += 1, NOT ++ or -- (those break param sync, see Panel #4925)
    _scripts: ClassVar = {
        'render': """
            const hidden = () => document.visibilityState === 'hidden';
            const beat = () => {
                // Increment counter, wrap around to avoid overflow
                const newVal = (data.counter + 1) % 2147483647;
                data.counter = newVal;
            };
            state.hidden_since = hidden() ? Date.now() : null;
            state.on_visibility = () => {
                state.hidden_since = hidden() ? Date.now() : null;
                // Beat at once when shown, not an interval later: the session
                // may be close to its stale timeout.
                if (state.hidden_since === null) beat();
            };
            document.addEventListener('visibilitychange', state.on_visibility);
            state.interval_id = setInterval(() => {
                if (state.hidden_since !== null
                    && Date.now() - state.hidden_since > data.hidden_grace_ms) {
                    return;
                }
                beat();
            }, data.interval_ms);
        """,
        'reaped': f"""
            if (data.reaped) {{
                try {{
                    sessionStorage.setItem('{RESUME_URL_KEY}', window.location.href);
                }} catch (e) {{}}
                window.location.replace('{SESSION_PAUSED_PATH}');
            }}
        """,
        'remove': """
            if (state.interval_id) {
                clearInterval(state.interval_id);
            }
            document.removeEventListener('visibilitychange', state.on_visibility);
        """,
    }

    def __init__(self, **params):
        # Set sizing to take no space
        params.setdefault('width', 0)
        params.setdefault('height', 0)
        params.setdefault('sizing_mode', 'fixed')
        params.setdefault('visible', False)
        super().__init__(**params)
