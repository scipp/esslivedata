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
    severed:
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
    severed = param.Boolean(
        default=False,
        doc="Set by the server on reap; the browser then leaves the page",
    )

    # Invisible container - the div exists but takes no space
    _template = """<div id="heartbeat" style="display:none;"></div>"""

    # The render script runs when the component loads.
    # It sets up a periodic interval that increments data.counter.
    # Note: Use -= 1 or += 1, NOT ++ or -- (those break param sync, see Panel #4925)
    _scripts: ClassVar = {
        'render': """
            state.interval_id = setInterval(() => {
                // Increment counter, wrap around to avoid overflow
                const newVal = (data.counter + 1) % 2147483647;
                data.counter = newVal;
            }, data.interval_ms);
        """,
        'severed': f"""
            if (data.severed) {{
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
        """,
    }

    def __init__(self, **params):
        # Set sizing to take no space
        params.setdefault('width', 0)
        params.setdefault('height', 0)
        params.setdefault('sizing_mode', 'fixed')
        params.setdefault('visible', False)
        super().__init__(**params)
