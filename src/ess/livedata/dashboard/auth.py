# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Dashboard login configuration, from environment variables.

See ``docs/user-guide/deployment.md``. The dashboard uses ``pn.serve``, which unlike
``panel serve`` installs no login unless passed the provider, never enables encryption
of the tokens stored in cookies, and validates nothing. :py:func:`configure_auth` makes
up for this, and fails at startup rather than serve an unprotected dashboard by mistake.
"""

import json
import os
from urllib.parse import urlsplit
from urllib.request import urlopen

import panel as pn
import structlog

logger = structlog.get_logger(__name__)

_ENDPOINTS = {
    'PANEL_OAUTH_AUTHORIZE_URL': 'authorization_endpoint',
    'PANEL_OAUTH_TOKEN_URL': 'token_endpoint',
    'PANEL_OAUTH_USER_URL': 'userinfo_endpoint',
}
_REQUIRED_VARS = (
    'PANEL_OAUTH_KEY',
    'PANEL_OAUTH_SECRET',
    'PANEL_OAUTH_REDIRECT_URI',
    'PANEL_OAUTH_ENCRYPTION',
    'PANEL_COOKIE_SECRET',
    *_ENDPOINTS,
)


def configure_auth() -> str | None:
    """Validate the login configuration and enable token encryption.

    Returns
    -------
    :
        The OAuth provider to pass to ``pn.serve``, or None if no login is
        configured, in which case the dashboard is open to anyone who can reach it.

    Raises
    ------
    ValueError
        If the configuration is incomplete. Names variables, never their values.
    """
    provider = pn.config.oauth_provider
    if provider is None:
        if os.environ.get('LIVEDATA_REQUIRE_LOGIN'):
            raise ValueError("Login required, but PANEL_OAUTH_PROVIDER is not set.")
        if os.environ.get('PANEL_OAUTH_KEY') or os.environ.get('PANEL_OAUTH_SECRET'):
            raise ValueError("OAuth client set without PANEL_OAUTH_PROVIDER: no login.")
        logger.warning("No OAuth provider configured: dashboard login is disabled.")
        return None

    if url := os.environ.get('LIVEDATA_OAUTH_DISCOVERY_URL'):
        _load_endpoints(url)
    if missing := [var for var in _REQUIRED_VARS if not os.environ.get(var)]:
        raise ValueError(f"OAuth login requires {', '.join(missing)} to be set.")

    pn.state.encryption = _create_encryption(os.environ['PANEL_OAUTH_ENCRYPTION'])
    logger.info("Dashboard login enabled.", provider=provider)
    return provider


def get_login_endpoint() -> str | None:
    """Return the path of the redirect URI, where Panel must serve its login route."""
    path = urlsplit(pn.config.oauth_redirect_uri or '').path
    return path.rstrip('/') or None


def _load_endpoints(discovery_url: str) -> None:
    """Fill in endpoints not set explicitly, as Panel cannot read discovery itself."""
    with urlopen(discovery_url, timeout=10) as response:  # noqa: S310
        document = json.load(response)
    for var, field in _ENDPOINTS.items():
        if not os.environ.get(var):
            os.environ[var] = document[field]


def _create_encryption(key: str):
    from cryptography.fernet import Fernet

    return Fernet(key)
