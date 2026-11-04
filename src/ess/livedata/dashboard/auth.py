# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Dashboard login configuration.

The dashboard is configured with Panel's standard ``PANEL_OAUTH_*`` environment
variables, see ``docs/user-guide/deployment.md``. Panel only acts on them fully
under ``panel serve``. The dashboard uses ``pn.serve``, which

- installs no login unless passed the provider, so the variables alone leave the
  dashboard open,
- never enables encryption of the tokens stored in cookies, and
- validates nothing.

:py:func:`configure_auth` makes up for this, failing at startup rather than serving
a dashboard with less protection than the deployment asked for.

Panel also cannot read an OpenID discovery document, so ``LIVEDATA_OAUTH_DISCOVERY_URL``
is resolved here into the endpoint variables Panel's generic provider reads.
"""

import json
import os
from urllib.parse import urlsplit
from urllib.request import urlopen

import panel as pn
import structlog

logger = structlog.get_logger(__name__)

_REMOVED_BASIC_AUTH_VARS = (
    'LIVEDATA_BASIC_AUTH_PASSWORD',
    'LIVEDATA_BASIC_AUTH_COOKIE_SECRET',
)
_DISCOVERY_VAR = 'LIVEDATA_OAUTH_DISCOVERY_URL'
_DISCOVERY_TIMEOUT_SECONDS = 10
_OAUTH_VARS_NEEDING_PROVIDER = (
    'PANEL_OAUTH_KEY',
    'PANEL_OAUTH_SECRET',
    'PANEL_OAUTH_ENCRYPTION',
    'PANEL_OAUTH_REDIRECT_URI',
    'PANEL_OAUTH_AUTHORIZE_URL',
    'PANEL_OAUTH_TOKEN_URL',
    'PANEL_OAUTH_USER_URL',
    _DISCOVERY_VAR,
)
_GENERIC_ENDPOINTS = (
    ('AUTHORIZE_URL', 'PANEL_OAUTH_AUTHORIZE_URL', 'authorization_endpoint'),
    ('TOKEN_URL', 'PANEL_OAUTH_TOKEN_URL', 'token_endpoint'),
    ('USER_URL', 'PANEL_OAUTH_USER_URL', 'userinfo_endpoint'),
)


def configure_auth() -> str | None:
    """Validate the login configuration and enable token encryption.

    Endpoints missing from the environment are looked up in the discovery document
    if one is configured. Setting them all explicitly avoids contacting the provider
    at startup. Error messages name variables but never include their values.

    Returns
    -------
    :
        The OAuth provider to pass to ``pn.serve``, or None if no login is
        configured, in which case the dashboard is open to anyone who can reach it.

    Raises
    ------
    ValueError
        If the configuration is incomplete or invalid, or if a required discovery
        document cannot be read.
    """
    if removed := [var for var in _REMOVED_BASIC_AUTH_VARS if os.environ.get(var)]:
        raise ValueError(
            f"Basic authentication was removed, but {', '.join(removed)} is set. "
            "Configure OAuth and unset it; see the deployment guide."
        )
    provider = pn.config.oauth_provider
    if provider is None:
        if orphaned := [v for v in _OAUTH_VARS_NEEDING_PROVIDER if os.environ.get(v)]:
            raise ValueError(
                f"{', '.join(orphaned)} set without PANEL_OAUTH_PROVIDER. No login "
                "would be enforced, so refusing to start."
            )
        logger.warning("No OAuth provider configured: dashboard login is disabled.")
        return None

    from panel.auth import AUTH_PROVIDERS

    if provider not in AUTH_PROVIDERS:
        raise ValueError(
            f"Unknown PANEL_OAUTH_PROVIDER; expected one of {sorted(AUTH_PROVIDERS)}."
        )
    required = {
        'PANEL_OAUTH_KEY': pn.config.oauth_key,
        'PANEL_OAUTH_SECRET': pn.config.oauth_secret,
        'PANEL_COOKIE_SECRET': pn.config.cookie_secret,
        'PANEL_OAUTH_ENCRYPTION': pn.config.oauth_encryption_key,
    }
    if provider == 'generic':
        required.update(_resolve_generic_endpoints())
    elif os.environ.get(_DISCOVERY_VAR):
        raise ValueError(
            f"{_DISCOVERY_VAR} is only supported by the 'generic' OAuth provider."
        )
    if missing := [var for var, value in required.items() if not value]:
        raise ValueError(
            f"OAuth provider {provider!r} requires {', '.join(missing)} to be set."
        )
    pn.state.encryption = _create_encryption(pn.config.oauth_encryption_key)
    if not pn.config.oauth_redirect_uri:
        logger.warning(
            "PANEL_OAUTH_REDIRECT_URI is not set: deriving it from each request, "
            "which is wrong behind a TLS-terminating proxy."
        )
    logger.info("Dashboard login enabled.", provider=provider)
    return provider


def get_login_endpoint() -> str | None:
    """Return the route that must handle the provider's redirect back to us.

    Panel handles the redirect in its login handler, so that handler has to be
    served at the path of the redirect URI registered with the provider.

    Returns
    -------
    :
        The path of ``PANEL_OAUTH_REDIRECT_URI``. None for Panel's default route if
        the variable is unset or has no path: the app itself answers there and
        forwards the redirect to the login handler.
    """
    path = urlsplit(pn.config.oauth_redirect_uri or '').path
    return path if path.strip('/') else None


def _resolve_generic_endpoints() -> dict[str, str | None]:
    """Return each endpoint variable's value, filling gaps from discovery."""
    extra_params = _read_extra_params()
    endpoints = {
        var: extra_params.get(param) or os.environ.get(var)
        for param, var, _ in _GENERIC_ENDPOINTS
    }
    if all(endpoints.values()) or not (url := os.environ.get(_DISCOVERY_VAR)):
        return endpoints
    document = _fetch_discovery_document(url)
    for _, var, field in _GENERIC_ENDPOINTS:
        if endpoints[var]:
            continue
        if not (value := document.get(field)):
            raise ValueError(
                f"The document at {_DISCOVERY_VAR} has no {field!r}; set {var} instead."
            )
        os.environ[var] = endpoints[var] = value
    return endpoints


def _fetch_discovery_document(url: str) -> dict:
    if not url.startswith(('https://', 'http://')):
        raise ValueError(f"{_DISCOVERY_VAR} must be an http(s) URL.")
    try:
        with urlopen(url, timeout=_DISCOVERY_TIMEOUT_SECONDS) as response:  # noqa: S310
            document = json.load(response)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read the discovery document at {_DISCOVERY_VAR}: {error}. "
            "Check the provider, or set the endpoint variables explicitly."
        ) from None
    if not isinstance(document, dict):
        raise ValueError(f"The document at {_DISCOVERY_VAR} is not a JSON object.")
    return document


def _read_extra_params() -> dict:
    try:
        return pn.config.oauth_extra_params
    except (ValueError, SyntaxError):
        raise ValueError(
            "PANEL_OAUTH_EXTRA_PARAMS is not a valid Python dict literal."
        ) from None


def _create_encryption(key: str | bytes):
    # Imported here so that a dashboard without login does not need the package.
    from cryptography.fernet import Fernet

    try:
        return Fernet(key)
    except ValueError:
        raise ValueError(
            "PANEL_OAUTH_ENCRYPTION is not a valid key; generate one with "
            "`panel oauth-secret`."
        ) from None
