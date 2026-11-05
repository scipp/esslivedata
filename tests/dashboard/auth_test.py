# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import panel as pn
import pytest

from ess.livedata.dashboard import auth
from ess.livedata.dashboard.auth import configure_auth
from ess.livedata.dashboard.reduction import ReductionApp

SECRET = 'do-not-leak-this-value'  # noqa: S105
ENCRYPTION = object()


@pytest.fixture(autouse=True)
def clean_auth_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate from the host's environment and restore what configure_auth sets."""
    monkeypatch.setattr(pn.state, 'encryption', None)
    # Encryption needs the `cryptography` package, which only the dashboard extra has.
    monkeypatch.setattr(auth, '_create_encryption', lambda key: ENCRYPTION)
    # A snapshot rather than monkeypatch: discovery writes to the environment.
    saved = dict(os.environ)
    for var in list(os.environ):
        if var.startswith(
            ('PANEL_', 'BOKEH_COOKIE', 'LIVEDATA_OAUTH', 'LIVEDATA_REQUIRE')
        ):
            del os.environ[var]
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def oauth_env() -> None:
    """A complete configuration."""
    os.environ.update(
        PANEL_OAUTH_PROVIDER='generic',
        PANEL_OAUTH_KEY='client-id',
        PANEL_OAUTH_SECRET=SECRET,
        PANEL_COOKIE_SECRET=SECRET,
        PANEL_OAUTH_ENCRYPTION=SECRET,
        PANEL_OAUTH_REDIRECT_URI='https://dashboard.example.org/oauth-redirect',
        PANEL_OAUTH_AUTHORIZE_URL='https://idp.example.org/authorize',
        PANEL_OAUTH_TOKEN_URL='https://idp.example.org/token',  # noqa: S106
        PANEL_OAUTH_USER_URL='https://idp.example.org/userinfo',
    )


def test_no_configuration_means_no_login() -> None:
    assert configure_auth() is None


def test_required_login_without_provider_refuses_to_start() -> None:
    """The published image sets this, so a deployment cannot run open by mistake."""
    os.environ['LIVEDATA_REQUIRE_LOGIN'] = '1'
    with pytest.raises(ValueError, match='PANEL_OAUTH_PROVIDER'):
        configure_auth()


def test_complete_configuration_enables_login_and_encryption(oauth_env: None) -> None:
    assert configure_auth() == 'generic'
    assert pn.state.encryption is ENCRYPTION


def test_server_is_given_the_provider(
    oauth_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the provider keyword pn.serve installs no login, whatever the env."""
    served: list[dict] = []
    monkeypatch.setattr(pn, 'serve', lambda *args, **kwargs: served.append(kwargs))
    monkeypatch.setattr('atexit.register', lambda func: func)
    app = ReductionApp(
        log_level=logging.INFO, transport='none', config_dir=str(tmp_path)
    )
    app.run_forever()
    assert served[0]['oauth_provider'] == 'generic'
    assert served[0]['login_endpoint'] == '/oauth-redirect'


def test_oauth_variable_without_provider_refuses_to_start() -> None:
    """No provider means no login, so this dashboard would be open by mistake."""
    os.environ['PANEL_OAUTH_KEY'] = 'client-id'
    with pytest.raises(ValueError, match='PANEL_OAUTH_PROVIDER'):
        configure_auth()


@pytest.mark.parametrize(
    'var', ['PANEL_OAUTH_SECRET', 'PANEL_COOKIE_SECRET', 'PANEL_OAUTH_ENCRYPTION']
)
def test_missing_secret_refuses_to_start(var: str, oauth_env: None) -> None:
    del os.environ[var]
    with pytest.raises(ValueError, match=var) as excinfo:
        configure_auth()
    assert SECRET not in str(excinfo.value)


def test_endpoints_come_from_the_discovery_document(
    oauth_env: None, tmp_path: Path
) -> None:
    document = {'token_endpoint': 'https://idp.example.org/discovered'}
    discovery = tmp_path / 'openid-configuration'
    discovery.write_text(json.dumps(document))
    del os.environ['PANEL_OAUTH_TOKEN_URL']
    # urlopen also reads file:// URLs, so no server is needed.
    os.environ['LIVEDATA_OAUTH_DISCOVERY_URL'] = discovery.as_uri()
    assert configure_auth() == 'generic'
    assert os.environ['PANEL_OAUTH_TOKEN_URL'] == document['token_endpoint']
    # Set explicitly, so not overwritten.
    assert os.environ['PANEL_OAUTH_USER_URL'] == 'https://idp.example.org/userinfo'
