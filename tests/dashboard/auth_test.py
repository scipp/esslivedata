# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

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
    # patch.dict rather than monkeypatch: discovery writes to the environment.
    with mock.patch.dict(os.environ):
        for var in list(os.environ):
            if var.startswith(
                ('PANEL_', 'BOKEH_COOKIE', 'LIVEDATA_BASIC_AUTH', 'LIVEDATA_OAUTH')
            ):
                del os.environ[var]
        yield


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


def test_complete_configuration_enables_login_and_encryption(oauth_env: None) -> None:
    assert configure_auth() == 'generic'
    assert pn.state.encryption is ENCRYPTION


def test_server_is_given_the_provider(oauth_env: None, tmp_path: Path) -> None:
    """Without the provider keyword pn.serve installs no login, whatever the env."""
    app = ReductionApp(
        log_level=logging.INFO, transport='none', config_dir=str(tmp_path)
    )
    with mock.patch.object(pn, 'serve') as serve, mock.patch('atexit.register'):
        app.run_forever()
    assert serve.call_args.kwargs['oauth_provider'] == 'generic'
    assert serve.call_args.kwargs['login_endpoint'] == '/oauth-redirect'


def test_removed_basic_auth_variable_refuses_to_start() -> None:
    """An upgraded deployment must not silently lose its password protection."""
    os.environ['LIVEDATA_BASIC_AUTH_PASSWORD'] = SECRET
    with pytest.raises(ValueError, match='LIVEDATA_BASIC_AUTH_PASSWORD') as excinfo:
        configure_auth()
    assert SECRET not in str(excinfo.value)


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


def test_endpoints_come_from_the_discovery_document(oauth_env: None) -> None:
    del os.environ['PANEL_OAUTH_TOKEN_URL']
    os.environ['LIVEDATA_OAUTH_DISCOVERY_URL'] = 'https://idp.example.org/discovery'
    document = {'token_endpoint': 'https://idp.example.org/discovered'}
    response = io.BytesIO(json.dumps(document).encode())
    with mock.patch.object(auth, 'urlopen', return_value=response):
        assert configure_auth() == 'generic'
    assert os.environ['PANEL_OAUTH_TOKEN_URL'] == document['token_endpoint']
    # Set explicitly, so not overwritten.
    assert os.environ['PANEL_OAUTH_USER_URL'] == 'https://idp.example.org/userinfo'
