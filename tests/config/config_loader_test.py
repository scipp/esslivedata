# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config


@pytest.fixture
def _upstream_env_setup(monkeypatch):
    """Setup environment variables needed for tests"""
    env_vars = {
        'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092',
        'KAFKA_SECURITY_PROTOCOL': 'SASL_PLAINTEXT',
        'KAFKA_SASL_MECHANISM': 'SCRAM-SHA-256',
        'KAFKA_SASL_USERNAME': 'admin',
        'KAFKA_SASL_PASSWORD': 'admin',
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def _downstream_env_setup(monkeypatch):
    """Setup environment variables needed for tests"""
    env_vars = {
        'KAFKA2_BOOTSTRAP_SERVERS': 'localhost:9092',
        'KAFKA2_SECURITY_PROTOCOL': 'SASL_PLAINTEXT',
        'KAFKA2_SASL_MECHANISM': 'SCRAM-SHA-256',
        'KAFKA2_SASL_USERNAME': 'admin',
        'KAFKA2_SASL_PASSWORD': 'admin',
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


def test_control_consumer():
    config = load_config(namespace=config_names.control_consumer, env='')
    assert config['auto.offset.reset'] == 'latest'


@pytest.mark.parametrize('env', [None, 'dev', 'docker'])
@pytest.mark.usefixtures('_downstream_env_setup')
def test_kafka_downstream(env: str | None):
    config = load_config(namespace=config_names.kafka_downstream, env=env)
    assert 'bootstrap.servers' in config


@pytest.mark.parametrize('env', [None, 'dev', 'docker'])
@pytest.mark.usefixtures('_upstream_env_setup')
def test_kafka_upstream(env: str | None):
    config = load_config(namespace=config_names.kafka_upstream, env=env)
    assert 'bootstrap.servers' in config


def test_raw_data_consumer():
    config = load_config(namespace=config_names.raw_data_consumer, env='')
    assert config['auto.offset.reset'] == 'latest'


def test_reduced_data_consumer():
    config = load_config(namespace=config_names.reduced_data_consumer, env='')
    assert config['auto.offset.reset'] == 'latest'
