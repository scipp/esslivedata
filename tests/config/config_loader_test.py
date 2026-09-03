# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest

from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config


@pytest.fixture
def _kafka_env_setup(monkeypatch):
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


@pytest.mark.parametrize('env', [None, 'dev', 'docker'])
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka(env: str | None):
    config = load_config(namespace=config_names.kafka, env=env)
    assert 'bootstrap.servers' in config


@pytest.mark.parametrize(
    'protocol', ['PLAINTEXT', 'SASL_PLAINTEXT', 'plaintext', 'sasl_plaintext']
)
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_omits_ca_location_for_plaintext(monkeypatch, protocol):
    monkeypatch.setenv('KAFKA_SECURITY_PROTOCOL', protocol)
    monkeypatch.delenv('KAFKA_SSL_CA_LOCATION', raising=False)

    config = load_config(namespace=config_names.kafka, env='docker')

    assert 'ssl.ca.location' not in config


@pytest.mark.parametrize('protocol', ['SSL', 'SASL_SSL', 'ssl', 'sasl_ssl'])
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_uses_ca_location_for_tls(monkeypatch, protocol):
    monkeypatch.setenv('KAFKA_SECURITY_PROTOCOL', protocol)
    monkeypatch.setenv('KAFKA_SSL_CA_LOCATION', '/etc/ssl/certs/ESS Kafka #1 CA.pem')

    config = load_config(namespace=config_names.kafka, env='docker')

    assert config['ssl.ca.location'] == '/etc/ssl/certs/ESS Kafka #1 CA.pem'


@pytest.mark.parametrize('protocol', ['PLAINTEXT', 'SSL'])
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_omits_sasl_for_non_sasl_protocol(monkeypatch, protocol):
    monkeypatch.setenv('KAFKA_SECURITY_PROTOCOL', protocol)
    if protocol == 'SSL':
        monkeypatch.setenv('KAFKA_SSL_CA_LOCATION', '/etc/ssl/certs/kafka-ca.pem')
    for variable in (
        'KAFKA_SASL_MECHANISM',
        'KAFKA_SASL_USERNAME',
        'KAFKA_SASL_PASSWORD',
    ):
        monkeypatch.delenv(variable)

    config = load_config(namespace=config_names.kafka, env='docker')

    assert 'sasl.mechanism' not in config
    assert 'sasl.username' not in config
    assert 'sasl.password' not in config


@pytest.mark.parametrize('protocol', ['SASL_PLAINTEXT', 'SASL_SSL'])
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_preserves_sasl_values_as_strings(monkeypatch, protocol):
    monkeypatch.setenv('KAFKA_SECURITY_PROTOCOL', protocol)
    monkeypatch.setenv('KAFKA_BOOTSTRAP_SERVERS', 'broker-1:9093,broker-2:9093')
    monkeypatch.setenv('KAFKA_SASL_MECHANISM', 'SCRAM-SHA-256')
    monkeypatch.setenv('KAFKA_SASL_USERNAME', '00123')
    monkeypatch.setenv('KAFKA_SASL_PASSWORD', 'true # password: []')
    if protocol == 'SASL_SSL':
        monkeypatch.setenv('KAFKA_SSL_CA_LOCATION', '/etc/ssl/certs/kafka-ca.pem')

    config = load_config(namespace=config_names.kafka, env='docker')

    assert config['bootstrap.servers'] == 'broker-1:9093,broker-2:9093'
    assert config['sasl.mechanism'] == 'SCRAM-SHA-256'
    assert config['sasl.username'] == '00123'
    assert config['sasl.password'] == 'true # password: []'


@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_requires_sasl_variables_for_sasl_protocol(monkeypatch):
    monkeypatch.delenv('KAFKA_SASL_PASSWORD')

    with pytest.raises(ValueError, match='KAFKA_SASL_PASSWORD'):
        load_config(namespace=config_names.kafka, env='docker')


@pytest.mark.parametrize('ca_location', [None, '', '   '])
@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_requires_ca_location_for_tls(monkeypatch, ca_location):
    monkeypatch.setenv('KAFKA_SECURITY_PROTOCOL', 'SASL_SSL')
    if ca_location is None:
        monkeypatch.delenv('KAFKA_SSL_CA_LOCATION', raising=False)
    else:
        monkeypatch.setenv('KAFKA_SSL_CA_LOCATION', ca_location)

    with pytest.raises(ValueError, match='KAFKA_SSL_CA_LOCATION'):
        load_config(namespace=config_names.kafka, env='docker')


@pytest.mark.usefixtures('_kafka_env_setup')
def test_kafka_docker_requires_other_template_variables(monkeypatch):
    monkeypatch.delenv('KAFKA_BOOTSTRAP_SERVERS')

    with pytest.raises(ValueError, match='KAFKA_BOOTSTRAP_SERVERS'):
        load_config(namespace=config_names.kafka, env='docker')


def test_raw_data_consumer():
    config = load_config(namespace=config_names.raw_data_consumer, env='')
    assert config['auto.offset.reset'] == 'latest'


def test_reduced_data_consumer():
    config = load_config(namespace=config_names.reduced_data_consumer, env='')
    assert config['auto.offset.reset'] == 'latest'
