# Services

## Overview

### Main services

The following main services are available:

```sh
python -m ess.livedata.services.monitor_data --instrument dummy
python -m ess.livedata.services.detector_data --instrument dummy
python -m ess.livedata.services.data_reduction --instrument dummy
python -m ess.livedata.services.timeseries --instrument dummy
```

For testing, each of these should be run with the `--dev` argument.
This will run the services with a simplified topic structure and make them compatible with the fake data services [below](#fake-data-services).

Note also the `--sink png` argument, which will save the outputs as PNG files instead of publishing them to Kafka.
This allows for testing the service outputs without running the dashboard.

### Dashboard

The dashboard can be run using:

```sh
python -m ess.livedata.dashboard.reduction --instrument dummy
```

Navigate to `http://localhost:5009` for the reduction dashboard.

### Fake data services

The following fake data services are available:

```sh
python -m ess.livedata.services.fake_monitors --mode ev44 --instrument dummy
python -m ess.livedata.services.fake_detectors --instrument dummy
python -m ess.livedata.services.fake_logdata --instrument dummy
```

## Example: Running the monitor data service

Services can be found in `ess.livedata.services`.
Configuration is in `ess.livedata.config.defaults`.
By default the files with the `dev` suffix are used.
Set `LIVEDATA_ENV` to, e.g., `staging` to use the `staging` configuration.

For a local demo, run the fake monitor data producer:

```sh
python -m ess.livedata.services.fake_monitors --mode ev44 --instrument dummy
```

Run the monitor data histogramming and accumulation service:

```sh
python -m ess.livedata.services.monitor_data --instrument dummy
```

Run the reduction dashboard:

```sh
python -m ess.livedata.dashboard.reduction --instrument dummy
```

Navigate to `http://localhost:5009` to see the dashboard.

## Running the services using Docker

Note: The docker is somewhat out of date and not all services are available currently.

You can also run all the services using Docker.
Use the provided `docker-compose-ess.livedata.yml` file to start Kafka:

```sh
LIVEDATA_INSTRUMENT=dummy docker-compose -f docker-compose-ess.livedata.yml up
```

This will start the Zookeeper, Kafka broker.
This can be then used with the services run manually as described above.

Alternatively, you can use profiles to start specific service groups:

```sh
# Start reduction dashboard
LIVEDATA_INSTRUMENT=dummy docker-compose --profile reduction -f docker-compose-ess.livedata.yml up
```

It will take a minute or two for the services to start fully.

When using the `reduction` profile, navigate to `http://localhost:5009` to see the reduction dashboard.

### Kafka Configuration

The services can be configured to connect to different Kafka brokers using environment variables.
There may be two distinct Kafka brokers: one upstream with raw data and one downstream for processed data and ESSlivedata control.

- `KAFKA_BOOTSTRAP_SERVERS`: Bootstrap servers for the upstream Kafka broker.
- `KAFKA_SECURITY_PROTOCOL`: Security protocol for the upstream Kafka broker.
- `KAFKA_SASL_MECHANISM`: SASL mechanism for the upstream Kafka broker.
- `KAFKA_SASL_USERNAME`: SASL username for the upstream Kafka broker.
- `KAFKA_SASL_PASSWORD`: SASL password for the upstream Kafka broker.

- `KAFKA2_BOOTSTRAP_SERVERS`: Bootstrap servers for the downstream Kafka broker.
- `KAFKA2_SECURITY_PROTOCOL`: Security protocol for the downstream Kafka broker.
- `KAFKA2_SASL_MECHANISM`: SASL mechanism for the downstream Kafka broker.
- `KAFKA2_SASL_USERNAME`: SASL username for the downstream Kafka broker.
- `KAFKA2_SASL_PASSWORD`: SASL password for the downstream Kafka broker.

Note that the security and authentication is not necessary when using the Kafka broker from the Docker container.
`KAFKA2_BOOTSTRAP_SERVERS` is also configured to default to use the Kafka broker from the Docker container.