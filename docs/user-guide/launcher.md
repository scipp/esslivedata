# ESSlivedata Launcher

The ESSlivedata launcher allows you to run all services in a single process using threads. This simplifies development and testing by eliminating the need to manage multiple terminal windows.

## Quick Start

### Run the DREAM demo with a single command

```bash
# Start Kafka (required)
docker-compose up kafka

# Run all DREAM services in one process
python -m ess.livedata.launcher --dream-demo
```

This is equivalent to the traditional method of running 8 separate terminal windows as described in the [DREAM demo guide](dream-demo.md).

### Run specific services

```bash
# Run only the fake data producers
python -m ess.livedata.launcher --fake-detectors --fake-monitors --instrument dummy --dev

# Run processing services and dashboard
python -m ess.livedata.launcher --monitor-data --data-reduction --dashboard --instrument dummy --dev

# Run everything
python -m ess.livedata.launcher --all --instrument dummy --dev
```

## Command-Line Options

### Basic Options

- `--instrument INSTRUMENT`: Select the instrument (default: dummy)
- `--dev`: Run in development mode with simplified topic naming
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Service Selection

Enable individual services:

- `--fake-detectors`: Fake detector data producer
- `--fake-monitors`: Fake monitor data producer
- `--fake-logdata`: Fake log data producer
- `--monitor-data`: Monitor data processing service
- `--detector-data`: Detector data processing service
- `--data-reduction`: Data reduction service
- `--timeseries`: Timeseries service
- `--dashboard`: Reduction dashboard (runs in main thread)

Or enable all services:

- `--all`: Enable all services

### Fake Service Options

- `--nexus-file FILE`: NeXus file for realistic fake detector data
- `--monitor-mode MODE`: Monitor data mode (ev44 or da00, default: da00)
- `--num-monitors N`: Number of monitors to simulate (1-10, default: 2)

### Preset Configurations

- `--dream-demo`: Run the complete DREAM demo setup (equivalent to `--instrument dream --dev --all`)

## Architecture

The launcher runs each service in a separate thread within a single Python process:

```
Main Process
├── Thread: fake-detectors
├── Thread: fake-monitors
├── Thread: fake-logdata
├── Thread: monitor-data
├── Thread: detector-data
├── Thread: data-reduction
├── Thread: timeseries
└── Main Thread: dashboard (Panel requirement)
```

Services communicate via Kafka exactly as they would when run in separate processes. This is Phase 1 of the migration away from requiring Kafka for local development.

## Comparison with Traditional Method

### Traditional (Multiple Terminals)

```bash
# Terminal 1
docker-compose up kafka

# Terminal 2
python -m ess.livedata.services.fake_detectors --instrument dream

# Terminal 3
python -m ess.livedata.services.fake_monitors --mode da00 --instrument dream

# Terminal 4
python -m ess.livedata.services.fake_logdata --instrument dream

# Terminal 5
python -m ess.livedata.services.monitor_data --instrument dream --dev

# Terminal 6
python -m ess.livedata.services.detector_data --instrument dream --dev

# Terminal 7
python -m ess.livedata.services.data_reduction --instrument dream --dev

# Terminal 8
python -m ess.livedata.services.timeseries --instrument dream --dev

# Terminal 9
python -m ess.livedata.dashboard.reduction --instrument dream
```

### With Launcher (Single Terminal)

```bash
# Terminal 1
docker-compose up kafka

# Terminal 2
python -m ess.livedata.launcher --dream-demo
```

## Graceful Shutdown

The launcher handles Ctrl+C gracefully:

1. Sends shutdown signal to all services
2. Waits for services to complete current work
3. Stops services in reverse order
4. Ensures clean resource cleanup

## Logging

All services log to the console with unified formatting:

```
2025-01-15 10:30:45 - launcher.fake-detectors - INFO - fake-detectors service started
2025-01-15 10:30:45 - launcher.monitor-data - INFO - monitor-data service started
2025-01-15 10:30:46 - ess.livedata.kafka - INFO - Subscribing to topics: ['dummy_monitor_ev44']
```

Use `--log-level DEBUG` for detailed debugging information.

## Troubleshooting

### Services not communicating

Ensure Kafka is running:
```bash
docker-compose up kafka
```

### Dashboard not showing data

1. Check that fake data producers are enabled (`--fake-detectors --fake-monitors`)
2. Verify the correct instrument is selected
3. Use `--dev` flag for local development

### High CPU usage

The launcher runs all services in threads. For production deployments, use the traditional separate-process approach or container orchestration.

## Future Enhancements (Phase 2)

The current launcher still requires Kafka. Phase 2 will introduce in-memory queues for local development, eliminating the Kafka dependency entirely for testing and development scenarios.