# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESSlivedata is a live data reduction visualization framework for the European Spallation Source (ESS). It processes real-time neutron detector data via Kafka streams and provides interactive dashboards for monitoring and data reduction workflows.

## Development Commands

### Setup and Installation

```sh
# Install development dependencies
pip install -r requirements/dev.txt

# Install package in editable mode
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

### Running Tests

All unit tests run without Kafka - no Docker container needed.

```sh
# Run all tests
tox

# Run tests for specific Python version
tox -e py311

# Run tests manually with pytest
python -m pytest

# Run specific test file
python -m pytest tests/core/processor_test.py

# Run tests with benchmarks
python -m pytest --benchmark-only
```

### Code Quality

```sh
# Run all pre-commit checks (formatting, linting, static analysis)
tox -e static

# Run ruff linting (primary linting tool)
ruff check .

# Run ruff formatting
ruff format .

# Type checking with mypy (minimize errors, but not strictly enforced)
tox -e mypy
```

**Note**: The project primarily relies on `ruff` for linting. `mypy` type checking is run but not strictly enforced - aim to minimize errors where practical.

### Documentation

```sh
# Build documentation
tox -e docs

# Build manually
python -m sphinx -v -b html -d .tox/docs_doctrees docs html

# Run doctest
python -m sphinx -v -b doctest -d .tox/docs_doctrees docs html

# Check links
tox -e linkcheck
```

### Dependency Management

```sh
# Update dependencies (runs pip-compile-multi)
tox -e deps
```

**Important**: After changing dependencies in `pyproject.toml`, always run `tox -e deps` to update the requirement files.

### Project Template Management

This project uses [Copier](https://copier.readthedocs.io/) with the Scipp template (https://github.com/scipp/copier_template):

```sh
# Update project skeleton from template
copier update
```

The template manages the project structure, configuration files, and development tooling. Configuration is stored in `.copier-answers.yml`.

### Running Services Locally

```sh
# Start Kafka using Docker
docker-compose up kafka

# Run fake data producers for testing
python -m ess.livedata.services.fake_monitors --mode ev44 --instrument dummy
python -m ess.livedata.services.fake_detectors --instrument dummy
python -m ess.livedata.services.fake_logdata --instrument dummy

# Run main processing services (use --dev for local testing)
python -m ess.livedata.services.monitor_data --instrument dummy --dev
python -m ess.livedata.services.detector_data --instrument dummy --dev
python -m ess.livedata.services.data_reduction --instrument dummy --dev
python -m ess.livedata.services.timeseries --instrument dummy --dev

# Run dashboard in development mode
python -m ess.livedata.dashboard.reduction --instrument dummy

# Run dashboard in production mode with gunicorn (port 5009)
LIVEDATA_INSTRUMENT=dummy gunicorn ess.livedata.dashboard.reduction_wsgi:application
```

Note: Use `--sink png` argument with processing services to save outputs as PNG files instead of publishing to Kafka for testing.

## Architecture Overview

### Core Architecture Pattern

The codebase follows a **message-driven service architecture** with these key abstractions:

- **Service**: Top-level lifecycle manager that runs processors in a loop
- **Processor**: Orchestrates message processing (typically `StreamProcessor`)
- **Handler**: Business logic for processing messages from specific streams
- **MessageSource**: Abstraction for consuming messages (e.g., from Kafka)
- **MessageSink**: Abstraction for publishing results (e.g., to Kafka)

### Message Flow

```
Kafka Topics → MessageSource → Processor → Handler → MessageSink → Kafka Topics
```

1. Messages arrive from Kafka via `MessageSource` (e.g., `BackgroundMessageSource` wrapping `KafkaConsumer`)
2. `StreamProcessor` batches messages by stream key and routes to appropriate handlers
3. Handlers (registered in `HandlerRegistry`) process messages and return results
4. Results are published via `MessageSink` (e.g., `KafkaSink`)

### Key Components

**Core Layer** (`src/ess/livedata/core/`):
- `service.py`: Service lifecycle management with signal handling
- `processor.py`: `StreamProcessor` routes messages to handlers
- `handler.py`: Base handler protocol and registry
- `message.py`: Core message types (`Message`, `StreamId`, `StreamKind`)

**Kafka Layer** (`src/ess/livedata/kafka/`):
- `source.py`: Kafka consumers with background polling
- `sink.py`: Kafka producers for publishing results
- `message_adapter.py`: Adapts raw Kafka messages to domain types
- `stream_mapping.py`: Maps Kafka topics to stream identifiers

**Configuration** (`src/ess/livedata/config/`):
- `config_loader.py`: Loads YAML/Jinja2 configurations per instrument
- `instruments/`: Instrument-specific configurations (DREAM, Bifrost, LOKI, etc.)
- `workflows.py`: Workflow definitions using sciline workflows

**Handlers** (`src/ess/livedata/handlers/`):
- `detector_data_handler.py`: Handles detector events
- `monitor_data_handler.py`: Handles monitor data
- `data_reduction_handler.py`: Executes reduction workflows
- `workflow_factory.py`: Creates workflow graphs

**Dashboard** (`src/ess/livedata/dashboard/`):
- Uses Panel/Holoviews for interactive visualizations
- Implements MVC pattern with controllers mediating between services and widgets
- `ConfigService`: Central configuration management with Pydantic models
- `DataService`: Manages data streams and notifies subscribers
- `WorkflowController`: Orchestrates workflow configuration and execution
- `ConfigBackedParam`: Translation layer between Param widgets and Pydantic models

### Dashboard Architecture

The dashboard follows a **layered MVC architecture**:

1. **Presentation Layer**: Panel widgets and Holoviews plots
2. **Application Layer**: Controllers (`WorkflowController`), Services (`ConfigService`, `DataService`)
3. **Infrastructure Layer**: Kafka integration (`KafkaTransport`, `BackgroundMessageBridge`)

**Key Pattern**: Separation between:
- **Pydantic models**: Backend validation, serialization, Kafka communication
- **Param models**: GUI widgets and user interaction
- **ConfigBackedParam**: Translation layer bridging the two (for simple controls)

**Threading Model**:
- Background threads handle Kafka polling/publishing
- Queue-based communication prevents blocking the GUI
- Batched message processing for efficiency

See [docs/developer/design/dashboard-architecture.md](docs/developer/design/dashboard-architecture.md) for detailed architecture diagrams.

## Service Factory Pattern

New services are created using `DataServiceBuilder`:

```python
builder = DataServiceBuilder(
    instrument='dummy',
    name='my_service',
    handler_factory=MyHandlerFactory(),
    adapter=MyMessageAdapter()  # optional
)
service = builder.build_from_config(topics=[...])
service.start()
```

See [src/ess/livedata/service_factory.py](src/ess/livedata/service_factory.py) for details.

## Configuration System

- Configuration files are in YAML (with Jinja2 templating support)
- Located in `src/ess/livedata/config/defaults/`
- Environment variable `LIVEDATA_ENV` selects configuration (dev, staging, production)
- Instrument-specific configs are in `src/ess/livedata/config/instruments/`
- Use `config_names.CONFIG_STREAM` etc. for accessing config stream names

## Testing

- Tests are in `tests/` mirroring the `src/ess/livedata/` structure
- Use fakes from `fakes.py` for testing (e.g., `FakeMessageSource`, `FakeMessageSink`)
- Test files follow pattern `*_test.py`
- Tests use `pytest` with `--import-mode=importlib`

## Code Style Conventions

- **Formatting**: Enforced by `ruff` (no manual conventions)
- **Docstrings**: NumPy format (see [docs/developer/coding-conventions.md](docs/developer/coding-conventions.md))
- **Type hints**: Required; `mypy` is used for type checking but not strictly enforced
- **Line length**: 88 characters
- Do not include type annotations in docstring parameters (handled by sphinx-autodocs-typehints)

### Docstring Example

```python
def process(x: int) -> float:
    """Short description.

    Parameters
    ----------
    x:
        Description of x.

    Returns
    -------
    :
        Description of return value.
    """
```

For single-sentence docstrings (no params/returns sections needed):

```python
def simple_method(self) -> int:
    """Returns the number of dimensions."""
```

## Important Patterns

### Adding a New Service

1. Create handler factory implementing `HandlerFactory[Tin, Tout]`
2. Create handlers implementing `Handler` protocol
3. Use `DataServiceBuilder` to construct service
4. Register service in `services/` module
5. Add instrument configuration in `config/defaults/`

### Adding Dashboard Widgets

1. Create Pydantic model for backend validation
2. Create Param model (subclass `ConfigBackedParam` for simple controls)
3. Subscribe widget to `ConfigService`
4. For complex workflows, use a centralized controller (e.g., `WorkflowController`)

### Message Processing

- Messages have `timestamp`, `stream`, and `value` fields
- `StreamId` identifies message type (kind + name)
- Use `compact_messages()` to deduplicate by keeping latest per stream
- Handlers receive batched messages for the same stream

### Background Processing

- Use `BackgroundMessageSource` for Kafka consumers (non-blocking polls)
- Use `BackgroundMessageBridge` in dashboards to prevent GUI blocking
- Queue-based communication between threads

## Key Environment Variables

- `LIVEDATA_ENV`: Configuration environment (dev, staging, production)
- `LIVEDATA_INSTRUMENT`: Instrument name (dummy, DREAM, Bifrost, LOKI, etc.)
- `KAFKA_BOOTSTRAP_SERVERS`: Upstream Kafka broker (raw data)
- `KAFKA2_BOOTSTRAP_SERVERS`: Downstream Kafka broker (processed data, defaults to Docker container)
- `JUPYTER_PLATFORM_DIRS`: Set to 1 for tests

## Instrument Support

Available instruments are registered in `src/ess/livedata/config/instruments/`:
- `dummy`: Test instrument
- `dream`: DREAM diffractometer (requires `essdiffraction`)
- `bifrost`: Bifrost spectrometer (requires `essspectroscopy`)
- `loki`: LOKI SANS (requires `esssans`)
- `odin`: ODIN imaging (requires `essimaging`)
- `nmx`: Macromolecular crystallography
- `tbl`: Test Beamline

Install optional dependencies: `pip install esslivedata[dream]`, `pip install esslivedata[bifrost]`, etc.

## Common Gotchas

- Always run `tox -e deps` after changing dependencies in `pyproject.toml`
- Use `--dev` flag when testing services locally to use simplified topic structure
- Services (not unit tests) require Kafka to be running; use `docker-compose up kafka` for local development
- All unit tests run independently without Kafka
- Dashboard runs on port 5009
- Pre-commit hooks will auto-format code; if they make changes, commit will be rejected (re-stage and commit again)
- When adding new message types, register them in `StreamKind` enum
- Project skeleton is managed by Copier template - use `copier update` to sync with upstream template changes
