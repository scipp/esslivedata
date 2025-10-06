# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESSlivedata is a live data reduction visualization framework for the European Spallation Source (ESS). It processes real-time neutron detector data via Kafka streams and provides interactive dashboards for monitoring and data reduction workflows.

## Development Commands

### Environment Setup

**IMPORTANT**: This project uses a mamba environment with Python 3.11. The devcontainer includes mamba pre-installed.

```sh
# Create mamba environment with Python 3.11
mamba create -n esslivedata python=3.11 -y

# Install development dependencies (note: skips docs.txt due to dependency conflict)
mamba run -n esslivedata pip install -r requirements/base.txt -r requirements/basetest.txt -r requirements/static.txt pre-commit

# Install package in editable mode
mamba run -n esslivedata pip install -e .

# Setup pre-commit hooks (automatically runs on git commit)
mamba run -n esslivedata pre-commit install
```

**For Claude Code**: Always use `mamba run -n esslivedata <command>` to run commands in the environment:
- Use `mamba run -n esslivedata <command>` instead of activating the environment
- Example: `mamba run -n esslivedata pytest` or `mamba run -n esslivedata python -m pytest`
- Pre-commit hooks will run automatically on `git commit` if properly installed
- The environment includes all tools needed for testing, linting, and development
- When making commits, use `mamba run -n esslivedata git commit` to ensure pre-commit hooks run correctly

### Running Tests

All unit tests run without Kafka - no Docker container needed.

```sh
# Run all tests
tox

# Run tests for specific Python version
tox -e py311

# Run tests manually with pytest (using mamba environment)
mamba run -n esslivedata python -m pytest

# Run specific test file
mamba run -n esslivedata python -m pytest tests/core/processor_test.py

# Run tests with benchmarks
mamba run -n esslivedata python -m pytest --benchmark-only
```

### Code Quality

```sh
# Run all pre-commit checks (formatting, linting, static analysis)
tox -e static

# Run ruff linting (primary linting tool) - using mamba environment
mamba run -n esslivedata ruff check .

# Run ruff formatting - using mamba environment
mamba run -n esslivedata ruff format .

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

# Run fake data producers for testing (using mamba environment)
mamba run -n esslivedata python -m ess.livedata.services.fake_monitors --mode ev44 --instrument dummy
mamba run -n esslivedata python -m ess.livedata.services.fake_detectors --instrument dummy
mamba run -n esslivedata python -m ess.livedata.services.fake_logdata --instrument dummy

# Run main processing services (use --dev for local testing)
mamba run -n esslivedata python -m ess.livedata.services.monitor_data --instrument dummy --dev
mamba run -n esslivedata python -m ess.livedata.services.detector_data --instrument dummy --dev
mamba run -n esslivedata python -m ess.livedata.services.data_reduction --instrument dummy --dev
mamba run -n esslivedata python -m ess.livedata.services.timeseries --instrument dummy --dev

# Run dashboard in development mode
mamba run -n esslivedata python -m ess.livedata.dashboard.reduction --instrument dummy

# Run dashboard in production mode with gunicorn (port 5009)
mamba run -n esslivedata bash -c "LIVEDATA_INSTRUMENT=dummy gunicorn ess.livedata.dashboard.reduction_wsgi:application"
```

Note: Use `--sink png` argument with processing services to save outputs as PNG files instead of publishing to Kafka for testing.

## Architecture Overview

### Core Architecture Pattern

The codebase follows a **message-driven service architecture** with these key abstractions:

- **Service**: Top-level lifecycle manager that runs processors in a loop
- **Processor**: Orchestrates message processing (always `OrchestratingProcessor`)
- **PreprocessorFactory**: Creates accumulators for different message stream types
- **Accumulator**: Preprocesses and accumulates messages before workflow execution
- **Workflow**: Scientific reduction logic that processes accumulated data
- **MessageSource**: Abstraction for consuming messages (e.g., from Kafka)
- **MessageSink**: Abstraction for publishing results (e.g., to Kafka)

### Message Flow

```
Kafka Topics → MessageSource → Processor → Preprocessor → JobManager → Workflow → MessageSink → Kafka Topics
```

1. Messages arrive from Kafka via `MessageSource` (e.g., `BackgroundMessageSource` wrapping `KafkaConsumer`)
2. `OrchestratingProcessor` batches messages by time window
3. Preprocessors (accumulators) transform and accumulate messages
4. `JobManager` schedules workflow execution with accumulated data
5. Workflows execute scientific reduction logic
6. Results are published via `MessageSink` (e.g., `KafkaSink`)

### Key Components

**Core Layer** (`src/ess/livedata/core/`):
- `service.py`: Service lifecycle management with signal handling
- `orchestrating_processor.py`: `OrchestratingProcessor` manages job-based processing
- `handler.py`: Preprocessor factory and accumulator protocols
- `message.py`: Core message types (`Message`, `StreamId`, `StreamKind`)
- `job_manager.py`: Manages workflow job scheduling and execution

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
- `detector_data_handler.py`: Preprocessor factory for detector events
- `monitor_data_handler.py`: Preprocessor factory for monitor data
- `data_reduction_handler.py`: Preprocessor factory for reduction workflows
- `accumulators.py`: Common preprocessor/accumulator implementations
- `workflow_factory.py`: Workflow protocol and factory interfaces

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

## Detailed Architecture Documentation

For in-depth understanding of ESSlivedata's architecture, see the following design documents:

- **[Backend Service Architecture](docs/developer/design/backend-service-architecture.md)**: Service-Processor-Workflow pattern, job management, and service lifecycle
- **[Message Flow and Transformation](docs/developer/design/message-flow-and-transformation.md)**: End-to-end message journey, adapters, stream mapping, and batching strategies
- **[Job-Based Processing](docs/developer/design/job-based-processing.md)**: Job lifecycle, scheduling, primary vs auxiliary data, and workflow protocol
- **[Testing Guide](docs/developer/testing-guide.md)**: Unit testing with fakes, integration testing, and testing strategies
- **[Dashboard Architecture](docs/developer/design/dashboard-architecture.md)**: MVC pattern, configuration management, and threading model

## Service Factory Pattern

New services are created using `DataServiceBuilder`:

```python
builder = DataServiceBuilder(
    instrument='dummy',
    name='my_service',
    preprocessor_factory=MyPreprocessorFactory(),
    adapter=MyMessageAdapter()  # optional
)
service = builder.build_from_config(topics=[...])
service.start()
```

All services use `OrchestratingProcessor` for job-based processing. See [src/ess/livedata/service_factory.py](src/ess/livedata/service_factory.py) for details.

## Configuration System

- Configuration files are in YAML (with Jinja2 templating support)
- Located in `src/ess/livedata/config/defaults/`
- Environment variable `LIVEDATA_ENV` selects configuration (dev, staging, production)
- Instrument-specific configs are in `src/ess/livedata/config/instruments/`
- Use `config_names.CONFIG_STREAM` etc. for accessing config stream names

## Testing

- Tests are in `tests/` mirroring the `src/ess/livedata/` structure
- **All unit tests run without Kafka** - use fakes from `fakes.py` (e.g., `FakeMessageSource`, `FakeMessageSink`)
- Test files follow pattern `*_test.py`
- Tests use `pytest` with `--import-mode=importlib`
- See [Testing Guide](docs/developer/testing-guide.md) for comprehensive testing strategies and examples

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

1. Create preprocessor factory extending `JobBasedPreprocessorFactoryBase[Tin, Tout]`
2. Implement `make_preprocessor()` to create accumulators for different stream types
3. Register workflows with the instrument configuration
4. Use `DataServiceBuilder` to construct service
5. Add service module in `services/`
6. Add instrument configuration in `config/defaults/`

### Adding Dashboard Widgets

1. Create Pydantic model for backend validation
2. Create Param model (subclass `ConfigBackedParam` for simple controls)
3. Subscribe widget to `ConfigService`
4. For complex workflows, use a centralized controller (e.g., `WorkflowController`)

### Message Processing

- Messages have `timestamp`, `stream`, and `value` fields
- `StreamId` identifies message type (kind + name)
- Use `compact_messages()` to deduplicate by keeping latest per stream
- Preprocessors accumulate messages via `add()`, return accumulated data via `get()`
- Workflows receive accumulated data and execute scientific reduction logic

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
