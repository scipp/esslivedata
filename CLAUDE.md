# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESSlivedata is a live data reduction visualization framework for the European Spallation Source (ESS). It processes real-time neutron detector data via Kafka streams and provides interactive dashboards for monitoring and data reduction workflows.

## Development Commands

### Environment Setup

**IMPORTANT**: In the devcontainer, this project uses micromamba with Python 3.11 in the base environment. The environment is automatically activated - you do not need to activate it manually.

**Note**: In the devcontainer, all Python commands (`python`, `pytest`, `tox`, etc.) automatically use the micromamba base environment. Pre-commit hooks will run automatically on `git commit` if properly installed.

### Running Tests

All unit tests run without Kafka - no Docker container needed.

```sh
# Run all tests
tox

# Run tests manually with pytest
python -m pytest

# Run specific test file
python -m pytest tests/core/processor_test.py
```

### Code Quality

Use the `linter` agent for code quality checks. Tools: `ruff` (primary, enforced in CI), `pylint` (optional), `mypy` (optional, not enforced). Run `tox -e static` for all pre-commit checks.

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

# Run dashboard (port 5009)
python -m ess.livedata.dashboard.reduction --instrument dummy
```

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

### Dashboard Architecture

The dashboard follows a **layered MVC architecture**:

1. **Presentation Layer**: Panel widgets and Holoviews plots
2. **Application Layer**: Controllers (`WorkflowController`), Services (`ConfigService`, `DataService`)
3. **Infrastructure Layer**: Kafka integration (`KafkaTransport`, `BackgroundMessageBridge`)

**Key Pattern**: Separation between:
- **Pydantic models**: Backend validation, serialization, Kafka communication
- **Param models**: GUI widgets and user interaction. Concrete fields (such as `int`, `bool`, ...) must be in a nested model or widget generation will fail.
- **ConfigBackedParam**: Translation layer bridging the two (for simple controls)

**Threading Model**:
- Background threads handle Kafka polling/publishing
- Queue-based communication prevents blocking the GUI
- Batched message processing for efficiency

See [docs/developer/design/dashboard-architecture.md](docs/developer/design/dashboard-architecture.md) for detailed architecture diagrams.

## Detailed Architecture Documentation

For in-depth understanding of ESSlivedata's architecture, see the following design documents:

- **[Backend Service Architecture](docs/developer/design/backend-service-architecture.md)**: Service-Processor-Workflow pattern, job management, and service lifecycle. Job lifecycle, scheduling, primary vs auxiliary data, and workflow protocol
- **[Message Flow and Transformation](docs/developer/design/message-flow-and-transformation.md)**: End-to-end message journey, adapters, stream mapping, and batching strategies
- **[Dashboard Architecture](docs/developer/design/dashboard-architecture.md)**: MVC pattern, configuration management, and threading model

## Service Factory Pattern

New services are created using `DataServiceBuilder`:
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

### Slow Tests

Tests marked with `@pytest.mark.slow` are excluded by default when running `pytest` directly:

```sh
# Default: runs fast tests only (~25s)
python -m pytest

# Include slow tests (~85s total)
python -m pytest -m "not integration"

# Run only slow tests
python -m pytest -m slow
```

Slow tests are automatically included when running via `tox` (CI). The slow marker is applied to tests taking >2s, primarily Bifrost and LOKI data reduction workflow tests.

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

### Message Processing

- Messages have `timestamp`, `stream`, and `value` fields
- `StreamId` identifies message type (kind + name)
- Preprocessors accumulate messages via `add()`, return accumulated data via `get()`
- Workflows receive accumulated data and execute scientific reduction logic

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
