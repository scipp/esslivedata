## Project Overview

ESSlivedata is a live data reduction visualization framework for the European Spallation Source (ESS). It processes real-time neutron detector data via Kafka streams and provides interactive dashboards for monitoring and data reduction workflows.

## Development Commands

### Environment Setup

In the devcontainer, micromamba (Python 3.11) is auto-activated -- `python`, `pytest`, `tox` etc. just work.

**Worktree Setup** (when launched with `claude -w`):

```sh
python -m venv .venv && source .venv/bin/activate && pip install -e ".[test]"
```

### Tests

- Tests in `tests/` mirror `src/ess/livedata/` structure, files follow `*_test.py`
- All unit tests run without Kafka -- use fakes from `fakes.py`
- `pytest` uses `--import-mode=importlib`

```sh
python -m pytest                      # fast tests only (~25s)
python -m pytest -m "not integration" # include @pytest.mark.slow (~85s)
tox                                   # CI: includes slow tests
```

### Code Quality

Use the `linter` agent. Tools: `ruff` (primary, CI-enforced), `pylint` (optional), `mypy` (optional). Run `tox -e static` for all pre-commit checks.

### Documentation

```sh
tox -e docs       # build HTML docs
tox -e linkcheck  # check links
# Manual: python -m sphinx -v -b html -d .tox/docs_doctrees docs html
# Doctest: python -m sphinx -v -b doctest -d .tox/docs_doctrees docs html
```

### Project Template

Uses [Copier](https://copier.readthedocs.io/) with [Scipp template](https://github.com/scipp/copier_template). Config in `.copier-answers.yml`. Update: `copier update`.

### Running Services Locally

Services require Kafka (`docker-compose up kafka`). Use `--dev` for simplified topic structure.

Services: `fake_monitors`, `fake_detectors`, `fake_logdata`, `monitor_data`, `detector_data`, `data_reduction`, `timeseries`.
Run as: `python -m ess.livedata.services.<name> --instrument dummy [--dev]`

Dashboard: `python -m ess.livedata.dashboard.reduction --instrument dummy`

## Tools

`src/ess/livedata/nexus_helpers.py` -- utilities for extracting Kafka topic and source names from NeXus files.

## Architecture Overview

### Core Abstractions

Message-driven service architecture:

- **Service**: Top-level lifecycle manager running processors in a loop
- **Workflow**: Scientific reduction logic processing accumulated data (instances run as `Job`)
- **MessageSource / MessageSink**: Abstractions for consuming/publishing messages (e.g., Kafka)

### Message Flow

```
Kafka Topics -> MessageSource -> OrchestratingProcessor -> Preprocessors -> JobManager -> Workflow -> MessageSink -> Kafka Topics
```

### Key Components

- **`core/`**: `service.py` (lifecycle), `orchestrating_processor.py` (job-based batching), `handler.py` (preprocessor factory/protocols), `message.py` (Message, StreamId, StreamKind), `job_manager.py` (scheduling)
- **`kafka/`**: `source.py` (consumers), `sink.py` (producers), `message_adapter.py` (raw -> domain), `stream_mapping.py` (topic -> stream)
- **`config/`**: YAML + Jinja2 configs. Defaults in `config/defaults/`, per-instrument in `config/instruments/`
- **`handlers/`**: Preprocessor factories for detector, monitor, and reduction data; accumulators; workflow protocol
- **`dashboard/`**: Panel/HoloViews visualizations, MVC pattern, `DataService` for data stream subscriptions

## Code Style

- Formatting: `ruff` (88 char lines)
- Type hints: required; `mypy` used but not strictly enforced
- Docstrings: NumPy format, no type annotations in params (sphinx-autodoc-typehints handles it)

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

Single-sentence: `"""Returns the number of dimensions."""`

## Instrument Support

Instruments registered in `src/ess/livedata/config/instruments/`: `dummy`, `dream`, `bifrost`, `loki`, `odin`, `nmx`, `tbl`.
Optional deps installed as extras: `pip install esslivedata[dream]`, etc.
