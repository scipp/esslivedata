# Integration Tests

This directory contains integration tests for ESSlivedata. Unlike unit tests that use fakes and mocks, integration tests run actual services (as subprocesses) and verify end-to-end behavior through real Kafka communication.

## Overview

Integration tests verify the interaction between:
- **Dashboard backend** (controllers, services, orchestrator)
- **Backend services** (monitor_data, detector_data, data_reduction, etc.)
- **Kafka** (message broker for communication)

The tests focus on dashboard ↔ backend service integration, using fake data producers to simulate raw data streams.

## Prerequisites

### Kafka

Integration tests require Kafka to be running. Start Kafka using Docker Compose:

```bash
docker-compose up kafka
```

Kafka should be available at `localhost:9092` (default configuration).

### Python Environment

Ensure you have all test dependencies installed:

```bash
# If using micromamba (devcontainer)
# Dependencies are already installed

# If using pip/venv
pip install -r requirements/test.txt
pip install -e .
```

## Running Integration Tests

### Using tox (recommended)

```bash
# Run all integration tests
tox -e integration

# Run specific test file
tox -e integration -- tests/integration/test_workflow_lifecycle.py

# Run specific test function
tox -e integration -- tests/integration/test_workflow_lifecycle.py::test_workflow_can_start_and_receive_data
```

### Using pytest directly

```bash
# Run all integration tests
pytest -m integration tests/integration/

# Run with verbose output
pytest -m integration -v tests/integration/

# Run specific test
pytest -m integration tests/integration/test_workflow_lifecycle.py::test_workflow_can_start_and_receive_data
```

### Running with different instruments

By default, tests use the `dummy` instrument. To test with a different instrument, use the `@pytest.mark.instrument()` marker:

```python
@pytest.mark.instrument('bifrost')
def test_bifrost_workflow(integration_env):
    ...
```

## Architecture

### Components

#### 1. DashboardBackend (`backend.py`)

Reusable dashboard backend without GUI components. Provides:
- All dashboard services (DataService, JobService, WorkflowController, etc.)
- Kafka consumer integration via BackgroundMessageSource
- `update()` method to process message batches
- Context manager for automatic cleanup

**Usage:**
```python
with DashboardBackend(instrument='dummy', dev=True) as backend:
    backend.workflow_controller.start_workflow(...)
    backend.update()  # Process messages
```

#### 2. ServiceProcess (`service_process.py`)

Manages service subprocesses with lifecycle control:
- Starts services with proper command-line arguments
- Graceful shutdown with timeout
- Captures stdout/stderr for debugging
- Context manager for automatic cleanup

**Usage:**
```python
with ServiceProcess('ess.livedata.services.fake_monitors',
                     instrument='dummy', mode='ev44') as service:
    # Service is running
    assert service.is_running()
```

#### 3. Test Helpers (`helpers.py`)

Synchronous utilities for waiting on conditions:
- `wait_for_data(data_service, key, timeout)` - Wait for data in DataService
- `wait_for_job_status(job_service, job_id, status, timeout)` - Wait for job status
- `wait_for_job_data(job_service, job_id, timeout)` - Wait for job data
- `wait_for_condition(condition, timeout)` - Wait for arbitrary condition
- `collect_updates(data_service)` - Context manager to collect all updates

**Usage:**
```python
from .helpers import wait_for_data, wait_for_job_status

# Wait for specific data
data = wait_for_data(backend.data_service, result_key, timeout=5.0)

# Wait for job status
status = wait_for_job_status(backend.job_service, job_id, 'running', timeout=5.0)
```

#### 4. Pytest Fixtures (`conftest.py`)

Reusable fixtures for common test setups:

- **`dashboard_backend`**: Provides a DashboardBackend instance
- **`monitor_services`**: Starts fake_monitors + monitor_data services
- **`detector_services`**: Starts fake_detectors + detector_data services
- **`reduction_services`**: Starts full reduction pipeline
- **`integration_env`**: Combines backend + services into IntegrationEnv

**Usage:**
```python
def test_something(integration_env):
    backend = integration_env.backend
    services = integration_env.services
    # Test implementation
```

## Writing Integration Tests

### Basic Pattern

```python
import pytest
from .conftest import IntegrationEnv
from .helpers import wait_for_data

@pytest.mark.integration
def test_my_workflow(integration_env: IntegrationEnv):
    """Test description."""
    backend = integration_env.backend

    # 1. Start workflow
    backend.workflow_controller.start_workflow(...)

    # 2. Process messages
    for _ in range(5):
        backend.update()

    # 3. Wait for expected behavior
    data = wait_for_data(backend.data_service, key, timeout=5.0)

    # 4. Assert expectations
    assert data is not None

    # 5. Stop workflow
    backend.workflow_controller.stop_workflow(...)
```

### Multi-turn Interaction Pattern

```python
@pytest.mark.integration
def test_workflow_reconfiguration(integration_env: IntegrationEnv):
    """Test workflow reconfiguration."""
    backend = integration_env.backend

    # Start workflow with initial config
    backend.workflow_controller.start_workflow(workflow_id, sources, config1)
    for _ in range(5):
        backend.update()

    data1 = wait_for_data(backend.data_service, key, timeout=5.0)

    # Reconfigure workflow
    backend.workflow_controller.update_config(workflow_id, sources, config2)
    for _ in range(5):
        backend.update()

    data2 = wait_for_data(backend.data_service, key, timeout=5.0)

    # Verify data changed
    assert data2 != data1

    # Stop workflow
    backend.workflow_controller.stop_workflow(sources)
```

### Using Different Instruments

```python
@pytest.mark.integration
@pytest.mark.instrument('bifrost')
def test_bifrost_specific_workflow(integration_env: IntegrationEnv):
    """Test Bifrost-specific workflow."""
    backend = integration_env.backend
    assert integration_env.instrument == 'bifrost'
    # Test implementation
```

### Custom Service Combinations

For tests needing specific services, use fixtures directly:

```python
@pytest.mark.integration
def test_with_custom_services(dashboard_backend):
    """Test with custom service setup."""
    from .service_process import ServiceGroup, ServiceProcess

    services = ServiceGroup({
        'fake_logdata': ServiceProcess('ess.livedata.services.fake_logdata',
                                       instrument='dummy'),
        'timeseries': ServiceProcess('ess.livedata.services.timeseries',
                                     instrument='dummy', dev=True),
    })

    with services:
        # Test implementation
        pass
```

### Writing Tests Robust to Fixture Scope Changes

**IMPORTANT**: Currently, integration test fixtures create new backend processes for every test (function scope). In the future, fixtures may be changed to session or module scope to improve performance by sharing processes across multiple tests.

Write tests that work regardless of whether fixtures are isolated or shared:

#### DO: Filter by Test-Specific Identifiers

Always filter global state by your test's workflow ID or source name before making assertions:

```python
@pytest.mark.integration
def test_my_workflow(integration_env: IntegrationEnv):
    backend = integration_env.backend
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )

    backend.workflow_controller.start_workflow(workflow_id, ['monitor1'], config)

    # ✅ GOOD: Filter by workflow_id before checking
    def check_for_job_data():
        backend.update()
        for job_number, source_data in backend.job_service.job_data.items():
            # Only check jobs belonging to OUR workflow
            if backend.job_service.job_info.get(job_number) == workflow_id:
                if 'monitor1' in source_data:
                    return True
        return False

    wait_for_condition(check_for_job_data, timeout=10.0)

    # ✅ GOOD: Filter workflow-specific jobs
    workflow_jobs = [
        job_num for job_num, wf_id in backend.job_service.job_info.items()
        if wf_id == workflow_id
    ]
    assert len(workflow_jobs) > 0, f"Expected at least one job for {workflow_id}"
```

#### DO: Use Existence Checks

Test for the presence of expected data, not exact counts:

```python
# ✅ GOOD: Check for existence
assert 'monitor1' in source_data

# ✅ GOOD: Check for at least one
assert len(workflow_jobs) > 0

# ❌ BAD: Assumes no other tests have run
assert len(backend.job_service.job_data) == 1
```

#### DO: Use Relative Assertions

Make assertions about your test's data, not global state:

```python
# ✅ GOOD: Compare your own data
assert data_after_config_change != data_before_config_change

# ✅ GOOD: Check properties of your results
assert result.sizes['time'] > 10

# ❌ BAD: Assumes you're the first test
assert job_number == 0
```

#### DON'T: Assert Global State

Avoid assumptions about global state or ordering:

```python
# ❌ BAD: Assumes isolation
assert len(backend.job_service.job_data) == 1
assert len(backend.data_service.get_all_keys()) == 1
assert backend.job_service.next_job_number == 1

# ❌ BAD: Assumes you're first
assert list(backend.job_service.job_info.keys())[0] == 0

# ❌ BAD: Assumes specific timing
time.sleep(2.0)  # Wait for data (use wait_for_condition instead)
```

#### Pattern Summary

**Filter → Check → Assert**

1. **Filter**: Narrow down to your test's data using workflow_id/source_name
2. **Check**: Verify presence/properties of your filtered data
3. **Assert**: Make claims only about your test's data

This pattern ensures tests remain valid whether fixtures are shared or isolated.

## Debugging Integration Tests

### Enable Debug Logging

```python
import logging

@pytest.mark.integration
def test_with_debug_logging(integration_env):
    logging.basicConfig(level=logging.DEBUG)
    # Test implementation
```

### Inspect Service Output

```python
@pytest.mark.integration
def test_check_service_output(monitor_services):
    service = monitor_services['fake_monitors']

    # Get captured output
    stdout = service.get_stdout()
    stderr = service.get_stderr()

    print(f"Service stdout: {stdout}")
    print(f"Service stderr: {stderr}")
```

### Manual Service Inspection

Sometimes it's useful to run services manually for debugging:

```bash
# Terminal 1: Start Kafka
docker-compose up kafka

# Terminal 2: Start fake monitors
python -m ess.livedata.services.fake_monitors --instrument dummy --mode ev44

# Terminal 3: Start monitor_data
python -m ess.livedata.services.monitor_data --instrument dummy --dev

# Terminal 4: Run test
pytest -m integration -s tests/integration/test_workflow_lifecycle.py::test_workflow_can_start_and_receive_data
```

## Common Issues

### Kafka Connection Errors

**Error:** `kafka.errors.NoBrokersAvailable`

**Solution:** Ensure Kafka is running (`docker-compose up kafka`)

### Service Startup Failures

**Error:** Service fails to start or exits immediately

**Solution:**
- Check service logs using `service.get_stderr()`
- Verify instrument name is valid
- Ensure all dependencies are installed

### Test Timeouts

**Error:** `WaitTimeout` raised from helpers

**Solution:**
- Increase timeout values in `wait_for_*` calls
- Ensure services are producing data (check service logs)
- Verify Kafka topics are correctly configured
- Add more `backend.update()` calls to process messages

### Port Conflicts

If services fail with port binding errors, check for other running instances:

```bash
# Check for running Python services
ps aux | grep ess.livedata

# Kill stray services
pkill -f ess.livedata.services
```

## Best Practices

1. **Write scope-agnostic tests**: Always filter by workflow_id/source_name before assertions. See "Writing Tests Robust to Fixture Scope Changes" above for detailed patterns.

2. **Use appropriate fixtures**: Choose the fixture that matches your test needs (monitor_services, detector_services, etc.)

3. **Process messages regularly**: Call `backend.update()` frequently to process Kafka messages

4. **Use helpers for waiting**: Don't use `time.sleep()` for synchronization - use `wait_for_*` helpers instead

5. **Clean up properly**: Use context managers or pytest fixtures to ensure services are stopped

6. **Test in isolation**: Each test should be independent and not rely on state from other tests

7. **Be mindful of timing**: Integration tests involve real services and network I/O - be generous with timeouts

8. **Document test purpose**: Add clear docstrings explaining what each test verifies

## Future Enhancements

- **Fixture scope optimization**: Change fixtures to session/module scope to share backend processes across tests for improved performance (currently function-scoped)
- **Testcontainers integration**: Automatically manage Kafka lifecycle
- **Parallel test execution**: Run tests concurrently with isolated Kafka topics
- **Performance testing**: Measure throughput and latency
- **Error injection**: Test error handling and recovery
- **CI integration**: Automated integration tests in CI pipeline

## Contributing

When adding new integration tests:

1. Follow the patterns in existing tests
2. Use the `@pytest.mark.integration` marker
3. Add clear docstrings
4. Use appropriate helper functions
5. Test cleanup and error cases
6. Update this README if introducing new patterns
