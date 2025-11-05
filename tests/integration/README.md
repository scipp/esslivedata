# Integration Tests

Integration tests verify end-to-end behavior of ESSlivedata by running actual services (as subprocesses) and communicating through real Kafka. Ensure Kafka is running (`docker-compose up kafka`) before running tests.

## Pytest Markers

- **`@pytest.mark.integration`**: Marks a test as an integration test
- **`@pytest.mark.instrument('name')`**: Runs test with specified instrument (default: 'dummy')
- **`@pytest.mark.services('name')`**: Runs test with specified service combination (e.g., 'monitor', 'detector', 'reduction')

## Available Fixtures

See `conftest.py` for fixture details:
- `integration_env`: Full integration environment (backend + services)
- `dashboard_backend`: Just the backend without services
- `monitor_services`, `detector_services`, `reduction_services`: Specific service combinations

## Test Helpers

See `helpers.py` for synchronous waiting utilities (e.g., `wait_for_data`, `wait_for_condition`). Avoid using `time.sleep()`—use helpers instead.

## Writing Integration Tests

### Basic Pattern

```python
@pytest.mark.integration
def test_my_workflow(integration_env):
    backend = integration_env.backend

    # Start workflow, process messages, wait for results
    backend.workflow_controller.start_workflow(...)
    for _ in range(5):
        backend.update()

    # Assert expectations
    assert result_is_valid(...)

    backend.workflow_controller.stop_workflow(...)
```

### Using Different Instruments

```python
@pytest.mark.integration
@pytest.mark.instrument('bifrost')
def test_bifrost_workflow(integration_env):
    backend = integration_env.backend
    assert integration_env.instrument == 'bifrost'
    # Test implementation
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

## Debugging

- Check service output via `service.get_stderr()` if a service fails to start
- Use `-s` flag with pytest to see print statements: `pytest -m integration -s test_file.py`
- For timeout issues, verify Kafka is running and services are producing data

## Common Issues

| Issue | Solution |
|-------|----------|
| `kafka.errors.NoBrokersAvailable` | Ensure Kafka is running (`docker-compose up kafka`) |
| Service startup failure | Check `service.get_stderr()` for error messages |
| Test timeout / `WaitTimeout` | Increase timeout values, verify services are running and producing data |
| Port binding errors | Check for stray services: `pkill -f ess.livedata.services` |

## Best Practices

1. **Write scope-agnostic tests**: Filter by `workflow_id`/`source_name` before assertions (see "Writing Tests Robust to Fixture Scope Changes")
2. **Choose the right fixture**: Use `integration_env`, `dashboard_backend`, or specific service combinations
3. **Call `backend.update()` regularly**: Ensures messages are processed from Kafka
4. **Use helpers, not `time.sleep()`**: Use `wait_for_*` helpers from `helpers.py` for synchronization
5. **Don't assume test ordering**: Tests may run in any order; don't rely on global state
6. **Add clear docstrings**: Explain what each test verifies
