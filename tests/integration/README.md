# Integration Tests

Integration tests verify end-to-end behavior of ESSlivedata by running actual services (as subprocesses) and communicating through real Kafka. Ensure Kafka is running (`docker-compose up kafka`) before running tests.

## Pytest Markers

- **`@pytest.mark.integration`**: Marks a test as an integration test
- **`@pytest.mark.instrument('name')`**: Runs test with specified instrument (default: 'dummy')
- **`@pytest.mark.services('name')`**: **Required** when using `integration_env` fixture. Valid values: `'monitor'`, `'detector'`, or `'reduction'` (specifies which services to run)

## Available Fixtures

See `conftest.py` for fixture details:
- `integration_env`: Full integration environment (backend + services)
- `dashboard_backend`: Just the backend without services
- `monitor_services`, `detector_services`, `reduction_services`: Specific service combinations

## Test Helpers

See `helpers.py` for utilities that wait for specific jobs:

- `wait_for_job_data()` - Wait for data to arrive for specific job(s)
- `wait_for_job_statuses()` - Wait for status updates for specific job(s)
- `wait_for_condition()` - Generic condition waiter

**Always use helpers instead of `time.sleep()` or manual `backend.update()` loops.**

## Writing Integration Tests

### Basic Pattern

```python
@pytest.mark.integration
@pytest.mark.services('monitor')
def test_my_workflow(integration_env):
    backend = integration_env.backend

    # Define the workflow type to test
    workflow_id = WorkflowId(
        instrument='dummy',
        namespace='monitor_data',
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    # Start workflow (returns job_ids with unique UUIDs for each source)
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id, source_names, config
    )

    # Use helper to wait for data for the specific jobs we created
    wait_for_job_data(backend, job_ids, timeout=10.0)

    # Make assertions about the jobs we created
    job_data = backend.job_service.job_data[job_ids[0].job_number]
    assert 'monitor1' in job_data

    # Clean up
    backend.workflow_controller.stop_workflow(workflow_id)
```

## Best Practices

1. **Use helpers from `helpers.py`**: They handle `backend.update()` and wait for specific jobs
2. **Wait for the specific jobs you created**: Pass the `job_ids` returned from `start_workflow()` to the helpers
3. **Check properties, not global state**: Assert on your test's data, not total job counts
4. **Add clear docstrings**: Explain what each test verifies

Example of what to avoid:
```python
# ❌ BAD: Assumes no other workflows or tests
assert len(backend.job_service.job_data) == 1
assert job_ids[0].job_number == 0
```
