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

See `helpers.py` for utilities that filter by `workflow_id` and wait for conditions:

- `wait_for_workflow_job_data()` - Wait for job data, handles `backend.update()` and filtering
- `wait_for_workflow_job_statuses()` - Wait for job status updates
- `get_workflow_jobs()` - Get job numbers for a workflow
- `get_workflow_job_data()` - Get job data filtered by workflow and sources
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

    # Use helper to wait for data (handles backend.update() and filtering)
    wait_for_workflow_job_data(backend, workflow_id, source_names, timeout=10.0)

    # Make assertions about the jobs we created
    job_data = backend.job_service.job_data[job_ids[0].job_number]
    assert 'monitor1' in job_data

    # Clean up
    backend.workflow_controller.stop_workflow(workflow_id)
```

## Best Practices

1. **Use helpers from `helpers.py`**: They handle `backend.update()` and filter by `workflow_id`
2. **Check properties, not global state**: Assert on your test's data, not total job counts or job numbers
3. **Add clear docstrings**: Explain what each test verifies

Example of what to avoid:
```python
# ❌ BAD: Assumes no other workflows or tests
assert len(backend.job_service.job_data) == 1
assert job_ids[0].job_number == 0
```
