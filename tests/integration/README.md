# Integration Tests

Integration tests verify end-to-end behavior of ESSlivedata by running actual services (as subprocesses) and communicating through real Kafka. Ensure Kafka is running (`docker-compose up kafka`) before running tests.

## Scope

Every test here needs a broker, is marked `@pytest.mark.integration`, and asserts something *only the wire can show*: that a command became a durable Kafka record, that a real service consumed it, that a real service's output is what the dashboard consumes. Behavior that a fake transport or an injected clock can pin belongs in `tests/dashboard/` or `tests/services/` — an integration test that duplicates it buys nothing and costs tens of seconds of wall clock. Tests that need no broker do not belong here at all: they would silently run in the default suite while `tox -e integration` never selects them.

`helpers.py` is a helper library rather than a test module, so it has its own unit tests in `helpers_test.py` (unmarked, broker-free, and intentionally part of the default suite).

## Pytest Markers

- **`@pytest.mark.integration`**: Marks a test as an integration test
- **`@pytest.mark.instrument('name')`**: Runs test with specified instrument (default: 'dummy')
- **`@pytest.mark.services('name')`**: **Required** when using `integration_env` fixture. Selects the `<name>_services` fixture. `'monitor'`, `'detector'` and `'reduction'` come from `conftest.py`; a test module can define its own group via `create_service_group()`

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
- `topic_high_watermark()`, `wait_for_watermark_advance()`, `wait_for_watermark_stall()` - Observe a topic's offsets without consuming, to prove a message was (or was not) written

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
        name='monitor_histogram',
        version=1,
    )
    source_names = ['monitor1']

    # Start workflow (returns job_ids with unique UUIDs for each source)
    job_ids = backend.workflow_controller.start_workflow(
        workflow_id, source_names, config
    )

    # Use helper to wait for data for the specific jobs we created
    # Returns dict[JobId, dict[ResultKey, data]]
    job_data = wait_for_job_data(backend, workflow_id, job_ids, timeout=10.0)

    # Make assertions about the jobs we created
    assert job_ids[0] in job_data
    assert len(job_data[job_ids[0]]) > 0  # At least one result key

    # Clean up
    backend.workflow_controller.stop_workflow(workflow_id)
```

## Best Practices

1. **Use helpers from `helpers.py`**: They handle `backend.update()` and wait for specific jobs
2. **Wait for the specific jobs you created**: Pass the `job_ids` returned from `start_workflow()` to the helpers
3. **Check properties, not global state**: Assert on your test's data, not total job counts
4. **Add clear docstrings**: Explain what each test verifies
