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

### Writing Tests Robust to Fixture Scope Changes

**IMPORTANT**: Currently, integration test fixtures create new backend processes for every test (function scope). In the future, fixtures may be changed to session or module scope to improve performance by sharing processes across multiple tests.

Write tests that work regardless of whether fixtures are isolated or shared:

#### DO: Use Helper Functions to Filter by Workflow

The helper functions in `helpers.py` automatically filter by `workflow_id`, isolating your test's jobs from other workflows:

```python
@pytest.mark.integration
@pytest.mark.services('monitor')
def test_my_workflow(integration_env: IntegrationEnv):
    backend = integration_env.backend
    workflow_id = WorkflowId(...)
    source_names = ['monitor1']

    job_ids = backend.workflow_controller.start_workflow(
        workflow_id, source_names, config
    )

    # ✅ GOOD: Use helpers that filter by workflow_id automatically
    wait_for_workflow_job_data(backend, workflow_id, source_names, timeout=10.0)

    # Retrieve filtered data using helpers
    job_data = get_workflow_job_data(backend.job_service, workflow_id, source_names)
    assert len(job_data) > 0
```

#### DO: Use Existence Checks, Not Exact Counts

```python
# ✅ GOOD: Check for existence in your filtered data
assert 'monitor1' in job_data
assert len(job_data) > 0

# ❌ BAD: Assumes no other workflows running
assert len(backend.job_service.job_data) == 1
```

#### DO: Assert Properties, Not Global State

```python
# ✅ GOOD: Properties of your test's results
assert result.sizes['time'] > 10
assert data_after_change != data_before_change

# ❌ BAD: Assumes you're the first/only test
assert job_ids[0].job_number == 0
assert backend.job_service.next_job_number == 1
```

#### Summary: Filter → Assert

1. **Filter**: Use helpers to get data for your `workflow_id`
2. **Assert**: Check properties/presence of your filtered data only

## Best Practices

1. **Use helpers from `helpers.py`**: They automatically filter by `workflow_id` and handle `backend.update()`
2. **Assert on filtered data only**: Never assert global state (job counts, job numbers, etc.)
3. **Use existence checks**: Test for presence (`> 0`), not exact counts
4. **Add clear docstrings**: Explain what each test verifies
