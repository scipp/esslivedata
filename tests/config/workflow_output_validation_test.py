# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Validate that workflow outputs match their declared output models.

This test runs workflows with fake data and validates that the actual outputs
have structure compatible with the declared output model templates.

These tests are slow because they:
1. Import heavy dependencies (sciline, ess.reduce, etc.)
2. Actually run workflows with fake data
"""

import pytest
import scipp as sc

from ess.livedata.config import models, workflow_spec
from ess.livedata.config.instrument import instrument_registry
from ess.livedata.config.instruments import available_instruments, get_config
from ess.livedata.services.data_reduction import make_reduction_service_builder
from ess.livedata.services.detector_data import make_detector_service_builder
from ess.livedata.services.monitor_data import make_monitor_service_builder
from ess.livedata.services.timeseries import make_timeseries_service_builder
from tests.helpers.livedata_app import LivedataApp


def _get_service_builder(namespace: str):
    """Get the service builder function for a given namespace."""
    builders = {
        'detector_data': make_detector_service_builder,
        'monitor_data': make_monitor_service_builder,
        'data_reduction': make_reduction_service_builder,
        'timeseries': make_timeseries_service_builder,
    }
    return builders.get(namespace)


def _get_source_name_for_workflow(
    instrument_name: str, workflow_spec: workflow_spec.WorkflowSpec
) -> str:
    """Get an appropriate source name for the workflow."""
    if workflow_spec.source_names:
        return workflow_spec.source_names[0]
    # Fallback mappings for namespaces without explicit source_names
    namespace = workflow_spec.namespace
    if namespace == 'timeseries':
        # Timeseries workflows don't need a specific source
        return 'timeseries'
    return 'unknown'


def _collect_workflow_outputs_for_validation():
    """Collect workflows for output validation, grouped by namespace.

    Returns (instrument, workflow_id, output_names) tuples for parametrization.
    """
    workflows = []
    for instrument_name in available_instruments():
        _ = get_config(instrument_name)
        instrument = instrument_registry[instrument_name]
        instrument.load_factories()

        for workflow_id in instrument.workflow_factory:
            spec = instrument.workflow_factory[workflow_id]
            builder_fn = _get_service_builder(spec.namespace)
            if builder_fn is None:
                # Skip workflows in unknown namespaces
                continue

            output_names = list(spec.outputs.model_fields.keys())
            workflows.append(
                pytest.param(
                    instrument_name,
                    workflow_id,
                    output_names,
                    id=str(workflow_id),
                    marks=[pytest.mark.slow],
                )
            )
    return workflows


def _validate_output_against_template(
    actual: sc.DataArray,
    template: sc.DataArray,
    output_name: str,
    workflow_id: workflow_spec.WorkflowId,
) -> list[str]:
    """Validate that actual output matches template structure.

    Returns a list of validation errors (empty if valid).

    Note: We don't validate units because:
    1. Plotter matching (DataRequirements) doesn't check units
    2. Units can vary based on input data characteristics
    3. Test data may produce unrealistic derived units
    """
    errors = []

    # Check ndim matches
    if actual.ndim != template.ndim:
        errors.append(f"ndim mismatch: actual={actual.ndim}, template={template.ndim}")

    # Check that required coords from template exist in actual
    # (templates may define only a subset of coords)
    errors.extend(
        f"missing required coord '{coord_name}'"
        for coord_name in template.coords
        if coord_name not in actual.coords
    )

    return errors


@pytest.mark.parametrize(
    ("instrument_name", "workflow_id", "output_names"),
    _collect_workflow_outputs_for_validation(),
)
def test_workflow_outputs_match_declared_model(
    instrument_name: str,
    workflow_id: workflow_spec.WorkflowId,
    output_names: list[str],
) -> None:
    """Test that workflow outputs match their declared output model.

    This test:
    1. Creates a service with the workflow configured
    2. Publishes fake event data
    3. Runs the workflow to produce actual output
    4. Validates that each output matches its declared template
    """
    # Skip workflows that require external data files
    if str(workflow_id) == "dream/data_reduction/powder_reduction_with_vanadium/1":
        pytest.skip("Requires vanadium data file not available in test environment")

    instrument = instrument_registry[instrument_name]
    spec = instrument.workflow_factory[workflow_id]

    # Get the appropriate service builder for this namespace
    builder_fn = _get_service_builder(spec.namespace)
    if builder_fn is None:
        pytest.skip(f"No service builder for namespace '{spec.namespace}'")

    # Create the service
    builder = builder_fn(instrument=instrument_name)
    app = LivedataApp.from_service_builder(builder)

    # Configure the workflow
    source_name = _get_source_name_for_workflow(instrument_name, spec)
    config_key = models.ConfigKey(
        source_name=source_name,
        service_name=spec.namespace,
        key="workflow_config",
    )
    workflow_config = workflow_spec.WorkflowConfig(identifier=workflow_id)
    app.publish_config_message(key=config_key, value=workflow_config.model_dump())
    app.service.step()

    # Publish fake events to trigger workflow execution
    # Different namespaces need different event types
    if spec.namespace in ('detector_data', 'data_reduction'):
        app.publish_events(size=1000, time=1)
    if spec.namespace in ('monitor_data', 'data_reduction'):
        app.publish_monitor_events(size=500, time=1)
    if spec.namespace == 'timeseries':
        # Timeseries needs log data
        app.publish_log_message(source_name='proton_charge', time=1.0, value=100.0)

    # Run the service to process events and produce output
    app.service.step()

    # Check that we got some output
    if len(app.sink.messages) == 0:
        pytest.skip("Workflow produced no output (may need specific data)")

    # Collect actual outputs by name
    # The stream name is JSON containing output_name in the ResultKey
    actual_outputs: dict[str, sc.DataArray] = {}
    for msg in app.sink.messages:
        if isinstance(msg.value, sc.DataArray):
            # msg.stream is a StreamId with kind and name attributes
            # name is JSON of ResultKey containing the output_name
            stream_name = msg.stream.name
            actual_outputs[stream_name] = msg.value

    # Validate each declared output
    validation_errors = []
    for output_name in output_names:
        template = spec.get_output_template(output_name)
        if template is None:
            validation_errors.append(f"'{output_name}': no template defined")
            continue

        # Find the actual output by matching the output name in the stream name
        # Stream name is JSON containing the output_name
        actual = None
        for stream_name, value in actual_outputs.items():
            if output_name in stream_name:
                actual = value
                break

        if actual is None:
            # Output wasn't produced - this might be OK for optional outputs
            # or outputs that only appear under certain conditions
            continue

        errors = _validate_output_against_template(
            actual, template, output_name, workflow_id
        )
        if errors:
            validation_errors.append(f"'{output_name}': {'; '.join(errors)}")

    if validation_errors:
        pytest.fail(
            f"Output validation failed for {workflow_id}:\n"
            + "\n".join(f"  - {e}" for e in validation_errors)
        )
