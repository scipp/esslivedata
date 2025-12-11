# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Screenshot test fixtures for the HTTP transport.

This module provides a registry-based system for defining test data fixtures.
Data generators are registered by workflow_id and output_name, and the system
automatically matches them with workflow configurations to inject data.

Usage:
    # In your fixture module (e.g., dummy/__init__.py):
    from scripts.screenshot_fixtures import fixture_registry

    @fixture_registry.register('dummy/detector_data/panel_0_xy/1', output='current')
    def make_panel_0_data() -> sc.DataArray:
        return sc.DataArray(...)

    # To inject all registered fixtures:
    from scripts.screenshot_fixtures import inject_fixtures
    inject_fixtures(port=5009, fixtures_dir=Path(...))
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import yaml
from streaming_data_types import dataarray_da00

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.kafka.scipp_da00_compat import scipp_to_da00

if TYPE_CHECKING:
    import scipp as sc

logger = logging.getLogger(__name__)

# Directory containing the fixture config files
FIXTURES_DIR = Path(__file__).parent

DataGenerator = Callable[[], 'sc.DataArray']


@dataclass
class FixtureKey:
    """Key for looking up a data generator in the registry."""

    workflow_id: str  # e.g., 'dummy/detector_data/panel_0_xy/1'
    output_name: str = 'current'

    def __hash__(self) -> int:
        return hash((self.workflow_id, self.output_name))


@dataclass
class FixtureRegistry:
    """
    Registry for screenshot data generators.

    Data generators are functions that return sc.DataArray. They are registered
    by workflow_id and output_name. When fixtures are injected, the registry
    looks up the appropriate generator for each workflow/output combination.
    """

    _generators: dict[FixtureKey, DataGenerator] = field(default_factory=dict)

    def register(
        self, workflow_id: str, *, output: str = 'current'
    ) -> Callable[[DataGenerator], DataGenerator]:
        """
        Decorator to register a data generator.

        Parameters
        ----------
        workflow_id:
            The workflow ID string, e.g., 'dummy/detector_data/panel_0_xy/1'
        output:
            The output name, e.g., 'current'. Defaults to 'current'.

        Example
        -------
        @fixture_registry.register('dummy/detector_data/panel_0_xy/1')
        def make_panel_0_data() -> sc.DataArray:
            return sc.DataArray(...)
        """
        key = FixtureKey(workflow_id=workflow_id, output_name=output)

        def decorator(func: DataGenerator) -> DataGenerator:
            self._generators[key] = func
            return func

        return decorator

    def get_generator(
        self, workflow_id: str, output_name: str = 'current'
    ) -> DataGenerator | None:
        """Get a registered generator, or None if not found."""
        key = FixtureKey(workflow_id=workflow_id, output_name=output_name)
        return self._generators.get(key)

    def clear(self) -> None:
        """Clear all registered generators (useful for testing)."""
        self._generators.clear()


# Global registry instance
fixture_registry = FixtureRegistry()


@dataclass
class WorkflowFixtureInfo:
    """Information extracted from workflow_configs.yaml for a single workflow."""

    workflow_id: WorkflowId
    job_number: UUID
    source_names: list[str]


def load_workflow_configs(
    fixtures_dir: Path, instrument: str
) -> list[WorkflowFixtureInfo]:
    """
    Load workflow configurations from YAML and extract fixture-relevant info.

    Parameters
    ----------
    fixtures_dir:
        Directory containing instrument subdirectories with config files.
    instrument:
        Instrument name (subdirectory name).

    Returns
    -------
    :
        List of workflow fixture info extracted from workflow_configs.yaml.
    """
    config_path = fixtures_dir / instrument / 'workflow_configs.yaml'
    if not config_path.exists():
        logger.warning("No workflow_configs.yaml found at %s", config_path)
        return []

    with open(config_path) as f:
        configs = yaml.safe_load(f) or {}

    results = []
    for workflow_id_str, config in configs.items():
        try:
            workflow_id = WorkflowId.from_string(workflow_id_str)
            current_job = config.get('current_job', {})
            job_number_str = current_job.get('job_number')
            if job_number_str is None:
                logger.warning(
                    "No job_number in current_job for %s, skipping", workflow_id_str
                )
                continue

            # Get source_names from current_job.jobs keys (the actual running jobs)
            jobs = current_job.get('jobs', {})
            source_names = list(jobs.keys())

            results.append(
                WorkflowFixtureInfo(
                    workflow_id=workflow_id,
                    job_number=UUID(job_number_str),
                    source_names=source_names,
                )
            )
        except Exception as e:
            logger.warning("Failed to parse workflow config %s: %s", workflow_id_str, e)

    return results


def serialize_to_da00(result_key: ResultKey, data: sc.DataArray) -> bytes:
    """Serialize a DataArray to da00 format with the given ResultKey."""
    return dataarray_da00.serialise_da00(
        source_name=result_key.model_dump_json(),
        timestamp_ns=int(time.time_ns()),
        data=scipp_to_da00(data),
    )


def inject_data(port: int, payload: bytes) -> dict:
    """Inject data via HTTP POST to the dashboard."""
    url = f'http://localhost:{port}/api/data'
    body = json.dumps({'payload_base64': base64.b64encode(payload).decode('utf-8')})
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body.encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=5.0) as response:  # noqa: S310
        return json.loads(response.read().decode('utf-8'))


def inject_fixtures(
    port: int = 5009,
    fixtures_dir: Path = FIXTURES_DIR,
    instrument: str = 'dummy',
    registry: FixtureRegistry | None = None,
) -> dict[str, dict]:
    """
    Inject all registered fixtures for the given instrument.

    This function:
    1. Loads workflow_configs.yaml to get job numbers and source names
    2. For each workflow, looks up registered data generators
    3. Generates data and injects it via the HTTP API

    Parameters
    ----------
    port:
        Dashboard port number.
    fixtures_dir:
        Directory containing instrument subdirectories with config files.
    instrument:
        Instrument name.
    registry:
        Fixture registry to use. Defaults to the global fixture_registry.

    Returns
    -------
    :
        Dict mapping "{workflow_id}/{source_name}/{output}" to injection results.
    """
    if registry is None:
        registry = fixture_registry

    workflow_infos = load_workflow_configs(fixtures_dir, instrument)
    results: dict[str, dict] = {}

    for info in workflow_infos:
        workflow_id_str = str(info.workflow_id)

        # For now, we only support 'current' output, but the system is extensible
        output_name = 'current'
        generator = registry.get_generator(workflow_id_str, output_name)

        if generator is None:
            logger.debug(
                "No generator registered for %s/%s, skipping",
                workflow_id_str,
                output_name,
            )
            continue

        # Generate data once, inject for each source_name
        data = generator()

        for source_name in info.source_names:
            result_key = ResultKey(
                workflow_id=info.workflow_id,
                job_id=JobId(job_number=info.job_number, source_name=source_name),
                output_name=output_name,
            )
            payload = serialize_to_da00(result_key, data)

            key = f"{workflow_id_str}/{source_name}/{output_name}"
            try:
                results[key] = inject_data(port, payload)
                logger.info("Injected data for %s", key)
            except Exception as e:
                logger.error("Failed to inject data for %s: %s", key, e)
                results[key] = {'error': str(e)}

    return results
