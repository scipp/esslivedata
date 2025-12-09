# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Screenshot test fixtures for the HTTP transport.

This module provides pre-configured workflows and test data for screenshot testing.
The fixtures use well-known, fixed job_numbers so that injected data matches
the persisted workflow configurations.

Usage:
    from scripts.screenshot_fixtures import (
        FIXTURES_DIR,
        PANEL_0_JOB_NUMBER,
        AREA_PANEL_JOB_NUMBER,
        make_panel_0_data,
        make_area_panel_data,
        inject_screenshot_data,
    )
"""

import base64
import json
import time
import urllib.request
from pathlib import Path
from uuid import UUID

import numpy as np
import scipp as sc
from streaming_data_types import dataarray_da00

from ess.livedata.config.workflow_spec import JobId, ResultKey, WorkflowId
from ess.livedata.kafka.scipp_da00_compat import scipp_to_da00

# Directory containing the fixture config files
FIXTURES_DIR = Path(__file__).parent

# Well-known job numbers matching workflow_configs.yaml
PANEL_0_JOB_NUMBER = UUID("00000000-0000-0000-0000-000000000001")
AREA_PANEL_JOB_NUMBER = UUID("00000000-0000-0000-0000-000000000002")

# Workflow IDs
PANEL_0_WORKFLOW_ID = WorkflowId(
    instrument='dummy', namespace='detector_data', name='panel_0_xy', version=1
)
AREA_PANEL_WORKFLOW_ID = WorkflowId(
    instrument='dummy', namespace='detector_data', name='area_panel_xy', version=1
)


def make_gaussian_blob(
    shape: tuple[int, int] = (64, 64),
    center: tuple[float, float] | None = None,
    sigma: float | None = None,
    amplitude: float = 500.0,
    noise_level: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Create a 2D Gaussian blob with Poisson noise."""
    if center is None:
        center = (shape[0] / 2, shape[1] / 2)
    if sigma is None:
        sigma = min(shape) / 6

    y, x = np.ogrid[: shape[0], : shape[1]]
    blob = np.exp(-((y - center[0]) ** 2 + (x - center[1]) ** 2) / (2 * sigma**2))
    rng = np.random.default_rng(seed)
    noise = rng.poisson(lam=noise_level, size=shape)
    return (blob * amplitude + noise).astype(np.float64)


def make_panel_0_data(shape: tuple[int, int] = (64, 64)) -> sc.DataArray:
    """Create test data for panel_0 detector."""
    data = make_gaussian_blob(shape, center=(shape[0] * 0.4, shape[1] * 0.6), seed=42)
    return sc.DataArray(
        sc.array(dims=['y', 'x'], values=data, unit='counts'),
        coords={
            'x': sc.arange('x', 0.0, float(shape[1]), unit=None),
            'y': sc.arange('y', 0.0, float(shape[0]), unit=None),
        },
    )


def make_area_panel_data(shape: tuple[int, int] = (128, 128)) -> sc.DataArray:
    """Create test data for area_panel detector (larger with multiple blobs)."""
    # Multiple overlapping blobs for a more interesting pattern
    blob1 = make_gaussian_blob(
        shape, center=(shape[0] * 0.3, shape[1] * 0.3), sigma=15, seed=42
    )
    blob2 = make_gaussian_blob(
        shape, center=(shape[0] * 0.7, shape[1] * 0.6), sigma=20, seed=43
    )
    blob3 = make_gaussian_blob(
        shape, center=(shape[0] * 0.5, shape[1] * 0.8), sigma=10, amplitude=300, seed=44
    )
    data = blob1 + blob2 + blob3

    return sc.DataArray(
        sc.array(dims=['y', 'x'], values=data, unit='counts'),
        coords={
            'x': sc.arange('x', 0.0, float(shape[1]), unit=None),
            'y': sc.arange('y', 0.0, float(shape[0]), unit=None),
        },
    )


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


def inject_screenshot_data(port: int = 5009) -> dict[str, dict]:
    """
    Inject all test data for screenshot capture.

    This injects data for both panel_0 and area_panel detectors using
    the well-known job_numbers that match the fixture configs.

    Returns dict mapping detector names to injection results.
    """
    results = {}

    # Inject panel_0 data
    panel_0_key = ResultKey(
        workflow_id=PANEL_0_WORKFLOW_ID,
        job_id=JobId(job_number=PANEL_0_JOB_NUMBER, source_name='panel_0'),
        output_name='current',
    )
    panel_0_data = make_panel_0_data()
    panel_0_payload = serialize_to_da00(panel_0_key, panel_0_data)
    results['panel_0'] = inject_data(port, panel_0_payload)

    # Inject area_panel data
    area_panel_key = ResultKey(
        workflow_id=AREA_PANEL_WORKFLOW_ID,
        job_id=JobId(job_number=AREA_PANEL_JOB_NUMBER, source_name='area_panel'),
        output_name='current',
    )
    area_panel_data = make_area_panel_data()
    area_panel_payload = serialize_to_da00(area_panel_key, area_panel_data)
    results['area_panel'] = inject_data(port, area_panel_payload)

    return results
