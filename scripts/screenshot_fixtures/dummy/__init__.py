# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Data generators for the dummy instrument screenshot fixtures.

This module registers data generators for each workflow defined in
workflow_configs.yaml. The generators create synthetic test data
for screenshot testing.
"""

import numpy as np
import scipp as sc

from screenshot_fixtures import fixture_registry


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


@fixture_registry.register('dummy/detector_data/panel_0_xy/1')
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


@fixture_registry.register('dummy/detector_data/area_panel_xy/1')
def make_area_panel_data(shape: tuple[int, int] = (128, 128)) -> sc.DataArray:
    """Create test data for area_panel detector (larger with multiple blobs)."""
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
