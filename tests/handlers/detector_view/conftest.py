# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Pytest configuration and shared fixtures for detector view tests."""

# Re-export test utilities for use in test modules
from .utils import (  # noqa: F401
    make_fake_detector_number,
    make_fake_empty_detector,
    make_fake_nexus_detector_data,
    make_fake_ungrouped_nexus_data,
    make_logical_transform,
    make_test_factory,
)
