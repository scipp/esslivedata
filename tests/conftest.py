# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest


@pytest.fixture
def infra_kwargs() -> dict:
    """Infrastructure topic kwargs for constructing StreamMapping in tests."""
    return {
        "livedata_commands_topic": "cmd",
        "livedata_data_topic": "data",
        "livedata_responses_topic": "resp",
        "livedata_roi_topic": "roi",
        "livedata_status_topic": "status",
    }


def pytest_configure(config: pytest.Config) -> None:
    """Turn off xdist when benchmarking.

    ``addopts`` carries ``-n auto``, but pytest-benchmark cannot measure anything
    under xdist and errors out on every benchmark. Benchmarks want a quiet machine
    anyway, so drop back to a single process instead of making every caller
    remember ``-n0``.
    """
    if config.option.benchmark_enable or not config.option.benchmark_skip:
        config.option.numprocesses = 0
        config.option.dist = "no"
