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
