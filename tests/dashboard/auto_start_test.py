# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging

import pytest

from ess.livedata.dashboard.reduction import ReductionApp


@pytest.mark.parametrize('transport', ['kafka', 'none'])
def test_auto_start_requires_fake_transport(transport: str) -> None:
    # The guard fires before any transport setup, so no broker is contacted.
    with pytest.raises(ValueError, match="auto_start requires"):
        ReductionApp(log_level=logging.INFO, transport=transport, auto_start=True)
