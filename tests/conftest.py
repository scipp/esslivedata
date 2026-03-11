# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest


def _is_download_failure(excinfo: pytest.ExceptionInfo) -> bool:
    """Check if an exception was caused by a failed file download.

    Upstream ESS packages use pooch to download data files from external
    servers. When those servers are unavailable, tests fail with network
    errors that are outside our control. This function detects such
    failures so they can be reported as xfail instead of errors.
    """
    try:
        import requests.exceptions
    except ImportError:
        return False

    network_errors = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
    # Walk the exception chain (including __cause__ and __context__)
    exc: BaseException | None = excinfo.value
    while exc is not None:
        if isinstance(exc, network_errors):
            return True
        # Move to the next exception in the chain
        cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
        if cause is exc:
            break
        exc = cause
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == 'call' and report.failed and call.excinfo is not None:
        if _is_download_failure(call.excinfo):
            report.outcome = 'skipped'
            report.wasxfail = 'External file download failed (server unavailable)'
