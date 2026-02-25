# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)


def extract_error_summary(error_message: str) -> str:
    """Extract a short summary from an error message or traceback.

    For multi-line messages (typically Python tracebacks), returns the last
    non-empty line, which contains the exception type and message.
    For single-line messages, returns the message as-is.

    Parameters
    ----------
    error_message:
        Full error message, potentially a multi-line traceback.

    Returns
    -------
    :
        The last non-empty line of the message.
    """
    return error_message.strip().rsplit('\n', 1)[-1]
