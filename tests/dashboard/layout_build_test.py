# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
from pathlib import Path

import pytest
from bokeh.document import Document
from panel.io.state import set_curdoc

from ess.livedata.dashboard.reduction import ReductionApp


@pytest.fixture
def app(tmp_path: Path) -> ReductionApp:
    return ReductionApp(
        log_level=logging.INFO, transport='none', config_dir=str(tmp_path)
    )


def test_layout_build_leaves_no_document_hold(app: ReductionApp) -> None:
    """
    Building the layout must not leave the document held.

    Bokeh constructs the ServerSession right after the app callable returns and
    registers every session callback the build put on the document. A hold
    surviving the build defers the corresponding SessionCallbackAdded events past
    that point, so unholding registers the same callbacks a second time and Bokeh
    raises "A callback of the same type has already been added with this ID".
    """
    doc = Document()
    with set_curdoc(doc):
        app.create_layout()
    assert doc.callbacks.hold_value is None
