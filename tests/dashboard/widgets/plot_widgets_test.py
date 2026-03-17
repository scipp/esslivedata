# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pydantic

from ess.livedata.dashboard.plot_params import WindowMode, WindowParams
from ess.livedata.dashboard.widgets.plot_widgets import _format_window_info


class _FakeParams(pydantic.BaseModel):
    window: WindowParams = pydantic.Field(default_factory=WindowParams)


class TestFormatWindowInfo:
    def test_returns_empty_when_supports_windowing_false(self) -> None:
        params = _FakeParams()
        assert _format_window_info(params, supports_windowing=False) == ''

    def test_returns_latest_for_latest_mode(self) -> None:
        params = _FakeParams(window=WindowParams(mode=WindowMode.latest))
        assert _format_window_info(params) == 'latest'

    def test_returns_window_info_for_window_mode(self) -> None:
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=10)
        )
        assert _format_window_info(params) == '10s window'

    def test_returns_empty_when_no_window_attr(self) -> None:
        class NoWindowParams(pydantic.BaseModel):
            pass

        assert _format_window_info(NoWindowParams()) == ''

    def test_supports_windowing_false_overrides_window_mode(self) -> None:
        """Even with window mode set, supports_windowing=False returns empty."""
        params = _FakeParams(
            window=WindowParams(mode=WindowMode.window, window_duration_seconds=5)
        )
        assert _format_window_info(params, supports_windowing=False) == ''
