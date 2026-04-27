# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import time

from ess.livedata import Service


class FakeProcessor:
    def __init__(self):
        self.call_count = 0

    def process(self) -> None:
        self.call_count += 1

    def finalize(self, *, error: str | None = None) -> None:
        pass


def test_create_start_stop_service() -> None:
    processor = FakeProcessor()
    service = Service(processor=processor)
    assert processor.call_count == 0
    service.start(blocking=False)
    assert service.is_running
    time.sleep(0.2)
    assert processor.call_count > 0
    service.stop()
    assert not service.is_running
