# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import gc
import threading
import weakref

import pytest

from ess.livedata.core.cyclic_gc import PeriodicGarbageCollector
from ess.livedata.core.service import Service


class _SelfReferential:
    """Object reclaimable only by the cyclic collector, like a materialised
    networkx graph view (see issue #1264)."""

    def __init__(self) -> None:
        self.cycle = self


@pytest.fixture
def gc_disabled():
    """Disable automatic collection so only explicit collects can reclaim."""
    was_enabled = gc.isenabled()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TestPeriodicGarbageCollector:
    def test_collects_cyclic_garbage(self, gc_disabled) -> None:
        clock = FakeClock()
        collector = PeriodicGarbageCollector(interval=1.0, clock=clock)
        obj = _SelfReferential()
        ref = weakref.ref(obj)
        del obj
        # Refcounting alone cannot reclaim the cycle.
        assert ref() is not None
        clock.now = 1.0
        assert collector.maybe_collect() is True
        assert ref() is None

    def test_respects_interval(self, gc_disabled) -> None:
        clock = FakeClock()
        collector = PeriodicGarbageCollector(interval=10.0, clock=clock)
        obj = _SelfReferential()
        ref = weakref.ref(obj)
        del obj
        clock.now = 9.9
        assert collector.maybe_collect() is False
        assert ref() is not None
        clock.now = 10.0
        assert collector.maybe_collect() is True
        assert ref() is None

    def test_interval_measured_from_last_collection(self, gc_disabled) -> None:
        clock = FakeClock()
        collector = PeriodicGarbageCollector(interval=10.0, clock=clock)
        clock.now = 25.0
        assert collector.maybe_collect() is True
        clock.now = 34.9
        assert collector.maybe_collect() is False
        clock.now = 35.0
        assert collector.maybe_collect() is True

    def test_freeze_exempts_preexisting_objects(self, gc_disabled) -> None:
        collector = PeriodicGarbageCollector(interval=0.0, clock=FakeClock())
        frozen_before = gc.get_freeze_count()
        obj = _SelfReferential()
        ref = weakref.ref(obj)
        try:
            collector.freeze()
            assert gc.get_freeze_count() > frozen_before
            # The cycle was alive at freeze time, so collection ignores it
            # even after it becomes garbage.
            del obj
            collector.maybe_collect()
            assert ref() is not None
        finally:
            gc.unfreeze()
        gc.collect()
        assert ref() is None


class _CyclicGarbageProcessor:
    """Processor that leaves one cyclic garbage object per process() call."""

    def __init__(self) -> None:
        self.refs: list[weakref.ref] = []
        self.processed = threading.Event()

    def process(self) -> None:
        obj = _SelfReferential()
        self.refs.append(weakref.ref(obj))
        del obj
        self.processed.set()

    def finalize(self, *, error: str | None = None) -> None:
        pass


class TestServiceGarbageCollection:
    def test_service_loop_collects_cyclic_garbage(self, gc_disabled) -> None:
        processor = _CyclicGarbageProcessor()
        service = Service(
            processor=processor,
            garbage_collector=PeriodicGarbageCollector(interval=0.0),
        )
        try:
            service.start(blocking=False)
            assert processor.processed.wait(timeout=5.0)
        finally:
            service.stop()
            gc.unfreeze()
        assert len(processor.refs) > 0
        # Every cycle except possibly the last (created after the loop's final
        # collect) was reclaimed without the automatic collector running.
        assert sum(ref() is not None for ref in processor.refs) <= 1

    def test_service_without_collector_leaves_cyclic_garbage(self, gc_disabled) -> None:
        processor = _CyclicGarbageProcessor()
        service = Service(processor=processor, collect_garbage=False)
        try:
            service.start(blocking=False)
            assert processor.processed.wait(timeout=5.0)
        finally:
            service.stop()
        assert len(processor.refs) > 0
        assert all(ref() is not None for ref in processor.refs)
        gc.collect()
        assert all(ref() is None for ref in processor.refs)
