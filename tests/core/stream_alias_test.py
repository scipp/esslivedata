# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import uuid
from dataclasses import FrozenInstanceError

import pytest

from ess.livedata.config.workflow_spec import JobId
from ess.livedata.core.stream_alias import (
    AliasedResult,
    BindingConflictError,
    BindStreamAlias,
    StreamAliasRegistry,
    UnbindStreamAlias,
)


def _make_job_id(source: str = 'src') -> JobId:
    return JobId(source_name=source, job_number=uuid.uuid4())


class TestStreamAliasRegistry:
    def test_bind_lookup_roundtrip(self) -> None:
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'counts_total')
        assert registry.has('fom-0')
        assert registry.lookup(job_id, 'counts_total') == 'fom-0'

    def test_lookup_returns_none_for_unbound(self) -> None:
        registry = StreamAliasRegistry()
        assert registry.lookup(_make_job_id(), 'output') is None

    def test_bind_idempotent_for_same_triple(self) -> None:
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'counts_total')
        registry.bind('fom-0', job_id, 'counts_total')  # No raise.
        assert registry.lookup(job_id, 'counts_total') == 'fom-0'

    def test_multi_source_under_same_alias(self) -> None:
        """One workflow run on N sources binds N pairs under the alias."""
        registry = StreamAliasRegistry()
        job_a = _make_job_id('det_1')
        job_b = _make_job_id('det_2')
        registry.bind('fom-0', job_a, 'counts_total')
        registry.bind('fom-0', job_b, 'counts_total')
        assert registry.lookup(job_a, 'counts_total') == 'fom-0'
        assert registry.lookup(job_b, 'counts_total') == 'fom-0'

    def test_conflicting_alias_for_same_output_rejected(self) -> None:
        """Binding the same (job, output) under a different alias raises."""
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'counts_total')
        with pytest.raises(BindingConflictError):
            registry.bind('fom-1', job_id, 'counts_total')

    def test_unbind_removes_all_bindings_under_alias(self) -> None:
        registry = StreamAliasRegistry()
        job_a = _make_job_id('det_1')
        job_b = _make_job_id('det_2')
        registry.bind('fom-0', job_a, 'counts_total')
        registry.bind('fom-0', job_b, 'counts_total')
        registry.unbind('fom-0')
        assert not registry.has('fom-0')
        assert registry.lookup(job_a, 'counts_total') is None
        assert registry.lookup(job_b, 'counts_total') is None

    def test_unbind_unknown_is_noop(self) -> None:
        registry = StreamAliasRegistry()
        registry.unbind('does-not-exist')  # No raise.

    def test_can_rebind_after_unbind(self) -> None:
        registry = StreamAliasRegistry()
        job_a = _make_job_id('a')
        job_b = _make_job_id('b')
        registry.bind('fom-0', job_a, 'x')
        registry.unbind('fom-0')
        registry.bind('fom-0', job_b, 'y')
        assert registry.lookup(job_b, 'y') == 'fom-0'
        assert registry.lookup(job_a, 'x') is None

    def test_multiple_aliases_independent(self) -> None:
        registry = StreamAliasRegistry()
        job_id = _make_job_id()
        registry.bind('fom-0', job_id, 'counts_total')
        registry.bind('fom-1', job_id, 'counts_in_toa')
        assert registry.lookup(job_id, 'counts_total') == 'fom-0'
        assert registry.lookup(job_id, 'counts_in_toa') == 'fom-1'


class TestBindStreamAlias:
    def test_roundtrip(self) -> None:
        job_id = _make_job_id()
        msg = BindStreamAlias(alias='fom-0', job_id=job_id, output_name='counts_total')
        restored = BindStreamAlias.model_validate_json(msg.model_dump_json())
        assert restored == msg

    def test_key_is_class_var(self) -> None:
        assert BindStreamAlias.key == 'bind_stream_alias'


class TestUnbindStreamAlias:
    def test_roundtrip(self) -> None:
        msg = UnbindStreamAlias(alias='fom-0')
        restored = UnbindStreamAlias.model_validate_json(msg.model_dump_json())
        assert restored == msg

    def test_key_is_class_var(self) -> None:
        assert UnbindStreamAlias.key == 'unbind_stream_alias'


class TestAliasedResult:
    def test_holds_data_and_alias(self) -> None:
        result = AliasedResult(data=42, alias='fom-0')
        assert result.data == 42
        assert result.alias == 'fom-0'

    def test_is_frozen(self) -> None:
        result = AliasedResult(data=42, alias='fom-0')
        with pytest.raises(FrozenInstanceError):
            result.data = 0  # type: ignore[misc]
