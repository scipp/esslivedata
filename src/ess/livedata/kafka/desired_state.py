# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Kafka reader and writer of the desired-state topic.

See :mod:`ess.livedata.core.desired_state` for the record format. Both classes
own their consumer or producer and do not go through ``KafkaMessageSource`` or
``KafkaSink``: the reader must see every record from offset 0 (a bounded
background queue may drop records), and the writer must report whether each
record was stored.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import confluent_kafka as kafka
import structlog
from confluent_kafka import KafkaError, KafkaException

from ..config.workflow_spec import WorkflowId
from ..core.desired_state import (
    DesiredStateDecodeError,
    DesiredStateWriteError,
    DesiredUpdate,
    DesiredWorkflow,
    decode_desired_record,
    encode_desired_record,
)
from .errors import is_fatal

logger = structlog.get_logger(__name__)

_MAX_BATCH = 1000
#: Upper bound of a single blocking ``consume`` during bootstrap.
_BOOTSTRAP_CONSUME_TIMEOUT_S = 0.5
#: Extra time ``flush`` waits beyond ``message.timeout.ms``, so that librdkafka
#: always reports the delivery (success or timeout) before ``flush`` returns.
_FLUSH_MARGIN_S = 1.0


class KafkaDesiredStateReader:
    """Reads the desired-state topic from offset 0 with its own consumer.

    The consumer is assigned partition 0 at ``OFFSET_BEGINNING`` without a
    consumer group or offset commits. The end of the bootstrap read is the
    partition EOF event (``enable.partition.eof``), not the record at
    ``high_watermark - 1``, which compaction may have removed.

    Undecodable records are logged, counted in :attr:`undecodable_records`, and
    skipped. Fatal consumer errors raise ``KafkaException``; other consumer
    errors are logged and yield no updates.

    Parameters
    ----------
    kafka_config:
        Broker and authentication settings, e.g. from
        ``load_config(namespace=config_names.kafka)``.
    topic:
        The desired-state topic. Must exist and have exactly one partition.
    consumer_factory:
        Creates the consumer from its configuration.
    """

    def __init__(
        self,
        *,
        kafka_config: dict[str, Any],
        topic: str,
        consumer_factory: Callable[[dict[str, Any]], kafka.Consumer] = kafka.Consumer,
    ) -> None:
        self._topic = topic
        self._undecodable_records = 0
        self._consumer = consumer_factory(
            {
                **kafka_config,
                'group.id': f'livedata_desired_{uuid.uuid4()}',
                'enable.auto.commit': False,
                'enable.partition.eof': True,
                # Only relevant if the assigned offset falls out of range; the
                # table must then be re-read, not skipped.
                'auto.offset.reset': 'earliest',
            }
        )
        try:
            topics = self._consumer.list_topics(timeout=5.0).topics
            if topic not in topics:
                raise ValueError(f"Topic not found: '{topic}'")
            if (n := len(topics[topic].partitions)) != 1:
                raise ValueError(f"Topic '{topic}' must have 1 partition, has {n}")
            self._consumer.assign(
                [kafka.TopicPartition(topic, 0, kafka.OFFSET_BEGINNING)]
            )
        except BaseException:
            self._consumer.close()
            raise

    @property
    def undecodable_records(self) -> int:
        """Number of records skipped because they could not be decoded."""
        return self._undecodable_records

    def bootstrap(self, timeout_s: float) -> list[DesiredUpdate]:
        """Read until partition EOF.

        Raises
        ------
        TimeoutError
            If partition EOF is not reached within ``timeout_s``.
        """
        deadline = time.monotonic() + timeout_s
        updates: list[DesiredUpdate] = []
        while (remaining := deadline - time.monotonic()) > 0:
            timeout = min(remaining, _BOOTSTRAP_CONSUME_TIMEOUT_S)
            if self._consume(updates, timeout=timeout):
                logger.info(
                    "desired_state_bootstrapped",
                    topic=self._topic,
                    records=len(updates),
                )
                return updates
        raise TimeoutError(
            f"No end of partition on '{self._topic}' within {timeout_s} s"
            f" ({len(updates)} records read)"
        )

    def poll(self) -> list[DesiredUpdate]:
        updates: list[DesiredUpdate] = []
        self._consume(updates, timeout=0)
        return updates

    def _consume(self, updates: list[DesiredUpdate], *, timeout: float) -> bool:
        """Append one batch of decoded records; return whether EOF was reached."""
        eof = False
        for msg in self._consumer.consume(num_messages=_MAX_BATCH, timeout=timeout):
            err = msg.error()
            if err is None:
                try:
                    updates.append(decode_desired_record(msg.key(), msg.value()))
                except DesiredStateDecodeError as e:
                    self._undecodable_records += 1
                    logger.error(
                        "desired_record_undecodable",
                        topic=self._topic,
                        offset=msg.offset(),
                        error=str(e),
                    )
            elif err.code() == KafkaError._PARTITION_EOF:
                eof = True
            elif is_fatal(err):
                raise KafkaException(err)
            else:
                logger.warning(
                    "desired_state_consumer_error", topic=self._topic, error=str(err)
                )
        return eof

    def close(self) -> None:
        self._consumer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class KafkaDesiredStateWriter:
    """Writes desired-state records with an idempotent producer.

    Idempotence keeps two writes to the same key in order across retries.
    ``message.timeout.ms`` is ``timeout_s``, and :meth:`write` waits slightly
    longer for the delivery report, so a record reported as failed is never
    stored later.

    Parameters
    ----------
    kafka_config:
        Broker and authentication settings, e.g. from
        ``load_config(namespace=config_names.kafka)``.
    topic:
        The desired-state topic.
    timeout_s:
        Time after which an unconfirmed record counts as failed.
    producer_factory:
        Creates the producer from its configuration.
    """

    def __init__(
        self,
        *,
        kafka_config: dict[str, Any],
        topic: str,
        timeout_s: float = 10.0,
        producer_factory: Callable[[dict[str, Any]], kafka.Producer] = kafka.Producer,
    ) -> None:
        self._topic = topic
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._producer = producer_factory(
            {
                **kafka_config,
                'enable.idempotence': True,
                'message.timeout.ms': int(timeout_s * 1000),
            }
        )

    def write(self, workflow_id: WorkflowId, desired: DesiredWorkflow | None) -> None:
        """Write one record (``None`` is a tombstone) and wait for its delivery.

        Raises
        ------
        DesiredStateWriteError
            If the delivery failed or was not reported in time.
        KafkaException
            On a fatal producer error.
        """
        key, value = encode_desired_record(workflow_id, desired)
        reports: list[KafkaError | None] = []
        # One write at a time, so each flush waits for its own delivery report.
        with self._lock:
            try:
                self._producer.produce(
                    topic=self._topic,
                    key=key,
                    value=value,
                    on_delivery=lambda err, _msg: reports.append(err),
                )
            except (BufferError, KafkaException) as e:
                if isinstance(e, KafkaException) and is_fatal(e.args[0]):
                    raise
                raise DesiredStateWriteError(
                    f"Failed to write '{workflow_id}': {e}"
                ) from e
            self._producer.flush(timeout=self._timeout_s + _FLUSH_MARGIN_S)
        if not reports:
            raise DesiredStateWriteError(
                f"No delivery report for '{workflow_id}' within {self._timeout_s} s"
            )
        if (err := reports[0]) is not None:
            if is_fatal(err):
                raise KafkaException(err)
            raise DesiredStateWriteError(f"Failed to write '{workflow_id}': {err}")
