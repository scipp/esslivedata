# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import confluent_kafka as kafka
import structlog
from confluent_kafka import KafkaError
from confluent_kafka.error import KafkaException

logger = structlog.get_logger(__name__)

#: Broker errors meaning "this partition has no usable leader yet, ask again".
#: Leadership is not established atomically with topic creation or a broker
#: restart, so a partition can be present in cluster metadata while the broker
#: hosting it has not yet applied the leadership change.
_LEADER_PENDING_ERRORS = frozenset(
    {
        KafkaError.UNKNOWN_TOPIC_OR_PART,
        KafkaError.LEADER_NOT_AVAILABLE,
        KafkaError.NOT_LEADER_FOR_PARTITION,
    }
)

#: How long to wait for partition leadership to settle before giving up.
_LEADER_TIMEOUT = 30.0
_LEADER_POLL_INTERVAL = 0.5


def validate_topics_exist(consumer: kafka.Consumer, topics: list[str]) -> None:
    """Check if all topics exist and are accessible."""
    logger.debug("validating_topics", topics=topics)
    try:
        cluster_metadata = consumer.list_topics(timeout=5.0)
        available_topics = cluster_metadata.topics
        missing_topics = [topic for topic in topics if topic not in available_topics]
        if missing_topics:
            logger.error("topics_not_found", missing_topics=missing_topics)
            raise ValueError(f"Topics not found: {missing_topics}")
        logger.info("topics_validated", topic_count=len(topics))
    except KafkaException as e:
        logger.exception("topic_metadata_fetch_failed")
        raise ValueError(f"Failed to fetch topic metadata: {e}") from e


def _high_watermark(
    consumer: kafka.Consumer, topic: str, partition: int, *, deadline: float
) -> int:
    """Fetch a partition's high watermark, waiting for its leader to settle.

    Only the partition leader answers ``ListOffsets``, and librdkafka does not
    retry this query on our behalf. Without the wait a consumer created inside a
    leader-election window -- a just-created topic, a restarted broker -- fails
    closed on an error the broker itself considers transient.
    """
    while True:
        try:
            _low, high = consumer.get_watermark_offsets(
                kafka.TopicPartition(topic, partition), timeout=5.0
            )
            return high
        except KafkaException as e:
            expired = time.monotonic() >= deadline
            if e.args[0].code() not in _LEADER_PENDING_ERRORS or expired:
                logger.exception(
                    "watermark_fetch_failed", topic=topic, partition=partition
                )
                raise ValueError(
                    f"Failed to fetch watermark for '{topic}' partition"
                    f" {partition}: {e}"
                ) from e
            logger.info(
                "awaiting_partition_leader",
                topic=topic,
                partition=partition,
                error=str(e.args[0]),
            )
            time.sleep(_LEADER_POLL_INTERVAL)


def assign_all_partitions(consumer: kafka.Consumer, topics: list[str]) -> None:
    """Manually assign every partition of every topic to a consumer.

    ``Consumer.assign`` replaces the entire assignment, so all partitions of all
    topics must be passed in a single call; assigning per topic in a loop would
    leave only the last topic assigned.

    Offsets are pinned to the current high watermark rather than left to
    ``auto.offset.reset``: with manual assignment, "latest" is resolved lazily
    when fetching starts, so a message produced between ``assign()`` and the
    first fetch would be skipped silently (e.g. a command sent right after a
    service reports ready). Pinning makes the contract deterministic: every
    message produced after assignment is consumed.

    Resolving the watermarks may block for up to ``_LEADER_TIMEOUT`` in total
    while partition leadership settles; see :func:`_high_watermark`.
    """
    deadline = time.monotonic() + _LEADER_TIMEOUT
    assignment: list[kafka.TopicPartition] = []
    for topic in topics:
        try:
            partitions = consumer.list_topics(topic).topics[topic].partitions
        except KafkaException as e:
            logger.exception("partition_assignment_failed", topic=topic)
            raise ValueError(
                f"Failed to assign partitions for topic '{topic}': {e}"
            ) from e
        if not partitions:
            logger.error("topic_has_no_partitions", topic=topic)
            raise ValueError(f"Topic '{topic}' exists but has no partitions")
        partition_ids = list(partitions.keys())
        offsets = {
            partition: _high_watermark(consumer, topic, partition, deadline=deadline)
            for partition in partition_ids
        }
        assignment.extend(
            kafka.TopicPartition(topic, p, offsets[p]) for p in partition_ids
        )
        logger.info(
            "partitions_resolved",
            topic=topic,
            partition_count=len(partition_ids),
            offsets=offsets,
        )
    consumer.assign(assignment)


@contextmanager
def make_bare_consumer(
    topics: list[str], config: dict[str, Any]
) -> Generator[kafka.Consumer, None, None]:
    """Create a bare confluent_kafka.Consumer that can be used by KafkaMessageSource.

    Partitions are assigned manually rather than via ``subscribe``; the two APIs
    are mutually exclusive in librdkafka, and manual assignment guarantees every
    partition is consumed immediately without waiting for a group rebalance.
    """
    consumer = kafka.Consumer(config)
    try:
        validate_topics_exist(consumer, topics)
        assign_all_partitions(consumer, topics)
        yield consumer
    finally:
        consumer.close()


@contextmanager
def make_consumer_from_config(
    *,
    topics: list[str],
    config: dict[str, Any],
    group: str,
    unique_group_id: bool = True,
) -> Generator[kafka.Consumer, None, None]:
    """Create a Kafka consumer from a configuration dictionary."""
    if unique_group_id:
        config['group.id'] = f'{group}_{uuid.uuid4()}'
    logger.info("kafka_consumer_created", topics=topics, group_id=config['group.id'])
    with make_bare_consumer(config=config, topics=topics) as consumer:
        yield consumer
