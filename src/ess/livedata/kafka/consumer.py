# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import confluent_kafka as kafka
import structlog
from confluent_kafka.error import KafkaException

logger = structlog.get_logger(__name__)


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
    """
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
        offsets: dict[int, int] = {}
        for partition in partition_ids:
            try:
                _low, high = consumer.get_watermark_offsets(
                    kafka.TopicPartition(topic, partition), timeout=5.0
                )
            except KafkaException as e:
                logger.exception(
                    "watermark_fetch_failed", topic=topic, partition=partition
                )
                raise ValueError(
                    f"Failed to fetch watermark for '{topic}' partition"
                    f" {partition}: {e}"
                ) from e
            offsets[partition] = high
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
