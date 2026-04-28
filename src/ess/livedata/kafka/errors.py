# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Shared error classification for Kafka source and sink."""

from confluent_kafka import KafkaError

# Auth/misconfiguration codes treated as fatal in addition to librdkafka-flagged
# fatal errors. Crashing on these surfaces the problem to the operator instead
# of silently spamming the broker with retries (consumer side) or dropping
# results into a metric counter (producer side). The set is the union of codes
# relevant to either side; codes that a given side cannot encounter are
# harmless to include.
FATAL_ERROR_CODES = frozenset(
    {
        KafkaError.TOPIC_AUTHORIZATION_FAILED,
        KafkaError.GROUP_AUTHORIZATION_FAILED,
        KafkaError.CLUSTER_AUTHORIZATION_FAILED,
        KafkaError.SASL_AUTHENTICATION_FAILED,
        KafkaError.TRANSACTIONAL_ID_AUTHORIZATION_FAILED,
    }
)


def is_fatal(err: KafkaError) -> bool:
    """Return True if the error should crash the service."""
    return err.fatal() or err.code() in FATAL_ERROR_CODES
