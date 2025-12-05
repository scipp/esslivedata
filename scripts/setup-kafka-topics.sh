#!/bin/bash
# This is run (automatically) inside the kafka container to (re)create topics.
# Usage (manually):
# docker exec kafka-broker sh /scripts/setup-kafka-topics.sh [--skip-delete] dream
set -e

SKIP_DELETE=false

# Parse optional flag
if [ "$1" = "--skip-delete" ]; then
  SKIP_DELETE=true
  shift
fi

if [ -z "$1" ]; then
  echo "Usage: $0 [--skip-delete] <LIVEDATA_INSTRUMENT>"
  exit 1
fi

LIVEDATA_INSTRUMENT=$1

if [ "$SKIP_DELETE" != "true" ]; then
  kafka-topics --bootstrap-server kafka:29092 --delete --topic '.*' || true
fi

kafka-leader-election --bootstrap-server kafka:29092 --election-type PREFERRED --all-topic-partitions

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_detector \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=10737418240 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_area_detector \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=10737418240 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_beam_monitor \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=10737418240 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_motion \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=10737418240 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_livedata_commands \
  --config cleanup.policy=delete \
  --config retention.ms=172800000 \
  --config delete.retention.ms=86400000 \
  --config max.message.bytes=1048576 \
  --config retention.bytes=-1 \
  --config segment.bytes=67108864 \
  --config segment.ms=86400000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_livedata_responses \
  --config cleanup.policy=delete \
  --config retention.ms=172800000 \
  --config delete.retention.ms=86400000 \
  --config max.message.bytes=1048576 \
  --config retention.bytes=-1 \
  --config segment.bytes=67108864 \
  --config segment.ms=86400000

# Status/heartbeat topic
kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_livedata_heartbeat \
  --config cleanup.policy=delete \
  --config retention.ms=172800000 \
  --config delete.retention.ms=86400000 \
  --config max.message.bytes=1048576 \
  --config retention.bytes=-1 \
  --config segment.bytes=67108864 \
  --config segment.ms=86400000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_livedata_data \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=1073741824 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000

kafka-topics --create --bootstrap-server kafka:29092 \
  --topic ${LIVEDATA_INSTRUMENT}_livedata_roi \
  --config cleanup.policy=delete \
  --config delete.retention.ms=60000 \
  --config max.message.bytes=104857600 \
  --config retention.bytes=1073741824 \
  --config retention.ms=30000 \
  --config segment.bytes=104857600 \
  --config segment.ms=60000
