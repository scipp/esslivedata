# Testing Guide

## Table of Contents

1. [Overview](#overview)
2. [Testing Philosophy](#testing-philosophy)
3. [Running Tests](#running-tests)
4. [Test Structure](#test-structure)
5. [Using Fakes for Unit Tests](#using-fakes-for-unit-tests)
6. [Testing Services](#testing-services)
7. [Testing Handlers](#testing-handlers)
8. [Testing Workflows](#testing-workflows)
9. [Integration Testing](#integration-testing)
10. [Testing with Fake Data Services](#testing-with-fake-data-services)

## Overview

ESSlivedata follows a **"test without Kafka"** philosophy for unit tests. All core logic is tested using fake implementations that don't require external dependencies. This makes tests:

- **Fast**: No Docker containers or network I/O
- **Reliable**: No flaky network issues
- **Portable**: Run anywhere without setup
- **Debuggable**: Deterministic, reproducible failures

**Key Principle:** Unit tests run without Kafka. Integration tests (services) require Kafka.

## Testing Philosophy

### Test Pyramid

```
         ╱╲
        ╱  ╲ Integration Tests
       ╱────╲ (with Kafka, few)
      ╱      ╲
     ╱        ╲ Service Tests
    ╱──────────╲ (with fakes, moderate)
   ╱            ╲
  ╱              ╲ Unit Tests
 ╱────────────────╲ (handlers, adapters, many)
```

**Layers:**

1. **Unit Tests** (80%): Core logic, handlers, adapters, message processing
   - Use fakes for all dependencies
   - No external dependencies
   - Fast, deterministic

2. **Service Tests** (15%): End-to-end service behavior
   - Use fake sources and sinks
   - Test service lifecycle, error handling
   - Still no Kafka

3. **Integration Tests** (5%): Real Kafka integration
   - Require Docker Kafka
   - Test actual message serialization/deserialization
   - Ensure compatibility with real systems

### What to Test

**Do Test:**
- Message routing logic
- Handler business logic
- Workflow accumulation and finalization
- Error handling and recovery
- Job lifecycle management
- Adapter transformations
- Service lifecycle (start, stop, signals)

**Don't Test:**
- Third-party library internals (confluent_kafka, streaming_data_types)
- Kafka broker behavior
- Network reliability
- GUI widget rendering (test controllers instead)

## Running Tests

### All Tests

```bash
# Using tox (recommended)
tox

# Specific Python version
tox -e py311

# Manual with pytest
python -m pytest
```

### Specific Test Files

```bash
# Single file
python -m pytest tests/core/processor_test.py

# Pattern matching
python -m pytest tests/handlers/
```

### With Coverage

```bash
# Coverage report
tox -e py311 -- --cov=ess.livedata --cov-report=html

# View in browser
open htmlcov/index.html
```

### Benchmarks Only

```bash
python -m pytest --benchmark-only
```

## Test Structure

### Directory Layout

```
tests/
├── core/                   # Core abstractions
│   ├── service_test.py
│   ├── processor_test.py
│   ├── handler_test.py
│   ├── message_test.py
│   └── job_manager_test.py
├── kafka/                  # Kafka integration
│   ├── source_test.py
│   ├── sink_test.py
│   ├── message_adapter_test.py
│   └── stream_mapping_test.py
├── handlers/               # Handler implementations
│   ├── detector_data_handler_test.py
│   ├── monitor_data_handler_test.py
│   └── data_reduction_handler_test.py
├── dashboard/              # Dashboard components
│   ├── config_service_test.py
│   ├── data_service_test.py
│   └── workflow_controller_test.py
└── fakes.py               # Fake implementations
```

### Naming Conventions

- Test files: `*_test.py`
- Test functions: `test_*`
- Test classes: `Test*`
- Fixtures: Descriptive names (no `test_` prefix)

### Example Test File

```python
import pytest
from ess.livedata.core.processor import StreamProcessor
from ess.livedata.fakes import FakeMessageSource, FakeMessageSink

def test_processor_routes_messages_to_handler():
    # Arrange
    source = FakeMessageSource([message1, message2])
    sink = FakeMessageSink()
    handler_registry = make_handler_registry()

    processor = StreamProcessor(
        source=source,
        sink=sink,
        handler_registry=handler_registry,
    )

    # Act
    processor.process()

    # Assert
    assert len(sink.messages) == 2
    assert sink.messages[0].value == expected_result
```

## Using Fakes for Unit Tests

### Available Fakes

Located in `ess.livedata.fakes`:

```python
from ess.livedata.fakes import (
    FakeMessageSource,      # In-memory message source
    FakeMessageSink,        # Collects messages for assertions
    FakeKafkaMessage,       # Simulate Kafka messages
    FakeKafkaConsumer,      # Simulate Kafka consumer
    FakeKafkaProducer,      # Simulate Kafka producer
)
```

### FakeMessageSource

```python
from ess.livedata.fakes import FakeMessageSource
from ess.livedata.core.message import Message, StreamId, StreamKind

# Create messages
messages = [
    Message(
        timestamp=1000,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='det1'),
        value=detector_events_1,
    ),
    Message(
        timestamp=2000,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='det1'),
        value=detector_events_2,
    ),
]

# Create source
source = FakeMessageSource(messages)

# Use in processor
processor = StreamProcessor(source=source, sink=sink, handler_registry=registry)
processor.process()

# First call returns all messages, subsequent calls return empty list
assert source.get_messages() == []
```

**Multiple Batches:**
```python
# Return different messages on each call
source = FakeMessageSource([])
source.add_messages([msg1, msg2])  # First call to get_messages()
source.add_messages([msg3, msg4])  # Second call to get_messages()
```

### FakeMessageSink

```python
from ess.livedata.fakes import FakeMessageSink

sink = FakeMessageSink()

# Processor publishes to sink
processor.process()

# Assert on published messages
assert len(sink.messages) == 3
assert sink.messages[0].stream.name == 'det1'
assert sink.messages[0].value == expected_value

# Clear for next test
sink.clear()
```

### FakeKafkaMessage

```python
from ess.livedata.fakes import FakeKafkaMessage
from streaming_data_types import eventdata_ev44

# Create serialized message
ev44_buffer = eventdata_ev44.serialise_ev44(
    source_name='detector_1',
    reference_time=[1000],
    time_of_flight=[100, 200, 300],
    detector_id=[0, 1, 2],
)

kafka_msg = FakeKafkaMessage(
    key=b'detector_1',
    value=ev44_buffer,
    topic='dream_detectors',
    timestamp=1000,
)

# Use with adapters
from ess.livedata.kafka.message_adapter import KafkaToEv44Adapter

adapter = KafkaToEv44Adapter(stream_lut=stream_lut)
message = adapter.adapt(kafka_msg)

assert message.stream.name == 'high_flux_detector'
```

## Testing Services

### Service Lifecycle Testing

```python
import pytest
from ess.livedata.core.service import Service
from ess.livedata.fakes import FakeMessageSource, FakeMessageSink

def test_service_processes_messages_in_loop():
    source = FakeMessageSource([msg1, msg2])
    sink = FakeMessageSink()
    processor = StreamProcessor(source=source, sink=sink, handler_registry=registry)

    service = Service(processor=processor, name='test_service')

    # Don't use blocking start() in tests, use step()
    service._running = True  # Simulate running state
    service.step()  # Single iteration

    assert len(sink.messages) == 2

def test_service_handles_processor_exceptions():
    # Processor that raises
    class FailingProcessor:
        def process(self):
            raise RuntimeError("Processing failed")

    service = Service(processor=FailingProcessor(), name='test_service')

    # Service should catch exception, log, and stop
    with pytest.raises(SystemExit):  # Service exits on error
        service._running = True
        service._run_loop()

def test_service_graceful_shutdown():
    import signal

    source = FakeMessageSource([])
    processor = StreamProcessor(source=source, sink=FakeMessageSink(), handler_registry=registry)
    service = Service(processor=processor)

    service.start(blocking=False)  # Start without blocking
    assert service.is_running

    # Simulate SIGTERM
    service._handle_shutdown(signal.SIGTERM, None)

    assert not service.is_running
```

### Testing with Context Manager

```python
def test_service_cleans_up_resources():
    resource_cleaned = False

    class FakeResource:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            nonlocal resource_cleaned
            resource_cleaned = True

    from contextlib import ExitStack
    resources = ExitStack()
    resources.enter_context(FakeResource())

    service = Service(
        processor=processor,
        resources=resources.pop_all(),
    )

    with service:
        pass  # Exit immediately

    assert resource_cleaned
```

## Testing Handlers

### Handler Unit Test Template

```python
import pytest
from ess.livedata.core.message import Message, StreamId, StreamKind
from ess.livedata.handlers.my_handler import MyHandler

@pytest.fixture
def handler():
    return MyHandler(logger=logging.getLogger(), config={})

def test_handler_processes_single_message(handler):
    # Arrange
    msg = Message(
        timestamp=1000,
        stream=StreamId(kind=StreamKind.DETECTOR_EVENTS, name='det1'),
        value=input_data,
    )

    # Act
    results = handler.handle([msg])

    # Assert
    assert len(results) == 1
    assert results[0].stream == msg.stream
    assert results[0].value == expected_output

def test_handler_batches_multiple_messages(handler):
    msgs = [
        Message(timestamp=1000, stream=stream_id, value=data1),
        Message(timestamp=2000, stream=stream_id, value=data2),
        Message(timestamp=3000, stream=stream_id, value=data3),
    ]

    results = handler.handle(msgs)

    # Handler may combine into single result
    assert len(results) == 1
    assert results[0].value == combined_output

def test_handler_handles_errors_gracefully(handler):
    # Invalid input
    msg = Message(timestamp=1000, stream=stream_id, value=invalid_data)

    # Should not raise, may return empty or error result
    results = handler.handle([msg])
    # Assert appropriate error handling
```

### Testing Accumulators

```python
from ess.livedata.handlers.accumulators import Cumulative

def test_accumulator_adds_data():
    acc = Cumulative(clear_on_get=False)

    acc.add(timestamp=1000, data=data1)
    acc.add(timestamp=2000, data=data2)

    result = acc.get()
    assert result == data1 + data2  # Cumulative sum

def test_accumulator_clears_on_get():
    acc = Cumulative(clear_on_get=True)

    acc.add(timestamp=1000, data=data1)
    result1 = acc.get()

    acc.add(timestamp=2000, data=data2)
    result2 = acc.get()

    # Second result should only have data2
    assert result2 == data2
```

## Testing Workflows

### Workflow Unit Test

```python
from ess.livedata.handlers.workflow_factory import Workflow

def test_workflow_accumulate_and_finalize():
    # Create workflow
    workflow = MyWorkflow(params=my_params)

    # Accumulate data
    workflow.accumulate({'detector_1': events1})
    workflow.accumulate({'detector_1': events2, 'sample_temp': temp})

    # Finalize
    result = workflow.finalize()

    assert 'histogram' in result
    assert result['histogram'].shape == expected_shape

def test_workflow_clear_resets_state():
    workflow = MyWorkflow(params=my_params)

    workflow.accumulate({'detector_1': events1})
    workflow.clear()

    # Should raise or return empty result
    with pytest.raises(ValueError):
        workflow.finalize()

def test_workflow_with_sciline():
    import sciline as sl
    from ess.reduce.streaming import StreamProcessor

    # Build sciline graph
    pipeline = sl.Pipeline(providers, params=params)
    workflow = StreamProcessor(pipeline)

    # Test accumulate/finalize/clear
    workflow.accumulate({'detector_1': events})
    result = workflow.finalize()
    workflow.clear()

    assert result is not None
```

## Integration Testing

### Testing with Real Kafka

Integration tests require Docker Kafka:

```bash
# Start Kafka
docker-compose up kafka

# Run integration tests
python -m pytest tests/integration/
```

**Example Integration Test:**

```python
import pytest
from confluent_kafka import Producer, Consumer

@pytest.mark.integration
def test_kafka_roundtrip():
    # This test requires Kafka running
    producer = Producer({'bootstrap.servers': 'localhost:29092'})
    consumer = Consumer({
        'bootstrap.servers': 'localhost:29092',
        'group.id': 'test_group',
        'auto.offset.reset': 'earliest',
    })

    # Produce message
    producer.produce('test_topic', value=b'test_message')
    producer.flush()

    # Consume message
    consumer.subscribe(['test_topic'])
    msg = consumer.poll(timeout=5.0)

    assert msg is not None
    assert msg.value() == b'test_message'

    consumer.close()
```

**Marking Integration Tests:**

```python
# In pytest.ini or conftest.py
markers =
    integration: marks tests as integration tests (require Kafka)

# Run only unit tests (skip integration)
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```

## Testing with Fake Data Services

### Running Fake Services

For manual/visual testing without real instruments:

```bash
# Terminal 1: Fake detector events
python -m ess.livedata.services.fake_detectors --instrument dummy

# Terminal 2: Fake monitor events
python -m ess.livedata.services.fake_monitors --mode ev44 --instrument dummy

# Terminal 3: Fake log data
python -m ess.livedata.services.fake_logdata --instrument dummy

# Terminal 4: Process data
python -m ess.livedata.services.detector_data --instrument dummy --dev

# Terminal 5: Dashboard
python -m ess.livedata.dashboard.reduction --instrument dummy
```

### Testing with PNG Sink

Avoid running full dashboard for testing service outputs:

```bash
# Save outputs as PNG files instead of Kafka
python -m ess.livedata.services.detector_data \
    --instrument dummy \
    --dev \
    --sink png

# Check PNG files in current directory
ls *.png
```

### Fake Service Patterns

**Configurable Fake Data:**
```python
# In fake service
from ess.livedata.fakes import FakeMessageSource

def generate_fake_events(num_events: int) -> DetectorEvents:
    return DetectorEvents(
        time_of_flight=np.random.exponential(1000, num_events),
        detector_id=np.random.randint(0, 10000, num_events),
    )

# Publish at regular intervals
while True:
    events = generate_fake_events(num_events=1000)
    producer.produce(topic, serialize(events))
    time.sleep(0.1)  # 10 Hz
```

## Best Practices

### Do's

✅ **Use fakes for unit tests** - Don't require Kafka
✅ **Test one thing per test** - Clear, focused assertions
✅ **Use descriptive names** - `test_processor_routes_messages_to_correct_handler()`
✅ **Test error cases** - Don't just test happy paths
✅ **Use fixtures** - Share common setup
✅ **Mock external dependencies** - Not ESSlivedata code
✅ **Test public interfaces** - Not private methods

### Don'ts

❌ **Don't test third-party code** - Trust confluent_kafka works
❌ **Don't use real Kafka in unit tests** - Use fakes
❌ **Don't test implementation details** - Test behavior
❌ **Don't write brittle tests** - Avoid exact float comparisons
❌ **Don't share state between tests** - Each test independent
❌ **Don't skip cleanup** - Use fixtures or context managers

### Example: Good vs Bad

**Bad:**
```python
def test_everything():
    # Test too broad, hard to debug
    service = create_service()
    service.start()
    time.sleep(5)
    result = get_result_from_kafka()
    assert result is not None  # What failed?
```

**Good:**
```python
def test_processor_routes_detector_events_to_detector_handler():
    # Focused, uses fakes, clear assertion
    source = FakeMessageSource([detector_message])
    sink = FakeMessageSink()
    processor = StreamProcessor(source, sink, detector_registry)

    processor.process()

    assert len(sink.messages) == 1
    assert sink.messages[0].stream.kind == StreamKind.DETECTOR_EVENTS
```

---

## Summary

ESSlivedata testing strategy:

- **Unit tests** (80%): Fast, deterministic, use fakes
- **Service tests** (15%): End-to-end behavior, still with fakes
- **Integration tests** (5%): Real Kafka, verify compatibility

**Key Testing Tools:**
- `FakeMessageSource`: In-memory message provider
- `FakeMessageSink`: Collect and assert on results
- `FakeKafkaMessage`: Simulate Kafka messages
- `pytest`: Test framework
- `tox`: Test automation across Python versions

**Philosophy:** Test business logic thoroughly without external dependencies. Use integration tests sparingly to verify real-world compatibility.

For more information on running tests, see [Getting Started](../getting-started.md#running-tests).
