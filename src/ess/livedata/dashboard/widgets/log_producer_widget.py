# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import panel as pn
import scipp as sc
import structlog

from ess.livedata import Message, StreamId, StreamKind
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.core.timestamp import Timestamp
from ess.livedata.kafka.sink import KafkaSink
from ess.livedata.kafka.sink_serializers import F144Serializer

logger = structlog.get_logger(__name__)


PublishFn = Callable[[float, str], None]


class _DeviceMotion:
    """Per-device state machine that simulates motor motion on slider changes.

    On every slider change the widget calls :meth:`on_change`. The state
    machine then publishes:

    1. ``target_stream`` = new slider value (VAL update, immediate)
    2. ``idle_stream`` = 0 (DMOV cleared, immediate)
    3. ``ramp_steps`` ``value_stream`` updates linearly from the previous
       reading to the new target over ``ramp_seconds``
    4. ``idle_stream`` = 1 on the final step (DMOV set, device settled)

    The initial publish in :meth:`__init__` seeds all three substreams so the
    downstream :class:`DeviceSynthesizer` bootstraps immediately.
    """

    def __init__(
        self,
        *,
        publish: PublishFn,
        value_stream: str,
        target_stream: str,
        idle_stream: str | None,
        initial: float,
        ramp_seconds: float = 2.0,
        ramp_steps: int = 10,
    ) -> None:
        self._publish = publish
        self._value_stream = value_stream
        self._target_stream = target_stream
        self._idle_stream = idle_stream
        self._ramp_seconds = ramp_seconds
        self._ramp_steps = max(ramp_steps, 1)
        self._current = float(initial)
        self._target = float(initial)
        self._ramp_start = float(initial)
        self._step_index = 0
        self._callback: pn.io.PeriodicCallback | None = None
        # Seed all substreams so the device bootstraps and emits immediately.
        self._publish(self._target, self._target_stream)
        self._publish(self._current, self._value_stream)
        if self._idle_stream is not None:
            self._publish(1.0, self._idle_stream)

    def on_change(self, new_target: float) -> None:
        self._cancel_pending()
        self._target = float(new_target)
        self._ramp_start = self._current
        self._step_index = 0
        # 1. New setpoint.
        self._publish(self._target, self._target_stream)
        # 2. Mark as moving.
        if self._idle_stream is not None:
            self._publish(0.0, self._idle_stream)
        # 3. Schedule the ramp.
        period_ms = max(50, int(self._ramp_seconds * 1000 / self._ramp_steps))
        self._callback = pn.state.add_periodic_callback(
            self._step, period=period_ms, count=self._ramp_steps
        )

    def _step(self) -> None:
        self._step_index += 1
        if self._step_index >= self._ramp_steps:
            self._current = self._target
        else:
            fraction = self._step_index / self._ramp_steps
            self._current = (
                self._ramp_start + (self._target - self._ramp_start) * fraction
            )
        self._publish(self._current, self._value_stream)
        if self._step_index >= self._ramp_steps:
            if self._idle_stream is not None:
                self._publish(1.0, self._idle_stream)
            self._callback = None

    def _cancel_pending(self) -> None:
        if self._callback is not None and self._callback.running:
            self._callback.stop()
        self._callback = None


class LogProducerWidget:
    """Widget for manually publishing log messages to Kafka streams.

    Each slider entry is either a simple log stream (single f144 publish per
    change, via ``stream_name``) or a synthesised device drive (``value_stream``
    plus ``target_stream`` plus optional ``idle_stream``) where the slider
    represents the target setpoint and the widget animates a readback ramp
    plus DMOV transitions on each change.
    """

    def __init__(self, instrument: str, exit_stack: ExitStack):
        self._instrument = instrument

        self._sink = exit_stack.enter_context(
            KafkaSink(
                kafka_config=load_config(namespace=config_names.kafka),
                serializer=F144Serializer(instrument=instrument),
            )
        )

        self._throttled_checkbox = pn.widgets.Checkbox(
            label='Continuous updates', value=True, width=300
        )
        pn.config.throttled = True
        self._throttled_checkbox.param.watch(self._on_throttled_change, 'value')

        self._sliders: list[pn.widgets.FloatSlider] = []
        self._device_motions: list[_DeviceMotion] = []
        self._load_and_create_sliders()

    def _load_and_create_sliders(self):
        """Load slider configuration from JSON file and create widgets."""
        config_path = self._get_config_path()

        if not config_path.exists():
            logger.warning("Log producer config file not found: %s", config_path)
            return

        try:
            with open(config_path) as f:
                config = json.load(f)

            for slider_config in config.get('sliders', []):
                slider = self._create_slider(slider_config)
                self._sliders.append(slider)

        except Exception as e:
            logger.error("Failed to load log producer config: %s", e)

    def _get_config_path(self) -> Path:
        """Get the path to the configuration file for the current instrument."""
        # Get the package root directory
        package_root = Path(__file__).resolve().parents[5]
        config_dir = package_root / 'configs'
        return config_dir / f'log_producer_{self._instrument}.json'

    def _create_slider(self, config: dict) -> pn.widgets.FloatSlider:
        """Create a slider widget from configuration."""
        slider = pn.widgets.FloatSlider(
            label=config['label'],
            start=config['min'],
            end=config['max'],
            step=config['step'],
            value=config['initial'],
            width=300,
        )

        if 'value_stream' in config and 'target_stream' in config:
            motion = _DeviceMotion(
                publish=self._publish_value,
                value_stream=config['value_stream'],
                target_stream=config['target_stream'],
                idle_stream=config.get('idle_stream'),
                initial=config['initial'],
                ramp_seconds=config.get('ramp_seconds', 2.0),
                ramp_steps=config.get('ramp_steps', 10),
            )
            self._device_motions.append(motion)
            slider.param.watch(lambda event: motion.on_change(event.new), 'value')
        else:
            stream_name = config['stream_name']
            slider.param.watch(
                lambda event, name=stream_name: self._publish_value(event.new, name),
                'value',
            )
        return slider

    def _publish_value(self, value: float, stream_name: str) -> None:
        """Publish a single f144 value to the named stream."""
        da = sc.DataArray(
            sc.scalar(float(value)),
            coords={'time': Timestamp.now().to_scipp()},
        )
        msg = Message(value=da, stream=StreamId(kind=StreamKind.LOG, name=stream_name))
        self._sink.publish_messages([msg])
        logger.info(
            "Stream '%s' - Published message with value: %s", stream_name, value
        )

    def _on_throttled_change(self, event):
        """Update pn.config.throttled when checkbox value changes."""
        pn.config.throttled = event.new
        logger.info("Throttled mode set to: %s", event.new)

    @property
    def panel(self) -> pn.viewable.Viewable:
        """Return the panel widget."""
        return pn.Column(
            pn.pane.Markdown("## Fake instrument controls"),
            *self._sliders,
            self._throttled_checkbox,
            pn.pane.Alert(
                "### Note\nThis **does not control the instrument**. The sliders feed "
                "motor \"readback\" values into Kafka for demonstration purposes. "
                "This widget is only shown in development mode and is not available in "
                "a production version of the ESSlivedata Dashboard.",
                alert_type="warning",
            ),
        )
