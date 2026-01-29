# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import json
import time
from contextlib import ExitStack
from pathlib import Path

import panel as pn
import scipp as sc
import structlog

from ess.livedata import Message, StreamId, StreamKind
from ess.livedata.config import config_names
from ess.livedata.config.config_loader import load_config
from ess.livedata.kafka.sink import KafkaSink, serialize_dataarray_to_f144

logger = structlog.get_logger(__name__)


class LogProducerWidget:
    """Widget for manually publishing log messages to Kafka streams."""

    def __init__(self, instrument: str, exit_stack: ExitStack):
        self._instrument = instrument

        self._sink = exit_stack.enter_context(
            KafkaSink(
                kafka_config=load_config(namespace=config_names.kafka_upstream),
                instrument=instrument,
                serializer=serialize_dataarray_to_f144,
            )
        )

        self._throttled_checkbox = pn.widgets.Checkbox(
            name='Continuous updates', value=True, width=300
        )
        pn.config.throttled = True
        self._throttled_checkbox.param.watch(self._on_throttled_change, 'value')

        self._sliders = []
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
        stream_name = config['stream_name']

        slider = pn.widgets.FloatSlider(
            name=config['label'],
            start=config['min'],
            end=config['max'],
            step=config['step'],
            value=config['initial'],
            width=300,
        )

        # Create callback with stream_name captured in closure
        def callback(event, stream=stream_name):
            self._publish_message(event.new, stream)

        slider.param.watch(callback, 'value')
        return slider

    def _publish_message(self, value: float, stream_name: str):
        """Publish a message to the specified stream."""
        da = sc.DataArray(
            sc.scalar(value), coords={'time': sc.scalar(time.time_ns(), unit='ns')}
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
