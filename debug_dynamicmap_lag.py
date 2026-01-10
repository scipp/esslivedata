"""Minimal example to reproduce DynamicMap initialization lag issue."""

import time
from datetime import datetime

import holoviews as hv
import numpy as np
import panel as pn
from holoviews import streams

hv.extension('bokeh')
pn.extension()

size = 1024


class ImageStreamer:
    """Simple class to stream random images through a Pipe."""

    def __init__(self) -> None:
        self.pipe = streams.Pipe(data=[])
        self.counter = 0
        self.start_time = time.time()
        self.first_send = None
        self.first_render_empty = None
        self.first_render_data = None
        # Add widgets for status tracking
        self.status_text = pn.pane.Markdown(
            "**Status:** Waiting for first update...", sizing_mode='stretch_width'
        )
        self.counter_indicator = pn.indicators.Number(
            name='Images Sent',
            value=0,
            format='{value:,.0f}',
            default_color='primary',
            sizing_mode='stretch_width',
        )
        self.last_update_indicator = pn.widgets.StaticText(
            name='Last Update',
            value='None',
            sizing_mode='stretch_width',
        )

    def generate_image(self) -> np.ndarray:
        """Generate a random image with a counter overlay pattern."""
        # Create a random 512x512 image
        img = np.random.rand(size, size)
        # Add a pattern based on counter to make changes visible
        img[::50, :] += 0.5 * (self.counter % 10) / 10
        return img

    def send_image(self) -> None:
        """Generate and send a new image to the pipe."""
        # Does hold increase the lag?
        with pn.io.hold():
            self._send_image()

    def _send_image(self) -> None:
        self.counter += 1
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if self.first_send is None:
            self.first_send = elapsed
            print(f"[{elapsed:.2f}s] First image sent (counter={self.counter})")

        # Update status widgets FIRST
        widget_update_start = time.time()

        self.status_text.object = f"**Status:** Active - **Counter:** {self.counter} - **Elapsed:** {elapsed:.2f}s - **Time:** {timestamp}"

        self.counter_indicator.value = self.counter
        self.last_update_indicator.value = timestamp

        widget_update_duration = time.time() - widget_update_start
        print(
            f"[{elapsed:.2f}s] Widgets updated (took {widget_update_duration * 1000:.1f}ms)"
        )

        img = self.generate_image()

        send_start = time.time()
        self.pipe.send(img)
        send_duration = time.time() - send_start

        print(
            f"[{elapsed:.2f}s] Sent image #{self.counter} (send took {send_duration * 1000:.1f}ms)"
        )

    def create_plot(self, data: np.ndarray) -> hv.Image:
        """Create a HoloViews Image from numpy array."""
        render_start = time.time()
        elapsed = render_start - self.start_time
        is_empty = len(data) == 0

        if is_empty:
            if self.first_render_empty is None:
                self.first_render_empty = elapsed
                print(f"[{elapsed:.2f}s] First render (EMPTY PLACEHOLDER)")
            # Return empty placeholder
            result = hv.Image(np.zeros((size, size))).opts(
                width=800, height=600, colorbar=True, title="Waiting for data..."
            )
        else:
            if self.first_render_data is None:
                self.first_render_data = elapsed
                lag_from_empty = self.first_render_data - (self.first_render_empty or 0)
                lag_from_send = self.first_render_data - (self.first_send or 0)
                print(f"[{elapsed:.2f}s] First render (REAL DATA)")
                print(f"    -> Lag from first send: {lag_from_send:.2f}s")
                print(f"    -> Lag from empty render: {lag_from_empty:.2f}s")

            result = hv.Image(data).opts(
                width=800,
                height=600,
                cmap='viridis',
                colorbar=True,
                # framewise does not appear to cause extra lag
                framewise=True,
                title=f"Image #{self.counter} @ {datetime.now().strftime('%H:%M:%S')}",
            )

        render_duration = time.time() - render_start
        print(
            f"[{elapsed:.2f}s] create_plot() completed ({'empty' if is_empty else 'data'}, took {render_duration * 1000:.1f}ms)"
        )

        return result


def create_app() -> pn.template.MaterialTemplate:
    """Create the Panel application."""
    setup_start = time.time()
    print(f"[0.00s] create_app() starting...")

    streamer = ImageStreamer()
    print(f"[{time.time() - setup_start:.2f}s] ImageStreamer created")

    # Create DynamicMap with Pipe stream
    dmap_start = time.time()
    dmap = hv.DynamicMap(
        streamer.create_plot,
        streams=[streamer.pipe],
        cache_size=1,
    ).opts(shared_axes=False)
    print(
        f"[{time.time() - setup_start:.2f}s] DynamicMap created (took {(time.time() - dmap_start) * 1000:.1f}ms)"
    )

    # Info panel
    info = pn.pane.Markdown(
        """
        # DynamicMap Initialization Lag Test

        This app sends a new random image every second via `hv.streams.Pipe`.
        Watch the console output to see timing information.

        **Bug:** Sometimes takes 10+ seconds for first image to render. Did not
        successfully reproduce this here yet.
        """
    )

    # Status widgets panel
    status_panel = pn.Column(
        "## Real-time Status",
        streamer.status_text,
        streamer.counter_indicator,
        streamer.last_update_indicator,
        sizing_mode='stretch_width',
    )

    pane_start = time.time()
    plot = pn.pane.HoloViews(dmap, sizing_mode='stretch_width')
    print(
        f"[{time.time() - setup_start:.2f}s] HoloViews pane created (took {(time.time() - pane_start) * 1000:.1f}ms)"
    )

    # Does plots in tabs affect anything? Not as far as I can tell.
    tabs = pn.Tabs(('Plot 1', plot))

    # Create layout
    template = pn.template.MaterialTemplate(
        title="DynamicMap Lag Debug",
        sidebar=[info],
        main=[status_panel, tabs],
        header_background='#2596be',
    )

    print(f"[{time.time() - setup_start:.2f}s] Template created")

    # Start periodic callback (1000ms = 1 second)
    pn.state.add_periodic_callback(streamer.send_image, period=1000)
    print(f"[{time.time() - setup_start:.2f}s] Periodic callback registered")

    print("=" * 60)
    print("App initialized. Watch for timing information below:")
    print("=" * 60)

    return template


if __name__ == '__main__':
    # For running standalone
    pn.serve(create_app, port=5111, show=True, autoreload=False)
else:
    # For panel serve
    create_app().servable()
