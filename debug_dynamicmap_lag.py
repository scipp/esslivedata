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

    def generate_image(self) -> np.ndarray:
        """Generate a random image with a counter overlay pattern."""
        # Create a random 512x512 image
        img = np.random.rand(size, size)
        # Add a pattern based on counter to make changes visible
        img[::50, :] += 0.5 * (self.counter % 10) / 10
        return img

    def send_image(self) -> None:
        """Generate and send a new image to the pipe."""
        self.counter += 1
        elapsed = time.time() - self.start_time

        if self.first_send is None:
            self.first_send = elapsed
            print(f"[{elapsed:.2f}s] First image sent (counter={self.counter})")

        img = self.generate_image()

        send_start = time.time()
        # Does hold increase the lag?
        # with pn.io.hold():
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
        
        **Expected behavior:** Image should appear within ~1 second of app load.
        
        **Bug:** Sometimes takes 10+ seconds for first image to render.
        
        Console shows:
        - When images are sent to the pipe
        - When first render occurs (empty vs. data)
        - The lag between empty placeholder and first real data
        - Timing for each render call
        """
    )

    pane_start = time.time()
    plot = pn.pane.HoloViews(dmap, sizing_mode='stretch_width')
    print(
        f"[{time.time() - setup_start:.2f}s] HoloViews pane created (took {(time.time() - pane_start) * 1000:.1f}ms)"
    )

    tabs = pn.Tabs(('Plot 1', plot))

    # Create layout
    template = pn.template.MaterialTemplate(
        title="DynamicMap Lag Debug",
        sidebar=[info],
        main=[tabs],
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
    pn.serve(create_app, port=5009, show=True, autoreload=False)
else:
    # For panel serve
    create_app().servable()
