"""
Plot composition prototype to validate the layer-based composition model.

This prototype demonstrates:
1. DynamicMap Curves that update via periodic callbacks
2. Dynamic addition of new Curve layers via button
3. Static overlay addition (VLines for peak markers) via button
4. Interactive stream overlays:
   - BoundsX: Tracks viewport/selection bounds (no toolbar tool)
   - BoxEdit: Interactive rectangle drawing (adds toolbar tool automatically)

Key insight from ROIDetectorPlotFactory:
- Each layer must be a separate DynamicMap
- Interactive streams (BoxEdit, PolyDraw) must attach to the DynamicMap, not the element
- Layers compose via * operator: dmap1 * dmap2 * dmap3
- This pattern enables both programmatic updates (via Pipe) and user interaction

Run with: panel serve prototype.py --show
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import holoviews as hv
import numpy as np
import panel as pn

hv.extension("bokeh")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Sources
# =============================================================================


@dataclass
class CurveData:
    """Simple curve data container."""

    x: np.ndarray
    y: np.ndarray
    label: str = "curve"


class DataGenerator:
    """Generates simulated curve data with noise."""

    def __init__(self, x_range: tuple[float, float] = (0, 10), n_points: int = 100):
        self.x = np.linspace(x_range[0], x_range[1], n_points)
        self._phase = 0.0
        self._amplitude = 1.0

    def generate(self, label: str = "curve") -> CurveData:
        """Generate curve data with random noise and evolving phase."""
        self._phase += 0.1
        noise = np.random.normal(0, 0.1, len(self.x))
        y = self._amplitude * np.sin(self.x + self._phase) + noise
        return CurveData(x=self.x, y=y, label=label)

    def set_amplitude(self, amplitude: float) -> None:
        """Set amplitude for generated curves."""
        self._amplitude = amplitude


# =============================================================================
# Layer Model
# =============================================================================


@dataclass(frozen=True)
class LayerSpec:
    """Specification for a single layer in a composed plot."""

    name: str
    element_type: str  # "curve", "vlines", "bounds_selector"
    params: dict = field(default_factory=dict)
    is_interactive: bool = False


@dataclass
class LayerState:
    """Runtime state for a layer."""

    spec: LayerSpec
    pipe: hv.streams.Pipe | None = None
    dmap: hv.DynamicMap | None = None
    stream: hv.streams.Stream | None = None  # For interactive layers
    data_generator: DataGenerator | None = None


# =============================================================================
# Element Factories
# =============================================================================


def curve_element(data: CurveData | None, **params) -> hv.Curve:
    """Create a Curve element from CurveData."""
    if data is None:
        return hv.Curve([])
    return hv.Curve((data.x, data.y), label=data.label).opts(**params)


def vlines_element(positions: list[float] | None, **params) -> hv.Overlay:
    """Create vertical lines at given positions."""
    if not positions:
        return hv.Overlay([])
    return hv.Overlay([hv.VLine(x).opts(**params) for x in positions])


def bounds_rectangles(data: list[tuple] | None, **params) -> hv.Rectangles:
    """Create rectangles for bounds selection display."""
    if not data:
        return hv.Rectangles([])
    return hv.Rectangles(data).opts(**params)


# =============================================================================
# Plot Composer
# =============================================================================


class PlotComposer:
    """
    Manages composition of multiple layers into a single plot.

    Handles:
    - Layer lifecycle (creation, updates, removal)
    - Periodic data updates for dynamic layers
    - Interactive stream setup (BoundsX)
    - Layer composition via overlay
    """

    def __init__(self, update_interval_ms: int = 500):
        self._layers: dict[str, LayerState] = {}
        self._composition: hv.DynamicMap | None = None
        self._update_callback: pn.io.PeriodicCallback | None = None
        self._update_interval_ms = update_interval_ms
        self._on_composition_change: Callable[[], None] | None = None
        self._bounds_callback: Callable[[dict], None] | None = None

    def set_composition_change_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be invoked when composition changes."""
        self._on_composition_change = callback

    def set_bounds_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for bounds selection events."""
        self._bounds_callback = callback

    def _update_all_dynamic_layers(self) -> None:
        """Update all dynamic curve layers at once using pn.io.hold()."""
        with pn.io.hold():
            for name, state in self._layers.items():
                if state.data_generator is not None and state.pipe is not None:
                    data = state.data_generator.generate(label=name)
                    state.pipe.send(data)

    def _ensure_update_callback(self) -> None:
        """Start the single update callback if not already running."""
        if self._update_callback is None:
            self._update_callback = pn.state.add_periodic_callback(
                self._update_all_dynamic_layers, period=self._update_interval_ms
            )
            logger.info(
                "Started periodic update callback (interval=%dms)",
                self._update_interval_ms,
            )

    def _stop_update_callback_if_no_dynamic_layers(self) -> None:
        """Stop the update callback if no dynamic layers remain."""
        has_dynamic = any(
            state.data_generator is not None for state in self._layers.values()
        )
        if not has_dynamic and self._update_callback is not None:
            self._update_callback.stop()
            self._update_callback = None
            logger.info("Stopped periodic update callback (no dynamic layers)")

    def add_dynamic_curve_layer(
        self,
        name: str,
        generator: DataGenerator,
        **params,
    ) -> None:
        """
        Add a dynamic curve layer that updates periodically.

        Parameters
        ----------
        name:
            Unique layer identifier.
        generator:
            Data generator for this curve.
        **params:
            HoloViews opts for the Curve element.
        """
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(name=name, element_type="curve", params=params)
        pipe = hv.streams.Pipe(data=None)

        # Capture variables to avoid closure issues with Panel sessions
        _curve_element = curve_element
        _params = params

        def make_curve(data, _fn=_curve_element, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_curve, streams=[pipe])

        state = LayerState(
            spec=spec,
            pipe=pipe,
            dmap=dmap,
            data_generator=generator,
        )
        self._layers[name] = state

        # Initial data
        initial_data = generator.generate(label=name)
        pipe.send(initial_data)

        # Ensure the shared update callback is running
        self._ensure_update_callback()

        logger.info("Added dynamic curve layer: %s", name)
        self._notify_composition_change()

    def add_static_vlines_layer(
        self, name: str, positions: list[float], **params
    ) -> None:
        """
        Add a static vertical lines layer.

        Parameters
        ----------
        name:
            Unique layer identifier.
        positions:
            X positions for vertical lines.
        **params:
            HoloViews opts for VLine elements.
        """
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(name=name, element_type="vlines", params=params)
        pipe = hv.streams.Pipe(data=positions)

        # Capture variables to avoid closure issues with Panel sessions
        _vlines_element = vlines_element
        _params = params

        def make_vlines(data, _fn=_vlines_element, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_vlines, streams=[pipe])

        state = LayerState(spec=spec, pipe=pipe, dmap=dmap)
        self._layers[name] = state

        logger.info("Added static vlines layer: %s with positions %s", name, positions)
        self._notify_composition_change()

    def add_bounds_selector_layer(self, name: str, **params) -> None:
        """
        Add an interactive bounds selection layer (BoundsX).

        This demonstrates the pattern from ROIDetectorPlotFactory:
        - Create a Pipe for programmatic rectangle updates
        - Create a DynamicMap wrapping Rectangles
        - Attach BoundsX stream to the DynamicMap (not the element)

        Parameters
        ----------
        name:
            Unique layer identifier.
        **params:
            HoloViews opts for Rectangles elements.
        """
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(
            name=name,
            element_type="bounds_selector",
            params=params,
            is_interactive=True,
        )

        # Create pipe for programmatic updates
        pipe = hv.streams.Pipe(data=[])

        # Capture variables explicitly to avoid closure issues with Panel sessions
        _bounds_rectangles = bounds_rectangles
        _params = params

        def make_rectangles(data, _fn=_bounds_rectangles, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_rectangles, streams=[pipe])

        # Add xbox_select tool to enable selection - BoundsX doesn't add it automatically
        # The tool must be on the DynamicMap that BoundsX is attached to
        dmap = dmap.opts(tools=['xbox_select'])

        # Create BoundsX stream attached to the DynamicMap
        # This is the critical pattern: source must be the DynamicMap
        bounds_stream = hv.streams.BoundsX(source=dmap)

        # Capture variables for closure
        _pipe = pipe
        _logger = logger
        _callback = self._bounds_callback

        def on_bounds_change(_pipe=_pipe, _logger=_logger, _cb=_callback, **kwargs):
            """Handle bounds selection changes."""
            bounds = kwargs.get("boundsx")
            if bounds is not None:
                x0, x1 = bounds
                # Update the rectangles display
                # BoundsX gives us x bounds, we need to create full rectangle
                # Using fixed y range since we don't have BoundsXY
                rect_data = [(x0, -2, x1, 2)]  # (x0, y0, x1, y1)
                _pipe.send(rect_data)
                _logger.info("Bounds selected: x=[%.2f, %.2f]", x0, x1)
                if _cb:
                    _cb({"x0": x0, "x1": x1})

        bounds_stream.add_subscriber(on_bounds_change)

        state = LayerState(spec=spec, pipe=pipe, dmap=dmap, stream=bounds_stream)
        self._layers[name] = state

        logger.info("Added bounds selector layer: %s", name)
        self._notify_composition_change()

    def add_box_edit_layer(self, name: str, num_objects: int = 4, **params) -> None:
        """
        Add an interactive BoxEdit layer for rectangle selection.

        Similar to ROIDetectorPlotFactory but simplified for the prototype.

        Parameters
        ----------
        name:
            Unique layer identifier.
        num_objects:
            Maximum number of rectangles allowed.
        **params:
            HoloViews opts for Rectangles elements.
        """
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(
            name=name, element_type="box_edit", params=params, is_interactive=True
        )

        # Create pipe for programmatic updates
        pipe = hv.streams.Pipe(data=[])

        # Capture variables to avoid closure issues with Panel sessions
        _bounds_rectangles = bounds_rectangles
        _params = params

        def make_rectangles(data, _fn=_bounds_rectangles, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_rectangles, streams=[pipe])

        # Create BoxEdit stream attached to the DynamicMap
        box_stream = hv.streams.BoxEdit(source=dmap, num_objects=num_objects, data={})

        # Capture variables for closure
        _pipe = pipe
        _logger = logger

        def on_box_change(data, _pipe=_pipe, _logger=_logger):
            """Handle box edit changes."""
            if data is None:
                return
            x0_list = data.get("x0", [])
            x1_list = data.get("x1", [])
            y0_list = data.get("y0", [])
            y1_list = data.get("y1", [])

            rect_data = list(zip(x0_list, y0_list, x1_list, y1_list, strict=True))
            _pipe.send(rect_data)
            _logger.info("BoxEdit: %d rectangles", len(rect_data))

        box_stream.param.watch(lambda event, _fn=on_box_change: _fn(event.new), "data")

        state = LayerState(spec=spec, pipe=pipe, dmap=dmap, stream=box_stream)
        self._layers[name] = state

        logger.info("Added BoxEdit layer: %s", name)
        self._notify_composition_change()

    def remove_layer(self, name: str) -> None:
        """Remove a layer by name."""
        if name not in self._layers:
            logger.warning("Layer %s not found", name)
            return

        del self._layers[name]
        logger.info("Removed layer: %s", name)

        # Stop the shared callback if no dynamic layers remain
        self._stop_update_callback_if_no_dynamic_layers()

        self._notify_composition_change()

    def get_composition(self) -> hv.DynamicMap | hv.Overlay:
        """
        Get the composed plot of all layers.

        Returns an overlay of all layer DynamicMaps.
        """
        if not self._layers:
            # Return empty curve if no layers
            return hv.DynamicMap(lambda: hv.Curve([]))

        # Note: Use `is not None` check, not truthiness - HoloViews DynamicMap
        # evaluates to False when empty, which would incorrectly filter out layers
        dmaps = [
            state.dmap for state in self._layers.values() if state.dmap is not None
        ]

        if not dmaps:
            return hv.DynamicMap(lambda: hv.Curve([]))

        if len(dmaps) == 1:
            return dmaps[0]

        # Compose all layers via overlay
        result = dmaps[0]
        for dmap in dmaps[1:]:
            result = result * dmap

        return result

    def _notify_composition_change(self) -> None:
        """Notify that composition has changed."""
        if self._on_composition_change:
            self._on_composition_change()

    def get_layer_names(self) -> list[str]:
        """Get list of current layer names."""
        return list(self._layers.keys())


# =============================================================================
# Dashboard Application
# =============================================================================


class CompositionDashboard:
    """
    Dashboard for testing plot composition.

    Provides controls for:
    - Adding dynamic curve layers
    - Adding static VLine layers
    - Adding interactive selection layers
    - Viewing the composed plot
    """

    def __init__(self):
        self._composer = PlotComposer()
        self._curve_counter = 0
        self._vlines_counter = 0
        self._bounds_info = pn.pane.Markdown("No bounds selected")

        # Setup composition change callback
        self._composer.set_composition_change_callback(self._update_plot_pane)
        self._composer.set_bounds_callback(self._on_bounds_selected)

        # Create UI components
        self._create_controls()
        initial_composition = self._composer.get_composition().opts(
            responsive=True, min_height=400
        )
        self._plot_pane = pn.pane.HoloViews(
            initial_composition,
            sizing_mode="stretch_both",
        )

    def _create_controls(self) -> None:
        """Create control buttons."""
        self._add_curve_btn = pn.widgets.Button(
            name="Add Dynamic Curve", button_type="primary"
        )
        self._add_curve_btn.on_click(self._on_add_curve)

        self._add_vlines_btn = pn.widgets.Button(
            name="Add Peak Markers (VLines)", button_type="success"
        )
        self._add_vlines_btn.on_click(self._on_add_vlines)

        self._add_bounds_btn = pn.widgets.Button(
            name="Add BoundsX Selector", button_type="warning"
        )
        self._add_bounds_btn.on_click(self._on_add_bounds)

        self._add_boxedit_btn = pn.widgets.Button(
            name="Add BoxEdit Layer", button_type="warning"
        )
        self._add_boxedit_btn.on_click(self._on_add_boxedit)

        self._layer_select = pn.widgets.Select(
            name="Remove Layer", options=[], width=200
        )
        self._remove_btn = pn.widgets.Button(name="Remove", button_type="danger")
        self._remove_btn.on_click(self._on_remove_layer)

        self._layer_list = pn.pane.Markdown("Layers: (none)")

    def _on_add_curve(self, event) -> None:
        """Add a new dynamic curve layer."""
        self._curve_counter += 1
        name = f"curve_{self._curve_counter}"

        # Create generator with random amplitude
        generator = DataGenerator()
        generator.set_amplitude(0.5 + random.random())

        # Assign a color from the default cycle
        colors = hv.Cycle.default_cycles["default_colors"]
        color = colors[(self._curve_counter - 1) % len(colors)]

        self._composer.add_dynamic_curve_layer(
            name=name,
            generator=generator,
            color=color,
            line_width=2,
        )
        self._update_layer_list()

    def _on_add_vlines(self, event) -> None:
        """Add a static VLines layer with random positions."""
        self._vlines_counter += 1
        name = f"peaks_{self._vlines_counter}"

        # Generate random peak positions
        n_peaks = random.randint(2, 5)
        positions = sorted([random.uniform(1, 9) for _ in range(n_peaks)])

        self._composer.add_static_vlines_layer(
            name=name,
            positions=positions,
            color="red",
            line_dash="dashed",
            line_width=1,
        )
        self._update_layer_list()

    def _on_add_bounds(self, event) -> None:
        """Add a BoundsX selector layer."""
        name = "bounds_selector"
        if name in self._composer.get_layer_names():
            logger.info("BoundsX selector already exists")
            return

        self._composer.add_bounds_selector_layer(
            name=name,
            fill_alpha=0.2,
            fill_color="blue",
            line_color="blue",
            line_width=2,
        )
        self._update_layer_list()

    def _on_add_boxedit(self, event) -> None:
        """Add a BoxEdit layer."""
        name = "box_edit"
        if name in self._composer.get_layer_names():
            logger.info("BoxEdit layer already exists")
            return

        self._composer.add_box_edit_layer(
            name=name,
            num_objects=4,
            fill_alpha=0.2,
            fill_color="green",
            line_color="green",
            line_width=2,
        )
        self._update_layer_list()

    def _on_remove_layer(self, event) -> None:
        """Remove selected layer."""
        name = self._layer_select.value
        if name:
            self._composer.remove_layer(name)
            self._update_layer_list()

    def _on_bounds_selected(self, bounds: dict) -> None:
        """Handle bounds selection."""
        self._bounds_info.object = f"Bounds: x=[{bounds['x0']:.2f}, {bounds['x1']:.2f}]"

    def _update_layer_list(self) -> None:
        """Update layer list display."""
        layers = self._composer.get_layer_names()
        if layers:
            layer_str = "\n".join(f"- {name}" for name in layers)
            self._layer_list.object = f"Layers:\n{layer_str}"
            self._layer_select.options = layers
        else:
            self._layer_list.object = "Layers: (none)"
            self._layer_select.options = []

    def _update_plot_pane(self) -> None:
        """Update plot pane with new composition."""
        composition = self._composer.get_composition()
        # Apply responsive sizing to the composition
        self._plot_pane.object = composition.opts(responsive=True, min_height=400)

    def get_layout(self) -> pn.Column:
        """Get the dashboard layout."""
        controls = pn.Column(
            "## Plot Composition Prototype",
            pn.pane.Markdown(
                """
                This prototype tests the layer-based composition model:
                1. **Add Dynamic Curve**: Updating curves (periodic callback)
                2. **Add Peak Markers**: Static vertical lines
                3. **Add BoundsX Selector**: Tracks viewport bounds (no toolbar tool)
                4. **Add BoxEdit Layer**: Interactive rectangles (**adds toolbar tool**)

                **Key insight**: Each layer is a separate `DynamicMap`. Interactive
                streams (BoxEdit, PolyDraw) must attach to the DynamicMap, not the
                element. This enables both programmatic updates and user interaction.
                """
            ),
            pn.Row(
                self._add_curve_btn,
                self._add_vlines_btn,
            ),
            pn.Row(
                self._add_bounds_btn,
                self._add_boxedit_btn,
            ),
            pn.Row(
                self._layer_select,
                self._remove_btn,
            ),
            self._layer_list,
            self._bounds_info,
            width=350,
        )

        plot = pn.Column(
            self._plot_pane,
            sizing_mode="stretch_both",
            min_height=500,
        )

        return pn.Row(
            controls,
            plot,
            sizing_mode="stretch_both",
        )


# =============================================================================
# Application Entry Point
# =============================================================================


def create_app():
    """Create the Panel application."""
    dashboard = CompositionDashboard()
    return dashboard.get_layout()


# For `panel serve prototype.py`
if __name__.startswith("bokeh"):
    pn.serve(create_app())
else:
    # For direct execution or testing
    app = create_app()
    app.servable()
