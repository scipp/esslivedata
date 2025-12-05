"""
Plot Composition Prototype
==========================

A reference implementation validating the layer-based composition model for ESSlivedata.

Run with: panel serve prototype.py --show

What This Prototype Demonstrates
--------------------------------
1. Multiple DynamicMap layers composed via the * operator
2. Dynamic layer addition/removal at runtime
3. Periodic data updates to curve layers
4. Static overlays (VLines for peak markers)
5. Interactive streams (BoundsX, BoxEdit) attached to layers

Key Learnings (see design doc for details)
------------------------------------------
1. DYNAMICMAP COMPOSITION: Each layer must be a separate DynamicMap. Interactive
   streams (BoxEdit, PolyDraw) must attach to the DynamicMap, not the element.
   Pattern: `DynamicMap(Curve) * DynamicMap(Rectangles)` - NOT
   `DynamicMap(Curve * Rectangles)`

2. TOOL REGISTRATION: BoxEdit/PolyDraw auto-add their toolbar tools.
   BoundsX/BoundsXY do NOT - must explicitly add via `.opts(tools=['xbox_select'])`.

3. CLOSURE CAPTURE: Panel sessions require explicit variable capture in callbacks.
   Use `def fn(data, _var=var)` pattern, not bare closures.

4. DYNAMICMAP TRUTHINESS: Empty DynamicMap evaluates to False.
   Use `if dmap is not None`, not `if dmap`.

5. SINGLE UPDATE CALLBACK: One shared periodic callback with `pn.io.hold()` is
   more efficient than per-layer callbacks.

6. FULL REBUILD OK: Rebuilding composition on layer add/remove is acceptable
   since these operations are rare.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import holoviews as hv
import numpy as np
import panel as pn

hv.extension("bokeh")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Model
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
        self._amplitude = amplitude


# =============================================================================
# Layer Specification
# =============================================================================


@dataclass(frozen=True)
class LayerSpec:
    """Specification for a single layer in a composed plot."""

    name: str
    element_type: str  # "curve", "vlines", "bounds_selector", "box_edit"
    params: dict = field(default_factory=dict)
    is_interactive: bool = False


@dataclass
class LayerState:
    """Runtime state for a layer.

    This tracks all artifacts needed for a layer's lifecycle:
    - spec: The immutable specification
    - pipe: For pushing data updates
    - dmap: The DynamicMap that renders this layer
    - stream: For interactive layers (BoundsX, BoxEdit, etc.)
    - data_generator: For layers with periodic updates
    """

    spec: LayerSpec
    pipe: hv.streams.Pipe | None = None
    dmap: hv.DynamicMap | None = None
    stream: hv.streams.Stream | None = None
    data_generator: DataGenerator | None = None


# =============================================================================
# Element Factories
#
# These are simple functions that transform data into HoloViews elements.
# In the real implementation, existing Plotter classes become element factories.
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


def rectangles_element(data: list[tuple] | None, **params) -> hv.Rectangles:
    """Create rectangles from list of (x0, y0, x1, y1) tuples."""
    if not data:
        return hv.Rectangles([])
    return hv.Rectangles(data).opts(**params)


# =============================================================================
# Plot Composer
#
# The composer manages layer lifecycle and composition. Each layer gets its own
# Pipe (for data updates) and DynamicMap (for rendering). Interactive streams
# attach to the DynamicMap, enabling both programmatic updates and user interaction.
# =============================================================================


class PlotComposer:
    """
    Manages composition of multiple layers into a single plot.

    Responsibilities:
    - Layer lifecycle (creation, updates, removal)
    - Periodic data updates for dynamic layers
    - Interactive stream setup
    - Layer composition via overlay
    """

    def __init__(self, update_interval_ms: int = 500):
        self._layers: dict[str, LayerState] = {}
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

    # -------------------------------------------------------------------------
    # Periodic Updates
    # -------------------------------------------------------------------------

    def _update_all_dynamic_layers(self) -> None:
        """Update all dynamic curve layers at once.

        KEY INSIGHT: Use pn.io.hold() to batch all pipe updates into a single
        render cycle. This reduces visual flicker and improves performance
        compared to per-layer callbacks.
        """
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

    # -------------------------------------------------------------------------
    # Layer Creation
    # -------------------------------------------------------------------------

    def add_dynamic_curve_layer(
        self,
        name: str,
        generator: DataGenerator,
        **params,
    ) -> None:
        """Add a dynamic curve layer that updates periodically."""
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(name=name, element_type="curve", params=params)
        pipe = hv.streams.Pipe(data=None)

        # KEY INSIGHT: Closure Capture
        # Panel sessions can have issues with closures that reference outer
        # variables. Capture variables explicitly as default arguments to
        # ensure each session gets its own copy.
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

        # Send initial data
        initial_data = generator.generate(label=name)
        pipe.send(initial_data)

        self._ensure_update_callback()
        logger.info("Added dynamic curve layer: %s", name)
        self._notify_composition_change()

    def add_static_vlines_layer(
        self, name: str, positions: list[float], **params
    ) -> None:
        """Add a static vertical lines layer (e.g., peak markers)."""
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(name=name, element_type="vlines", params=params)
        pipe = hv.streams.Pipe(data=positions)

        # Closure capture pattern
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
        """Add an interactive bounds selection layer (BoundsX).

        KEY INSIGHT: Tool Registration
        BoundsX does NOT automatically add its toolbar tool. We must explicitly
        add it via .opts(tools=['xbox_select']). In contrast, BoxEdit and
        PolyDraw DO auto-add their tools.
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

        pipe = hv.streams.Pipe(data=[])

        # Closure capture pattern
        _rectangles_element = rectangles_element
        _params = params

        def make_rectangles(data, _fn=_rectangles_element, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_rectangles, streams=[pipe])

        # KEY INSIGHT: BoundsX requires explicit tool registration
        dmap = dmap.opts(tools=["xbox_select"])

        # KEY INSIGHT: Interactive Stream Attachment
        # The stream must attach to the DynamicMap (source=dmap), not to
        # an element. This is required for both:
        # 1. Click detection (HoloViews routes events to the source)
        # 2. Programmatic updates (Pipe → DynamicMap → re-render)
        bounds_stream = hv.streams.BoundsX(source=dmap)

        # Closure capture for subscriber
        _pipe = pipe
        _logger = logger
        _callback = self._bounds_callback

        def on_bounds_change(_pipe=_pipe, _logger=_logger, _cb=_callback, **kwargs):
            bounds = kwargs.get("boundsx")
            if bounds is not None:
                x0, x1 = bounds
                # Update rectangles display (fixed y range since BoundsX only gives x)
                rect_data = [(x0, -2, x1, 2)]
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
        """Add an interactive BoxEdit layer for rectangle drawing.

        KEY INSIGHT: BoxEdit auto-adds its toolbar tool, unlike BoundsX.
        No explicit .opts(tools=[...]) needed.
        """
        if name in self._layers:
            logger.warning("Layer %s already exists, removing first", name)
            self.remove_layer(name)

        spec = LayerSpec(
            name=name, element_type="box_edit", params=params, is_interactive=True
        )

        pipe = hv.streams.Pipe(data=[])

        # Closure capture pattern
        _rectangles_element = rectangles_element
        _params = params

        def make_rectangles(data, _fn=_rectangles_element, _p=_params):
            return _fn(data, **_p)

        dmap = hv.DynamicMap(make_rectangles, streams=[pipe])

        # KEY INSIGHT: BoxEdit attaches to DynamicMap and auto-adds its tool
        box_stream = hv.streams.BoxEdit(source=dmap, num_objects=num_objects, data={})

        # Closure capture for watcher
        _pipe = pipe
        _logger = logger

        def on_box_change(data, _pipe=_pipe, _logger=_logger):
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

    # -------------------------------------------------------------------------
    # Layer Removal and Composition
    # -------------------------------------------------------------------------

    def remove_layer(self, name: str) -> None:
        """Remove a layer by name."""
        if name not in self._layers:
            logger.warning("Layer %s not found", name)
            return

        del self._layers[name]
        logger.info("Removed layer: %s", name)

        self._stop_update_callback_if_no_dynamic_layers()
        self._notify_composition_change()

    def get_composition(self) -> hv.DynamicMap | hv.Overlay:
        """Get the composed plot of all layers.

        KEY INSIGHT: DynamicMap Composition
        Each layer is a separate DynamicMap. We compose them via the * operator:
        `dmap1 * dmap2 * dmap3`. This produces an Overlay where each layer can
        update independently and interactive streams work correctly.
        """
        if not self._layers:
            return hv.DynamicMap(lambda: hv.Curve([]))

        # KEY INSIGHT: DynamicMap Truthiness
        # HoloViews DynamicMap evaluates to False when empty. We must use
        # explicit `is not None` check, not truthiness, or we'd incorrectly
        # filter out valid but empty layers.
        dmaps = [
            state.dmap for state in self._layers.values() if state.dmap is not None
        ]

        if not dmaps:
            return hv.DynamicMap(lambda: hv.Curve([]))

        if len(dmaps) == 1:
            return dmaps[0]

        # Compose via * operator - first layer is "bottom", subsequent on top
        result = dmaps[0]
        for dmap in dmaps[1:]:
            result = result * dmap

        return result

    def _notify_composition_change(self) -> None:
        """Notify that composition has changed (layer added/removed)."""
        if self._on_composition_change:
            self._on_composition_change()

    def get_layer_names(self) -> list[str]:
        """Get list of current layer names."""
        return list(self._layers.keys())


# =============================================================================
# Dashboard Application
# =============================================================================


class CompositionDashboard:
    """Dashboard for testing plot composition."""

    def __init__(self):
        self._composer = PlotComposer()
        self._curve_counter = 0
        self._vlines_counter = 0
        self._bounds_info = pn.pane.Markdown("No bounds selected")

        self._composer.set_composition_change_callback(self._update_plot_pane)
        self._composer.set_bounds_callback(self._on_bounds_selected)

        self._create_controls()
        initial_composition = self._composer.get_composition().opts(
            responsive=True, min_height=400
        )
        self._plot_pane = pn.pane.HoloViews(
            initial_composition,
            sizing_mode="stretch_both",
        )

    def _create_controls(self) -> None:
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
        self._curve_counter += 1
        name = f"curve_{self._curve_counter}"

        generator = DataGenerator()
        generator.set_amplitude(0.5 + np.random.random())

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
        self._vlines_counter += 1
        name = f"peaks_{self._vlines_counter}"

        n_peaks = np.random.randint(2, 6)
        positions = sorted([np.random.uniform(1, 9) for _ in range(n_peaks)])

        self._composer.add_static_vlines_layer(
            name=name,
            positions=positions,
            color="red",
            line_dash="dashed",
            line_width=1,
        )
        self._update_layer_list()

    def _on_add_bounds(self, event) -> None:
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
        name = self._layer_select.value
        if name:
            self._composer.remove_layer(name)
            self._update_layer_list()

    def _on_bounds_selected(self, bounds: dict) -> None:
        self._bounds_info.object = f"Bounds: x=[{bounds['x0']:.2f}, {bounds['x1']:.2f}]"

    def _update_layer_list(self) -> None:
        layers = self._composer.get_layer_names()
        if layers:
            layer_str = "\n".join(f"- {name}" for name in layers)
            self._layer_list.object = f"Layers:\n{layer_str}"
            self._layer_select.options = layers
        else:
            self._layer_list.object = "Layers: (none)"
            self._layer_select.options = []

    def _update_plot_pane(self) -> None:
        """Update plot pane with new composition.

        Called when layers are added/removed. Full rebuild is acceptable
        since layer changes are rare (user-initiated).
        """
        composition = self._composer.get_composition()
        self._plot_pane.object = composition.opts(responsive=True, min_height=400)

    def get_layout(self) -> pn.Column:
        controls = pn.Column(
            "## Plot Composition Prototype",
            pn.pane.Markdown(
                """
This prototype validates the layer-based composition model:

1. **Add Dynamic Curve**: Periodic updates via Pipe
2. **Add Peak Markers**: Static VLines overlay
3. **Add BoundsX Selector**: Selection tracking (requires explicit tool)
4. **Add BoxEdit Layer**: Interactive rectangles (auto-adds tool)

**Key insight**: Each layer is a separate `DynamicMap`. Interactive
streams attach to the DynamicMap, not the element, enabling both
programmatic updates and user interaction.
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
    app = create_app()
    app.servable()
