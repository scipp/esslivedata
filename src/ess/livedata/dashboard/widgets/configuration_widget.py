# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import panel as pn
import pydantic
import structlog

from ess.livedata.config.workflow_spec import AuxSources
from ess.livedata.dashboard.configuration_adapter import ConfigurationAdapter

from .model_widget import ModelWidget
from .styles import Colors, ErrorBox, ModalSizing, StatusColors

_VTABS_STYLESHEET = f"""
.bk-tab {{
    border-right: 1px solid {Colors.TAB_BORDER} !important;
    text-align: left !important;
}}
.bk-tab.bk-active {{
    background-color: {Colors.TAB_ACTIVE_BG} !important;
    border: 1px solid {Colors.TAB_BORDER} !important;
    border-right: none !important;
}}
"""


class ConfigurationWidget:
    """Generic widget for configuring parameters and source selection."""

    def __init__(self, config: ConfigurationAdapter) -> None:
        """
        Initialize generic configuration widget.

        Parameters
        ----------
        config
            Configuration adapter providing data and callbacks
        """
        self._config = config
        self._source_selector = self._create_source_selector()
        self._aux_sources_widget = self._create_aux_sources_widget()
        self._source_error_pane = pn.pane.HTML("", sizing_mode='stretch_width')
        self._model_widget = self._create_model_widget()
        self._tabs: pn.Tabs | None = None
        self._tab_field_order: list[str | None] = []
        self._title_pane = self._create_title_pane()
        self._widget = pn.Column(self._build_body(), sizing_mode='stretch_both')

    def _create_source_selector(
        self,
    ) -> pn.widgets.MultiChoice | pn.widgets.Select | None:
        """Create source selection widget, or None if no sources available.

        Returns a single-choice ``Select`` when the adapter declares
        ``single_source=True``; otherwise the default multi-choice widget.
        """
        # No source selector needed when there are no sources (e.g., static overlays)
        if not self._config.source_names:
            return None

        if (
            not self._config.initial_source_names
            and len(self._config.source_names) == 1
        ):
            initial_source_names = self._config.source_names
        else:
            initial_source_names = self._config.initial_source_names

        # Build options dict: {display_title: internal_name}
        # Sort by display title for consistent ordering
        options = {
            self._config.get_source_title(name): name
            for name in self._config.source_names
        }
        sorted_options = dict(sorted(options.items()))

        if self._config.single_source:
            initial_value = (
                initial_source_names[0]
                if initial_source_names
                else next(iter(sorted_options.values()))
            )
            return pn.widgets.Select(
                name="Source Name",
                options=sorted_options,
                value=initial_value,
                sizing_mode='stretch_width',
            )

        return pn.widgets.MultiChoice(
            name="Source Names",
            options=sorted_options,
            value=sorted(initial_source_names),
            placeholder="Select source names to apply workflow to",
            sizing_mode='stretch_width',
        )

    def _create_aux_sources_widget(self) -> AuxSourcesWidget | None:
        """Create auxiliary sources widget."""
        aux_sources = self._config.aux_sources
        if aux_sources is None:
            return None

        initial_values = self._config.initial_aux_source_names

        widget = AuxSourcesWidget(
            aux_sources,
            initial_values=initial_values,
            get_source_title=self._config.get_source_title,
        )

        # Watch for changes to trigger model_widget recreation
        for select in widget.select_widgets.values():
            select.param.watch(self._on_aux_source_changed, 'value')

        return widget

    def _create_model_widget(self) -> ModelWidget | NoParamsWidget | ErrorWidget:
        """Create model widget based on current aux source selections."""
        if self._aux_sources_widget is not None:
            aux_selections = self._aux_sources_widget.get_values()
        else:
            aux_selections = None

        try:
            model_class = self._config.set_aux_sources(aux_selections)
        except Exception as e:
            return ErrorWidget(str(e))

        if model_class is None or not model_class.model_fields:
            return NoParamsWidget()
        return ModelWidget(
            model_class=model_class,
            initial_values=self._config.initial_parameter_values,
            show_descriptions=True,
            hidden_fields=self._config.hidden_fields,
        )

    def _create_title_pane(self) -> pn.pane.HTML:
        """Build the title/description pane.

        Title and description come from source-controlled specs, not user input,
        so HTML in descriptions (e.g. ``<br>``, ``<ul>``) is rendered as markup.
        """
        return pn.pane.HTML(
            f"<h2 style='margin:0 0 4px 0;'>{self._config.title}</h2>"
            "<p style='margin:0; "
            f"color:{Colors.TEXT_MUTED};'>{self._config.description}</p>",
            sizing_mode='stretch_width',
        )

    def _general_tab_items(self) -> list[Any]:
        """Collect items that belong in the 'General' tab (sources/aux/errors)."""
        items: list[Any] = []
        if self._source_selector is not None:
            items.append(self._source_selector)
            items.append(self._source_error_pane)
        if self._aux_sources_widget is not None:
            items.append(self._aux_sources_widget.panel)
        if isinstance(self._model_widget, ErrorWidget):
            items.append(self._model_widget.widget)
        return items

    def _build_body(self) -> pn.Tabs | pn.Column | pn.pane.HTML:
        """Build the body: vertical tabs when multiple sections exist, else the
        model widget directly."""
        general_items = self._general_tab_items()
        param_tabs = self._model_widget.param_group_tabs
        if not general_items and not param_tabs:
            # Nothing tabbable (typically NoParamsWidget with no sources).
            self._tabs = None
            self._tab_field_order = []
            return self._model_widget.widget

        tab_entries: list[tuple[str, pn.Column]] = []
        field_order: list[str | None] = []
        if general_items:
            tab_entries.append(
                (
                    'General',
                    pn.Column(
                        *general_items, sizing_mode='stretch_width', margin=(10, 10)
                    ),
                )
            )
            field_order.append(None)
        for field_name, title, content in param_tabs:
            tab_entries.append((title, content))
            field_order.append(field_name)

        self._tabs = pn.Tabs(
            *tab_entries,
            tabs_location='left',
            dynamic=True,
            sizing_mode='stretch_width',
            min_height=350,
            stylesheets=[_VTABS_STYLESHEET],
        )
        self._tab_field_order = field_order
        return self._tabs

    def _on_aux_source_changed(self, event) -> None:
        """Handle auxiliary source selection change."""
        self._model_widget = self._create_model_widget()
        with pn.io.hold():
            self._widget.objects = [self._build_body()]

    @property
    def widget(self) -> pn.Column:
        """Configuration body (tabs or fallback widget); does not include title."""
        return self._widget

    @property
    def title_pane(self) -> pn.pane.HTML:
        """Title/description pane.

        Separate so callers can pin it above a scroll area.
        """
        return self._title_pane

    @property
    def selected_sources(self) -> list[str]:
        """Get the selected source names."""
        if self._source_selector is None:
            return []
        if isinstance(self._source_selector, pn.widgets.Select):
            return [self._source_selector.value] if self._source_selector.value else []
        return self._source_selector.value

    @property
    def parameter_values(self):
        """Get current parameter values as a model instance."""
        return self._model_widget.parameter_values

    def validate_configuration(self) -> tuple[bool, list[str]]:
        """
        Validate that required fields are configured.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_error_messages)
        """
        errors = []

        # Validate source selection (only if sources are available)
        if self._source_selector is not None:
            if len(self.selected_sources) == 0:
                errors.append("Please select at least one source name.")
                self._highlight_source_error(True)
            else:
                self._highlight_source_error(False)

        # Validate parameter widgets
        param_valid, param_errors = self._model_widget.validate_parameters()
        if not param_valid:
            errors.extend(param_errors)

        return len(errors) == 0, errors

    def _highlight_source_error(self, has_error: bool) -> None:
        """Highlight source selector with error state."""
        if self._source_selector is None:
            return

        if has_error:
            self._source_selector.styles = {
                'border': f'2px solid {StatusColors.ERROR}',
                'border-radius': '4px',
            }
            self._source_error_pane.object = (
                f"<p style='color: {StatusColors.ERROR}; "
                f"margin: 5px 0; font-size: 0.9em;'>"
                "Please select at least one source name.</p>"
            )
        else:
            self._source_selector.styles = {'border': 'none'}
            self._source_error_pane.object = ""

    def clear_validation_errors(self) -> None:
        """Clear all validation error states."""
        self._highlight_source_error(False)
        self._model_widget.clear_validation_errors()

    def activate_first_error_tab(self) -> None:
        """Switch to the first tab containing a validation error.

        Source errors and whole-model errors resolve to the 'General' tab.
        Per-field errors resolve to the tab owning that field. No-op when the
        body falls back to a non-tabbed widget.
        """
        if self._tabs is None:
            return
        general_index = (
            self._tab_field_order.index(None) if None in self._tab_field_order else -1
        )
        if (
            self._source_selector is not None
            and len(self.selected_sources) == 0
            and general_index >= 0
        ):
            self._tabs.active = general_index
            return
        if isinstance(self._model_widget, ErrorWidget) and general_index >= 0:
            self._tabs.active = general_index
            return
        failing = self._model_widget.get_failing_field_names()
        if not failing:
            return
        first = failing[0]
        for i, field_name in enumerate(self._tab_field_order):
            if field_name == first:
                self._tabs.active = i
                return


class ConfigurationPanel:
    """Reusable configuration panel with validation and action execution."""

    def __init__(
        self,
        config: ConfigurationAdapter,
    ) -> None:
        """
        Initialize configuration panel.

        Parameters
        ----------
        config
            Configuration adapter providing data and callbacks
        """
        self._config = config
        self._config_widget = ConfigurationWidget(config)
        self._error_pane = pn.pane.HTML("", sizing_mode='stretch_width')
        self._logger = structlog.get_logger()
        self._body = pn.Column(
            self._config_widget.widget,
            self._error_pane,
            sizing_mode='stretch_both',
        )
        self._combined_panel: pn.Column | None = None

    def split_for_sticky_header(self) -> tuple[pn.pane.HTML, pn.Column]:
        """Return (title pane, body) separately for modal layouts that pin the
        title above an independently scrolling body."""
        return self._config_widget.title_pane, self._body

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration and show errors inline.

        Returns
        -------
        :
            Tuple of (is_valid, list_of_error_messages)
        """
        self._config_widget.clear_validation_errors()
        self._error_pane.object = ""

        is_valid, errors = self._config_widget.validate_configuration()

        if not is_valid:
            self._show_validation_errors(errors)
            self._config_widget.activate_first_error_tab()

        return is_valid, errors

    def execute_action(self) -> bool:
        """
        Execute the configuration action.

        Assumes validation has already passed. If validation is needed,
        use validate() first or use validate_and_execute().

        Returns
        -------
        :
            True if action succeeded, False if action raised error
        """
        try:
            self._config.start_action(
                self._config_widget.selected_sources,
                self._config_widget.parameter_values,
            )
        except Exception as e:
            self._logger.exception("Error starting '%s'", self._config.title)
            error_message = f"Error starting '{self._config.title}': {e!s}"
            self._show_action_error(error_message)
            return False

        return True

    def validate_and_execute(self) -> bool:
        """
        Convenience method: validate then execute if valid.

        Returns
        -------
        :
            True if both validation and execution succeeded, False otherwise
        """
        is_valid, _ = self.validate()
        if not is_valid:
            return False
        return self.execute_action()

    def _show_validation_errors(self, errors: list[str]) -> None:
        """Show validation errors inline."""
        error_html = (
            f"<div style='background-color: {ErrorBox.BG}; "
            f"border: 1px solid {ErrorBox.BORDER}; "
            f"border-radius: 4px; padding: 10px; "
            f"margin: 10px 0;'>"
            f"<h6 style='color: {ErrorBox.TEXT}; "
            f"margin: 0 0 10px 0;'>"
            f"Please fix the following errors:</h6>"
            f"<ul style='color: {ErrorBox.TEXT}; "
            f"margin: 0; padding-left: 20px;'>"
        )
        for error in errors:
            error_html += f"<li>{error}</li>"
        error_html += "</ul></div>"

        self._error_pane.object = error_html

    def _show_action_error(self, message: str) -> None:
        """Show action error inline."""
        error_html = (
            f"<div style='background-color: {ErrorBox.BG}; "
            f"border: 1px solid {ErrorBox.BORDER}; "
            f"border-radius: 4px; padding: 10px; "
            f"margin: 10px 0;'>"
            f"<p style='color: {ErrorBox.TEXT}; "
            f"margin: 0;'>{message}</p>"
            "</div>"
        )
        self._error_pane.object = error_html

    @property
    def panel(self) -> pn.Column:
        """Combined panel: title + body (for inline/non-modal placement).

        Mutually exclusive with :meth:`split_for_sticky_header`; do not use both
        for the same ``ConfigurationPanel`` instance.
        """
        if self._combined_panel is None:
            self._combined_panel = pn.Column(
                self._config_widget.title_pane,
                self._body,
                sizing_mode='stretch_both',
            )
        return self._combined_panel


class ConfigurationModal:
    """Modal wrapper around ConfigurationPanel with action buttons."""

    def __init__(
        self,
        config: ConfigurationAdapter,
        start_button_text: str = "Start",
        success_callback: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize configuration modal.

        Parameters
        ----------
        config
            Configuration adapter providing data and callbacks
        start_button_text
            Text for the start button
        success_callback
            Called when action completes successfully
        """
        self._config = config
        self._success_callback = success_callback

        # Create panel
        self._panel = ConfigurationPanel(config=config)

        # Create action buttons
        self._start_button = pn.widgets.Button(
            name=start_button_text, button_type="primary"
        )
        self._start_button.on_click(self._on_start_clicked)

        self._cancel_button = pn.widgets.Button(name="Cancel", button_type="light")
        self._cancel_button.on_click(self._on_cancel_clicked)

        # Create modal with panel + buttons
        self._modal = self._create_modal()

    def _create_modal(self) -> pn.Modal:
        """Create the modal dialog with sticky header + footer."""
        footer = pn.Row(
            pn.Spacer(),
            self._cancel_button,
            self._start_button,
            margin=(10, 0),
            sizing_mode='stretch_width',
        )
        title, body = self._panel.split_for_sticky_header()
        scroll_body = pn.Column(
            body,
            sizing_mode='stretch_width',
            max_height=ModalSizing.SCROLL_BODY_MAX_HEIGHT,
            scroll=True,
        )
        content = pn.Column(
            title,
            scroll_body,
            footer,
            sizing_mode='stretch_width',
        )
        modal = pn.Modal(
            content,
            name=f"Configure {self._config.title}",
            margin=20,
            width=ModalSizing.WIDTH,
        )

        modal.param.watch(self._on_modal_closed, 'open')
        return modal

    def _on_start_clicked(self, event) -> None:
        """Handle start button click."""
        with pn.io.hold():
            if self._panel.validate_and_execute():
                self._modal.open = False
                if self._success_callback:
                    self._success_callback()

    def _on_cancel_clicked(self, event) -> None:
        """Handle cancel button click."""
        self._modal.open = False

    def _on_modal_closed(self, event) -> None:
        """Handle modal being closed (cleanup)."""
        if not event.new:  # Modal was closed
            # Remove modal from its parent container after a short delay
            # to allow the close animation to complete
            def cleanup():
                try:
                    if hasattr(self._modal, '_parent') and self._modal._parent:
                        self._modal._parent.remove(self._modal)
                except Exception:  # noqa: S110
                    pass  # Ignore cleanup errors

            pn.state.add_periodic_callback(cleanup, period=100, count=1)

    def show(self) -> None:
        """Show the modal dialog."""
        self._modal.open = True

    @property
    def modal(self) -> pn.Modal:
        """Get the modal widget."""
        return self._modal


class AuxSourcesWidget:
    """Widget for selecting auxiliary data sources from an AuxSources spec."""

    def __init__(
        self,
        aux_sources: AuxSources,
        initial_values: dict[str, str] | None = None,
        get_source_title: Callable[[str], str] | None = None,
    ) -> None:
        self._aux_sources = aux_sources
        self._get_source_title = get_source_title or (lambda x: x)
        self._select_widgets: dict[str, pn.widgets.Select] = {}

        for name, inp in aux_sources.inputs.items():
            # Build options: {display_title: stream_name}
            options = {self._get_source_title(c): c for c in inp.choices}
            initial = (
                initial_values.get(name, inp.default) if initial_values else inp.default
            )
            self._select_widgets[name] = pn.widgets.Select(
                name=inp.title or name,
                options=options,
                value=initial,
                sizing_mode='stretch_width',
            )

        self._panel = pn.Column(
            *self._select_widgets.values(), sizing_mode='stretch_width'
        )

    @property
    def select_widgets(self) -> dict[str, pn.widgets.Select]:
        """Exposed for watching changes."""
        return self._select_widgets

    @property
    def panel(self) -> pn.Column:
        """Panel widget for display."""
        return self._panel

    def get_values(self) -> dict[str, str]:
        """Return current selections as {input_name: stream_name}."""
        return {name: w.value for name, w in self._select_widgets.items()}


class NoParamsWidget:
    class EmptyModel(pydantic.BaseModel): ...

    def __init__(self):
        self.widget = pn.pane.HTML(
            "<div style='padding: 20px; text-align: center; color: #666; "
            "font-style: italic; border: 1px solid #ddd; border-radius: 4px; "
            "background-color: #f9f9f9;'>"
            "There are no parameters to configure."
            "</div>",
            sizing_mode='stretch_width',
        )

    @property
    def param_group_tabs(self) -> list[tuple[str, str, pn.Column]]:
        """No parameter groups when the model has no fields."""
        return []

    @property
    def parameter_values(self) -> pydantic.BaseModel:
        """Return empty model serializing to empty dict."""
        return self.EmptyModel()

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """Always valid when no parameters."""
        return True, []

    def get_failing_field_names(self) -> list[str]:
        """No fields to fail."""
        return []

    def clear_validation_errors(self) -> None:
        """No-op for no parameters."""


class ErrorWidget:
    """Widget to display error messages."""

    def __init__(self, error_message: str):
        self.widget = pn.pane.HTML(
            f"<div style='padding: 20px; text-align: center; "
            f"color: {ErrorBox.TEXT}; font-weight: bold; "
            f"border: 2px solid {ErrorBox.BORDER}; "
            f"border-radius: 4px; "
            f"background-color: {ErrorBox.BG};'>"
            f"Error: {error_message}"
            f"</div>",
            sizing_mode='stretch_width',
            margin=0,
        )

    @property
    def param_group_tabs(self) -> list[tuple[str, str, pn.Column]]:
        """No tabs when the model failed to build."""
        return []

    @property
    def parameter_values(self) -> None:
        """Return None when in error state."""
        return None

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """Always invalid when in error state."""
        return False, [
            "Configuration error - please check auxiliary source selections."
        ]

    def get_failing_field_names(self) -> list[str]:
        """The error is a whole-model error, not a per-field error."""
        return []

    def clear_validation_errors(self) -> None:
        """No-op for error widget."""
