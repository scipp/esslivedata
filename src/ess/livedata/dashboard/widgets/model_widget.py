# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import Any

import panel as pn
import pydantic
from pydantic_core import PydanticUndefined

from .param_widget import ParamWidget
from .styles import Colors


def get_defaults(model: type[pydantic.BaseModel]) -> dict[str, Any]:
    """
    Get default values for all fields in a Pydantic model.

    Parameters
    ----------
    model
        Pydantic model class

    Returns
    -------
    dict[str, Any]
        Dictionary of field names and their default values
    """
    defaults = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.default is not PydanticUndefined:
            defaults[field_name] = field_info.default
        elif callable(field_info.default_factory):
            defaults[field_name] = field_info.default_factory()
    return defaults


class ModelWidget:
    """Builds per-field tab content for configuring a Pydantic model with nested
    model fields."""

    def __init__(
        self,
        model_class: type[pydantic.BaseModel],
        initial_values: dict[str, Any] | None = None,
        show_descriptions: bool = True,
        hidden_fields: frozenset[str] = frozenset(),
    ) -> None:
        """
        Initialize model configuration widget.

        Parameters
        ----------
        model_class
            Pydantic model class where each field is also a Pydantic model
        initial_values
            Initial values to populate the widgets with
        show_descriptions
            Whether to show field descriptions
        hidden_fields
            Field names to exclude from the UI. Hidden fields use their
            model defaults in ``parameter_values``.
        """
        self._model_class = model_class
        self._initial_values = initial_values or {}
        self._show_descriptions = show_descriptions
        self._hidden_fields = hidden_fields
        self._parameter_widgets: dict[str, ParamWidget] = {}
        self._tab_entries: list[tuple[str, str, pn.Column]] = []
        self._failing_field_names: list[str] = []
        try:
            self._build_tabs()
        except Exception as e:
            raise ValueError(
                f"Failed to create ModelWidget for {model_class}: {e}"
            ) from e

    def _build_tabs(self) -> None:
        """Build one (field_name, title, content) entry per parameter group."""
        for field_name, data in self._get_parameter_widget_data().items():
            param_widget = ParamWidget(data['field_type'])
            param_widget.set_values(data['values'])
            self._parameter_widgets[field_name] = param_widget

            content: list[Any] = [param_widget.panel()]
            if self._show_descriptions and data['description']:
                content.insert(
                    0,
                    pn.pane.HTML(
                        "<p style='margin: 0 0 10px 0; "
                        f"color: {Colors.TEXT_MUTED}; font-size: 0.9em;'>"
                        f"{data['description']}</p>",
                        margin=(5, 5),
                    ),
                )
            self._tab_entries.append(
                (
                    field_name,
                    data['title'],
                    pn.Column(
                        *content,
                        sizing_mode='stretch_width',
                        margin=(10, 10),
                    ),
                )
            )

    def _get_parameter_widget_data(self) -> dict[str, dict[str, Any]]:
        """Get parameter widget data for the model."""
        root_defaults = get_defaults(self._model_class)
        widget_data = {}

        for field_name, field_info in self._model_class.model_fields.items():
            if field_name.startswith('_'):
                continue  # Skip private fields
            if field_name in self._hidden_fields:
                continue
            field_type: type[pydantic.BaseModel] = field_info.annotation  # type: ignore[assignment]
            values = get_defaults(field_type)
            values.update(root_defaults.get(field_name, {}))
            values.update(self._initial_values.get(field_name, {}))

            title = field_info.title or field_name.replace('_', ' ').title()
            widget_data[field_name] = {
                'field_type': field_type,
                'values': values,
                'title': title,
                'description': field_info.description,
            }

        return widget_data

    @property
    def param_group_tabs(self) -> list[tuple[str, str, pn.Column]]:
        """Return (field_name, title, content) for each parameter group."""
        return list(self._tab_entries)

    @property
    def parameter_values(self) -> pydantic.BaseModel:
        """Get current parameter values as a model instance."""
        widget_values = {
            name: widget.create_model()
            for name, widget in self._parameter_widgets.items()
        }
        return self._model_class(**widget_values)

    def validate_parameters(self) -> tuple[bool, list[str]]:
        """
        Validate all parameter widgets.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_error_messages)
        """
        errors = []
        self._failing_field_names = []
        for field_name, widget in self._parameter_widgets.items():
            is_valid, error_msg = widget.validate()
            if not is_valid:
                errors.append(f"{field_name}: {error_msg}")
                self._failing_field_names.append(field_name)
                widget.set_error_state(True, error_msg)
            else:
                widget.set_error_state(False, "")
        return len(errors) == 0, errors

    def get_failing_field_names(self) -> list[str]:
        """Return field names that failed the most recent validate_parameters call."""
        return list(self._failing_field_names)

    def clear_validation_errors(self) -> None:
        """Clear all validation error states."""
        self._failing_field_names = []
        for widget in self._parameter_widgets.values():
            widget.set_error_state(False, "")

    def set_values(self, values: dict[str, Any]) -> None:
        """Set values for the parameter widgets."""
        for field_name, field_values in values.items():
            if field_name in self._parameter_widgets:
                self._parameter_widgets[field_name].set_values(field_values)

    def get_parameter_widget(self, field_name: str) -> ParamWidget | None:
        """Get a specific parameter widget by field name."""
        return self._parameter_widgets.get(field_name)
