# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import Enum
from pathlib import Path
from typing import Any

import panel as pn
import pydantic
from pydantic_core import PydanticUndefined


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)


class ParamWidget:
    """Widget for creating and validating Pydantic model instances."""

    def __init__(self, model_class: type[pydantic.BaseModel]):
        self.model_class = model_class
        self.widgets = {}
        self._create_widgets()
        self._error_pane = pn.pane.HTML("", sizing_mode='stretch_width', margin=(0, 5))
        self._create_layout()

    def _create_widgets(self):
        """Create Panel widgets for each field in the model."""
        for field_name, field_info in self.model_class.model_fields.items():
            widget = self._create_widget_for_field(field_name, field_info)
            self.widgets[field_name] = widget

    def _create_layout(self):
        """Create the layout with widgets and error pane."""
        widget_row = pn.Row(*self.widgets.values(), sizing_mode='stretch_width')
        self.layout = pn.Column(
            widget_row, self._error_pane, sizing_mode='stretch_width'
        )

    def _create_widget_for_field(
        self, field_name: str, field_info: pydantic.fields.FieldInfo
    ):
        """Create appropriate widget based on field type."""
        field_type = field_info.annotation
        default_value = (
            field_info.default if field_info.default is not PydanticUndefined else None
        )
        description = field_info.description or field_name
        display_name = snake_to_camel(field_name)

        # Handle Optional types and get the actual type
        origin = getattr(field_type, '__origin__', None)
        if origin is type(None) or (
            hasattr(field_type, '__args__') and type(None) in field_type.__args__
        ):
            # Extract non-None type from Union
            args = getattr(field_type, '__args__', ())
            field_type = next(
                (arg for arg in args if arg is not type(None)), field_type
            )

        shared_options = {
            'name': display_name,
            'description': description,
            'sizing_mode': 'stretch_width',
            'margin': (0, 5),
            'disabled': field_info.frozen or False,
        }

        # Create widget based on type
        if field_type is float:
            return pn.widgets.FloatInput(
                value=default_value or 0.0,
                placeholder=description,
                **shared_options,
            )
        elif field_type is int:
            return pn.widgets.IntInput(
                value=default_value or 0,
                placeholder=description,
                **shared_options,
            )
        elif field_type is bool:
            options = shared_options.copy()
            description = options.pop('description', None)
            # Checkbox does not support description directly, add as tooltip.
            if description is None:
                checkbox = pn.widgets.Checkbox(value=default_value or False, **options)
                # Returning Row is is redundant but makes reading the value consistent
                return pn.layout.Row(checkbox, align='center')
            tooltip = pn.widgets.TooltipIcon(
                value=description,
                margin=options.get('margin'),
                # Pop so tooltip is right next to checkbox, stretch applied to row
                sizing_mode=options.pop('sizing_mode', None),
            )
            checkbox = pn.widgets.Checkbox(value=default_value or False, **options)
            return pn.layout.Row(checkbox, tooltip, align='center')
        elif field_type == Path or field_type is str:
            return pn.widgets.TextInput(
                value=str(default_value) if default_value else "",
                placeholder=description,
                **shared_options,
            )
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            options = {}
            for enum_val in field_type:
                if isinstance(enum_val.value, str):
                    # Use the value for string enums
                    display_key = enum_val.value
                else:
                    # Use string repr without enum class name for other enums
                    display_key = str(enum_val).split('.')[-1]
                options[display_key] = enum_val

            # Set the actual enum instance as the default value
            default_widget_value = (
                default_value if default_value else next(iter(options.values()))
            )
            return pn.widgets.Select(
                options=options, value=default_widget_value, **shared_options
            )
        else:
            # Fallback to text input
            return pn.widgets.TextInput(
                value=str(default_value) if default_value else "",
                placeholder=description,
                **shared_options,
            )

    def get_values(self) -> dict[str, Any]:
        """Get current values from all widgets."""
        values = {}
        for field_name, widget in self.widgets.items():
            field_type = self.model_class.model_fields[field_name].annotation
            if field_type is bool:
                widget = widget[0]
            value = widget.value
            # Convert Path strings back to Path objects
            if field_type == Path and isinstance(value, str):
                value = Path(value) if value else None
            # Handle enum values - widget.value will be the enum instance for Select
            # widgets with enum options
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                # value is already the enum instance from the Select widget
                pass
            values[field_name] = value
        return values

    def set_values(self, values: dict[str, Any]):
        """Set values for widgets."""
        for field_name, value in values.items():
            if field_name in self.widgets:
                field_type = self.model_class.model_fields[field_name].annotation
                if isinstance(value, Path):
                    value = str(value)
                elif field_type is bool:
                    # Checkbox widget is wrapped in a Row
                    self.widgets[field_name][0].value = value
                    continue
                elif isinstance(field_type, type) and issubclass(field_type, Enum):
                    value = field_type(value)
                self.widgets[field_name].value = value

    def create_model(self) -> pydantic.BaseModel:
        """Create and validate a model instance from current widget values."""
        values = self.get_values()
        return self.model_class(**values)

    def validate(self) -> tuple[bool, str]:
        """Validate current values without creating the model."""
        try:
            self.create_model()
            return True, "Valid"
        except pydantic.ValidationError as e:
            # Extract first error message for display
            error_details = e.errors()
            if error_details:
                field_name = error_details[0].get('loc', [''])[0]
                error_msg = error_details[0].get('msg', str(e))
                return False, f"{field_name}: {error_msg}" if field_name else error_msg
            return False, str(e)
        except ValueError as e:
            # Handle ValueError from model validators
            return False, str(e)

    def set_error_state(self, has_error: bool, error_message: str) -> None:
        """Set error state for the widget."""
        if has_error:
            # Clear all widgets first
            for widget in self.widgets.values():
                widget.styles = {'border': 'none'}

            # Try to highlight the specific failing field
            try:
                self.create_model()
            except pydantic.ValidationError as e:
                error_details = e.errors()
                if error_details:
                    field_name = error_details[0].get('loc', [''])[0]
                    if field_name in self.widgets:
                        # Apply error styling to the failing widget
                        self.widgets[field_name].styles = {
                            'border': '2px solid #dc3545',
                            'border-radius': '4px',
                        }

            # Show error message in the error pane
            self._error_pane.object = (
                f"<p style='color: #dc3545; margin: 5px 0; font-size: 0.9em;'>"
                f"{error_message}</p>"
            )
        else:
            # Clear error styling from all widgets
            for widget in self.widgets.values():
                widget.styles = {'border': 'none'}

            # Clear error message
            self._error_pane.object = ""

    def panel(self):
        """Return the Panel layout."""
        return self.layout
