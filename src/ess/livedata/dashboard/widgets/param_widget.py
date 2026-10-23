# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

import panel as pn
import pydantic
from pydantic_core import PydanticUndefined

from ess.livedata.dashboard.static_plots import Color

from .styles import Colors


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)


def _enum_options(enum_type: type[Enum]) -> dict[str, Enum]:
    """Build a Panel ``options`` dict mapping display label → enum member.

    String enums use the value as the label (e.g. an enum with member values
    that are user-readable strings); other enums use the member name —
    ``.name`` is robust to ``__str__`` overrides on plain ``Enum``s and to
    ``IntEnum``/``IntFlag`` inheriting their integer ``__str__``.
    """
    options: dict[str, Enum] = {}
    for enum_val in enum_type:
        if isinstance(enum_val.value, str):
            options[enum_val.value] = enum_val
        else:
            options[enum_val.name] = enum_val
    return options


def _is_set_of_enum(field_type) -> bool:
    """True for ``set[E]`` / ``frozenset[E]`` where ``E`` is an ``Enum``."""
    if get_origin(field_type) not in (set, frozenset):
        return False
    args = get_args(field_type)
    if len(args) != 1:
        return False
    inner = args[0]
    return isinstance(inner, type) and issubclass(inner, Enum)


def _is_group(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, pydantic.BaseModel)


def _is_constant(tp: Any) -> bool:
    """True for a ``Literal`` of one value, such as the tag of a union's alternative.

    It offers no choice, so a group shows no widget for it.
    """
    return get_origin(tp) is Literal and len(get_args(tp)) == 1


def _union_args(tp: Any) -> tuple[Any, ...]:
    return get_args(tp) if get_origin(tp) in (Union, UnionType) else ()


def _field_default(field_info: pydantic.fields.FieldInfo) -> Any:
    """The field's default, ``PydanticUndefined`` for a required field."""
    if field_info.default_factory is not None:
        return field_info.default_factory()  # type: ignore[call-arg]
    return field_info.default


def _tag_value(model: type[pydantic.BaseModel], discriminator: str) -> str:
    (tag,) = get_args(model.model_fields[discriminator].annotation)
    return tag


def _title_row(title: str, description: str | None) -> list[Any]:
    """The title of a nested group, with its description as a tooltip."""
    row: list[Any] = [pn.pane.HTML(f"<b>{title}</b>", margin=(0, 5))]
    if description:
        row.append(pn.widgets.TooltipIcon(value=description, margin=(0, 0)))
    return row


def _card(header: list[Any], *body: Any) -> pn.Card:
    """A box with a header row, for a group or a choice nested in a group."""
    return pn.Card(
        *body,
        header=pn.Row(
            *header,
            align=('start', 'center'),
            sizing_mode='stretch_width',
            margin=(3, 0),
        ),
        collapsible=False,
        header_background=Colors.BG_LIGHT,
        sizing_mode='stretch_width',
        margin=(5, 5),
        # Panel's stretched width ignores the margins, so a card would overflow
        # the box it is nested in.
        stylesheets=[':host { width: calc(100% - 10px) !important; }'],
    )


def make_group_widget(
    annotation: Any,
    discriminator: str | pydantic.Discriminator | None = None,
    hidden_fields: frozenset[str] = frozenset(),
) -> 'ParamWidget | UnionWidget | None':
    """Widget for a group or a discriminated union of groups, None for other types.

    Parameters
    ----------
    annotation:
        The type of the field.
    discriminator:
        The field's discriminator, the name of the tag field, required for a
        union of groups.
    hidden_fields:
        Fields of a group to exclude from the UI.
    """
    if _is_group(annotation):
        return ParamWidget(annotation, hidden_fields=hidden_fields)
    alternatives = _union_args(annotation)
    if not alternatives or not all(_is_group(tp) for tp in alternatives):
        return None
    if not isinstance(discriminator, str):
        raise TypeError(
            f"A union of groups needs a discriminator, the name of its tag field, "
            f"to be shown: {annotation}"
        )
    return UnionWidget(alternatives, discriminator)


class ParamWidget:
    """Widget for creating and validating Pydantic model instances.

    A field that is itself a group, a discriminated union of groups, or a group
    typed ``X | None`` is shown as a nested block. A ``Literal`` of one value is
    not shown, since it offers no choice.
    """

    def __init__(
        self,
        model_class: type[pydantic.BaseModel],
        hidden_fields: frozenset[str] = frozenset(),
    ):
        """
        Parameters
        ----------
        model_class:
            Pydantic model to configure.
        hidden_fields:
            Field names to exclude from the UI. Hidden fields keep their model
            defaults in the created model.
        """
        self.model_class = model_class
        self._hidden_fields = hidden_fields
        self.widgets = {}
        self._nested: dict[str, ParamWidget | UnionWidget | OptionalGroupWidget] = {}
        self._create_widgets()
        self._error_pane = pn.pane.HTML("", sizing_mode='stretch_width', margin=(0, 5))
        self._create_layout()

    def _create_widgets(self):
        """Create Panel widgets for each field in the model."""
        for field_name, field_info in self.model_class.model_fields.items():
            if field_name in self._hidden_fields or _is_constant(field_info.annotation):
                continue
            if (nested := self._create_nested(field_name, field_info)) is not None:
                widget, self.widgets[field_name] = nested
                if (default := _field_default(field_info)) is not PydanticUndefined:
                    widget.set_values(default)
                self._nested[field_name] = widget
                continue
            widget = self._create_widget_for_field(field_name, field_info)
            self.widgets[field_name] = widget

    def _create_nested(
        self, field_name: str, field_info: pydantic.fields.FieldInfo
    ) -> 'tuple[ParamWidget | UnionWidget | OptionalGroupWidget, Any] | None':
        """Widget and layout for a field that holds a group, None for other fields."""
        title = field_info.title or snake_to_camel(field_name)
        annotation = field_info.annotation
        args = _union_args(annotation)
        if NoneType in args:
            others = tuple(tp for tp in args if tp is not NoneType)
            if not any(_is_group(tp) for tp in others):
                return None
            if len(others) > 1:
                raise TypeError(
                    f"Field {field_name!r}: a union of groups typed U | None "
                    "cannot be shown"
                )
            optional = OptionalGroupWidget(
                ParamWidget(others[0]), title, field_info.description
            )
            return optional, optional.panel()
        group = make_group_widget(annotation, field_info.discriminator)
        if group is None:
            return None
        return group, _card(_title_row(title, field_info.description), group.panel())

    def _create_layout(self):
        """Create the layout with widgets and error pane."""
        widget_column = pn.Column(*self.widgets.values(), sizing_mode='stretch_width')
        self.layout = pn.Column(
            widget_column, self._error_pane, sizing_mode='stretch_width'
        )

    def _create_widget_for_field(
        self, field_name: str, field_info: pydantic.fields.FieldInfo
    ):
        """Create appropriate widget based on field type."""
        field_type = field_info.annotation
        default_value = _field_default(field_info)
        if default_value is PydanticUndefined:
            default_value = None
        description = field_info.description or field_name
        display_name = field_info.title or snake_to_camel(field_name)

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
            'label': display_name,
            'description': description,
            'sizing_mode': 'stretch_width',
            'margin': (5, 5),
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
        elif field_type is Color:
            return pn.widgets.ColorPicker(
                value=default_value or '#ff0000',
                **shared_options,
            )
        elif field_type == Path or field_type is str:
            return pn.widgets.TextInput(
                value=str(default_value) if default_value else "",
                placeholder=description,
                **shared_options,
            )
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            options = _enum_options(field_type)
            # Set the actual enum instance as the default value
            default_widget_value = (
                default_value if default_value else next(iter(options.values()))
            )
            return pn.widgets.Select(
                options=options, value=default_widget_value, **shared_options
            )
        elif _is_set_of_enum(field_type):
            (enum_type,) = get_args(field_type)
            options = _enum_options(enum_type)
            default_widget_value = list(default_value) if default_value else []
            return pn.widgets.MultiSelect(
                options=options,
                value=default_widget_value,
                size=min(len(options), 10),
                **shared_options,
            )
        elif get_origin(field_type) is Literal:
            # Handle Literal types (e.g., Literal['a', 'b', 'c']). A field may
            # supply human-readable labels via Field(json_schema_extra={'labels':
            # {value: label}}); the underlying (serialized) value is unchanged.
            literal_values = get_args(field_type)
            extra = field_info.json_schema_extra
            labels = extra.get('labels', {}) if isinstance(extra, dict) else {}
            options = {labels.get(val, str(val)): val for val in literal_values}

            default_widget_value = default_value if default_value else literal_values[0]
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
            if field_name in self._nested:
                values[field_name] = self._nested[field_name].get_values()
                continue
            field_type = self.model_class.model_fields[field_name].annotation
            if field_type is bool:
                widget = widget[0]
            value = widget.value
            # Convert Path strings back to Path objects
            if field_type == Path and isinstance(value, str):
                value = Path(value) if value else None
            elif _is_set_of_enum(field_type):
                # MultiSelect emits a list of enum members; pydantic accepts
                # any iterable for set/frozenset, so just pass the list through.
                pass
            # Handle enum values - widget.value will be the enum instance for Select
            # widgets with enum options
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                # value is already the enum instance from the Select widget
                pass
            values[field_name] = value
        return values

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel):
        """Set values for widgets.

        ``values`` maps field names to values, or is an instance of the model.
        """
        if isinstance(values, pydantic.BaseModel):
            values = dict(values)
        for field_name, value in values.items():
            if field_name in self._nested:
                self._nested[field_name].set_values(value)
            elif field_name in self.widgets:
                field_type = self.model_class.model_fields[field_name].annotation
                if isinstance(value, Path):
                    value = str(value)
                elif field_type is bool:
                    # Checkbox widget is wrapped in a Row
                    self.widgets[field_name][0].value = value
                    continue
                elif _is_set_of_enum(field_type):
                    (enum_type,) = get_args(field_type)
                    value = [enum_type(v) for v in value]
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
                # Model-level validators report an empty ``loc``; only field
                # errors carry a field path to prefix the message with.
                loc = error_details[0].get('loc', ())
                path = '.'.join(str(part) for part in loc)
                error_msg = error_details[0].get('msg', str(e))
                return False, f"{path}: {error_msg}" if path else error_msg
            return False, str(e)
        except ValueError as e:
            # Handle ValueError from model validators
            return False, str(e)

    def set_error_state(self, has_error: bool, error_message: str) -> None:
        """Set error state for the widget."""
        if has_error:
            # Clear all widgets first. An outline, unlike a border, leaves the
            # border of a nested group's card alone.
            for widget in self.widgets.values():
                widget.styles = {'outline': 'none'}

            # Try to highlight the specific failing field
            try:
                self.create_model()
            except pydantic.ValidationError as e:
                error_details = e.errors()
                if error_details:
                    loc = error_details[0].get('loc', ())
                    field_name = loc[0] if loc else ''
                    if field_name in self.widgets:
                        # Apply error styling to the failing widget
                        self.widgets[field_name].styles = {
                            'outline': '2px solid #dc3545',
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
                widget.styles = {'outline': 'none'}

            # Clear error message
            self._error_pane.object = ""

    def panel(self):
        """Return the Panel layout."""
        return self.layout


class UnionWidget:
    """Choice between the alternatives of a discriminated union of groups.

    A box whose header is a row of buttons, one per alternative, labelled by the
    title of its tag field. The box holds the fields of the chosen alternative.
    The fields of the others keep their values while hidden, so switching back
    restores them.
    """

    def __init__(
        self, alternatives: tuple[type[pydantic.BaseModel], ...], discriminator: str
    ) -> None:
        self._discriminator = discriminator
        self._alternatives = {
            _tag_value(model, discriminator): ParamWidget(model)
            for model in alternatives
        }
        tag_fields = {
            tag: model.model_fields[discriminator]
            for tag, model in zip(self._alternatives, alternatives, strict=True)
        }
        self._descriptions = {
            tag: field.description or '' for tag, field in tag_fields.items()
        }
        self._selector = pn.widgets.RadioButtonGroup(
            options={
                field.title or tag.replace('_', ' ').title(): tag
                for tag, field in tag_fields.items()
            },
            color='primary',
            variant='outline',
            sizing_mode='stretch_width',
            margin=(0, 0),
        )
        self._description = pn.pane.HTML(
            "", margin=(5, 5), styles={'color': Colors.TEXT_MUTED}
        )
        self._selector.param.watch(lambda event: self._show(event.new), 'value')
        self.layout = _card(
            [self._selector],
            self._description,
            *(widget.panel() for widget in self._alternatives.values()),
        )
        self._show(self._selector.value)

    def _show(self, tag: str) -> None:
        with pn.io.hold():
            for alternative_tag, widget in self._alternatives.items():
                widget.panel().visible = alternative_tag == tag
            self._description.object = self._descriptions[tag]

    @property
    def _active(self) -> ParamWidget:
        return self._alternatives[self._selector.value]

    def get_values(self) -> dict[str, Any]:
        """Values of the chosen alternative, its tag included."""
        return {self._discriminator: self._selector.value, **self._active.get_values()}

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel) -> None:
        """Choose the alternative that ``values`` names by its tag and set its fields.

        Without a tag, ``values`` are fields of the alternative chosen now.
        """
        if isinstance(values, pydantic.BaseModel):
            values = dict(values)
        values = dict(values)
        if (tag := values.pop(self._discriminator, None)) is not None:
            self._selector.value = tag
        self._active.set_values(values)

    def create_model(self) -> pydantic.BaseModel:
        """Create and validate an instance of the chosen alternative."""
        return self._active.create_model()

    def validate(self) -> tuple[bool, str]:
        """Validate the chosen alternative."""
        return self._active.validate()

    def set_error_state(self, has_error: bool, error_message: str) -> None:
        """Set error state for the chosen alternative."""
        self._active.set_error_state(has_error, error_message)

    def panel(self) -> pn.Card:
        """Return the Panel layout."""
        return self.layout


class OptionalGroupWidget:
    """A group typed ``X | None``: a box with a switch that shows its fields when on.

    Off means ``None``, "without X". The fields keep their values while hidden.
    """

    def __init__(self, group: ParamWidget, title: str, description: str | None):
        self._group = group
        self._toggle = pn.widgets.Switch(value=False, margin=(0, 5), width=40)
        self.layout = _card(
            [self._toggle, *_title_row(title, description)], group.panel()
        )
        self.layout.collapsed = True
        self._toggle.param.watch(
            lambda event: setattr(self.layout, 'collapsed', not event.new), 'value'
        )

    def get_values(self) -> dict[str, Any] | None:
        """Values of the group, None when the switch is off."""
        return self._group.get_values() if self._toggle.value else None

    def set_values(self, values: dict[str, Any] | pydantic.BaseModel | None) -> None:
        """Turn the switch off for None, else on with the group set to ``values``."""
        self._toggle.value = values is not None
        if values is not None:
            self._group.set_values(values)

    def panel(self) -> pn.Card:
        """Return the Panel layout."""
        return self.layout
