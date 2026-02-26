# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import Enum, StrEnum
from pathlib import Path
from typing import Literal

import panel as pn
import pydantic
import pytest

from ess.livedata.dashboard.widgets.param_widget import ParamWidget, snake_to_camel


class TestSnakeToCamel:
    """Tests for snake_to_camel utility function."""

    def test_single_word(self):
        assert snake_to_camel("word") == "Word"

    def test_two_words(self):
        assert snake_to_camel("snake_case") == "SnakeCase"

    def test_multiple_words(self):
        assert snake_to_camel("multiple_word_example") == "MultipleWordExample"

    def test_with_numbers(self):
        assert snake_to_camel("test_123_value") == "Test123Value"

    def test_already_camel(self):
        assert snake_to_camel("CamelCase") == "Camelcase"


class TestParamWidgetCreation:
    """Tests for ParamWidget creation and initialization."""

    def test_create_widget_with_basic_types(self):
        class TestModel(pydantic.BaseModel):
            int_field: int = 5
            float_field: float = 3.14
            str_field: str = "test"
            bool_field: bool = True

        widget = ParamWidget(TestModel)

        assert len(widget.widgets) == 4
        assert "int_field" in widget.widgets
        assert "float_field" in widget.widgets
        assert "str_field" in widget.widgets
        assert "bool_field" in widget.widgets

    def test_widget_has_layout(self):
        class TestModel(pydantic.BaseModel):
            field: int = 1

        widget = ParamWidget(TestModel)

        assert hasattr(widget, "layout")
        assert isinstance(widget.layout, pn.Column)

    def test_widget_has_error_pane(self):
        class TestModel(pydantic.BaseModel):
            field: int = 1

        widget = ParamWidget(TestModel)

        assert hasattr(widget, "_error_pane")
        assert isinstance(widget._error_pane, pn.pane.HTML)


class TestWidgetTypeCreation:
    """Tests for creating appropriate widget types for different field types."""

    def test_int_field_creates_int_input(self):
        class TestModel(pydantic.BaseModel):
            value: int = 5

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["value"], pn.widgets.IntInput)
        assert widget.widgets["value"].value == 5

    def test_float_field_creates_float_input(self):
        class TestModel(pydantic.BaseModel):
            value: float = 3.14

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["value"], pn.widgets.FloatInput)
        assert widget.widgets["value"].value == 3.14

    def test_str_field_creates_text_input(self):
        class TestModel(pydantic.BaseModel):
            value: str = "test"

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["value"], pn.widgets.TextInput)
        assert widget.widgets["value"].value == "test"

    def test_bool_field_creates_checkbox_in_row(self):
        class TestModel(pydantic.BaseModel):
            value: bool = True

        widget = ParamWidget(TestModel)

        # Boolean widgets are wrapped in a Row
        assert isinstance(widget.widgets["value"], pn.Row)
        checkbox = widget.widgets["value"][0]
        assert isinstance(checkbox, pn.widgets.Checkbox)
        assert checkbox.value is True

    def test_path_field_creates_text_input(self):
        class TestModel(pydantic.BaseModel):
            value: Path = Path("/tmp/test")  # noqa: S108

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["value"], pn.widgets.TextInput)
        assert widget.widgets["value"].value == "/tmp/test"  # noqa: S108

    def test_enum_field_creates_select(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"
            GREEN = "green"

        class TestModel(pydantic.BaseModel):
            color: Color = Color.RED

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["color"], pn.widgets.Select)
        assert widget.widgets["color"].value == Color.RED

    def test_literal_field_creates_select(self):
        class TestModel(pydantic.BaseModel):
            choice: Literal["a", "b", "c"] = "a"

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["choice"], pn.widgets.Select)
        assert widget.widgets["choice"].value == "a"

    def test_optional_field_extracts_type(self):
        class TestModel(pydantic.BaseModel):
            value: int | None = None

        widget = ParamWidget(TestModel)

        # Should still create an IntInput, not a TextInput fallback
        assert isinstance(widget.widgets["value"], pn.widgets.IntInput)

    def test_color_field_creates_color_picker(self):
        from ess.livedata.dashboard.static_plots import Color

        class TestModel(pydantic.BaseModel):
            color: Color = Color('#ff0000')

        widget = ParamWidget(TestModel)

        assert isinstance(widget.widgets["color"], pn.widgets.ColorPicker)
        assert widget.widgets["color"].value == '#ff0000'


class TestWidgetDefaultValues:
    """Tests for widget default values."""

    def test_uses_pydantic_default_for_int(self):
        class TestModel(pydantic.BaseModel):
            value: int = 42

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].value == 42

    def test_uses_zero_when_no_default_for_int(self):
        class TestModel(pydantic.BaseModel):
            value: int

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].value == 0

    def test_uses_zero_for_float_when_no_default(self):
        class TestModel(pydantic.BaseModel):
            value: float

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].value == 0.0

    def test_uses_empty_string_when_no_default_for_str(self):
        class TestModel(pydantic.BaseModel):
            value: str

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].value == ""

    def test_uses_false_when_no_default_for_bool(self):
        class TestModel(pydantic.BaseModel):
            value: bool

        widget = ParamWidget(TestModel)

        checkbox = widget.widgets["value"][0]
        assert checkbox.value is False


class TestWidgetLabelsAndDescriptions:
    """Tests for widget labels and descriptions."""

    def test_field_name_converted_to_camel_case(self):
        class TestModel(pydantic.BaseModel):
            my_field_name: int = 1

        widget = ParamWidget(TestModel)

        assert widget.widgets["my_field_name"].name == "MyFieldName"

    def test_field_description_used(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(default=1, description="Test description")

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].description == "Test description"

    def test_field_name_used_when_no_description(self):
        class TestModel(pydantic.BaseModel):
            my_value: int = 1

        widget = ParamWidget(TestModel)

        assert widget.widgets["my_value"].description == "my_value"

    def test_bool_field_with_description_creates_tooltip(self):
        class TestModel(pydantic.BaseModel):
            flag: bool = pydantic.Field(default=True, description="A flag")

        widget = ParamWidget(TestModel)

        row = widget.widgets["flag"]
        assert isinstance(row, pn.Row)
        # Should have checkbox and tooltip
        assert len(row) == 2
        assert isinstance(row[0], pn.widgets.Checkbox)
        assert isinstance(row[1], pn.widgets.TooltipIcon)


class TestWidgetFrozenFields:
    """Tests for frozen/disabled fields."""

    def test_frozen_field_is_disabled(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(default=5, frozen=True)

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].disabled is True

    def test_non_frozen_field_is_not_disabled(self):
        class TestModel(pydantic.BaseModel):
            value: int = 5

        widget = ParamWidget(TestModel)

        assert widget.widgets["value"].disabled is False


class TestEnumHandling:
    """Tests for enum field handling."""

    def test_string_enum_uses_values_as_display_keys(self):
        class Status(StrEnum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class TestModel(pydantic.BaseModel):
            status: Status = Status.ACTIVE

        widget = ParamWidget(TestModel)

        select = widget.widgets["status"]
        # For string enums, the display keys should be the string values
        assert "active" in select.options
        assert "inactive" in select.options

    def test_int_enum_uses_member_names_as_display_keys(self):
        class Priority(Enum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        class TestModel(pydantic.BaseModel):
            priority: Priority = Priority.LOW

        widget = ParamWidget(TestModel)

        select = widget.widgets["priority"]
        # For non-string enums, display keys are member names
        assert "LOW" in select.options
        assert "MEDIUM" in select.options
        assert "HIGH" in select.options


class TestLiteralHandling:
    """Tests for Literal type handling."""

    def test_literal_creates_select_with_options(self):
        class TestModel(pydantic.BaseModel):
            mode: Literal["fast", "normal", "slow"] = "normal"

        widget = ParamWidget(TestModel)

        select = widget.widgets["mode"]
        assert isinstance(select, pn.widgets.Select)
        assert "fast" in select.options
        assert "normal" in select.options
        assert "slow" in select.options
        assert select.value == "normal"


class TestGetValues:
    """Tests for get_values method."""

    def test_get_values_returns_dict(self):
        class TestModel(pydantic.BaseModel):
            a: int = 1
            b: float = 2.0

        widget = ParamWidget(TestModel)
        values = widget.get_values()

        assert isinstance(values, dict)
        assert "a" in values
        assert "b" in values

    def test_get_values_returns_current_widget_values(self):
        class TestModel(pydantic.BaseModel):
            value: int = 5

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = 42

        values = widget.get_values()
        assert values["value"] == 42

    def test_get_values_handles_bool_wrapped_in_row(self):
        class TestModel(pydantic.BaseModel):
            flag: bool = True

        widget = ParamWidget(TestModel)
        values = widget.get_values()

        assert values["flag"] is True

    def test_get_values_converts_path_string_to_path(self):
        class TestModel(pydantic.BaseModel):
            path: Path = Path("/tmp")  # noqa: S108

        widget = ParamWidget(TestModel)
        widget.widgets["path"].value = "/home/user"

        values = widget.get_values()
        assert isinstance(values["path"], Path)
        assert values["path"] == Path("/home/user")

    def test_get_values_handles_empty_path_string(self):
        class TestModel(pydantic.BaseModel):
            path: Path

        widget = ParamWidget(TestModel)
        widget.widgets["path"].value = ""

        values = widget.get_values()
        assert values["path"] is None

    def test_get_values_returns_enum_instance(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        class TestModel(pydantic.BaseModel):
            color: Color = Color.RED

        widget = ParamWidget(TestModel)
        widget.widgets["color"].value = Color.BLUE

        values = widget.get_values()
        assert values["color"] == Color.BLUE
        assert isinstance(values["color"], Color)


class TestSetValues:
    """Tests for set_values method."""

    def test_set_values_updates_widgets(self):
        class TestModel(pydantic.BaseModel):
            a: int = 1
            b: float = 2.0
            c: str = "test"

        widget = ParamWidget(TestModel)
        widget.set_values({"a": 42, "b": 3.14, "c": "updated"})

        assert widget.widgets["a"].value == 42
        assert widget.widgets["b"].value == 3.14
        assert widget.widgets["c"].value == "updated"

    def test_set_values_handles_bool_wrapped_in_row(self):
        class TestModel(pydantic.BaseModel):
            flag: bool = False

        widget = ParamWidget(TestModel)
        widget.set_values({"flag": True})

        checkbox = widget.widgets["flag"][0]
        assert checkbox.value is True

    def test_set_values_converts_path_to_string(self):
        class TestModel(pydantic.BaseModel):
            path: Path = Path("/tmp")  # noqa: S108

        widget = ParamWidget(TestModel)
        widget.set_values({"path": Path("/home/user")})

        assert widget.widgets["path"].value == "/home/user"

    def test_set_values_handles_enum_values(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        class TestModel(pydantic.BaseModel):
            color: Color = Color.RED

        widget = ParamWidget(TestModel)
        widget.set_values({"color": "blue"})

        assert widget.widgets["color"].value == Color.BLUE

    def test_set_values_ignores_unknown_fields(self):
        class TestModel(pydantic.BaseModel):
            value: int = 1

        widget = ParamWidget(TestModel)
        # Should not raise error for unknown field
        widget.set_values({"value": 42, "unknown_field": 99})

        assert widget.widgets["value"].value == 42


class TestCreateModel:
    """Tests for create_model method."""

    def test_create_model_returns_valid_model(self):
        class TestModel(pydantic.BaseModel):
            a: int = 1
            b: float = 2.0

        widget = ParamWidget(TestModel)
        model = widget.create_model()

        assert isinstance(model, TestModel)
        assert model.a == 1
        assert model.b == 2.0

    def test_create_model_uses_current_widget_values(self):
        class TestModel(pydantic.BaseModel):
            value: int = 5

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = 42

        model = widget.create_model()
        assert model.value == 42

    def test_create_model_raises_validation_error_for_invalid_values(self):
        class TestModel(pydantic.BaseModel):
            positive: int = pydantic.Field(gt=0)

        widget = ParamWidget(TestModel)
        widget.widgets["positive"].value = -5

        with pytest.raises(pydantic.ValidationError):
            widget.create_model()


class TestValidate:
    """Tests for validate method."""

    def test_validate_returns_true_for_valid_values(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(gt=0)

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = 5

        is_valid, message = widget.validate()
        assert is_valid is True
        assert message == "Valid"

    def test_validate_returns_false_for_invalid_values(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(gt=0)

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = -5

        is_valid, message = widget.validate()
        assert is_valid is False
        assert "value" in message

    def test_validate_does_not_create_model(self):
        class TestModel(pydantic.BaseModel):
            value: int = 5

        widget = ParamWidget(TestModel)
        original_value = widget.widgets["value"].value

        widget.validate()
        # Widget value should remain unchanged
        assert widget.widgets["value"].value == original_value


class TestSetErrorState:
    """Tests for set_error_state method."""

    def test_set_error_state_shows_error_message(self):
        class TestModel(pydantic.BaseModel):
            value: int = 1

        widget = ParamWidget(TestModel)
        widget.set_error_state(True, "Test error")

        assert "Test error" in widget._error_pane.object

    def test_set_error_state_clears_error_message(self):
        class TestModel(pydantic.BaseModel):
            value: int = 1

        widget = ParamWidget(TestModel)
        widget.set_error_state(True, "Test error")
        widget.set_error_state(False, "")

        assert widget._error_pane.object == ""

    def test_set_error_state_highlights_failing_field(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(gt=0)

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = -5
        widget.set_error_state(True, "value: must be greater than 0")

        # Field should have error border
        assert "border" in widget.widgets["value"].styles
        assert "#dc3545" in widget.widgets["value"].styles["border"]

    def test_set_error_state_clears_field_highlighting(self):
        class TestModel(pydantic.BaseModel):
            value: int = pydantic.Field(gt=0)

        widget = ParamWidget(TestModel)
        widget.widgets["value"].value = -5
        widget.set_error_state(True, "Test error")
        widget.widgets["value"].value = 5
        widget.set_error_state(False, "")

        # Field should have no border
        assert widget.widgets["value"].styles.get("border") == "none"


class TestPanel:
    """Tests for panel method."""

    def test_panel_returns_layout(self):
        class TestModel(pydantic.BaseModel):
            value: int = 1

        widget = ParamWidget(TestModel)
        panel = widget.panel()

        assert panel is widget.layout
        assert isinstance(panel, pn.Column)


class TestComplexModels:
    """Tests for complex model scenarios."""

    def test_multiple_field_types(self):
        class Color(StrEnum):
            RED = "red"
            BLUE = "blue"

        class ComplexModel(pydantic.BaseModel):
            name: str = "test"
            count: int = 10
            ratio: float = 0.5
            enabled: bool = True
            color: Color = Color.RED
            path: Path = Path("/tmp")  # noqa: S108
            mode: Literal["fast", "slow"] = "fast"

        widget = ParamWidget(ComplexModel)

        assert len(widget.widgets) == 7
        model = widget.create_model()
        assert isinstance(model, ComplexModel)

    def test_model_with_validation_constraints(self):
        class ConstrainedModel(pydantic.BaseModel):
            positive_int: int = pydantic.Field(gt=0, description="Must be positive")
            bounded_float: float = pydantic.Field(ge=0.0, le=1.0)
            non_empty_str: str = pydantic.Field(min_length=1)

        widget = ParamWidget(ConstrainedModel)
        widget.widgets["positive_int"].value = 5
        widget.widgets["bounded_float"].value = 0.75
        widget.widgets["non_empty_str"].value = "test"

        model = widget.create_model()
        assert model.positive_int == 5
        assert model.bounded_float == 0.75
        assert model.non_empty_str == "test"

    def test_model_with_optional_fields(self):
        class OptionalModel(pydantic.BaseModel):
            required: int
            optional_int: int | None = None
            optional_str: str | None = None

        widget = ParamWidget(OptionalModel)
        widget.widgets["required"].value = 42

        model = widget.create_model()
        assert model.required == 42
        assert model.optional_int is None or model.optional_int == 0
        assert model.optional_str is None or model.optional_str == ""

    def test_roundtrip_get_and_set_values(self):
        class TestModel(pydantic.BaseModel):
            a: int = 1
            b: float = 2.0
            c: str = "test"
            d: bool = True

        widget = ParamWidget(TestModel)
        original_values = widget.get_values()

        # Modify values
        widget.set_values({"a": 42, "b": 3.14, "c": "modified", "d": False})
        modified_values = widget.get_values()

        assert modified_values["a"] == 42
        assert modified_values["b"] == 3.14
        assert modified_values["c"] == "modified"
        assert modified_values["d"] is False

        # Set back to original
        widget.set_values(original_values)
        restored_values = widget.get_values()

        assert restored_values == original_values
