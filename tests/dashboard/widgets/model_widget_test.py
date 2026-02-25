# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from enum import StrEnum

import pydantic

from ess.livedata.dashboard.widgets.model_widget import ModelWidget, get_defaults


def test_get_defaults_handles_default_factory():
    """Test that get_defaults correctly handles fields with default_factory."""

    class ModelWithFactory(pydantic.BaseModel):
        with_default: int = 42
        with_factory: int = pydantic.Field(default_factory=lambda: 10, ge=5, le=20)
        no_default: int

    defaults = get_defaults(ModelWithFactory)
    assert defaults == {'with_default': 42, 'with_factory': 10}


class TestModelWidget:
    def test_create_from_dynamic_model(self) -> None:
        def make_model(description: str, option: str) -> type[pydantic.BaseModel]:
            class Option(StrEnum):
                OPTION1 = option

            class InnerModel(pydantic.BaseModel):
                a: int = 1
                b: float = 2.0
                c: Option = Option.OPTION1

            class OuterModel(pydantic.BaseModel):
                inner: InnerModel = pydantic.Field(
                    default_factory=InnerModel, title='Title', description=description
                )

            return OuterModel

        widget = ModelWidget(make_model(description='abc', option='opt1'))
        valid, errors = widget.validate_parameters()
        assert valid
        assert not errors

    def test_widget_with_constrained_default_factory(self) -> None:
        """Test that ModelWidget correctly initializes fields with default_factory."""

        class InnerModel(pydantic.BaseModel):
            constrained_value: int = pydantic.Field(
                default_factory=lambda: 5, ge=1, le=10, description='Constrained field'
            )

        class OuterModel(pydantic.BaseModel):
            inner: InnerModel = pydantic.Field(default_factory=InnerModel)

        widget = ModelWidget(OuterModel)
        valid, errors = widget.validate_parameters()
        assert valid, f"Validation failed: {errors}"
        assert not errors

        # Verify the widget correctly uses the default from default_factory
        params = widget.parameter_values
        assert params.inner.constrained_value == 5
