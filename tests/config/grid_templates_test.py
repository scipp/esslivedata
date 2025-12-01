# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for grid template loading."""

import pytest

from ess.livedata.config.grid_templates import GridTemplate, load_grid_templates


class TestGridTemplate:
    """Tests for GridTemplate dataclass."""

    def test_properties_from_config(self):
        """Test that properties extract values from config dict."""
        config = {
            'title': 'Test Grid',
            'nrows': 3,
            'ncols': 4,
            'cells': [
                {
                    'geometry': {'row': 0, 'col': 0, 'row_span': 1, 'col_span': 1},
                    'config': {'workflow_id': 'test/ns/wf/1'},
                }
            ],
        }
        template = GridTemplate(name='Test', config=config)

        assert template.name == 'Test'
        assert template.title == 'Test Grid'
        assert template.nrows == 3
        assert template.ncols == 4
        assert len(template.cells) == 1

    def test_min_rows_from_cells(self):
        """Test that min_rows is computed from cell geometries."""
        config = {
            'title': 'Test',
            'nrows': 2,
            'ncols': 2,
            'cells': [
                {'geometry': {'row': 0, 'col': 0, 'row_span': 1, 'col_span': 1}},
                {'geometry': {'row': 1, 'col': 0, 'row_span': 2, 'col_span': 1}},
            ],
        }
        template = GridTemplate(name='Test', config=config)

        # Cell at row 1 with row_span 2 requires at least 3 rows
        assert template.min_rows == 3

    def test_min_cols_from_cells(self):
        """Test that min_cols is computed from cell geometries."""
        config = {
            'title': 'Test',
            'nrows': 2,
            'ncols': 2,
            'cells': [
                {'geometry': {'row': 0, 'col': 1, 'row_span': 1, 'col_span': 3}},
            ],
        }
        template = GridTemplate(name='Test', config=config)

        # Cell at col 1 with col_span 3 requires at least 4 columns
        assert template.min_cols == 4

    def test_min_rows_cols_with_no_cells(self):
        """Test that min_rows/cols falls back to config values when no cells."""
        config = {
            'title': 'Empty Grid',
            'nrows': 5,
            'ncols': 6,
            'cells': [],
        }
        template = GridTemplate(name='Empty', config=config)

        assert template.min_rows == 5
        assert template.min_cols == 6

    def test_defaults_for_missing_config_keys(self):
        """Test defaults when config keys are missing."""
        template = GridTemplate(name='Minimal', config={})

        assert template.title == 'Minimal'  # Falls back to name
        assert template.nrows == 3
        assert template.ncols == 3
        assert template.cells == []

    def test_is_immutable(self):
        """Test that GridTemplate is frozen."""
        template = GridTemplate(name='Test', config={'title': 'Test'})

        with pytest.raises(AttributeError):
            template.name = 'Changed'


class TestLoadGridTemplates:
    """Tests for load_grid_templates function."""

    def test_loads_templates_for_dummy_instrument(self):
        """Test loading templates from dummy instrument."""
        templates = load_grid_templates('dummy')

        assert len(templates) >= 1
        # Check the detector_overview template we created
        names = [t.name for t in templates]
        assert 'Detector Overview' in names

    def test_returns_empty_for_unknown_instrument(self):
        """Test that unknown instruments return empty list."""
        templates = load_grid_templates('nonexistent_instrument')

        assert templates == []

    def test_returns_empty_for_instrument_without_templates(self):
        """Test instruments without grid_templates directory return empty list."""
        # LOKI exists but doesn't have grid_templates
        templates = load_grid_templates('loki')

        assert templates == []

    def test_template_has_correct_structure(self):
        """Test that loaded templates have the expected structure."""
        templates = load_grid_templates('dummy')

        # Find the detector overview template
        template = next((t for t in templates if t.name == 'Detector Overview'), None)
        assert template is not None

        # Check structure
        assert template.nrows == 2
        assert template.ncols == 2
        assert len(template.cells) == 2

        # Check cell structure
        cell = template.cells[0]
        assert 'geometry' in cell
        assert 'config' in cell
        assert 'workflow_id' in cell['config']
