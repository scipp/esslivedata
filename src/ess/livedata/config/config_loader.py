# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import os
from importlib import resources

import yaml
from jinja2 import Environment, Template
from jinja2.meta import find_undeclared_variables

from .environment import DEFAULT_ENV, ENV_VAR


def get_template_variables(template_content: str) -> set[str]:
    """Extract variables from Jinja template using AST parser."""
    env = Environment(autoescape=True)
    ast = env.parse(template_content)
    return find_undeclared_variables(ast)


def get_env_vars(template_content: str) -> dict[str, str]:
    """Get environment variables needed for template."""
    variables = get_template_variables(template_content)
    return {var: os.getenv(var) for var in variables}


def load_config(*, namespace: str, env: str | None = None) -> dict:
    """Load configuration based on environment.

    Parameters
    ----------
    namespace:
        Configuration namespace (e.g. 'monitor_data')
    env:
        Environment name ('dev', 'staging', 'prod').
        Defaults to value of LIVEDATA_ENV environment variable. Set to an empty string
        if the config file is independent of an environment.
    """
    env = env if env is not None else os.getenv(ENV_VAR, DEFAULT_ENV)
    env = f'_{env}' if env else ''
    config_file = f'{namespace}{env}.yaml'
    template_file = f'{namespace}{env}.yaml.jinja'

    config_path = resources.files('ess.livedata.config.defaults')

    # Try direct YAML first
    try:
        with config_path.joinpath(config_file).open() as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fall back to template
        try:
            with config_path.joinpath(template_file).open() as f:
                template_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Neither {config_file} nor {template_file} found in config defaults"
            ) from None
        template = Template(template_content)
        env_vars = get_env_vars(template_content)
        for var, value in env_vars.items():
            if value is None:
                raise ValueError(f"Environment variable {var} not set") from None

        # Render template and parse YAML
        rendered = template.render(**env_vars)
        return yaml.safe_load(rendered)
