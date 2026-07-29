# Getting started

## Setting up

### Dependencies

Create a virtual environment and install the package in editable mode together with its development dependencies (see [Dependency Management](./dependency-management.md) for more information about how dependencies are managed):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Additionally, building the documentation requires [pandoc](https://pandoc.org/) which is not on PyPI and needs to be installed through other means, e.g. with your OS package manager.

### Set up git hooks

The CI pipeline runs a number of code formatting and static analysis tools.
If they fail, a build is rejected.
To avoid that, you can run the same tools locally.
This can be done conveniently using [pre-commit](https://pre-commit.com/):

```sh
pre-commit install
```

Alternatively, if you want a different workflow, take a look at ``tox.ini`` or ``.pre-commit-config.yaml`` to see what tools are run and how.

## Running tests

`````{tab-set}
````{tab-item} tox
Run the tests using

```sh
tox -e py311
```

(or just `tox` if you want to run all environments).

````
````{tab-item} Manually
Run the tests using

```sh
python -m pytest
```
````
`````

## Building the docs

`````{tab-set}
````{tab-item} tox
Build the documentation using

```sh
tox -e docs
```

This builds the docs and also runs `doctest`.
`linkcheck` can be run separately using

```sh
tox -e linkcheck
```
````

````{tab-item} Manually

Build the documentation using

```sh
python -m sphinx -v -b html -d .tox/docs_doctrees docs html
```

Additionally, test the documentation using

```sh
python -m sphinx -v -b doctest -d .tox/docs_doctrees docs html
python -m sphinx -v -b linkcheck -d .tox/docs_doctrees docs html
```
````
`````