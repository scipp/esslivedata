# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType

import pandas as pd
from environments import BenchmarkResultFilePath, BenchmarkRootDir
from runner import BenchmarkReport

ResultPath = BenchmarkResultFilePath


def _extract_only_if_one[T](values: list[T], target_name: str) -> T | None:
    if len(cands := set(values)) > 1:
        raise ValueError(f"More than 1 {target_name} found in the list.")
    elif len(cands) == 0:
        return None
    else:
        return cands.pop()


def _extract_unit(units: list[str | None]) -> str | None:
    return _extract_only_if_one([unit for unit in units if unit is not None], "unit")


def _extract_dtype[T](values: list[T]) -> type[T] | str | None:
    none_type = type(None)
    cands = [valt for val in values if (valt := type(val)) is not none_type]
    if (extracted := _extract_only_if_one(cands, "type")) is str:
        return 'string'
    return extracted


def _reconstruct(contents: dict | list | tuple, target_type: type, strict: bool):
    """Reconstruct"""
    from dataclasses import is_dataclass

    if is_dataclass(target_type) and isinstance(contents, dict):
        return reconstruct_nested_dataclass(contents, target_type, strict)
    elif isinstance(target_type, type) and issubclass(target_type, tuple):
        # Reconstruct ``NamedTuple``.
        # Could not use ``NamedTuple`` since it is not a class.
        return target_type(*contents)
    else:
        return contents


def reconstruct_nested_dataclass[D](
    contents: dict, root_type: type[D], type_strict: bool = True
) -> D:
    """Reconstructs dataclass object from a dictionary.

    This helper works only if the signature of ``__init__`` method is not overwritten
    and all dataclass attributes are directly accessible from the object.

    For example, it will not reconstruct a dataclass if it is in a list or dictionary.
    """
    from dataclasses import is_dataclass
    from inspect import signature
    from typing import get_type_hints

    if not is_dataclass(root_type):
        raise ValueError(f"Expected a dataclass type. Got {root_type =}")

    constructor_sig = signature(root_type.__init__)
    constructor_args = constructor_sig.bind_partial(**contents)
    for arg_name in constructor_sig.parameters.keys():
        if arg_name not in constructor_args.arguments:
            constructor_args.arguments[arg_name] = None

    constructor_args.arguments.pop('self', None)
    reconstructed = root_type(**constructor_args.arguments)

    for attr_name, attr_type in get_type_hints(root_type).items():
        attr = getattr(reconstructed, attr_name)
        reconstructed_attr = _reconstruct(attr, attr_type, type_strict)

        setattr(reconstructed, attr_name, reconstructed_attr)

    return reconstructed


def list_to_pd_series(values: list) -> pd.Series:
    if dtype := _extract_dtype(values):
        try:
            return pd.Series(values, dtype=dtype)
        except TypeError:
            return pd.Series(values)
    else:
        return pd.Series(values)


def compose_name(dim: str, units: list[str | None]) -> str:
    return f"{dim} [{_extract_unit(units)}]"


def compose_pandas_column(dim: str, items: dict | list) -> tuple[str, pd.Series]:
    if isinstance(items, dict):
        return compose_name(dim, items['unit']), list_to_pd_series(items['value'])
    else:
        return dim, list_to_pd_series(items)


@dataclass
class ReportTemplate(BenchmarkReport):
    """
    This inheritance is needed to restore the overwritten ``__init__`` method.

    """

    def asdataframe(self) -> pd.DataFrame:
        all_fields = {
            **self.measurements,
            **self.arguments,
            'target-name': self.target_names,
            'environment': [self.environment] * len(self.target_names),
        }

        return pd.DataFrame(
            {
                col_name: col
                for dim, values in all_fields.items()
                for col_name, col in [compose_pandas_column(dim, values)]
            }
        )


def collect_result_paths(root_dir: BenchmarkRootDir) -> list[ResultPath]:
    """Collect all result paths in the benchmark root directory."""
    import os

    return [
        ResultPath(root_dir / file_name)
        for file_name in os.listdir(root_dir)
        if file_name.startswith('result-')
    ]


BenchmarkResultDict = NewType('BenchmarkResultDict', dict)


def read_report(path: ResultPath) -> BenchmarkResultDict:
    """Read the json report as dict."""
    import json

    return BenchmarkResultDict(json.loads(path.read_text()))


def reconstruct_report(raw_report: BenchmarkResultDict) -> ReportTemplate:
    """Reconstruct report saved in the ``path``."""

    return reconstruct_nested_dataclass(raw_report, ReportTemplate)


ResultMap = dict[ResultPath, BenchmarkResultDict]


def collect_results(result_paths: list[ResultPath]) -> ResultMap:
    """Collect all results from result_paths."""
    return ResultMap({path: read_report(path) for path in result_paths})


ReportMap = dict[ResultPath, ReportTemplate]


def collect_reports(result_map: ResultMap) -> ReportMap:
    """Collect all results from result_paths."""
    return ReportMap(
        {path: reconstruct_report(result) for path, result in result_map.items()}
    )


MergedMeasurementsDF = NewType('MergedMeasurementsDF', pd.DataFrame)


def merge_measurements(results: ReportMap) -> MergedMeasurementsDF:
    """Convert all reports to Dataset and merge."""
    dfs = [
        report.asdataframe().dropna(axis=1, how='all') for report in results.values()
    ]
    return MergedMeasurementsDF(
        pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    )
