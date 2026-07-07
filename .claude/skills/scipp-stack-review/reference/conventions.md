# Review culture and project conventions (scipp org)

Cross-repo conventions that human reviewers enforce, distilled from review
threads on scipp, scippnexus, sciline, essreduce, and esslivedata. Apply
these when reviewing any project in the family.

## Tests

- **Assert values, not proxies.** Counts, key presence, or shapes are weak;
  `sc.testing.assert_identical` also verifies unit and dtype. Canonical
  failure: selecting 0 pulses returned the *full* event buffer and the test
  passed because it only asserted the pulse count (scippnexus#137).
- One behavior per test, even at the cost of duplication; test names state
  the distinguishing feature and are updated when behavior changes; no
  placeholder names.
- `pytest.raises`/`pytest.warns` with `match=`; failure *and* success cases;
  parametrize instead of copy-paste — and a parametrized test must actually
  vary something that matters ("Why do you only test int64? Why do you even
  have a parametrized test then?").
- Reviewer "what happens if ...?" questions (repeated values, empty
  selection, single bin, exact multiples of step, zero events) convert into
  tests before merge.
- Fixtures: no single-use fixtures (inline them); fixtures must be realistic
  — check against real files (is that field really static, or an NXlog?).
  Don't test upstream libraries' behavior in downstream repos; don't freeze
  incidental behavior that impedes change.
- Copy semantics get tests: "modifying `a` after the copy does not affect
  `b`".
- Loading/behavior changes are verified against real files and downstream
  consumers (Mantid, scippneutron, the actual instrument workflow), not
  just unit tests.

## Errors, warnings, defaults

- Fail loudly and early with the semantically right exception
  (`sc.UnitError`, `sc.DimensionError`, `UnboundTypeVar`) and an actionable
  message naming the offending key/file/node.
- No broad `except Exception` fallbacks; the try body is the one failing
  statement; the caught types are concrete. "Is there a risk of hiding a
  genuine error?" is the test every fallback must pass.
- No silent defaults that mask misconfiguration (no `defaultdict` for
  user-supplied mappings; explicit allowlists over ignore-everything-else).
  Acceptable defaults are explicit no-ops (`inf`, `None`), documented.
- `assert` is stripped under `-O`: library code uses `if ... raise`.
- Warnings must be accurate about *what* failed (not the enclosing
  operation) and identify the culprit path.
- No `print` in library code; some libraries (scippnexus) deliberately use
  warnings, not `logging` — match the house style.

## Naming and terminology

- Names are reviewed as hard requirements. Spell out abbreviations in
  public names; one name per concept and one concept per name; rename
  rather than comment around a misleading name; method names are verbs;
  functions named for what they do now, not their history.
- Match established vocabulary: DAG terms (sink/output keys, targets),
  NeXus/file conventions (use the term the file format uses), no pandas
  vocabulary in scipp docs. Names must reflect approximations
  ("straight-line Ltotal").
- Quantifiers in names (`all_are_label_based_indices`); avoid names
  colliding with stack concepts (`ds` reads as Dataset).
- User-facing text (titles, parameter descriptions) must state what the
  number physically means, including reference points and units.

## API design

- Minimal public surface: every new public method/property/class must
  justify itself; prefer free functions and composable primitives; "easier
  to make things more complicated later than to remove". Everything public
  gets a docstring — or gets underscored. Export via `__init__` or keep
  private; don't expose half of a helper family.
- Named constructors for value types (`Timestamp.from_ns`), keyword-only
  optional flags, symmetric conversion names (`to_ns`/`to_seconds`),
  unit-explicit APIs.
- `__call__` only for things that are conceptually functions;
  `__getitem__` must be cheap; dunders reserved for cross-library
  conventions.
- Accept `Mapping`/`Sequence` (collections.abc) in signatures rather than
  `dict`/`list`; return the type that preserves key identity (a dict of
  results rather than a tuple, since `NewType` results lose identity).
- Sentinel defaults: module-level `_not_provided = object()` compared with
  `is not` — needed when "not passed" differs from "passed the default";
  check kwargs-forwarding APIs for clashes between alternative spellings
  (`dims`/`shape` vs `sizes`).
- Don't break downstream: renames of public types/providers require an
  org-wide usage check, deprecated aliases, or a coordinated release.
  Deprecation sequence: docstring directive first, runtime warning a
  release later, then error.

## Typing

- Type hints required on public functions; hints must match actual
  behavior — a wrong hint is a bug (fix code or hint). Dead branches
  contradicting the hints are rejected.
- No runtime `isinstance` argument validation (the hint plus mypy is the
  contract) — except where it keeps mypy sound and replaces stringly
  dispatch. No decorative `cast` (fix the annotation); minimize
  `type: ignore`; `Any` over `object`; `collections.abc` over deprecated
  `typing` aliases; mind the minimum supported Python version; enums over
  magic ints.
- `@dataclass(slots=True)` preferred for internal value classes.

## Code structure

- Split long/deeply nested functions — "the probability that all bugs are
  caught in code review approaches zero" for a long function. Extract
  repeated non-trivial code (rule of three) into helpers; extraction also
  stops scope leakage of loop variables.
- Don't delete explanatory comments during refactors; non-obvious logic
  (parsing one-liners, periodic-boundary tricks, ordering dependencies)
  needs a comment with an example.
- Prefer stdlib/scipp facilities over manual arithmetic (`std::midpoint`,
  `sc.lookup`, integer-array indexing, existing coord-union helpers).
- Performance changes need measurement; the deciding argument is memory
  traffic/allocation, not FLOPs. But asymptotic regressions are rejected
  regardless of current benchmarks, and speculative caching without
  benchmarks is rejected too (cache invalidation, thread-safety, leaks —
  `lru_cache` on methods leaks).
- Concurrency: lock scope must cover every field shared across lock
  domains; check cleanup paths (who calls `shutdown`?), unbounded growth
  of registries/state dicts on restart, and atomicity of
  swap-then-flush sequences.

## Docs and docstrings

- NumPy format that actually renders in Sphinx: blank lines around lists,
  `Parameters`/`Returns` sections, double backticks in rst, `:py:func:`
  cross-references. No type annotations in docstrings (autodoc handles it);
  don't restate defaults from the signature — they rot.
- Docstrings describe the general contract, not one use case, and must be
  updated with renames/behavior changes; contract notes go in the class
  docstring, not inline comments; approximations are stated.
- Notebook/recipe docs: standalone (local imports, copy-pasteable),
  motivation before code, TL;DR snippet before the long example, relative
  intra-doc links.
- ADRs/design docs are line-by-line checked for consistency with the code;
  contradictions between doc and implementation are treated as bugs.

## Process

- No unrelated changes smuggled into a PR (dependency refreshes, format
  churn, drive-by refactors) — they mask regressions and cause
  copier-template conflicts. PR title/description updated if the scope
  changes during review.
- TODOs become issues before merge ("Please do this now or open an
  issue."). Design disagreements go to ADRs.
- Dependencies: no upper pins in main deps (pins in test/requirements
  files); every direct import declared even if transitively available;
  optional heavy deps imported lazily with a helpful error; prefer
  widely-used, long-lived deps; locks generated in the oldest supported
  Python; released packages never depend on git branches.
- Small concrete fixes are delivered as GitHub ```suggestion blocks.
- Reasoned pushback on review comments is expected and respected — comply
  or argue with technical rationale, and verify that a reviewer's proposed
  simplification preserves guard semantics before adopting it (accepted
  "simplifications" have introduced OR-for-AND bugs and unconditional
  side effects).
