# sciline — reviewer reference

sciline builds task graphs by dependency injection on **type hints**: a
provider is a function whose return-type annotation is its node key and whose
argument types are its inputs. Most bugs come from misunderstanding what the
type-keyed graph does and does not check.

## Graph semantics

- Keys are matched by type only; argument *names* are irrelevant. Every
  provider argument and the return value must have a type hint (enforced at
  `insert`/construction with a clear error). A provider cannot take two
  arguments of the same type ("duplicate type hints" error).
- **One provider per key, last wins, silently.** `insert` or
  `pipeline[Key] = ...` for an existing key replaces the previous provider
  or param without any warning. There is no "ambiguous provider" error —
  collisions are silent bugs. This is why distinct domain types matter.
- Classes cannot be providers (annotations not reliably extractable) — use
  functions/staticmethods/instances. Decorated providers need
  `functools.wraps` or their hints are lost.
- Terminology the org enforces: "key" = the domain type connecting
  providers; "target" = what you request from `compute`/`get`.

## Domain types

- Simple keys: `X = NewType('X', underlying)`. Never reuse plain types
  (`float`, `str`, `dict`, `sc.DataArray`) as keys — they collide.
- Generic families: `NewType` cannot be generic, so subclass
  `sciline.Scope`: `class Filename(sciline.Scope[RunType, str], str): ...`
  — the trailing `, str` (inheriting the supertype) is mandatory.
- TypeVars used in keys must be constrained — either at definition
  (`TypeVar('T', A, B)`) or via `Pipeline(..., constraints={T: [...]})`.
  On `insert`, generic providers are eagerly expanded into one concrete
  provider per constraint combination.
- `Optional[T]` / `T | None` and **default argument values get no special
  handling** (ADR 0002): a default does not make an input optional, and
  `Optional[T]` is a distinct key from `T` (wire it explicitly:
  `pl[Optional[T]] = pl[T]`).

## Mutation, copying, laziness, caching

- `insert` and `__setitem__` **mutate in place**; `copy`, `map`, `reduce`,
  `__getitem__` return new graphs. Two consequences reviewers check:
  - Library/shared pipelines must be `.copy()`-ed before modification.
  - `pl.map(table)` without assigning the result is a no-op.
- `pipeline[Key] = value` sets a parameter if `value` is a plain object, but
  **splices a subgraph** if `value` is a Pipeline/DataGraph — a powerful and
  easy-to-misread overload.
- `get(target)` builds a `TaskGraph` (pruned to ancestors of targets);
  `compute(target)` = `get(...).compute()`. Providers run in any order or
  not at all — only ancestors of the targets execute.
- **No caching.** Repeated `compute()` calls re-run all shared ancestors.
  Idioms: `compute((A, B))` returns a dict from one execution;
  `bind_and_call(fn)` injects by type in one execution; pin an intermediate
  via `pl[Key] = precomputed` to continue from it.
- Missing inputs raise `UnsatisfiedRequirement` at build time (pass
  `handler=HandleAsComputeTimeException()` to defer, e.g. to `visualize`
  an incomplete graph).

## Provider purity

The scheduler decides when/if/how often providers run (dask threaded by
default). Therefore providers must not:
- write files or talk to networks/SciCat (side effects belong in a final
  `bind_and_call` step),
- mutate their inputs (inputs may be shared across branches),
- read wall-clock/random/global state.

## Parameters vs providers

- Params (`pl[Key] = value`) are constant leaves shown with their value;
  providers are computed nodes. Don't wrap constants in zero-arg providers;
  don't hard-code a constant where a computation belongs.
- One small typed parameter per knob. A giant `Config` blob as a dependency
  recreates the flat-namespace problem sciline exists to solve: everything
  depends on everything and nothing is individually replaceable.
- Judgment call reviewed case-by-case: parametrize by TypeVar only when it
  pays. A dict keyed by detector name was accepted over per-bank domain
  types to avoid making every provider generic (essreduce#319).

## map / reduce

- Column keys of the param table must be actual graph keys (the types).
- Insert providers/params on the base pipeline **before** `.map(...)`.
- `reduce(func=...)` ignores type hints; `func` receives all mapped values
  positionally, output addressed by `name=`. All inputs are held in memory
  at once. No index/axis → reduces over all mapped dims.
- Access mapped results with `sciline.compute_mapped` /
  `get_mapped_node_names` (require pandas).

## Anti-patterns (wrong → right)

```python
# Plain-type keys colliding silently
def f(...) -> float: ...           # → Result = NewType('Result', float)

# God-provider: load+clean+process in one function
# → one provider per domain-type transition; intermediates addressable/replaceable

# Repeated compute expecting caching
a = pl.compute(A); b = pl.compute(B)     # re-runs shared ancestors
# → res = pl.compute((A, B))

# Mutating a shared pipeline
workflow_from_library[Key] = my_value
# → pl = workflow_from_library.copy(); pl[Key] = my_value

# Map without reassignment
pl.map(table)                       # no-op → pl = pl.map(table)

# Relying on defaults / Optional pruning
def f(x: Mask | None = None) -> R: ...   # sciline still demands a provider
# → provide Optional[Mask] explicitly, or restructure

# Missing functools.wraps on a provider decorator → hints lost
```

## Review-history signals (sciline repo)

- Naming is reviewed hard: consistent DAG vocabulary (`output_keys`, not
  `sink_node_names`), no double meanings, rename instead of commenting
  around a misleading name.
- Minimal public API: new `Pipeline` methods must justify themselves;
  prefer free functions and composable primitives ("always easier to make
  things more complicated later than to remove").
- Complexity regressions rejected even when benchmarks look tolerable for
  today's workflows ("sciline is a *generic* package" — an O(N) → O(N²)
  change in `Pipeline.get` was blocked at +8 ms).
- Reprs must stay cheap and summary-level for thousand-row param tables.
- Correct exception types (`UnboundTypeVar` mirrors what mypy would say);
  no accidental `defaultdict` masking a should-raise path.
- Typing: `Any` over `object`; fix the annotation rather than `cast`;
  `collections.abc` over deprecated `typing` aliases; mind minimum Python.
