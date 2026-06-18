---
paths: requirements/*.in, requirements/*.txt, pyproject.toml
---

# Dependency Updates

## Bumping a single dependency — keep the lock diff minimal

`tox -e deps` runs a bare `pip-compile-multi`, which upgrades **every**
transitive dependency to its latest allowed version. For a focused change
("update essreduce to X") this produces a noisy diff (matplotlib, pytest,
jupyterlab, … all move) that can mask unrelated regressions.

To bump only the requested package(s) plus whatever they genuinely force:

1. Edit the floor in `pyproject.toml` and the relevant `requirements/*.in`
   (`base.in`, `nightly.in`, …).
2. Regenerate locks targeting only those packages — `--no-upgrade` pins
   everything else at its current version, `-P` forces the named ones:

   ```sh
   cd requirements
   python ./make_base.py
   pip-compile-multi -d . --backtracking --annotate-index \
     --no-upgrade -P essreduce -P scippneutron
   ```

   Pass `-P` for **each** package that must move, including transitives the
   new version requires (e.g. essreduce 26.6.1 needs `scippneutron>=26.6.0`,
   so the old `scippneutron` pin violates the constraint and must be bumped
   too). Plain `--no-upgrade` without the needed `-P` fails with a
   cross-environment "resolved to different versions" error.
3. Confirm the diff is minimal: `git diff requirements/*.txt | grep '^[-+][a-zA-Z]'`.

A full refresh is a deliberate, separate change — don't fold it into a
feature or single-dependency PR.
