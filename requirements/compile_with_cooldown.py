"""Run pip-compile-multi with a supply-chain cooldown.

Sets UV_EXCLUDE_NEWER to a date in the past before invoking
pip-compile-multi, so that recently uploaded package versions
are excluded from dependency resolution. This reduces the risk
of supply-chain attacks via newly published malicious packages.

The cooldown can be overridden by setting UV_EXCLUDE_NEWER in
the environment before running this script.
"""

import datetime
import os
import subprocess
import sys

COOLDOWN_DAYS = 7


def main() -> None:
    if "UV_EXCLUDE_NEWER" not in os.environ:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=COOLDOWN_DAYS
        )
        os.environ["UV_EXCLUDE_NEWER"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    sys.stderr.write(f"UV_EXCLUDE_NEWER={os.environ['UV_EXCLUDE_NEWER']}\n")
    sys.exit(
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip_compile_multi.cli", "--uv", *sys.argv[1:]],
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    main()
