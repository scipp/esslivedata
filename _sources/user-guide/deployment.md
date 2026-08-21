# Deployment Configuration

## Dashboard Authentication

The reduction dashboard supports optional basic password authentication via [Panel's built-in basic auth](https://panel.holoviz.org/how_to/authentication/basic.html). When enabled, users must enter a username and password before accessing the dashboard. Any username is accepted; only the password is validated.

### Enabling Authentication

Set two environment variables before starting the dashboard:

```sh
export LIVEDATA_BASIC_AUTH_PASSWORD=my_password
export LIVEDATA_BASIC_AUTH_COOKIE_SECRET=my_super_safe_cookie_secret
python -m ess.livedata.dashboard.reduction --instrument dummy
```

- `LIVEDATA_BASIC_AUTH_PASSWORD`: The password required to log in. Any username will be accepted.
- `LIVEDATA_BASIC_AUTH_COOKIE_SECRET`: A secret string used to sign session cookies. Use a long, random value in production.

The same options are available as CLI arguments (`--basic-auth-password`, `--basic-auth-cookie-secret`), but environment variables are preferred since command-line arguments may be visible in process listings and shell history.

### How It Works

When authentication is configured, Panel serves a login page before granting access to the dashboard. After a successful login, a signed session cookie is set in the browser so the user remains authenticated. The authenticated username is available via `pn.state.user` within the dashboard session.

When neither variable is set, the dashboard runs without authentication (the default).

## Dashboard Configuration Persistence

The ESSlivedata dashboard stores user configurations (workflow settings, plotter configurations, etc.) locally on disk to preserve settings across restarts.

### Configuration Directory

Dashboard configurations are stored in a per-instrument configuration directory. By default, this directory is resolved as follows:

1. If `LIVEDATA_CONFIG_DIR` environment variable is set, use `$LIVEDATA_CONFIG_DIR/<instrument>/`
2. Otherwise, use the platform-specific user config directory (e.g., `~/.config/esslivedata/<instrument>/` on Linux)

The platform-specific directory is determined using the `platformdirs` library, which follows standard conventions for each operating system (XDG on Linux, AppData on Windows, etc.).

### Example: Setting a Custom Configuration Directory

To store configurations in a custom location, set the `LIVEDATA_CONFIG_DIR` environment variable:

```sh
export LIVEDATA_CONFIG_DIR=/var/lib/esslivedata/configs
python -m ess.livedata.dashboard.reduction --instrument dummy
```

This will store all dashboard configurations for the `dummy` instrument in `/var/lib/esslivedata/configs/dummy/`.

### Configuration Files

The configuration directory contains YAML files for different types of configurations:

- `workflow_configs.yaml`: Workflow parameter settings
- `plotter_configs.yaml`: Plotter/visualization configurations

Each file is automatically created and managed by the dashboard. Configurations are persisted after each modification using atomic file operations (write to temp file, then rename) to prevent corruption if the process crashes during a write.
