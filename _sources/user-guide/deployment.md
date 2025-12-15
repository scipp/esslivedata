# Deployment Configuration

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
