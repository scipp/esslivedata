# Deployment Configuration

## Dashboard Configuration Persistence

The ESSlivedata dashboard stores user configurations (workflow settings, plotter configurations, etc.) locally on disk to preserve settings across restarts.

### Configuration Directory

Dashboard configurations are stored in a per-instrument configuration directory. By default, this directory is resolved as follows:

1. If `LIVEDATA_CONFIG_DIR` environment variable is set, use `$LIVEDATA_CONFIG_DIR/<instrument>/`
2. Otherwise, if `XDG_CONFIG_HOME` is set, use `$XDG_CONFIG_HOME/esslivedata/<instrument>/`
3. Otherwise, use `~/.config/esslivedata/<instrument>/`

### Example: Setting a Custom Configuration Directory

To store configurations in a custom location:

```sh
export LIVEDATA_CONFIG_DIR=/var/lib/esslivedata/configs
python -m ess.livedata.dashboard.reduction --instrument dummy
```

This will store all dashboard configurations for the `dummy` instrument in `/var/lib/esslivedata/configs/dummy/`.

### Example: Using XDG_CONFIG_HOME

Alternatively, if you want to use the XDG Base Directory specification:

```sh
export XDG_CONFIG_HOME=/etc/esslivedata
python -m ess.livedata.dashboard.reduction --instrument dummy
```

This will store configurations in `/etc/esslivedata/esslivedata/dummy/`.

### Configuration Files

The configuration directory contains YAML files for different types of configurations:

- `workflow_configs.yaml`: Workflow parameter settings
- `plotter_configs.yaml`: Plotter/visualization configurations

Each file is automatically created and managed by the dashboard. Configurations are persisted after each modification using atomic file operations with file locking to prevent corruption from concurrent access.
