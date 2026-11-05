# Deployment Configuration

## Instrument Selection

The published `esslivedata-backend` and `esslivedata-dashboard` images are not
instrument-specific. Deployment tooling must pass `--instrument <name>` to each
backend service and the dashboard. Since the application otherwise defaults to
`dummy`, deployment tooling should validate the instrument and always include
the argument explicitly.

Pin production deployments to a version tag or image digest rather than
`latest`, so that the same artifact can be promoted and rolled back across
instruments. The backend image installs every instrument extra. The dashboard
image installs the `dashboard` extra without instrument extras, since
instrument-specific processing runs in the backend services.

## Runtime Reference Data

Some instrument reduction workflows download checksum-verified lookup tables
when they are first used. The backend image sets `SCIPP_DATA_DIR` to
`/app/data/cache`. Mount a writable persistent directory at that path so the
data survives container replacement. The container runs as UID/GID 1000, so a
host-mounted directory must be writable by that identity. Deployments without
outbound network access must pre-populate this cache before starting production
workflows.

## Dashboard Authentication

The dashboard supports OAuth login through an OpenID Connect provider such as PING. Without OAuth configured it runs **without any login**, which is intended for local development only. The published image sets `LIVEDATA_REQUIRE_LOGIN=1`, so it refuses to start without OAuth; set it to empty to run that image without login.

Register a client with the provider using the authorization code flow and `client_secret_post` authentication. Then set these variables at runtime, never in the `Dockerfile`, as the images are published:

| Variable | Value |
|---|---|
| `PANEL_OAUTH_PROVIDER` | `generic` |
| `PANEL_OAUTH_KEY` | Client ID |
| `PANEL_OAUTH_SECRET` | Client secret |
| `PANEL_OAUTH_REDIRECT_URI` | Redirect URI registered with the provider, such as `https://<dashboard-host>/oauth-redirect` |
| `PANEL_OAUTH_SCOPE` | `openid,profile` |
| `PANEL_OAUTH_JWT_USER` | Token claim used as the username. At ESS: `sub`, as PING returns no email |
| `LIVEDATA_OAUTH_DISCOVERY_URL` | At ESS: `https://login.ess.eu/.well-known/openid-configuration` |
| `PANEL_COOKIE_SECRET` | Long random value |
| `PANEL_OAUTH_ENCRYPTION` | Output of `panel oauth-secret` |
| `BOKEH_ALLOW_WS_ORIGIN` | The dashboard's public hostname |

Serve the dashboard through an HTTPS proxy that also forwards WebSocket connections.

The dashboard refuses to start if this configuration is incomplete, or if the provider cannot be reached at startup.

There are no access levels: any account the provider accepts for this client has full control of the dashboard. To limit who can log in, restrict the client at the provider.

## Kafka TLS in Containers

When a container connects to Kafka with `KAFKA_SECURITY_PROTOCOL=SSL` or
`SASL_SSL`, set `KAFKA_SSL_CA_LOCATION` to the certificate path inside the
container. The deployment must also mount the CA certificate read-only at that
same path. Setting the environment variable alone does not make the host file
available in the container.

For example, the relevant Compose configuration is:

```yaml
services:
  livedata:
    environment:
      KAFKA_SECURITY_PROTOCOL: SASL_SSL
      KAFKA_SSL_CA_LOCATION: /etc/esslivedata/kafka-ca.pem
    volumes:
      - /host/path/kafka-ca.pem:/etc/esslivedata/kafka-ca.pem:ro
```

The CA variable and mount are not required for `PLAINTEXT` or
`SASL_PLAINTEXT` connections.

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
