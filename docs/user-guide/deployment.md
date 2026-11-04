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

The reduction dashboard supports login through an OAuth 2.0 / OpenID Connect provider, using [Panel's OAuth support](https://panel.holoviz.org/en/docs/latest/how_to/authentication/configuration.html). ESS uses PING, through Panel's `generic` provider.

```{warning}
Without OAuth configured, the dashboard runs **without any login**, and anyone who can reach it has full control. This is intended for local development only.
```

### Registering the Client

Register a client for the dashboard with the provider, using the authorization code flow:

- **Redirect URI**: for example `https://<dashboard-host>/oauth-redirect`. It must match `PANEL_OAUTH_REDIRECT_URI` exactly. The dashboard serves its login route at this path.
- **Client authentication**: `client_secret_post`. Panel sends the client secret in the body of the token request.
- **Scopes**: `openid` and `profile`.

### Configuration

Set these environment variables on the dashboard container at runtime:

| Variable | Value | Sensitive |
|---|---|---|
| `PANEL_OAUTH_PROVIDER` | `generic` | no |
| `PANEL_OAUTH_KEY` | Client ID | no |
| `PANEL_OAUTH_SECRET` | Client secret | **yes** |
| `PANEL_OAUTH_REDIRECT_URI` | `https://<dashboard-host>/oauth-redirect` | no |
| `LIVEDATA_OAUTH_DISCOVERY_URL` | The provider's OpenID discovery document (`.../.well-known/openid-configuration`), read at startup to find its endpoints. At ESS: `https://login.ess.eu/.well-known/openid-configuration`. | no |
| `PANEL_OAUTH_SCOPE` | `openid,profile` | no |
| `PANEL_OAUTH_JWT_USER` | Token claim shown as the username, such as `sub` or `mail`. Default: `email`, so this is required when the `email` scope is not requested. Login fails with a 401 if the claim is missing. | no |
| `PANEL_COOKIE_SECRET` | Long random value that signs the login cookies | **yes** |
| `PANEL_OAUTH_ENCRYPTION` | Key that encrypts the tokens stored in cookies. Generate one with `panel oauth-secret`. | **yes** |
| `PANEL_OAUTH_EXPIRY` | Optional login lifetime in days. Default: 1. | no |
| `BOKEH_ALLOW_WS_ORIGIN` | The dashboard's public hostname, replacing the image's default of `*` | no |

Set `PANEL_OAUTH_SCOPE` explicitly: Panel's default also requests `offline_access`, which the provider may reject.

### Starting While the Provider Is Down

With `LIVEDATA_OAUTH_DISCOVERY_URL`, the dashboard must reach the provider to start, and refuses to start if it cannot. To remove that dependency, set the endpoints explicitly using the values from the discovery document:

| Variable | Discovery document field |
|---|---|
| `PANEL_OAUTH_AUTHORIZE_URL` | `authorization_endpoint` |
| `PANEL_OAUTH_TOKEN_URL` | `token_endpoint` |
| `PANEL_OAUTH_USER_URL` | `userinfo_endpoint` |

An endpoint set this way takes priority over the discovery document. If all three are set, the document is not fetched at all.

The dashboard refuses to start if the configuration is incomplete, or if OAuth variables are set without `PANEL_OAUTH_PROVIDER`.

```{warning}
Never put these values in the `Dockerfile` or pass them as image build arguments. The images are published, and values set there remain readable in the image layers. Use a Kubernetes Secret or your deployment tool's equivalent.
```

### HTTPS

Serve the dashboard over HTTPS through a TLS-terminating proxy that also forwards WebSocket connections. Panel's login cookies are signed and `HttpOnly` but are not marked `Secure`, so a browser would also send them over plain HTTP. Do not expose the dashboard port directly, and consider enabling HSTS on the proxy.

### Upgrading from Basic Authentication

Password authentication has been removed, along with `--basic-auth-password`, `--basic-auth-cookie-secret` and their environment variables. It had no user accounts, so there is nothing to migrate.

Before upgrading a deployment, configure OAuth as described above and remove `LIVEDATA_BASIC_AUTH_PASSWORD` and `LIVEDATA_BASIC_AUTH_COOKIE_SECRET`. The dashboard refuses to start while either is still set, so that a deployment cannot silently lose its login.

### Limitations

- **There are no access levels.** Any account the provider accepts for this client has full control of the dashboard, including starting and stopping workflows. To limit who can log in, restrict the client at the provider.
- **Logging out is local.** It clears the dashboard's cookies but not the provider's session, so "Log in again" may sign the user straight back in.
- **Tokens are not refreshed.** Users log in again once the login expires. See `PANEL_OAUTH_EXPIRY`.
- **Only the dashboard is protected.** This does not cover Kafka or the backend services.

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
