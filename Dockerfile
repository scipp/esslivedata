# ============================================================
# Backend — processing services for all supported instruments
# ============================================================
FROM python:3.13-slim AS backend

ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 livedata && \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin livedata

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[all-instruments]"

ENV LIVEDATA_DATA_DIR=/app/data/geometry \
    SCIPP_DATA_DIR=/app/data/cache
RUN python -m ess.livedata.scripts.download_geometry && \
    mkdir -p "${SCIPP_DATA_DIR}" && \
    chown livedata:livedata "${SCIPP_DATA_DIR}"

USER livedata

ENV LIVEDATA_ENV=docker \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["tini", "--"]

# TODO: Replace with HTTP health endpoint (e.g., curl -sf http://localhost:8080/health)
#       once backend services expose liveness/readiness checks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import ess.livedata; print('ok')" || exit 1


# ============================================================
# Dashboard — reduction dashboard for all supported instruments
# ============================================================
FROM python:3.13-slim AS dashboard

ARG SETUPTOOLS_SCM_PRETEND_VERSION

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 livedata && \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin livedata

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e ".[dashboard]"

USER livedata

EXPOSE 5009

ENV LIVEDATA_ENV=docker \
    BOKEH_ALLOW_WS_ORIGIN=* \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "ess.livedata.dashboard.reduction"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:5009/ || exit 1
