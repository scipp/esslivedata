# ============================================================
# Backend — processing services for a specific instrument
# ============================================================
FROM python:3.13-slim AS backend

ARG SETUPTOOLS_SCM_PRETEND_VERSION
ARG INSTRUMENT

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

RUN pip install --no-cache-dir -e ".[${INSTRUMENT}]"

USER livedata

ENV LIVEDATA_ENV=docker \
    LIVEDATA_INSTRUMENT=${INSTRUMENT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "ess.livedata.services.monitor_data"]

# TODO: Replace with HTTP health endpoint (e.g., curl -sf http://localhost:8080/health)
#       once backend services expose liveness/readiness checks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import ess.livedata; print('ok')" || exit 1


# ============================================================
# Dashboard — reduction dashboard for a specific instrument
# ============================================================
FROM python:3.13-slim AS dashboard

ARG SETUPTOOLS_SCM_PRETEND_VERSION
ARG INSTRUMENT

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

RUN pip install --no-cache-dir -e ".[${INSTRUMENT},dashboard]"

USER livedata

EXPOSE 5009

ENV LIVEDATA_ENV=docker \
    LIVEDATA_INSTRUMENT=${INSTRUMENT} \
    BOKEH_ALLOW_WS_ORIGIN=* \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "ess.livedata.dashboard.reduction"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:5009/ || exit 1
