1:1 Correspondence: Kafka Topics ↔ HTTP Endpoints
Kafka Architecture:
Backend Service subscribes to:
  - LIVEDATA_CONFIG topic (workflow_config from dashboard)
  - detector/monitor topics (raw data)

Backend Service publishes to:
  - LIVEDATA_CONFIG topic (workflow_status responses)
  - LIVEDATA_DATA topic (processed data)
  - LIVEDATA_STATUS topic (job status)
HTTP Architecture (should mirror this):
Backend HTTP Service exposes:
  - GET /config → LIVEDATA_CONFIG messages (workflow_status)
  - GET /data → LIVEDATA_DATA messages
  - GET /status → LIVEDATA_STATUS messages

Backend polls:
  - Dashboard's GET /config → LIVEDATA_CONFIG messages (workflow_config)

Dashboard HTTP Service exposes:
  - GET /config → LIVEDATA_CONFIG messages (workflow_config, job_command)

Dashboard polls:
  - Backend's GET /config → LIVEDATA_CONFIG messages (workflow_status)
  - Backend's GET /data → LIVEDATA_DATA messages
  - Backend's GET /status → LIVEDATA_STATUS messages
The mapping:
Kafka topic = HTTP endpoint (URL path)
Kafka consumer = HTTP polling client (GET requests)
Kafka producer = HTTP server exposing endpoint
This makes perfect sense and is very clean! The current RoutingMessageSerializer that mixes everything is not the right abstraction for HTTP - that would be like having all Kafka topics merged into one. So the fix is:
Backend should expose separate endpoints for config vs data vs status
Dashboard already exposes config endpoint correctly (port 5011)
Each endpoint uses the appropriate serializer (not routing serializer)
Should I investigate the current implementation to see what needs changing to achieve this clean separation?