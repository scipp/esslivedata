# Integration Tests Requiring Kafka

This document lists integration tests that require Kafka infrastructure and therefore cannot be implemented as unit tests. These tests should be performed manually or in an integration test environment with Kafka running.

## ROI Feature Integration Tests

### Dashboard ROI Publisher Integration
**Location**: `src/ess/livedata/dashboard/dashboard.py:139-148`

**What needs testing**:
- Verify `ROIPublisher` is correctly instantiated in `DashboardBase.__init__()`
- Verify `ROIPublisher` uses correct Kafka sink (upstream broker)
- Verify `ROIPublisher` is passed to `PlottingController`
- Verify ROI updates from UI reach Kafka topic

**Why it needs Kafka**:
- Requires actual Kafka broker to verify message publishing
- Needs integration with `KafkaSink` and serialization
- Requires end-to-end message flow from UI through Kafka to backend

**Test scenario**:
1. Start dashboard with Kafka running
2. Create a workflow with detector view
3. Draw ROI rectangle in UI
4. Verify ROI message appears in `{instrument}_livedata_roi` Kafka topic
5. Verify message is correctly serialized (da00 format with ROI data)
6. Verify backend service receives and processes ROI message

### Detector Data Service ROI Route Integration
**Location**: `src/ess/livedata/services/detector_data.py:24`

**What needs testing**:
- Verify detector_data service subscribes to `livedata_roi` topic
- Verify ROI messages are correctly routed through `RoutingAdapter`
- Verify ROI messages reach `DetectorHandlerFactory.make_preprocessor()`
- Verify `LatestValue` accumulator receives ROI DataArrays
- Verify DetectorView workflow receives ROI configuration
- Verify ROI histogram outputs are published back to Kafka

**Why it needs Kafka**:
- Requires actual Kafka consumer to subscribe to topics
- Needs integration with streaming message flow
- Requires verification of bidirectional Kafka communication (ROI in, histograms out)

**Test scenario**:
1. Start detector_data service with Kafka running
2. Publish ROI rectangle message to `{instrument}_livedata_roi` topic
3. Verify service consumes and processes ROI message
4. Verify DetectorView workflow is updated with ROI configuration
5. Send detector event data
6. Verify `roi_current_0` and `roi_cumulative_0` outputs appear in data topic
7. Modify ROI (delete/update)
8. Verify workflow responds to ROI changes

### End-to-End ROI Workflow
**What needs testing**:
- Complete flow: UI ’ Kafka (ROI) ’ Backend ’ Kafka (histograms) ’ UI
- Multiple ROIs (create, update, delete)
- Multi-detector workflows (verify stream isolation with job_id)
- ROI persistence across dashboard restarts (if implemented)

**Why it needs Kafka**:
- Requires full stack integration
- Tests distributed system behavior
- Validates message serialization/deserialization round-trip

**Test scenario**:
1. Start full stack (dashboard + detector_data service + Kafka)
2. Create detector view workflow from UI
3. Draw multiple ROIs (0, 1, 2)
4. Verify all ROI spectra appear in UI with correct labels and colors
5. Delete ROI 1
6. Verify ROI 1 spectrum disappears from UI
7. Update ROI 0 (resize/move)
8. Verify ROI 0 histogram updates correctly
9. Restart dashboard
10. Verify ROI state is consistent (or reset, depending on design)

## Testing Recommendations

### Manual Testing Checklist
- [ ] Dashboard initializes with ROI publisher
- [ ] ROI rectangles can be drawn in detector plot
- [ ] ROI messages appear in Kafka topic with correct format
- [ ] Backend consumes ROI messages
- [ ] ROI histograms are generated and published
- [ ] ROI spectrum plots appear in UI
- [ ] Multiple ROIs display with correct colors
- [ ] ROI updates are reflected in backend and UI
- [ ] ROI deletions remove corresponding spectra
- [ ] Multi-detector workflows isolate ROIs correctly

### Integration Test Environment Setup
```bash
# Start Kafka
docker-compose up kafka

# Start fake data producers
python -m ess.livedata.services.fake_detectors --instrument dummy

# Start backend service
python -m ess.livedata.services.detector_data --instrument dummy --dev

# Start dashboard
python -m ess.livedata.dashboard.reduction --instrument dummy
```

### Kafka Topic Verification
```bash
# List topics
kafka-topics --bootstrap-server localhost:9092 --list

# Monitor ROI topic
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic dummy_livedata_roi --from-beginning

# Monitor data topic
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic dummy_livedata_data --from-beginning
```

## Notes

- All unit tests in this branch run **without** Kafka (using fakes)
- Integration tests listed here require actual Kafka infrastructure
- Consider adding automated integration tests in CI/CD if Kafka test environment is available
- Manual testing should be performed before merging to main
