# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Auto-generated NeXus f144 stream declarations.

Do not edit by hand. Regenerate with
``python -m ess.livedata.nexus_helpers <geometry.nxs> --generate``.

Source: coda_freia_999999_00022031.hdf
"""

from ess.livedata.config import F144Stream

PARSED_STREAMS: dict[str, F144Stream] = {
    '/entry/data': F144Stream(
        nexus_path='/entry/data',
        source='MISSING4',
        topic='freia_misc_devices',
        units='dimensionless',
    ),
    '/entry/instrument/FOC-1_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/FOC-1_chopper/delay',
        source='TODO21',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/FOC-1_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/FOC-1_chopper/rotation_speed',
        source='TODO23',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/FOC-1_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/FOC-1_chopper/rotation_speed_setpoint',
        source='TODO24',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/FOC-2_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/FOC-2_chopper/delay',
        source='TODO25',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/FOC-2_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/FOC-2_chopper/rotation_speed',
        source='TODO27',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/FOC-2_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/FOC-2_chopper/rotation_speed_setpoint',
        source='TODO28',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/FOC-3_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/FOC-3_chopper/delay',
        source='TODO29',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/FOC-3_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/FOC-3_chopper/rotation_speed',
        source='TODO31',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/FOC-3_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/FOC-3_chopper/rotation_speed_setpoint',
        source='TODO32',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-1_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/WBC-1_chopper/delay',
        source='TODO1',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/WBC-1_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/WBC-1_chopper/rotation_speed',
        source='TODO3',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-1_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/WBC-1_chopper/rotation_speed_setpoint',
        source='TODO4',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-2_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/WBC-2_chopper/delay',
        source='TODO5',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/WBC-2_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/WBC-2_chopper/rotation_speed',
        source='TODO7',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-2_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/WBC-2_chopper/rotation_speed_setpoint',
        source='TODO8',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-3_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/WBC-3_chopper/delay',
        source='TODO9',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/WBC-3_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/WBC-3_chopper/rotation_speed',
        source='TODO11',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WBC-3_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/WBC-3_chopper/rotation_speed_setpoint',
        source='TODO12',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WFM-1_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/WFM-1_chopper/delay',
        source='TODO13',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/WFM-1_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/WFM-1_chopper/rotation_speed',
        source='TODO15',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WFM-1_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/WFM-1_chopper/rotation_speed_setpoint',
        source='TODO16',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WFM-2_chopper/delay': F144Stream(
        nexus_path='/entry/instrument/WFM-2_chopper/delay',
        source='TODO17',
        topic='freia_choppers',
        units='ns',
    ),
    '/entry/instrument/WFM-2_chopper/rotation_speed': F144Stream(
        nexus_path='/entry/instrument/WFM-2_chopper/rotation_speed',
        source='TODO19',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/WFM-2_chopper/rotation_speed_setpoint': F144Stream(
        nexus_path='/entry/instrument/WFM-2_chopper/rotation_speed_setpoint',
        source='TODO20',
        topic='freia_choppers',
        units='Hz',
    ),
    '/entry/instrument/heavy_shutter': F144Stream(
        nexus_path='/entry/instrument/heavy_shutter',
        source='MISSING2',
        topic='freia_misc_devices',
        units='dimensionless',
    ),
    '/entry/instrument/light_shutter': F144Stream(
        nexus_path='/entry/instrument/light_shutter',
        source='MISSING1',
        topic='freia_misc_devices',
        units='dimensionless',
    ),
    '/entry/instrument/source': F144Stream(
        nexus_path='/entry/instrument/source',
        source='MISSING3',
        topic='freia_misc_devices',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_cycle_id': F144Stream(
        nexus_path='/entry/instrument/source/beam_cycle_id',
        source='TD-M:Ctrl-EVR-1:DbufCycleId-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_destination': F144Stream(
        nexus_path='/entry/instrument/source/beam_destination',
        source='TD-M:Ctrl-EVR-1:DbufBDest-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_destination_mode': F144Stream(
        nexus_path='/entry/instrument/source/beam_destination_mode',
        source='TD-M:Ctrl-EVR-1:DbufBDest-II',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_present': F144Stream(
        nexus_path='/entry/instrument/source/beam_present',
        source='TD-M:Ctrl-EVR-1:DbufBPresent-II',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_pulse_end_event_counter': F144Stream(
        nexus_path='/entry/instrument/source/beam_pulse_end_event_counter',
        source='TD-M:Ctrl-EVR-1:EvtBPulseEndCnt-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_pulse_end_event_id': F144Stream(
        nexus_path='/entry/instrument/source/beam_pulse_end_event_id',
        source='TD-M:Ctrl-EVR-1:EvtBPulseEndId-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_pulse_start_event_counter': F144Stream(
        nexus_path='/entry/instrument/source/beam_pulse_start_event_counter',
        source='TD-M:Ctrl-EVR-1:EvtBPulseStCnt-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/beam_state': F144Stream(
        nexus_path='/entry/instrument/source/beam_state',
        source='TD-M:Ctrl-EVR-1:DbufBState-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/bunch_length': F144Stream(
        nexus_path='/entry/instrument/source/bunch_length',
        source='TD-M:Ctrl-EVR-1:DbufBLen-I',
        topic='tn_data_general',
        units='us',
    ),
    '/entry/instrument/source/bunch_pattern': F144Stream(
        nexus_path='/entry/instrument/source/bunch_pattern',
        source='TD-M:Ctrl-EVR-1:DbufTgRast-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/cryo_tt_82025_temperature': F144Stream(
        nexus_path='/entry/instrument/source/cryo_tt_82025_temperature',
        source='CrS-CMS:Cryo-TT-82025:MeasValue',
        topic='tn_data_general',
        units='K',
    ),
    '/entry/instrument/source/cryo_tt_82027_temperature': F144Stream(
        nexus_path='/entry/instrument/source/cryo_tt_82027_temperature',
        source='CrS-CMS:Cryo-TT-82027:MeasValue',
        topic='tn_data_general',
        units='K',
    ),
    '/entry/instrument/source/cryo_tt_82029_temperature': F144Stream(
        nexus_path='/entry/instrument/source/cryo_tt_82029_temperature',
        source='CrS-CMS:Cryo-TT-82029:MeasValue',
        topic='tn_data_general',
        units='K',
    ),
    '/entry/instrument/source/cryo_tt_82031_temperature': F144Stream(
        nexus_path='/entry/instrument/source/cryo_tt_82031_temperature',
        source='CrS-CMS:Cryo-TT-82031:MeasValue',
        topic='tn_data_general',
        units='K',
    ),
    '/entry/instrument/source/cryo_tt_82033_temperature': F144Stream(
        nexus_path='/entry/instrument/source/cryo_tt_82033_temperature',
        source='CrS-CMS:Cryo-TT-82033:MeasValue',
        topic='tn_data_general',
        units='K',
    ),
    '/entry/instrument/source/current': F144Stream(
        nexus_path='/entry/instrument/source/current',
        source='TD-M:Ctrl-EVR-1:DbufBCurr-I',
        topic='tn_data_general',
        units='mA',
    ),
    '/entry/instrument/source/cycle_start_event_counter': F144Stream(
        nexus_path='/entry/instrument/source/cycle_start_event_counter',
        source='TD-M:Ctrl-EVR-1:EvtF14HzCnt-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/data_buffer_sent_offset': F144Stream(
        nexus_path='/entry/instrument/source/data_buffer_sent_offset',
        source='TD-M:Ctrl-SCE-1:DbufSentOffset-SP',
        topic='tn_data_general',
        units='us',
    ),
    '/entry/instrument/source/energy': F144Stream(
        nexus_path='/entry/instrument/source/energy',
        source='TD-M:Ctrl-EVR-1:DbufBEn-I',
        topic='tn_data_general',
        units='keV',
    ),
    '/entry/instrument/source/flat_top_current': F144Stream(
        nexus_path='/entry/instrument/source/flat_top_current',
        source='A2T-130LWU:PBI-BCM-001:FlatTopCurrentR',
        topic='tn_data_general',
        units='mA',
    ),
    '/entry/instrument/source/fsm_machine_active': F144Stream(
        nexus_path='/entry/instrument/source/fsm_machine_active',
        source='CrS-CMS:SC-FSM-007x:STS_Active',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/high_beta_flat_top_current': F144Stream(
        nexus_path='/entry/instrument/source/high_beta_flat_top_current',
        source='HEBT-160LWU:PBI-BCM-001:FlatTopCurrentR',
        topic='tn_data_general',
        units='mA',
    ),
    '/entry/instrument/source/high_beta_pulse_charge': F144Stream(
        nexus_path='/entry/instrument/source/high_beta_pulse_charge',
        source='HEBT-160LWU:PBI-BCM-001:PulseChargeR',
        topic='tn_data_general',
        units='uC',
    ),
    '/entry/instrument/source/high_beta_pulse_width': F144Stream(
        nexus_path='/entry/instrument/source/high_beta_pulse_width',
        source='HEBT-160LWU:PBI-BCM-001:PulseWidthR',
        topic='tn_data_general',
        units='us',
    ),
    '/entry/instrument/source/mode': F144Stream(
        nexus_path='/entry/instrument/source/mode',
        source='TD-M:Ctrl-EVR-1:DbufBMod-II',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/ni_acq_start_event_counter': F144Stream(
        nexus_path='/entry/instrument/source/ni_acq_start_event_counter',
        source='TD-M:Ctrl-EVR-1:EvtNiAcqStCnt-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/ni_sync_event_counter': F144Stream(
        nexus_path='/entry/instrument/source/ni_sync_event_counter',
        source='TD-M:Ctrl-EVR-1:EvtNiSyncCnt-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
    '/entry/instrument/source/pulse_charge': F144Stream(
        nexus_path='/entry/instrument/source/pulse_charge',
        source='A2T-130LWU:PBI-BCM-001:PulseChargeR',
        topic='tn_data_general',
        units='uC',
    ),
    '/entry/instrument/source/pulse_width': F144Stream(
        nexus_path='/entry/instrument/source/pulse_width',
        source='A2T-130LWU:PBI-BCM-001:PulseWidthR',
        topic='tn_data_general',
        units='us',
    ),
    '/entry/instrument/source/target_segment': F144Stream(
        nexus_path='/entry/instrument/source/target_segment',
        source='TD-M:Ctrl-EVR-1:DbufTgSeg-I',
        topic='tn_data_general',
        units='dimensionless',
    ),
}
