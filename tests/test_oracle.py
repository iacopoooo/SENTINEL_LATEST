"""
SENTINEL Test Suite - ORACLE Tests
====================================
Tests for ORACLE early warning system.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestOracleDetection:
    """Test ORACLE drift detection capabilities."""
    
    def test_oracle_import(self):
        """Test that ORACLE can be imported."""
        from temporal_engine.oracle_v3 import SentinelOracleV3
        assert SentinelOracleV3 is not None
    
    def test_oracle_detects_drift_before_diagnosis(self, oracle_drift_patient):
        """Test that ORACLE detects cancer drift before diagnosis date."""
        from temporal_engine.oracle_v3 import SentinelOracleV3
        
        history = oracle_drift_patient['visits']
        diagnosis_date = datetime.strptime(
            oracle_drift_patient['baseline']['diagnosis_date'], 
            '%Y-%m-%d'
        )
        
        # Prepare NGS data
        raw_ngs = [
            {'date': v.get('date'), 'noise_variants': v.get('noise_variants', [])}
            for v in history
        ]
        
        # Test progressive replay (stops before diagnosis)
        first_trigger_date = None
        first_trigger_prob = 0
        
        for i in range(3, len(history) + 1):
            slice_hist = history[:i]
            slice_ngs = raw_ngs[:i]
            
            current_date = datetime.strptime(slice_hist[-1]['date'], '%Y-%m-%d')
            if current_date >= diagnosis_date:
                break
            
            oracle = SentinelOracleV3(slice_hist, patient_id='TEST')
            alerts = oracle.run_oracle(raw_ngs_visits=slice_ngs)
            
            if alerts and alerts[0].probability >= 50:
                first_trigger_date = current_date
                first_trigger_prob = alerts[0].probability
                break
        
        # Assert: ORACLE should detect drift before diagnosis
        assert first_trigger_date is not None, "ORACLE should detect drift"
        assert first_trigger_date < diagnosis_date, "Detection should be before diagnosis"
        assert first_trigger_prob >= 50, f"Probability should be >=50%, got {first_trigger_prob}"
        
        # Calculate lead time
        lead_days = (diagnosis_date - first_trigger_date).days
        assert lead_days > 180, f"Lead time should be >6 months, got {lead_days} days"
    
    def test_oracle_clonal_expansion_signal(self, oracle_drift_patient):
        """Test that ORACLE identifies clonal expansion as signal source."""
        from temporal_engine.oracle_v3 import SentinelOracleV3
        
        history = oracle_drift_patient['visits'][:3]  # Use first 3 visits
        raw_ngs = [
            {'date': v.get('date'), 'noise_variants': v.get('noise_variants', [])}
            for v in history
        ]
        
        oracle = SentinelOracleV3(history, patient_id='TEST')
        alerts = oracle.run_oracle(raw_ngs_visits=raw_ngs)
        
        # Should have alerts
        assert len(alerts) > 0, "Should generate alerts"
        
        # Check for clonal expansion signal
        alert = alerts[0]
        signal_keys = [s.key for s in alert.signal_sources] if alert.signal_sources else []
        
        clonal_signals = ['driver_detected', 'clonal_expansion', 'vaf_increase']
        has_clonal_signal = any(s in str(signal_keys).lower() for s in clonal_signals)
        
        # Either has clonal signal or high probability
        assert has_clonal_signal or alert.probability >= 50
    
    def test_oracle_healthy_patient_no_alert(self):
        """Test that ORACLE doesn't alert on healthy patients."""
        from temporal_engine.oracle_v3 import SentinelOracleV3
        
        # Create healthy patient history
        healthy_history = [
            {"date": "2020-01-01", "blood_markers": {"ldh": 150, "crp": 0.3}},
            {"date": "2020-06-01", "blood_markers": {"ldh": 155, "crp": 0.4}},
            {"date": "2021-01-01", "blood_markers": {"ldh": 148, "crp": 0.3}},
        ]
        raw_ngs = [{"date": v["date"], "noise_variants": []} for v in healthy_history]
        
        oracle = SentinelOracleV3(healthy_history, patient_id='HEALTHY_TEST')
        alerts = oracle.run_oracle(raw_ngs_visits=raw_ngs)
        
        # Should have low probability or no alerts
        if alerts:
            assert alerts[0].probability < 30, "Healthy patient should have low probability"
