#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL TEMPORAL FUSION - L'Orchestratore
============================================
Combina Prophet (pazienti in trattamento) e Oracle (sani/remissione)
per il ciclo completo di navigazione temporale della malattia.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .prophet_v4 import (
    SentinelProphetV4,
    TemporalSignal,
    FusionEngineV2,
    FusionResult,
    Urgency,
    VAF_CLINICAL_THRESH,
)

from .oracle_v3 import (
    SentinelOracleV3,
    OracleAlert,
)

from digital_twin import DigitalTwin


class PatientPhase(Enum):
    HEALTHY_SCREENING = "healthy_screening"
    REMISSION_SURVEILLANCE = "remission_surveillance"
    ACTIVE_TREATMENT = "active_treatment"
    UNKNOWN = "unknown"


@dataclass
class PhaseDetectionResult:
    phase: PatientPhase
    confidence: float
    indicators: List[str]
    recommended_engine: str


def detect_patient_phase(patient_data: Dict[str, Any]) -> PhaseDetectionResult:
    indicators = []
    scores = {
        PatientPhase.ACTIVE_TREATMENT: 0.0,
        PatientPhase.REMISSION_SURVEILLANCE: 0.0,
        PatientPhase.HEALTHY_SCREENING: 0.0,
    }
    
    baseline = patient_data.get('baseline', patient_data)
    visits = patient_data.get('visits', [])
    
    if baseline.get('current_therapy'):
        scores[PatientPhase.ACTIVE_TREATMENT] += 0.4
        indicators.append("Current therapy specified")
    
    has_week = any('week_on_therapy' in v for v in visits)
    has_date = any('date' in v or 'month' in v for v in visits)
    
    if has_week:
        scores[PatientPhase.ACTIVE_TREATMENT] += 0.3
        indicators.append("Has week_on_therapy data")
    
    if has_date and len(visits) >= 5:
        scores[PatientPhase.HEALTHY_SCREENING] += 0.4
        indicators.append("Long history with dates")
    
    if baseline.get('remission_status') or baseline.get('post_treatment'):
        scores[PatientPhase.REMISSION_SURVEILLANCE] += 0.5
        indicators.append("Remission status indicated")
    
    if baseline.get('screening'):
        scores[PatientPhase.HEALTHY_SCREENING] += 0.5
        indicators.append("Screening indicated")
    
    best_phase = max(scores, key=scores.get)
    best_score = scores[best_phase]
    
    if best_score < 0.2:
        best_phase = PatientPhase.ACTIVE_TREATMENT if has_week else PatientPhase.UNKNOWN
        indicators.append("Phase inferred from data format")
    
    if best_phase == PatientPhase.ACTIVE_TREATMENT:
        engine = "prophet"
    elif best_phase in (PatientPhase.REMISSION_SURVEILLANCE, PatientPhase.HEALTHY_SCREENING):
        engine = "oracle"
    else:
        engine = "both"
    
    return PhaseDetectionResult(best_phase, min(1.0, best_score), indicators, engine)


@dataclass
class TemporalAnalysis:
    patient_id: str
    phase: PatientPhase
    phase_confidence: float
    engines_used: List[str]
    prophet_signals: Optional[Dict[str, TemporalSignal]] = None
    prophet_fusion: Optional[FusionResult] = None
    oracle_alerts: Optional[List[OracleAlert]] = None
    overall_risk_level: str = "LOW"
    primary_concern: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    next_assessment: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "patient_id": self.patient_id,
            "phase": self.phase.value,
            "phase_confidence": self.phase_confidence,
            "engines_used": self.engines_used,
            "overall_risk_level": self.overall_risk_level,
            "primary_concern": self.primary_concern,
            "recommended_actions": self.recommended_actions,
            "next_assessment": self.next_assessment,
        }
        if self.prophet_fusion:
            result["prophet_fusion"] = {
                "archetype": self.prophet_fusion.archetype,
                "urgency": self.prophet_fusion.urgency.value,
                "action": self.prophet_fusion.action,
            }
        if self.oracle_alerts:
            result["oracle_alerts"] = [
                {"risk_type": a.risk_type, "probability": a.probability}
                for a in self.oracle_alerts
            ]
        return result


class TemporalEngine:
    def __init__(self, auto_detect_phase: bool = True):
        self.auto_detect_phase = auto_detect_phase
    
    def analyze(
        self,
        patient_data: Dict[str, Any],
        force_phase: Optional[PatientPhase] = None,
        raw_ngs_visits: Optional[List[Dict[str, Any]]] = None,
        vaf_threshold: float = VAF_CLINICAL_THRESH,
    ) -> TemporalAnalysis:
        baseline = patient_data.get('baseline', patient_data)
        patient_id = baseline.get('patient_id', 'UNKNOWN')
        
        if force_phase:
            phase, phase_conf = force_phase, 1.0
            indicators = [f"Forced: {force_phase.value}"]
        else:
            det = detect_patient_phase(patient_data)
            phase, phase_conf, indicators = det.phase, det.confidence, det.indicators
        
        engines_used = []
        prophet_signals = None
        prophet_fusion = None
        oracle_alerts = None
        
        # Prophet
        if phase in (PatientPhase.ACTIVE_TREATMENT, PatientPhase.UNKNOWN):
            try:
                visits = patient_data.get('visits', [])
                if visits:
                    prophet = SentinelProphetV4(visits)
                    prophet_signals = {
                        "ldh": prophet.analyze_metric("ldh"),
                        "vaf": prophet.analyze_metric("vaf"),
                        "crp": prophet.analyze_metric("crp"),
                        "neutrophils": prophet.analyze_metric("neutrophils"),
                    }
                    prophet_fusion = FusionEngineV2.diagnose(
                        prophet, prophet_signals["ldh"], prophet_signals["vaf"], vaf_threshold
                    )
                    engines_used.append("prophet")
            except Exception as e:
                print(f"Prophet error: {e}")
        
        # Oracle
        if phase in (PatientPhase.HEALTHY_SCREENING, PatientPhase.REMISSION_SURVEILLANCE, PatientPhase.UNKNOWN):
            try:
                visits = patient_data.get('visits', [])
                oracle_history = []
                for v in visits:
                    if 'date' in v or 'month' in v:
                        oracle_history.append({
                            "date": v.get("date") or v.get("month"),
                            "blood": v.get("blood", v.get("blood_markers", {})),
                            "clinical": v.get("clinical", {}),
                        })
                
                if len(oracle_history) >= 3:
                    oracle = SentinelOracleV3(oracle_history, patient_id=patient_id)
                    oracle_alerts = oracle.run_oracle(raw_ngs_visits=raw_ngs_visits, max_alerts=3)
                    engines_used.append("oracle")
            except Exception as e:
                print(f"Oracle error: {e}")
        
        # Synthesize
        risk, concern, actions, next_assess = self._synthesize(phase, prophet_fusion, oracle_alerts, patient_data)
        
        return TemporalAnalysis(
            patient_id=patient_id,
            phase=phase,
            phase_confidence=phase_conf,
            engines_used=engines_used,
            prophet_signals=prophet_signals,
            prophet_fusion=prophet_fusion,
            oracle_alerts=oracle_alerts,
            overall_risk_level=risk,
            primary_concern=concern,
            recommended_actions=actions,
            next_assessment=next_assess,
            debug={"phase_indicators": indicators}
        )
    
    def _synthesize(self, phase, prophet_fusion, oracle_alerts, patient_data: Dict[str, Any] = None):
        risk = "LOW"
        concern = "No significant concerns"
        actions = ["Continue routine monitoring"]
        next_assess = "Per standard schedule"
        is_critical_external = False
        
        if patient_data:
            try:
                twin = DigitalTwin(patient_data)
                dt_res = twin.simulate_disease_trajectory()
                if dt_res.mortality_risk_percent > 75 or patient_data.get("baseline", {}).get("ecog_ps", 0) >= 2:
                    is_critical_external = True
            except:
                pass
        
        if prophet_fusion:
            urg = prophet_fusion.urgency
            if urg == Urgency.CRITICAL or is_critical_external:
                risk, concern = "CRITICAL", f"🔴 {prophet_fusion.archetype if urg == Urgency.CRITICAL else 'Severe Clinical Deterioration'}"
                next_assess = "Immediate (1-2 weeks max)"
                if is_critical_external and urg != Urgency.CRITICAL:
                    prophet_fusion.urgency = Urgency.CRITICAL
                    prophet_fusion.action = "Immediate Clinical Intervention Required"
            elif urg == Urgency.HIGH:
                risk, concern = "HIGH", f"🟠 {prophet_fusion.archetype}"
                next_assess = "Soon (4-8 weeks)"
            elif urg == Urgency.MEDIUM:
                risk, concern = "MEDIUM", f"🟡 {prophet_fusion.archetype}"
                next_assess = "Closer monitoring (6-8 weeks)"
            else:
                risk, concern = "LOW", f"🟢 {prophet_fusion.archetype}"
            actions = [prophet_fusion.action]
        else:
            if is_critical_external:
                risk, concern = "CRITICAL", "🔴 Severe Clinical Deterioration"
                next_assess = "Immediate (1-2 weeks max)"
                actions = ["Immediate Clinical Intervention Required"]
        
        if oracle_alerts:
            top = oracle_alerts[0]
            if risk in ("LOW", "MEDIUM"):
                if top.probability >= 70:
                    risk, concern = "HIGH", f"🔮 {top.risk_type}"
                    next_assess = "Accelerated workup"
                elif top.probability >= 40 and risk == "LOW":
                    risk, concern = "MEDIUM", f"🔮 {top.risk_type}"
                    next_assess = "Repeat in 4-8 weeks"
            for alert in oracle_alerts:
                actions.extend(alert.recommended_actions[:2])
        
        return risk, concern, list(dict.fromkeys(actions)), next_assess


def analyze_patient_temporal(
    patient_data: Dict[str, Any],
    raw_ngs_visits: Optional[List[Dict[str, Any]]] = None,
) -> TemporalAnalysis:
    return TemporalEngine().analyze(patient_data, raw_ngs_visits=raw_ngs_visits)


def run_prophet_only(patient_visits: List[Dict[str, Any]]):
    prophet = SentinelProphetV4(patient_visits)
    signals = {k: prophet.analyze_metric(k) for k in ["ldh", "vaf", "crp", "neutrophils"]}
    fusion = FusionEngineV2.diagnose(prophet, signals["ldh"], signals["vaf"])
    return signals, fusion


def run_oracle_only(patient_history, raw_ngs_visits=None, patient_id="UNKNOWN"):
    oracle = SentinelOracleV3(patient_history, patient_id=patient_id)
    return oracle.run_oracle(raw_ngs_visits=raw_ngs_visits)


if __name__ == "__main__":
    print("TEMPORAL ENGINE - DEMO")
    
    patient = {
        "baseline": {"patient_id": "TEST-001", "current_therapy": "Pembrolizumab"},
        "visits": [
            {"week_on_therapy": 0, "blood_markers": {"ldh": 180}, "genetics": {"tp53_vaf": 0.4}},
            {"week_on_therapy": 4, "blood_markers": {"ldh": 200}, "genetics": {"tp53_vaf": 0.8}},
            {"week_on_therapy": 8, "blood_markers": {"ldh": 250}, "genetics": {"tp53_vaf": 1.5}},
            {"week_on_therapy": 12, "blood_markers": {"ldh": 320}, "genetics": {"tp53_vaf": 3.0}},
        ]
    }
    
    result = TemporalEngine().analyze(patient)
    print(f"Risk: {result.overall_risk_level}")
    print(f"Concern: {result.primary_concern}")
    print(f"Actions: {result.recommended_actions}")
