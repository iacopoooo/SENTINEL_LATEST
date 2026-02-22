"""
SENTINEL TEMPORAL ENGINE
=========================
Prophet + Oracle = Superpoteri temporali
"""

from .prophet_v4 import (
    SentinelProphetV4,
    TemporalSignal,
    ConfidenceScore,
    ConfidenceLevel,
    ConfidenceEngineV2,
    FusionEngineV2,
    FusionResult,
    Urgency,
)

from .oracle_v3 import (
    SentinelOracleV3,
    OracleAlert,
    Evidence,
)

from .temporal_fusion import (
    TemporalEngine,
    TemporalAnalysis,
    PatientPhase,
    detect_patient_phase,
    analyze_patient_temporal,
    run_prophet_only,
    run_oracle_only,
)

__version__ = "1.0.0"
__all__ = [
    # Prophet
    "SentinelProphetV4",
    "TemporalSignal", 
    "ConfidenceScore",
    "ConfidenceLevel",
    "ConfidenceEngineV2",
    "FusionEngineV2",
    "FusionResult",
    "Urgency",
    # Oracle
    "SentinelOracleV3",
    "OracleAlert",
    "Evidence",
    # Fusion
    "TemporalEngine",
    "TemporalAnalysis",
    "PatientPhase",
    "detect_patient_phase",
    "analyze_patient_temporal",
    "run_prophet_only",
    "run_oracle_only",
]
