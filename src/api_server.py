"""
SENTINEL REST API
==================
FastAPI server for SENTINEL Early Warning System.

Endpoints:
- POST /analyze/patient         Full patient analysis
- POST /analyze/oracle          ORACLE risk assessment
- POST /analyze/pharmacogenomics PGx analysis
- POST /analyze/safety          Clinical safety check
- GET  /health                   Health check

Usage:
    uvicorn api_server:app --reload --port 8000
    
    Or run directly:
    python api_server.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn")
    print("   Running in mock mode...")
    FastAPI = None

# Import SENTINEL modules
from temporal_engine.oracle_v3 import SentinelOracleV3, OracleAlert
from safety_alerts import ClinicalSafetyEngine, ClinicalAlert, AlertSeverity

# Lazy import PGx to avoid slow startup
_pgx_modules = None

def get_pgx_modules():
    global _pgx_modules
    if _pgx_modules is None:
        from farmacogenomica.pgx_extractor import PGxExtractor
        from farmacogenomica.metabolizer_classifier import MetabolizerClassifier
        _pgx_modules = {
            'extractor': PGxExtractor(),
            'classifier': MetabolizerClassifier()
        }
    return _pgx_modules


# =============================================================================
# MODELS
# =============================================================================

class PatientData(BaseModel):
    """Patient input data model"""
    patient_id: str = Field(default="UNKNOWN")
    baseline: Dict[str, Any] = Field(default_factory=dict)
    visits: List[Dict[str, Any]] = Field(default_factory=list)
    raw_ngs_visits: Optional[List[Dict[str, Any]]] = None

class OracleRequest(BaseModel):
    """ORACLE analysis request"""
    patient_data: PatientData
    max_alerts: int = Field(default=3, ge=1, le=10)

class SafetyRequest(BaseModel):
    """Safety analysis request"""
    patient_data: PatientData

class PGxRequest(BaseModel):
    """Pharmacogenomics analysis request"""
    patient_data: PatientData
    check_drug: Optional[str] = None

class FullAnalysisRequest(BaseModel):
    """Full patient analysis request"""
    patient_data: PatientData
    include_oracle: bool = True
    include_pgx: bool = True
    include_safety: bool = True
    max_oracle_alerts: int = 3


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class OracleAlertResponse(BaseModel):
    risk_type: str
    probability: float
    confidence_level: str
    confidence_score: float
    lead_time: str
    summary: str
    recommended_actions: List[str]
    signal_sources: List[Dict[str, Any]]

class SafetyAlertResponse(BaseModel):
    category: str
    severity: str
    title: str
    message: str
    recommended_actions: List[str]
    requires_immediate_action: bool

class PGxVariantResponse(BaseModel):
    gene: str
    genotype: str
    phenotype: Optional[str] = None
    recommendations: List[str] = []

class FullAnalysisResponse(BaseModel):
    patient_id: str
    analysis_timestamp: str
    oracle_alerts: List[OracleAlertResponse] = []
    safety_alerts: List[SafetyAlertResponse] = []
    pgx_variants: List[PGxVariantResponse] = []
    summary: Dict[str, Any] = {}


# =============================================================================
# API APP
# =============================================================================

if FastAPI:
    app = FastAPI(
        title="SENTINEL API",
        description="Advanced Bayesian Early Warning System for Cancer Detection",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # ENDPOINTS
    # =========================================================================

    @app.get("/")
    async def root():
        return {
            "name": "SENTINEL API",
            "version": "3.0.0",
            "status": "operational",
            "endpoints": [
                "/analyze/patient",
                "/analyze/oracle",
                "/analyze/safety",
                "/analyze/pharmacogenomics",
                "/health"
            ]
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "oracle": "ready",
                "safety": "ready",
                "pharmacogenomics": "lazy-loaded"
            }
        }

    @app.post("/analyze/oracle", response_model=List[OracleAlertResponse])
    async def analyze_oracle(request: OracleRequest):
        """Run ORACLE risk assessment on patient data."""
        try:
            patient = request.patient_data.dict()
            history = patient.get('visits', [])
            
            if not history and patient.get('baseline'):
                history = [patient['baseline']]
            
            oracle = SentinelOracleV3(
                patient_history=history,
                patient_id=patient.get('patient_id', 'API_PATIENT')
            )
            
            raw_ngs = patient.get('raw_ngs_visits')
            alerts = oracle.run_oracle(
                raw_ngs_visits=raw_ngs,
                max_alerts=request.max_alerts
            )
            
            return [
                OracleAlertResponse(
                    risk_type=a.risk_type,
                    probability=a.probability,
                    confidence_level=a.confidence.level.value,
                    confidence_score=a.confidence.score,
                    lead_time=a.lead_time,
                    summary=a.summary,
                    recommended_actions=a.recommended_actions,
                    signal_sources=[
                        {"key": e.key, "weight": e.weight_lr, "score": e.score, "details": e.details}
                        for e in a.signal_sources
                    ]
                )
                for a in alerts
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/safety", response_model=List[SafetyAlertResponse])
    async def analyze_safety(request: SafetyRequest):
        """Run clinical safety checks on patient data."""
        try:
            patient = request.patient_data.dict()
            engine = ClinicalSafetyEngine()
            
            alerts = engine.run_full_safety_check(patient)
            
            return [
                SafetyAlertResponse(
                    category=a.category.value,
                    severity=a.severity.value,
                    title=a.title,
                    message=a.message,
                    recommended_actions=a.recommended_actions,
                    requires_immediate_action=a.requires_immediate_action
                )
                for a in alerts
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/pharmacogenomics", response_model=List[PGxVariantResponse])
    async def analyze_pharmacogenomics(request: PGxRequest):
        """Run pharmacogenomics analysis on patient data."""
        try:
            patient = request.patient_data.dict()
            pgx = get_pgx_modules()
            
            extractor = pgx['extractor']
            classifier = pgx['classifier']
            
            result = extractor.extract_from_sentinel(patient)
            
            responses = []
            for variant in result.variants_found:
                phenotype_result = classifier.classify(variant.gene, variant.genotype)
                
                responses.append(PGxVariantResponse(
                    gene=variant.gene,
                    genotype=variant.genotype,
                    phenotype=phenotype_result.phenotype.value if phenotype_result else None,
                    recommendations=result.recommendations if result.recommendations else []
                ))
            
            return responses
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/patient", response_model=FullAnalysisResponse)
    async def analyze_patient(request: FullAnalysisRequest):
        """Run complete patient analysis (ORACLE + Safety + PGx)."""
        try:
            patient = request.patient_data.dict()
            patient_id = patient.get('patient_id', 'API_PATIENT')
            
            response = FullAnalysisResponse(
                patient_id=patient_id,
                analysis_timestamp=datetime.now().isoformat(),
                summary={}
            )
            
            # ORACLE Analysis
            if request.include_oracle:
                history = patient.get('visits', [])
                if not history and patient.get('baseline'):
                    history = [patient['baseline']]
                
                oracle = SentinelOracleV3(
                    patient_history=history,
                    patient_id=patient_id
                )
                
                raw_ngs = patient.get('raw_ngs_visits')
                alerts = oracle.run_oracle(
                    raw_ngs_visits=raw_ngs,
                    max_alerts=request.max_oracle_alerts
                )
                
                response.oracle_alerts = [
                    OracleAlertResponse(
                        risk_type=a.risk_type,
                        probability=a.probability,
                        confidence_level=a.confidence.level.value,
                        confidence_score=a.confidence.score,
                        lead_time=a.lead_time,
                        summary=a.summary,
                        recommended_actions=a.recommended_actions,
                        signal_sources=[
                            {"key": e.key, "weight": e.weight_lr, "score": e.score, "details": e.details}
                            for e in a.signal_sources
                        ]
                    )
                    for a in alerts
                ]
                response.summary['oracle_risk_count'] = len(alerts)
                if alerts:
                    response.summary['max_oracle_probability'] = max(a.probability for a in alerts)
            
            # Safety Analysis
            if request.include_safety:
                engine = ClinicalSafetyEngine()
                safety_alerts = engine.run_full_safety_check(patient)
                
                response.safety_alerts = [
                    SafetyAlertResponse(
                        category=a.category.value,
                        severity=a.severity.value,
                        title=a.title,
                        message=a.message,
                        recommended_actions=a.recommended_actions,
                        requires_immediate_action=a.requires_immediate_action
                    )
                    for a in safety_alerts
                ]
                response.summary['safety_alert_count'] = len(safety_alerts)
                response.summary['critical_safety_alerts'] = sum(
                    1 for a in safety_alerts if a.severity == AlertSeverity.CRITICAL
                )
            
            # PGx Analysis
            if request.include_pgx:
                pgx = get_pgx_modules()
                extractor = pgx['extractor']
                classifier = pgx['classifier']
                
                result = extractor.extract_from_sentinel(patient)
                
                response.pgx_variants = []
                for variant in result.variants_found:
                    phenotype_result = classifier.classify(variant.gene, variant.genotype)
                    response.pgx_variants.append(PGxVariantResponse(
                        gene=variant.gene,
                        genotype=variant.genotype,
                        phenotype=phenotype_result.phenotype.value if phenotype_result else None,
                        recommendations=result.recommendations if result.recommendations else []
                    ))
                response.summary['pgx_variants_found'] = len(result.variants_found)
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

else:
    # Mock app for when FastAPI is not installed
    app = None
    print("⚠️ FastAPI not available - API server cannot run")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if app:
        try:
            import uvicorn
            print("🚀 Starting SENTINEL API server...")
            print("   Documentation: http://localhost:8000/docs")
            print("   Alternative docs: http://localhost:8000/redoc")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("❌ uvicorn not installed. Install with: pip install uvicorn")
    else:
        print("❌ Cannot start server - FastAPI not installed")
        print("   Install with: pip install fastapi uvicorn")
