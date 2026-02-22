"""
SENTINEL Test Suite - Pytest Fixtures
======================================
Shared fixtures for SENTINEL testing.
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# PATIENT FIXTURES
# =============================================================================

@pytest.fixture
def oracle_drift_patient():
    """Patient with healthy baseline progressing to cancer."""
    return {
        "baseline": {
            "patient_id": "TEST_ORACLE_001",
            "age": 62,
            "sex": "M",
            "diagnosis_date": "2024-02-15"
        },
        "visits": [
            {
                "date": "2020-03-15",
                "blood_markers": {"ldh": 145, "crp": 0.3, "cea": 1.8},
                "clinical_status": {"weight_kg": 82},
                "noise_variants": []
            },
            {
                "date": "2022-06-10",
                "blood_markers": {"ldh": 185, "crp": 1.5, "cea": 4.2},
                "clinical_status": {"weight_kg": 80},
                "noise_variants": [{"gene": "KRAS", "vaf": 0.08}]
            },
            {
                "date": "2023-02-15",
                "blood_markers": {"ldh": 220, "crp": 3.2, "cea": 8.5},
                "clinical_status": {"weight_kg": 77},
                "noise_variants": [
                    {"gene": "KRAS", "vaf": 0.5},
                    {"gene": "TP53", "vaf": 0.15}
                ]
            },
            {
                "date": "2024-02-15",
                "blood_markers": {"ldh": 420, "crp": 14.0, "cea": 52.0},
                "clinical_status": {"weight_kg": 68},
                "noise_variants": [
                    {"gene": "KRAS", "vaf": 32.0},
                    {"gene": "TP53", "vaf": 28.0}
                ]
            }
        ]
    }


@pytest.fixture
def pgx_poor_metabolizer_patient():
    """Patient with DPYD *2A/*2A on FOLFOX - contraindicated."""
    return {
        "baseline": {
            "patient_id": "TEST_PGX_001",
            "age": 55,
            "sex": "F",
            "current_therapy": "FOLFOX (5-FU + Oxaliplatino)",
            "pgx_profile": {
                "DPYD": "*2A/*2A",
                "UGT1A1": "*28/*28",
                "CYP2D6": "*1/*1"
            }
        }
    }


@pytest.fixture
def neutropenic_fever_patient():
    """Patient with neutropenic fever."""
    return {
        "baseline": {
            "patient_id": "TEST_NEUTRO_001",
            "age": 45,
            "sex": "F",
            "current_therapy": "FOLFOX"
        },
        "visits": [{
            "date": "2024-01-15",
            "blood_markers": {
                "neutrophils": 250,
                "platelets": 85000,
                "hemoglobin": 9.5
            },
            "clinical_status": {
                "temperature": 38.8,
                "blood_pressure_systolic": 95,
                "respiratory_rate": 24
            }
        }]
    }


@pytest.fixture
def critical_labs_patient():
    """Patient with multiple critical lab values."""
    return {
        "baseline": {
            "patient_id": "TEST_LABS_001",
            "age": 70,
            "sex": "M"
        },
        "visits": [{
            "date": "2024-01-15",
            "blood_markers": {
                "potassium": 6.8,
                "sodium": 118,
                "hemoglobin": 6.2,
                "platelets": 12000
            }
        }]
    }


@pytest.fixture
def drug_interaction_patient():
    """Patient with dangerous drug combinations."""
    return {
        "baseline": {
            "patient_id": "TEST_DDI_001",
            "age": 65,
            "sex": "M",
            "current_therapy": "Capecitabine",
            "medications": ["Warfarin", "Methotrexate", "Ibuprofen"]
        }
    }


@pytest.fixture
def qtc_risk_patient():
    """Patient on multiple QT-prolonging drugs."""
    return {
        "baseline": {
            "patient_id": "TEST_QTC_001",
            "age": 58,
            "sex": "F",
            "current_therapy": "Osimertinib",
            "medications": ["Ondansetron", "Haloperidol"]
        },
        "visits": [{
            "blood_markers": {"potassium": 3.2, "magnesium": 1.4}
        }]
    }


@pytest.fixture
def tls_risk_patient():
    """Patient at high risk for Tumor Lysis Syndrome."""
    return {
        "baseline": {
            "patient_id": "TEST_TLS_001",
            "histology": "Burkitt Lymphoma",
            "stage": "IV"
        },
        "visits": [{
            "blood_markers": {
                "ldh": 950,
                "uric_acid": 10.2,
                "creatinine": 2.1,
                "potassium": 5.8,
                "phosphate": 6.2
            }
        }]
    }


@pytest.fixture
def vte_risk_patient():
    """Patient with high VTE risk (Khorana score)."""
    return {
        "baseline": {
            "patient_id": "TEST_VTE_001",
            "primary_site": "Pancreatic adenocarcinoma"
        },
        "visits": [{
            "blood_markers": {
                "platelets": 450000,
                "hemoglobin": 9.2,
                "leukocytes": 13500
            },
            "clinical_status": {
                "weight_kg": 95,
                "height_cm": 170
            }
        }]
    }


# =============================================================================
# DATA LOADING
# =============================================================================

@pytest.fixture
def load_patient_file():
    """Factory fixture to load patient JSON files."""
    def _load(filename: str):
        path = PROJECT_ROOT / "data" / "patients" / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    return _load
