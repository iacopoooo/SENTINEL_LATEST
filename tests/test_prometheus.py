"""
Test suite per PROMETHEUS — Epistatic Discovery Engine.
=======================================================
Test difensivi che verificano:
  - Feature extraction da patient JSON
  - Information gain + binarizzazione
  - FDR (Benjamini-Hochberg)
  - Discovery con dataset piccoli (graceful degradation)
  - Oracle bridge (regola → Evidence)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from prometheus.feature_engineering import extract_patient_features, engineer_features
from prometheus.epistatic_engine import (
    information_gain, binarize, permutation_test,
    benjamini_hochberg, discover_epistatic, DiscoveryResult,
)
from prometheus.oracle_bridge import check_patient_rules


# ═══════════════════════════════════════════════════════════════════
# TEST: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Test estrazione features da JSON pazienti."""

    def test_minimal_patient(self):
        """Paziente con solo baseline → features valide."""
        patient = {
            "baseline": {
                "patient_id": "TEST-001",
                "age": 55,
                "sex": "M",
                "tp53_status": "mutated",
            }
        }
        features = extract_patient_features(patient)
        assert features["age"] == 55.0
        assert features["sex"] == 1.0
        assert features["tp53_status"] == 1.0

    def test_rich_patient(self):
        """Paziente con visits e blood_markers."""
        patient = {
            "baseline": {
                "patient_id": "TEST-002",
                "age": 68,
                "sex": "F",
                "genetics": {
                    "tp53_status": "wt",
                    "kras_mutation": "G12C",
                },
            },
            "visits": [{
                "blood_markers": {"ldh": 450, "crp": 8.5, "neutrophils": 6000},
                "clinical_status": {"weight_kg": 72},
            }],
        }
        features = extract_patient_features(patient)
        assert features["sex"] == 0.0
        assert features["tp53_status"] == 0.0
        assert features["kras_mut"] == 1.0  # G12C → mutato
        assert features["ldh"] == 450
        assert features["crp"] == 8.5

    def test_empty_patient(self):
        """Paziente vuoto → features con NaN, no crash."""
        features = extract_patient_features({})
        assert isinstance(features, dict)
        assert np.isnan(features.get("age", np.nan))

    def test_mutation_name_detected(self):
        """Nome mutazione specifico (es. G12C) → mutato."""
        patient = {"baseline": {"kras_mutation": "G12D"}}
        features = extract_patient_features(patient)
        assert features["kras_mut"] == 1.0

    def test_wt_not_mutated(self):
        """wt / wild-type → non mutato."""
        patient = {"baseline": {"tp53_status": "wt"}}
        features = extract_patient_features(patient)
        assert features["tp53_status"] == 0.0


class TestEngineerFeatures:
    """Test feature engineering derivate."""

    def test_nlr_calculation(self):
        """NLR calcolato da neutrophils/lymphocytes."""
        df = pd.DataFrame({
            "neutrophils": [6000.0, 3000.0],
            "lymphocytes": [2000.0, 1000.0],
        })
        df_eng, ft = engineer_features(df)
        assert "nlr" in df_eng.columns
        assert ft["nlr"] == "derived"
        assert abs(df_eng["nlr"].iloc[0] - 3.0) < 0.1

    def test_glasgow_score(self):
        """Glasgow Prognostic Score calcolato."""
        df = pd.DataFrame({
            "crp": [15.0, 2.0],
            "albumin": [30.0, 45.0],
        })
        df_eng, ft = engineer_features(df)
        assert "mgps" in df_eng.columns
        assert df_eng["mgps"].iloc[0] == 2.0  # CRP>10 AND albumin<35
        assert df_eng["mgps"].iloc[1] == 0.0  # Normal


# ═══════════════════════════════════════════════════════════════════
# TEST: STATISTICAL TOOLS
# ═══════════════════════════════════════════════════════════════════

class TestStatisticalTools:
    """Test information gain, binarizzazione, FDR."""

    def test_information_gain_perfect(self):
        """IG massimo quando x predice perfettamente y."""
        x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ig = information_gain(x, y)
        assert ig > 0.9  # Quasi 1.0

    def test_information_gain_random(self):
        """IG ~0 quando x è indipendente da y."""
        rng = np.random.RandomState(42)
        x = rng.randint(0, 2, 1000)
        y = rng.randint(0, 2, 1000)
        ig = information_gain(x, y)
        assert ig < 0.01

    def test_information_gain_empty(self):
        """IG = 0 con array vuoto."""
        assert information_gain(np.array([]), np.array([])) == 0.0

    def test_binarize_snp(self):
        """SNP: >0 → 1."""
        x = np.array([0, 1, 2, 0, 1])
        b = binarize(x, "snp")
        assert list(b) == [0, 1, 1, 0, 1]

    def test_binarize_biomarker(self):
        """Biomarker: > mediana → 1."""
        x = np.array([1, 2, 3, 4, 5])
        b = binarize(x, "biomarker")
        # mediana = 3, > 3 → 1
        assert list(b) == [0, 0, 0, 1, 1]

    def test_fdr_all_significant(self):
        """FDR: tutti p<0.01 → tutti significativi."""
        p = [0.001, 0.002, 0.003]
        sig = benjamini_hochberg(p, alpha=0.05)
        assert all(sig)

    def test_fdr_none_significant(self):
        """FDR: tutti p>0.5 → nessuno significativo."""
        p = [0.6, 0.7, 0.8]
        sig = benjamini_hochberg(p, alpha=0.05)
        assert not any(sig)

    def test_fdr_empty(self):
        """FDR: lista vuota → lista vuota."""
        assert benjamini_hochberg([]) == []


# ═══════════════════════════════════════════════════════════════════
# TEST: DISCOVERY ENGINE (DEFENSIVE)
# ═══════════════════════════════════════════════════════════════════

class TestDiscoveryEngine:
    """Test discovery con dataset piccoli."""

    def test_too_small_dataset(self):
        """N < 20 → result vuoto con warning, no crash."""
        df = pd.DataFrame({"snp_a": [1, 0, 1], "bio_b": [3, 2, 5]})
        y = np.array([1, 0, 1])
        ft = {"snp_a": "snp", "bio_b": "biomarker"}

        result = discover_epistatic(df, y, ft)
        assert isinstance(result, DiscoveryResult)
        assert len(result.all_rules) == 0
        assert len(result.warnings) > 0

    def test_imbalanced_dataset(self):
        """Tutti casi o tutti controlli → warning, no crash."""
        df = pd.DataFrame({"snp_a": [1, 0, 1] * 10, "bio_b": [3, 2, 5] * 10})
        y = np.ones(30)  # Tutti casi
        ft = {"snp_a": "snp", "bio_b": "biomarker"}

        result = discover_epistatic(df, y, ft)
        assert isinstance(result, DiscoveryResult)
        assert len(result.all_rules) == 0

    def test_json_serialization(self, tmp_path):
        """DiscoveryResult.to_json produce JSON valido."""
        result = DiscoveryResult(n_patients=0, n_features=0)
        outfile = str(tmp_path / "rules.json")
        result.to_json(outfile)

        # Verifica che il file contiene []
        loaded = DiscoveryResult.load_rules(outfile)
        assert loaded == []

    def test_load_missing_file(self):
        """DiscoveryResult.load_rules su file inesistente → []."""
        rules = DiscoveryResult.load_rules("/non/esiste.json")
        assert rules == []


# ═══════════════════════════════════════════════════════════════════
# TEST: ORACLE BRIDGE
# ═══════════════════════════════════════════════════════════════════

class TestOracleBridge:
    """Test ponte PROMETHEUS → ORACLE."""

    def test_no_rules_no_evidence(self):
        """Se rules=[], nessuna evidence generata."""
        patient = {
            "baseline": {"tp53_status": "mutated", "blood_markers": {"ldh": 500}}
        }
        evidence = check_patient_rules(patient, rules=[])
        assert evidence == []

    def test_matching_rule(self):
        """Paziente che matcha una regola → evidence generata."""
        rules = [{
            "markers": ["tp53_status", "ldh"],
            "types": "snp×biomarker",
            "phase": "B_snp×bio",
            "conditional_risk": 0.80,
            "risk_amplification": 2.5,
            "interaction_info": 0.015,
            "p_value_fdr": 0.01,
        }]
        patient = {
            "baseline": {
                "tp53_status": "mutated",
                "blood_markers": {"ldh": 500},
            }
        }
        evidence = check_patient_rules(patient, rules=rules)
        assert len(evidence) == 1
        assert "PROMETHEUS" in evidence[0]["key"]
        assert evidence[0]["weight_lr"] > 2.0

    def test_non_matching_rule(self):
        """Paziente senza TP53 mutato → nessun match."""
        rules = [{
            "markers": ["tp53_status", "ldh"],
            "types": "snp×biomarker",
            "phase": "B_snp×bio",
            "conditional_risk": 0.80,
            "risk_amplification": 2.0,
            "interaction_info": 0.01,
        }]
        patient = {
            "baseline": {
                "tp53_status": "wt",
                "blood_markers": {"ldh": 500},
            }
        }
        evidence = check_patient_rules(patient, rules=rules)
        assert evidence == []

    def test_missing_rules_file(self):
        """File regole inesistente → nessuna evidence, no crash."""
        patient = {"baseline": {}}
        evidence = check_patient_rules(patient, rules_path="/tmp/non_esiste_xyz.json")
        assert evidence == []
