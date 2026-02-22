"""
PROMETHEUS — Oracle Bridge
===========================
Traduce le regole epistatiche scoperte in Evidence objects
per il motore Bayesiano ORACLE.

Flusso:
  1. Carica discovered_rules.json (può essere [])
  2. Per ogni paziente, verifica se le condizioni di una regola sono soddisfatte
  3. Se sì, genera Evidence con LR proporzionale al risk_amplification
  4. ORACLE fonde l'evidenza nella probabilità finale
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("prometheus.bridge")

# Percorso di default per le regole scoperte
DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "discovered_rules.json"


# ═══════════════════════════════════════════════════════════════════
# RULE MATCHER
# ═══════════════════════════════════════════════════════════════════

# Soglie per binarizzare i biomarkers (mediana popolazione tipica)
BIOMARKER_THRESHOLDS = {
    "ldh": 250, "crp": 3.0, "glucose": 100, "albumin": 40,
    "neutrophils": 4000, "lymphocytes": 1.5, "platelets": 250000,
    "hemoglobin": 12.0, "creatinine": 1.2, "cea": 5.0,
    "wbc": 7500, "rdw": 14, "nlr": 3.0, "sii": 500,
    "weight": 70, "bmi": 25, "age": 60, "ecog": 1,
    "smoking": 0.5, "tmb": 10, "pdl1": 50,
}

# Valori "mutato" per SNPs
MUTATED_VALS = {"mutated", "mut", "loss", "amplification", "positive", "yes", "1"}


def _extract_marker_value(marker: str, patient_data: Dict) -> Optional[float]:
    """
    Estrae il valore di un marker dai dati paziente SENTINEL.
    Cerca in: baseline, genetics, blood_markers, clinical_status, visits.
    """
    base = patient_data.get("baseline", patient_data)
    genetics = base.get("genetics", {}) or {}
    blood = base.get("blood_markers", {}) or {}
    biomarkers = base.get("biomarkers", {}) or {}

    # Cerca nelle visits (ultima)
    visits = patient_data.get("visits", []) or []
    last_blood = {}
    last_clinical = {}
    for v in visits:
        for k, val in (v.get("blood_markers", {}) or {}).items():
            if val is not None:
                last_blood[k] = val
        for k, val in (v.get("clinical_status", {}) or {}).items():
            if val is not None:
                last_clinical[k] = val

    # SNP markers (terminano con _mut, _status, ecc.)
    if marker.endswith(("_mut", "_status")):
        # Cerca il gene nel genetics o baseline
        gene_key = marker  # es. "tp53_status"
        val = genetics.get(gene_key) or base.get(gene_key)
        if val is not None:
            s = str(val).lower().strip()
            if s in MUTATED_VALS or (len(s) >= 2 and any(c.isdigit() for c in s)):
                return 1.0
            return 0.0
        return None

    # Copy number
    if marker.endswith("_cn"):
        val = genetics.get(marker) or base.get(marker)
        return float(val) if val is not None else None

    # PGx
    if marker.startswith("pgx_"):
        gene = marker[4:].upper()
        pgx = base.get("pgx_profile", {}) or {}
        allele = pgx.get(gene)
        if allele is not None:
            return 0.0 if str(allele).strip() == "*1/*1" else 1.0
        return None

    # Biomarkers — cerca ovunque
    for source in [last_blood, blood, last_clinical, biomarkers, base]:
        if marker in source and source[marker] is not None:
            return float(source[marker])

    return None


def _is_marker_active(marker: str, value: float, feature_type: str) -> bool:
    """Determina se un marker è 'attivo' (sopra soglia / mutato)."""
    if feature_type == "snp":
        return value > 0
    threshold = BIOMARKER_THRESHOLDS.get(marker)
    if threshold is not None:
        return value > threshold
    # Default: sopra mediana = attivo
    return value > 0


def check_patient_rules(
    patient_data: Dict,
    rules: Optional[List[dict]] = None,
    rules_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Controlla se un paziente matcha le regole epistatiche scoperte.

    Args:
        patient_data: Dati paziente SENTINEL (formato standard)
        rules: Lista di regole (dict). Se None, carica da file.
        rules_path: Percorso al JSON regole. Default: data/discovered_rules.json

    Returns:
        Lista di Evidence-like dicts (vuota se nessun match o nessuna regola).
        Ogni dict ha: key, weight_lr, score, details
    """
    # Carica regole
    if rules is None:
        path = rules_path or str(DEFAULT_RULES_PATH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rules = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # File non esiste o vuoto → nessuna regola → OK
            return []

    if not rules:
        return []

    matched_evidence = []

    for rule in rules:
        markers = rule.get("markers", [])
        if not markers:
            continue

        # Verifica se il paziente ha TUTTI i marker attivi
        all_active = True
        marker_details = []

        for marker in markers:
            value = _extract_marker_value(marker, patient_data)
            if value is None:
                all_active = False
                break

            # Determina il tipo dal rule
            types_str = rule.get("types", "")
            is_snp = marker.endswith(("_mut", "_status", "_cn")) or "snp" in types_str.split("×")[0]
            ftype = "snp" if is_snp else "biomarker"

            if not _is_marker_active(marker, value, ftype):
                all_active = False
                break

            marker_details.append(f"{marker}={value:.2f}")

        if all_active:
            # Calcola LR dal risk amplification
            risk_amp = rule.get("risk_amplification", 1.0)
            cond_risk = rule.get("conditional_risk", 0.5)
            ii = rule.get("interaction_info", 0.0)

            # LR proporzionale: amplificazione 2x → LR 4, 3x → LR 6, etc.
            lr = max(2.0, risk_amp * 2.0)
            # Score basato sulla solidità della regola
            score = min(1.0, ii * 50 + 0.3)

            evidence = {
                "key": f"PROMETHEUS:{'+'.join(markers)}",
                "weight_lr": round(lr, 2),
                "score": round(score, 2),
                "details": (
                    f"Epistatic rule ({rule.get('phase', '?')}): "
                    f"{' + '.join(marker_details)} → "
                    f"risk {cond_risk:.0%} "
                    f"(amp {risk_amp:.1f}x, p_fdr={rule.get('p_value_fdr', '?')})"
                ),
                "phase": rule.get("phase", "unknown"),
                "cond_risk": cond_risk,
            }
            matched_evidence.append(evidence)
            log.info(f"  MATCH: {evidence['key']} | LR={lr:.1f} | {', '.join(marker_details)}")

    return matched_evidence
