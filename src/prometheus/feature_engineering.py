"""
PROMETHEUS — Feature Engineering per dati pazienti SENTINEL.
=============================================================
Legge i JSON pazienti dal database e costruisce una matrice
features con tipi annotati (snp, biomarker, lifestyle, derived, prs).

Supporta formati eterogenei: pazienti minimali (solo baseline)
e pazienti ricchi (baseline + visits + genetics + blood_markers).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("prometheus.features")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════════════

# Chiavi genetiche riconosciute nei JSON pazienti
# Mappa: chiave_json → (nome_feature, tipo)
GENETIC_KEYS = {
    # Mutazioni driver
    "kras_mutation": ("kras_mut", "snp"),
    "egfr_mutation": ("egfr_mut", "snp"),
    "tp53_status": ("tp53_status", "snp"),
    "stk11_status": ("stk11_status", "snp"),
    "keap1_status": ("keap1_status", "snp"),
    "rb1_status": ("rb1_status", "snp"),
    "met_status": ("met_status", "snp"),
    "braf_mutation": ("braf_mut", "snp"),
    "alk_status": ("alk_status", "snp"),
    "ros1_status": ("ros1_status", "snp"),
    "her2_status": ("her2_status", "snp"),
    "ret_status": ("ret_status", "snp"),
    "nras_mutation": ("nras_mut", "snp"),
    "pik3ca_status": ("pik3ca_mut", "snp"),
    "pik3ca_mutation": ("pik3ca_mut", "snp"),
    # Copy number
    "met_cn": ("met_cn", "snp"),
    "her2_cn": ("her2_cn", "snp"),
    # Scores
    "tmb_score": ("tmb", "biomarker"),
    "pdl1_percent": ("pdl1", "biomarker"),
}

# Biomarker chiavi dai blood_markers
BLOOD_MARKER_KEYS = [
    "ldh", "crp", "glucose", "albumin", "neutrophils", "lymphocytes",
    "monocytes", "platelets", "hemoglobin", "potassium", "sodium",
    "creatinine", "calcium", "cea", "alt", "alp", "hba1c",
    "wbc", "rdw", "mpv", "igf1", "vitamin_d", "testosterone",
    "triglycerides", "hdl", "ldl",
]

# Valori che indicano "mutato" / "positivo"
MUTATED_VALUES = {"mutated", "mut", "loss", "amplification", "positive", "yes", "1"}


def _is_mutated(val) -> float:
    """Converte un valore genetico in 0.0/1.0."""
    if val is None:
        return np.nan
    s = str(val).lower().strip()
    if s in ("", "none", "null", "nan", "wt", "wild-type", "negative", "no", "0"):
        return 0.0
    if s in MUTATED_VALUES:
        return 1.0
    # Se è un nome di mutazione specifico (es. "G12C", "L858R") → mutato
    if len(s) >= 2 and any(c.isdigit() for c in s):
        return 1.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# ESTRAZIONE FEATURES DA SINGOLO PAZIENTE
# ═══════════════════════════════════════════════════════════════════

def extract_patient_features(patient_data: Dict) -> Dict[str, float]:
    """
    Estrae un vettore di features da un singolo paziente SENTINEL.

    Gestisce sia pazienti minimali (solo baseline) che ricchi
    (baseline + visits + genetics + blood_markers).

    Returns:
        Dict[str, float] con chiave = nome feature, valore = float/nan
    """
    features: Dict[str, float] = {}
    base = patient_data.get("baseline", patient_data)

    # === Demographics ===
    features["age"] = float(base.get("age", np.nan) or np.nan)
    features["sex"] = 1.0 if str(base.get("sex", "")).upper() == "M" else 0.0

    # ECOG (vari formati)
    ecog = base.get("ecog_ps") or base.get("ecog") or base.get("ps")
    features["ecog"] = float(ecog) if ecog is not None else np.nan

    # Smoking
    smoking_map = {"never": 0, "former": 1, "current": 2, "ex": 1}
    smoking = str(base.get("smoking_status", "") or "").lower()
    features["smoking"] = float(smoking_map.get(smoking, np.nan))

    # === Genetics (baseline level) ===
    genetics = base.get("genetics", {}) or {}
    # Merge top-level genetics into genetics dict
    for key in GENETIC_KEYS:
        if key in base and key not in genetics:
            genetics[key] = base[key]

    for json_key, (feat_name, _ftype) in GENETIC_KEYS.items():
        val = genetics.get(json_key)
        if val is None:
            val = base.get(json_key)
        if feat_name.endswith("_cn"):
            new_val = float(val) if val is not None else np.nan
        elif feat_name in ("tmb", "pdl1"):
            new_val = float(val) if val is not None else np.nan
        else:
            new_val = _is_mutated(val)

        # Avoid overwriting a valid feature with NaN from a secondary alias
        if feat_name not in features or pd.isna(features[feat_name]) or (not pd.isna(new_val) and new_val > 0):
            features[feat_name] = new_val

    # === PGx profile ===
    pgx = base.get("pgx_profile", {}) or {}
    for gene, allele in pgx.items():
        feat = f"pgx_{gene.lower()}"
        # Encode: *1/*1 = 0 (normal), anything else = 1 (variant)
        features[feat] = 0.0 if str(allele).strip() == "*1/*1" else 1.0

    # === Blood markers (ultimo valore disponibile) ===
    # Cerca nella visita più recente, poi in baseline
    last_blood = {}

    # Da baseline.blood_markers
    base_blood = base.get("blood_markers", {}) or {}
    last_blood.update(base_blood)

    # Da visits (ultima visita sovrascrive)
    visits = patient_data.get("visits", []) or []
    for visit in visits:
        vb = visit.get("blood_markers", {}) or {}
        for k, v in vb.items():
            if v is not None and isinstance(v, (int, float)):
                last_blood[k] = v

    for key in BLOOD_MARKER_KEYS:
        val = last_blood.get(key)
        features[key] = float(val) if val is not None else np.nan

    # NLR (può essere pre-calcolato o calcolato)
    if "nlr" in last_blood:
        features["nlr_raw"] = float(last_blood["nlr"])

    # === Clinical status (ultimo) ===
    last_clinical = {}
    for visit in visits:
        vc = visit.get("clinical_status", {}) or {}
        last_clinical.update({k: v for k, v in vc.items() if v is not None})

    features["weight"] = float(last_clinical.get("weight_kg", np.nan) or np.nan)
    features["bmi"] = float(last_clinical.get("bmi", np.nan) or np.nan)

    # === Outcome proxy ===
    # Se il paziente ha una diagnosi confermata → outcome=1
    stage = str(base.get("stage", "") or "").strip()
    has_therapy = bool(base.get("current_therapy") or base.get("therapy"))
    histology = str(base.get("histology", "") or "").strip()
    features["has_cancer"] = 1.0 if (stage and stage != "none") or has_therapy or histology else 0.0

    return features


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (derivate + PRS)
# ═══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Crea feature derivate clinicamente rilevanti da un DataFrame grezzo.

    Returns:
        (df_engineered, feature_types_dict)
    """
    df = df.copy()
    ft: Dict[str, str] = {}

    # === Inflammation indices ===
    if "neutrophils" in df and "lymphocytes" in df:
        denom = df["lymphocytes"].replace(0, np.nan) + 0.01
        df["nlr"] = df["neutrophils"] / denom
        ft["nlr"] = "derived"

    if "platelets" in df and "lymphocytes" in df:
        denom = df["lymphocytes"].replace(0, np.nan) + 0.01
        df["plr"] = df["platelets"] / denom
        ft["plr"] = "derived"

        if "neutrophils" in df:
            df["sii"] = df["platelets"] * df["neutrophils"] / denom
            ft["sii"] = "derived"

    if "monocytes" in df and "lymphocytes" in df:
        denom = df["lymphocytes"].replace(0, np.nan) + 0.01
        df["mlr"] = df["monocytes"] / denom
        ft["mlr"] = "derived"

    # === Glasgow Prognostic Score ===
    if "crp" in df and "albumin" in df:
        df["mgps"] = 0.0
        df.loc[(df["crp"] > 10) & (df["albumin"] < 35), "mgps"] = 2.0
        mask_1 = ((df["crp"] > 10) | (df["albumin"] < 35)) & (df["mgps"] != 2)
        df.loc[mask_1, "mgps"] = 1.0
        ft["mgps"] = "derived"

    # === PRS compositi ===
    snp_cols = [c for c in df.columns if any(
        g in c for g in ["kras", "egfr", "tp53", "stk11", "braf", "alk", "ret", "nras"]
    ) and c.endswith(("_mut", "_status"))]

    dna_repair = [c for c in snp_cols if any(g in c for g in ["tp53", "rb1"])]
    if dna_repair:
        df["prs_dna_repair"] = df[dna_repair].sum(axis=1, min_count=1)
        ft["prs_dna_repair"] = "prs"

    driver = [c for c in snp_cols if any(g in c for g in ["kras", "egfr", "braf", "alk"])]
    if driver:
        df["prs_driver"] = df[driver].sum(axis=1, min_count=1)
        ft["prs_driver"] = "prs"

    immune_markers = [c for c in df.columns if c.startswith("pgx_")]
    if immune_markers:
        df["prs_pgx_variants"] = df[immune_markers].sum(axis=1, min_count=1)
        ft["prs_pgx_variants"] = "prs"

    # === Classify unclassified features ===
    for c in df.columns:
        if c in ft:
            continue
        if c.endswith(("_mut", "_status")) or c.startswith("pgx_"):
            ft[c] = "snp"
        elif c in BLOOD_MARKER_KEYS or c in ("tmb", "pdl1", "nlr_raw"):
            ft[c] = "biomarker"
        elif c in ("age", "sex", "ecog", "smoking", "weight", "bmi"):
            ft[c] = "lifestyle"
        elif c in ("has_cancer",):
            ft[c] = "outcome"
        elif c.endswith("_cn"):
            ft[c] = "snp"

    n_derived = sum(1 for v in ft.values() if v == "derived")
    n_prs = sum(1 for v in ft.values() if v == "prs")
    n_snp = sum(1 for v in ft.values() if v == "snp")
    n_bio = sum(1 for v in ft.values() if v == "biomarker")
    log.info(f"Engineered: {n_derived} derived, {n_prs} PRS, {n_snp} SNPs, {n_bio} biomarkers")

    return df, ft


# ═══════════════════════════════════════════════════════════════════
# LOADER: Database pazienti → DataFrame
# ═══════════════════════════════════════════════════════════════════

def load_patient_database(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Carica tutti i JSON pazienti e costruisce il DataFrame features.

    Args:
        data_dir: Percorso alla directory con i JSON pazienti

    Returns:
        (df_features, feature_types, patient_ids)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        log.warning(f"Directory non trovata: {data_dir}")
        return pd.DataFrame(), {}, []

    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        log.warning(f"Nessun file JSON in {data_dir}")
        return pd.DataFrame(), {}, []

    rows = []
    patient_ids = []
    skipped = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                patient_data = json.load(f)
            features = extract_patient_features(patient_data)
            pid = patient_data.get("baseline", {}).get("patient_id", fpath.stem)
            features["_patient_id"] = pid
            rows.append(features)
            patient_ids.append(pid)
        except Exception as e:
            log.debug(f"Skipping {fpath.name}: {e}")
            skipped += 1

    if not rows:
        log.warning("Nessun paziente caricato")
        return pd.DataFrame(), {}, []

    df_raw = pd.DataFrame(rows)

    # Drop _patient_id from features (not a feature)
    if "_patient_id" in df_raw.columns:
        df_raw = df_raw.drop(columns=["_patient_id"])

    # Engineering
    df_eng, feat_types = engineer_features(df_raw)

    log.info(
        f"Caricati {len(rows)} pazienti ({skipped} skipped) | "
        f"{len(df_eng.columns)} features"
    )
    return df_eng, feat_types, patient_ids
