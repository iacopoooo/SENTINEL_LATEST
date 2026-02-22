#!/usr/bin/env python3
"""
SENTINEL ML WEIGHTS MODULE v2.0
===============================
Carica e applica i pesi ML-validated per il calcolo del risk score.

Uso:
    from sentinel_ml_weights import calculate_risk_score, load_weights
    
    weights = load_weights()
    score = calculate_risk_score(patient_data, weights)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Path default per i pesi
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "sentinel_unified_weights.json"


def load_weights(filepath: Optional[Path] = None) -> Dict:
    """Carica i pesi dal file JSON"""
    
    if filepath is None:
        filepath = DEFAULT_WEIGHTS_PATH
    
    if not filepath.exists():
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_clinical_score(patient: Dict, weights: Dict, model: str = "pfs") -> Tuple[int, Dict]:
    """
    Calcola lo score clinico ML-validated.
    
    Args:
        patient: Dict con dati paziente (stage, age, sex, tmb_score, etc.)
        weights: Dict con pesi caricati
        model: "os" o "pfs"
    
    Returns:
        score: Score numerico
        breakdown: Dict con contributi di ogni feature
    """
    
    clinical_weights = weights.get(f"clinical_weights_{model}", weights.get("clinical_weights_pfs", {}))
    thresholds = weights.get("thresholds", {})
    
    score = 0
    breakdown = {}
    
    # Stage
    stage = str(patient.get("stage", "")).upper()
    if "IV" in stage or "4" in stage:
        score += clinical_weights.get("stage_IV", 100)
        breakdown["stage"] = f"Stage IV: +{clinical_weights.get('stage_IV', 100)}"
    elif "III" in stage or "3" in stage:
        score += clinical_weights.get("stage_III", 75)
        breakdown["stage"] = f"Stage III: +{clinical_weights.get('stage_III', 75)}"
    elif "II" in stage or "2" in stage:
        score += clinical_weights.get("stage_II", 50)
        breakdown["stage"] = f"Stage II: +{clinical_weights.get('stage_II', 50)}"
    elif "I" in stage or "1" in stage:
        score += clinical_weights.get("stage_I", 25)
        breakdown["stage"] = f"Stage I: +{clinical_weights.get('stage_I', 25)}"
    
    # Sex
    sex = str(patient.get("sex", "")).upper()
    if sex in ["M", "MALE"]:
        score += clinical_weights.get("sex_male", 27)
        breakdown["sex"] = f"Male: +{clinical_weights.get('sex_male', 27)}"
    
    # Age
    age = patient.get("age")
    if age is not None:
        try:
            age = float(age)
            age_elderly = thresholds.get("age_elderly", 70)
            age_middle = thresholds.get("age_middle", 60)
            
            if age >= age_elderly:
                pts = clinical_weights.get("age_over_70", 5)
                score += pts
                breakdown["age"] = f"Age ≥{age_elderly}: +{pts}"
            elif age >= age_middle:
                pts = clinical_weights.get("age_60_70", 3)
                score += pts
                breakdown["age"] = f"Age {age_middle}-{age_elderly}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # TMB
    tmb = patient.get("tmb_score") or patient.get("tmb")
    if tmb is not None:
        try:
            tmb = float(tmb)
            tmb_threshold = thresholds.get("tmb_high", 10)
            if tmb >= tmb_threshold:
                pts = clinical_weights.get("tmb_high", 22)
                score += pts
                breakdown["tmb"] = f"TMB ≥{tmb_threshold}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # Hypoxia (se disponibile)
    hypoxia = patient.get("hypoxia_buffa") or patient.get("hypoxia")
    if hypoxia is not None:
        try:
            hypoxia = float(hypoxia)
            hypoxia_threshold = thresholds.get("hypoxia_buffa_high", 0)
            if hypoxia > hypoxia_threshold:
                pts = clinical_weights.get("hypoxia_high", 25)
                score += pts
                breakdown["hypoxia"] = f"Hypoxia high: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # Aneuploidy
    aneuploidy = patient.get("aneuploidy_score") or patient.get("aneuploidy")
    if aneuploidy is not None:
        try:
            aneuploidy = float(aneuploidy)
            aneuploidy_threshold = thresholds.get("aneuploidy_high", 10)
            if aneuploidy >= aneuploidy_threshold:
                pts = clinical_weights.get("aneuploidy_high", 13)
                score += pts
                breakdown["aneuploidy"] = f"Aneuploidy ≥{aneuploidy_threshold}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # Fraction Genome Altered
    fga = patient.get("fraction_genome_altered")
    if fga is not None:
        try:
            fga = float(fga)
            fga_threshold = thresholds.get("fraction_genome_altered_high", 0.3)
            if fga >= fga_threshold:
                pts = clinical_weights.get("fraction_genome_altered_high", 17)
                score += pts
                breakdown["fga"] = f"FGA ≥{fga_threshold}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # MSI
    msi = patient.get("msi_score") or patient.get("msi")
    if msi is not None:
        try:
            msi = float(msi)
            msi_threshold = thresholds.get("msi_high", 10)
            if msi >= msi_threshold:
                pts = clinical_weights.get("msi_high", 9)
                score += pts
                breakdown["msi"] = f"MSI ≥{msi_threshold}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    return score, breakdown


def calculate_mutation_score(patient: Dict, weights: Dict) -> Tuple[int, Dict]:
    """
    Calcola lo score mutazionale basato su letteratura.
    
    Args:
        patient: Dict con dati paziente (genetics field)
        weights: Dict con pesi caricati
    
    Returns:
        score: Score numerico
        breakdown: Dict con contributi di ogni mutazione
    """
    
    mutation_weights = weights.get("mutation_weights", {})
    resistance_weights = mutation_weights.get("resistance_drivers", {})
    targetable_weights = mutation_weights.get("targetable_mutations", {})
    co_mutation_mods = mutation_weights.get("co_mutation_modifiers", {})
    
    score = 0
    breakdown = {}
    detected_mutations = []
    
    # Estrai genetics (può essere nested o flat)
    genetics = patient.get("genetics", patient.get("baseline", {}).get("genetics", {}))
    if not genetics:
        # Prova a estrarre direttamente dal paziente
        genetics = {k: v for k, v in patient.items() if '_status' in k.lower()}
    
    # Helper per check mutazione
    def is_mutated(gene: str) -> bool:
        # Check vari formati
        status = genetics.get(f"{gene}_status") or genetics.get(f"{gene.lower()}_status") or genetics.get(gene)
        if status is None:
            return False
        return str(status).lower() in ["mutated", "amplified", "amplification", "deleted", "fusion", "rearranged"]
    
    def get_variant(gene: str) -> Optional[str]:
        return genetics.get(f"{gene}_variant") or genetics.get(f"{gene.lower()}_variant")
    
    # Resistance drivers
    for gene, pts in resistance_weights.items():
        if is_mutated(gene):
            score += pts
            breakdown[gene] = f"{gene} mutated: +{pts}"
            detected_mutations.append(gene)
    
    # Targetable mutations
    if is_mutated("EGFR"):
        variant = get_variant("EGFR") or ""
        variant_upper = variant.upper()
        
        if "T790M" in variant_upper:
            pts = targetable_weights.get("EGFR_T790M", 40)
            score += pts
            breakdown["EGFR_T790M"] = f"EGFR T790M: +{pts}"
            detected_mutations.append("EGFR_T790M")
        elif "C797S" in variant_upper:
            pts = targetable_weights.get("EGFR_C797S", 50)
            score += pts
            breakdown["EGFR_C797S"] = f"EGFR C797S: +{pts}"
            detected_mutations.append("EGFR_C797S")
        else:
            # Activating mutation (L858R, exon19del, etc.) - no penalty
            detected_mutations.append("EGFR_activating")
    
    if is_mutated("KRAS"):
        variant = get_variant("KRAS") or ""
        if "G12C" in variant.upper():
            pts = targetable_weights.get("KRAS_G12C", 15)
            score += pts
            breakdown["KRAS_G12C"] = f"KRAS G12C: +{pts}"
            detected_mutations.append("KRAS_G12C")
        else:
            pts = targetable_weights.get("KRAS_other", 25)
            score += pts
            breakdown["KRAS_other"] = f"KRAS (non-G12C): +{pts}"
            detected_mutations.append("KRAS_other")
    
    if is_mutated("MET") or is_mutated("MET_amplification"):
        pts = targetable_weights.get("MET_amplification", 35)
        score += pts
        breakdown["MET_amp"] = f"MET amplification: +{pts}"
        detected_mutations.append("MET")
    
    if is_mutated("BRAF"):
        variant = get_variant("BRAF") or ""
        if "V600" in variant.upper():
            # V600E is targetable - no penalty
            detected_mutations.append("BRAF_V600E")
        else:
            pts = targetable_weights.get("BRAF_non_V600E", 20)
            score += pts
            breakdown["BRAF_non_V600E"] = f"BRAF non-V600E: +{pts}"
            detected_mutations.append("BRAF_non_V600E")
    
    if is_mutated("HER2") or is_mutated("ERBB2"):
        pts = targetable_weights.get("HER2_mutation", 25)
        score += pts
        breakdown["HER2"] = f"HER2/ERBB2 mutation: +{pts}"
        detected_mutations.append("HER2")
    
    # Co-mutation modifiers
    for co_mut, modifier in co_mutation_mods.items():
        genes = co_mut.split("_")
        if len(genes) == 2:
            if genes[0] in detected_mutations and genes[1] in detected_mutations:
                # Applica modificatore
                extra = int(score * (modifier - 1))
                score += extra
                breakdown[f"co_{co_mut}"] = f"{co_mut} interaction: +{extra} (x{modifier})"
    
    return score, breakdown


def calculate_blood_score(patient: Dict, weights: Dict) -> Tuple[int, Dict]:
    """
    Calcola lo score da blood markers (Elephant Protocol).
    
    Args:
        patient: Dict con dati paziente (blood_markers field)
        weights: Dict con pesi caricati
    
    Returns:
        score: Score numerico
        breakdown: Dict con contributi
    """
    
    blood_weights = weights.get("blood_marker_weights", {})
    thresholds = weights.get("thresholds", {})
    
    score = 0
    breakdown = {}
    
    # Estrai blood markers
    blood = patient.get("blood_markers", patient.get("baseline", {}).get("blood_markers", {}))
    if not blood:
        blood = patient  # Flat structure
    
    # LDH
    ldh = blood.get("ldh") or blood.get("LDH")
    if ldh is not None:
        try:
            ldh = float(ldh)
            ldh_very_high = thresholds.get("ldh_very_high", 500)
            ldh_high = thresholds.get("ldh_high", 350)
            
            if ldh >= ldh_very_high:
                pts = blood_weights.get("ldh_very_high", 50)
                score += pts
                breakdown["ldh"] = f"LDH ≥{ldh_very_high}: +{pts} (ELEPHANT ALERT)"
            elif ldh >= ldh_high:
                pts = blood_weights.get("ldh_high", 30)
                score += pts
                breakdown["ldh"] = f"LDH ≥{ldh_high}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # NLR
    nlr = blood.get("nlr") or blood.get("NLR")
    if nlr is not None:
        try:
            nlr = float(nlr)
            nlr_high = thresholds.get("nlr_high", 5.0)
            if nlr >= nlr_high:
                pts = blood_weights.get("nlr_high", 25)
                score += pts
                breakdown["nlr"] = f"NLR ≥{nlr_high}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    # Albumin
    albumin = blood.get("albumin") or blood.get("Albumin")
    if albumin is not None:
        try:
            albumin = float(albumin)
            albumin_low = thresholds.get("albumin_low", 3.5)
            if albumin < albumin_low:
                pts = blood_weights.get("albumin_low", 20)
                score += pts
                breakdown["albumin"] = f"Albumin <{albumin_low}: +{pts}"
        except (ValueError, TypeError):
            pass
    
    return score, breakdown


def calculate_risk_score(patient: Dict, 
                         weights: Optional[Dict] = None,
                         model: str = "pfs") -> Dict:
    """
    Calcola il risk score totale per un paziente.
    
    Args:
        patient: Dict con tutti i dati paziente
        weights: Dict con pesi (caricati automaticamente se None)
        model: "os" o "pfs" per scegliere i pesi clinici
    
    Returns:
        Dict con:
            - total_score: Score totale
            - clinical_score: Score clinico (ML-validated)
            - mutation_score: Score mutazionale (literature-based)
            - blood_score: Score blood markers
            - breakdown: Dettaglio contributi
            - risk_level: Interpretazione (Low/Moderate/High/Very High)
    """
    
    if weights is None:
        weights = load_weights()
    
    # Calcola componenti
    clinical_score, clinical_breakdown = calculate_clinical_score(patient, weights, model)
    mutation_score, mutation_breakdown = calculate_mutation_score(patient, weights)
    blood_score, blood_breakdown = calculate_blood_score(patient, weights)
    
    # Score totale
    total_score = clinical_score + mutation_score + blood_score
    
    # Interpretazione
    risk_interpretation = weights.get("risk_interpretation", {})
    risk_level = "Unknown"
    risk_color = "gray"
    
    for level, config in risk_interpretation.items():
        if config.get("min", 0) <= total_score <= config.get("max", 999):
            risk_level = config.get("label", level)
            risk_color = config.get("color", "gray")
            break
    
    # Breakdown completo
    breakdown = {
        "clinical": clinical_breakdown,
        "mutations": mutation_breakdown,
        "blood_markers": blood_breakdown
    }
    
    return {
        "total_score": total_score,
        "clinical_score": clinical_score,
        "mutation_score": mutation_score,
        "blood_score": blood_score,
        "breakdown": breakdown,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "model_used": model
    }


def get_risk_summary(result: Dict) -> str:
    """Genera un summary testuale del risk score"""
    
    lines = []
    lines.append(f"═══════════════════════════════════════════")
    lines.append(f"  SENTINEL RISK ASSESSMENT (ML-Validated)")
    lines.append(f"═══════════════════════════════════════════")
    lines.append(f"")
    lines.append(f"  TOTAL SCORE: {result['total_score']} [{result['risk_level']}]")
    lines.append(f"")
    lines.append(f"  Components:")
    lines.append(f"    Clinical (ML):    {result['clinical_score']}")
    lines.append(f"    Mutations (Lit):  {result['mutation_score']}")
    lines.append(f"    Blood Markers:    {result['blood_score']}")
    lines.append(f"")
    lines.append(f"  Breakdown:")
    
    for category, items in result['breakdown'].items():
        if items:
            lines.append(f"    [{category.upper()}]")
            for key, value in items.items():
                lines.append(f"      • {value}")
    
    lines.append(f"═══════════════════════════════════════════")
    
    return "\n".join(lines)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test con paziente esempio
    test_patient = {
        "stage": "IV",
        "age": 65,
        "sex": "M",
        "tmb_score": 15,
        "genetics": {
            "TP53_status": "mutated",
            "KRAS_status": "mutated",
            "KRAS_variant": "G12C",
            "STK11_status": "mutated",
            "EGFR_status": "wt"
        },
        "blood_markers": {
            "ldh": 380,
            "nlr": 6.2,
            "albumin": 3.2
        }
    }
    
    print("Testing SENTINEL ML Weights Module...")
    print()
    
    try:
        weights = load_weights()
        print(f"✅ Weights loaded successfully")
        print(f"   Clinical features: {len(weights.get('clinical_weights_pfs', {}))}")
        print(f"   Mutation weights: {len(weights.get('mutation_weights', {}).get('resistance_drivers', {}))}")
        print()
    except FileNotFoundError:
        print("⚠️  Weights file not found, using defaults")
        weights = None
    
    result = calculate_risk_score(test_patient, weights, model="pfs")
    
    print(get_risk_summary(result))
