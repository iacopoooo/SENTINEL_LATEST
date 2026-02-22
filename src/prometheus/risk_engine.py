"""
PROMETHEUS — Risk Engine
========================
Scoring Clinico Multi-Dimensionale.
"""

from typing import Dict, List
import numpy as np

# Pesi per ogni fattore di rischio (calibrati su letteratura)
RISK_WEIGHTS = {
    # Genetica (driver mutations)
    "kras_mut": {"weight": 12, "category": "genetica", "label": "KRAS mutato",
                 "correction": "Sotorasib/Adagrasib (G12C), MEK inhibitors"},
    "egfr_mut": {"weight": 10, "category": "genetica", "label": "EGFR mutato",
                 "correction": "Osimertinib, Monitoraggio T790M/C797S"},
    "pik3ca_mut": {"weight": 15, "category": "genetica", "label": "PIK3CA mutato",
                   "correction": "Alpelisib in combinazione, mTOR inhibitors"},
    "tp53_status": {"weight": 15, "category": "genetica", "label": "TP53 mutato",
                    "correction": "Monitoraggio intensivo, potenziale radio-resistenza"},
    "stk11_status": {"weight": 8, "category": "genetica", "label": "STK11 Loss",
                     "correction": "Resistenza IO, considerare chemio standard"},
    "braf_mut": {"weight": 10, "category": "genetica", "label": "BRAF mutato",
                 "correction": "Dabrafenib + Trametinib"},
    "alk_status": {"weight": 9, "category": "genetica", "label": "ALK+",
                   "correction": "Lorlatinib, Alectinib"},
    "ret_status": {"weight": 8, "category": "genetica", "label": "RET fusion",
                   "correction": "Selpercatinib, Pralsetinib"},

    # PGx variants
    "pgx_dpyd": {"weight": 6, "category": "farmacogenomica", "label": "DPYD variante",
                 "correction": "Evitare 5-FU/Capecitabina o ridurre dose 50%"},
    "pgx_ugt1a1": {"weight": 4, "category": "farmacogenomica", "label": "UGT1A1 *28/*28",
                    "correction": "Ridurre Irinotecan 30%, monitorare bilirubina"},

    # Biomarkers elevati
    "ldh_high": {"weight": 8, "category": "biomarker", "label": "LDH elevato",
                 "correction": "Indagare progressione, imaging urgente"},
    "crp_high": {"weight": 6, "category": "biomarker", "label": "CRP elevata",
                 "correction": "Escludere infezione, anti-infiammatori mirati"},
    "nlr_high": {"weight": 7, "category": "biomarker", "label": "NLR elevato (>5)",
                 "correction": "Immunomodulazione, considerare IO + chemio"},
    "albumin_low": {"weight": 5, "category": "biomarker", "label": "Albumina bassa",
                    "correction": "Supporto nutrizionale, sospettare cachessia"},
    "cea_high": {"weight": 6, "category": "biomarker", "label": "CEA elevato",
                 "correction": "Monitoraggio seriale, imaging se raddoppio <3 mesi"},

    # Lifestyle
    "smoking_active": {"weight": 8, "category": "lifestyle", "label": "Fumatore attivo",
                       "correction": "Cessazione immediata, supporto farmacologico"},
    "ecog_high": {"weight": 10, "category": "lifestyle", "label": "ECOG ≥2",
                  "correction": "Riabilitazione, rivalutare fitness per terapia"},
    "age_high": {"weight": 4, "category": "lifestyle", "label": "Età >65",
                 "correction": "Dosaggio personalizzato, monitoraggio aumentato"},
}

# Fattori protettivi (riducono il rischio)
PROTECTIVE_FACTORS = {
    "pdl1_high": {"weight": -5, "label": "PD-L1 ≥50%",
                  "benefit": "Candidato a immunoterapia (Pembrolizumab)"},
    "tmb_high": {"weight": -4, "label": "TMB alto (≥10)",
                 "benefit": "Risposta immunoterapia molto probabile"},
    "vitamin_d_ok": {"weight": -2, "label": "Vitamina D adeguata",
                     "benefit": "Effetto protettivo immunomodulante"},
    "normal_weight": {"weight": -2, "label": "BMI normale",
                      "benefit": "Farmacocinetica ottimale"},
}

# Cluster profiles (basati su letteratura oncologica)
CLUSTER_PROFILES = {
    0: {"name": "Basso Rischio", "color": "#4CAF50", "emoji": "🟢",
        "desc": "Pochi fattori di rischio attivi. Monitoraggio standard.",
        "monitoring": "Ogni 6 mesi", "risk_range": (0, 25)},
    1: {"name": "Rischio Moderato", "color": "#FF9800", "emoji": "🟡",
        "desc": "Pattern infiammatorio o genetico singolo. Sorveglianza attiva.",
        "monitoring": "Ogni 3 mesi", "risk_range": (25, 50)},
    2: {"name": "Rischio Elevato", "color": "#FF5722", "emoji": "🟠",
        "desc": "Multipli driver attivi o biomarker alterati. Intervento raccomandato.",
        "monitoring": "Ogni 4-6 settimane", "risk_range": (50, 75)},
    3: {"name": "Rischio Critico", "color": "#D32F2F", "emoji": "🔴",
        "desc": "Convergenza genetica + biomarker + lifestyle. Azione immediata.",
        "monitoring": "Ogni 2 settimane", "risk_range": (75, 100)},
}


def compute_risk_score(features: Dict[str, float], patient_data: dict = None) -> Dict:
    """
    Calcola il rischio base e individua fattori/mismatch. per il paziente.

    Returns:
        Dict con risk_score, cluster, risk_factors, protections, corrections
    """
    total_risk = 0
    total_protection = 0
    risk_factors = []
    protections = []
    corrections = []
    category_scores = {"genetica": 0, "farmacogenomica": 0, "biomarker": 0, "lifestyle": 0}

    # === Valuta ogni fattore di rischio ===
    for key, info in RISK_WEIGHTS.items():
        active = False

        if key == "kras_mut":
            active = features.get("kras_mut", 0) > 0
        elif key == "tp53_status":
            active = features.get("tp53_status", 0) > 0
        elif key == "pik3ca_mut":
            active = features.get("pik3ca_mut", 0) > 0
        elif key == "egfr_mut":
            active = features.get("egfr_mut", 0) > 0
        elif key == "stk11_status":
            active = features.get("stk11_status", 0) > 0
        elif key == "braf_mut":
            active = features.get("braf_mut", 0) > 0
        elif key == "alk_status":
            active = features.get("alk_status", 0) > 0
        elif key == "ret_status":
            active = features.get("ret_status", 0) > 0
        elif key == "pgx_dpyd":
            active = features.get("pgx_dpyd", 0) > 0
        elif key == "pgx_ugt1a1":
            active = features.get("pgx_ugt1a1", 0) > 0
        elif key == "ldh_high":
            active = features.get("ldh", 0) > 250
        elif key == "crp_high":
            active = features.get("crp", 0) > 5
        elif key == "nlr_high":
            nlr_val = features.get("nlr_raw", 0) or features.get("nlr", 0)
            active = nlr_val > 5 if nlr_val else False
        elif key == "albumin_low":
            alb = features.get("albumin", 99)
            active = 0 < alb < 35
        elif key == "cea_high":
            active = features.get("cea", 0) > 5
        elif key == "smoking_active":
            active = features.get("smoking", 0) >= 2
        elif key == "ecog_high":
            active = features.get("ecog", 0) >= 2
        elif key == "age_high":
            active = features.get("age", 0) > 65

        if active:
            w = int(info["weight"])
            cat = str(info["category"])
            total_risk += w
            category_scores[cat] = category_scores.get(cat, 0) + w
            risk_factors.append({
                "label": info["label"],
                "weight": w,
                "category": cat,
            })
            corrections.append({
                "target": info["label"],
                "action": info["correction"],
                "category": cat,
                "weight": w,
                "priority": "ALTA" if w >= 10 else (
                    "MEDIA" if w >= 6 else "BASSA"),
            })

    # === Fattori protettivi ===
    pdl1 = features.get("pdl1", 0)
    if pdl1 and pdl1 >= 50:
        total_protection += abs(int(PROTECTIVE_FACTORS["pdl1_high"]["weight"]))
        protections.append(PROTECTIVE_FACTORS["pdl1_high"])

    tmb = features.get("tmb", 0)
    if tmb and tmb >= 10:
        total_protection += abs(int(PROTECTIVE_FACTORS["tmb_high"]["weight"]))
        protections.append(PROTECTIVE_FACTORS["tmb_high"])

    bmi = features.get("bmi", 0) or features.get("weight", 0)
    if bmi and 18.5 <= bmi <= 25:
        total_protection += abs(int(PROTECTIVE_FACTORS["normal_weight"]["weight"]))
        protections.append(PROTECTIVE_FACTORS["normal_weight"])

    # === NEURO-SYMBOLIC VETO INTEGRATION ===
    if patient_data:
        try:
            from sentinel_engine import VetoSystem
            veto_sys = VetoSystem()
            # VetoSystem check_therapy expects the baseline dict
            baseline_data = patient_data.get("baseline", patient_data)
                
            veto = veto_sys.check_therapy(baseline_data)
            if veto and getattr(veto, 'active', False) and getattr(veto, 'severity', '') == 'HIGH':
                total_risk += 45
                cat = "VETO TERAPEUTICO"
                category_scores[cat] = category_scores.get(cat, 0) + 45
                risk_factors.append({
                    "label": f"🛑 {veto.reason}",
                    "weight": 45,
                    "category": cat,
                })
                corrections.append({
                    "target": f"🛑 {veto.reason}",
                    "action": veto.recommendation,
                    "category": cat,
                    "weight": 45,
                    "priority": "ALTA",
                })
        except Exception as e:
            pass

    # === Calcola score finale (0-100) integrando il DigitalTwin ===
    # Uso DigitalTwin per avere lo score di base (allineato con Engine Principale)
    base_core_risk = 0
    if patient_data:
        try:
            from digital_twin import DigitalTwin
            twin = DigitalTwin(patient_data)
            
            dt_res = twin.simulate_disease_trajectory()
            base_core_risk = dt_res.mortality_risk_percent
        except Exception as e:
            base_core_risk = 0
        
    # Se il DigitalTwin ci dà un rischio alto, lo usiamo come base
    max_possible = sum(i["weight"] for i in RISK_WEIGHTS.values())
    proprietary_raw_score = max(0, total_risk - total_protection)
    proprietary_risk = min(100, int((proprietary_raw_score / max_possible) * 100 * 2))  
    
    risk_score = max(int(base_core_risk), proprietary_risk)

    # === Assegna cluster ===
    if risk_score < 25:
        cluster_id = 0
    elif risk_score < 50:
        cluster_id = 1
    elif risk_score < 75:
        cluster_id = 2
    else:
        cluster_id = 3

    cluster = CLUSTER_PROFILES[cluster_id]

    # === Timeline Adattiva ===
    timeline = []
    current_risk = risk_score
    
    is_critical = risk_score > 75
    months_to_project = 12 if is_critical else 60
    step = 1 if is_critical else 6

    base_monthly_increase = risk_score * 0.003  # Progressione basale
    for month in range(1, months_to_project + 1):
        # Risk increases faster with more factors
        acceleration = 1.0 + len(risk_factors) * 0.02
        current_risk = min(100, current_risk + base_monthly_increase * acceleration * (1 + month * 0.005))
        
        if month % step == 0 or month == months_to_project:
            label = f"{month}m" if is_critical or month % 12 != 0 else f"Anno {month // 12}"
            timeline.append({
                "month": month,
                "year": month / 12,
                "risk": round(current_risk, 1),
                "label": label,
            })

    # === Epistatic rules check ===
    epistatic_warnings = []
    try:
        from prometheus.oracle_bridge import check_patient_rules
        patient_for_bridge = {"baseline": {"blood_markers": {}}}
        for k, v in features.items():
            if not np.isnan(v) if isinstance(v, float) else True:
                if k in ("ldh", "crp", "glucose", "albumin", "neutrophils", "lymphocytes",
                         "cea", "creatinine", "platelets", "hemoglobin"):
                    patient_for_bridge["baseline"]["blood_markers"][k] = v
                else:
                    patient_for_bridge["baseline"][k] = v

        epistatic = check_patient_rules(patient_for_bridge)
        if epistatic:
            epistatic_warnings = epistatic
            # Boost risk if epistatic rules match
            risk_score = min(100, risk_score + len(epistatic) * 5)
    except Exception:
        pass

    return {
        "risk_score": risk_score,
        "cluster_id": cluster_id,
        "cluster": cluster,
        "risk_factors": sorted(risk_factors, key=lambda x: -x["weight"]),
        "protections": protections,
        "corrections": sorted(corrections, key=lambda x: -x.get("weight", 0)),
        "category_scores": category_scores,
        "timeline": timeline,
        "total_risk_points": total_risk,
        "total_protection_points": total_protection,
        "n_active_factors": len(risk_factors),
        "epistatic_warnings": epistatic_warnings,
    }
