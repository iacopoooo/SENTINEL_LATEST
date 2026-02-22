#!/usr/bin/env python3
"""
SENTINEL TRIAL - AI CONTROL TOWER v12.0
================================================
+ Dual-Core Engine Check
+ Hybrid Logging
+ Time Travel & Follow-up Module v2.0
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# LLM per note cliniche
try:
    import sys
    sys.path.insert(0, 'src')
    from clinical_notes_llm import enrich_visit_data, is_available as llm_available
    LLM_AVAILABLE = llm_available()
    print(f"✅ LLM Module Loaded (Available: {LLM_AVAILABLE})")
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"⚠️ LLM Module not available: {e}")


# CONFIGURAZIONE PATH CRITICA
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))  # Aggiunge la root al path

LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR = BASE_DIR / 'scripts'
DATA_DIR = BASE_DIR / 'data' / 'patients'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# COLORI
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"

# IMPORT IBRIDO (Tenta di caricare il modulo Follow-Up v2)
try:
    from src.follow_up_v2 import (
        Visit, PatientTimeline, load_patient_timeline, save_patient_timeline,
        get_current_patient_state, print_timeline_summary,
        BloodMarkers, ImagingResult, GeneticSnapshot, TherapyInfo, ClinicalStatus,
        AdverseEvent, RECISTResponse, AdverseEventGrade, TherapyChangeReason, ComplianceLevel
    )

    FOLLOW_UP_AVAILABLE = True
    FOLLOW_UP_VERSION = "v2.0"
except ImportError:
    # Fallback a v1
    try:
        from src.follow_up import (
            Visit, PatientTimeline, load_patient_timeline, save_patient_timeline,
            get_current_patient_state, print_timeline_summary
        )

        FOLLOW_UP_AVAILABLE = True
        FOLLOW_UP_VERSION = "v1.0"
    except ImportError as e:
        logger.warning(f"Follow-Up module not loaded: {e}")
        FOLLOW_UP_AVAILABLE = False
        FOLLOW_UP_VERSION = None


def header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{C_CYAN}")
    print("╔══════════════════════════════════════════════╗")
    print("║   SENTINEL TRIAL - AI CONTROL TOWER v12.0    ║")
    print("║   Hybrid Engine: Weighted + Bayesian         ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"{C_RESET}")


def check_environment():
    """Verifica che il cervello ibrido sia presente"""
    required_files = [
        'src/sentinel_engine.py',
        'src/sentinel_bayesian_v3.py',
    ]

    optional_files = [
        'src/follow_up_v2.py',
        'src/follow_up.py'
    ]

    missing = []
    for f in required_files:
        if not (BASE_DIR / f).exists():
            missing.append(f)

    if missing:
        print(f"{C_RED}[CRITICAL ERROR] Moduli AI mancanti:{C_RESET}")
        for m in missing:
            print(f" - {m}")
        print("\nAlcune funzionalità potrebbero essere disabilitate.")

    return True


# =============================================================================
# INPUT HELPERS
# =============================================================================

def get_input(prompt: str, default: str = "") -> str:
    """Input con valore default"""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def get_float(prompt: str, default: float = 0.0) -> float:
    """Input numerico float"""
    try:
        val = get_input(prompt, str(default) if default else "")
        return float(val) if val else default
    except ValueError:
        print("⚠️ Valore non valido, uso default")
        return default


def get_int(prompt: str, default: int = 0) -> int:
    """Input numerico int"""
    try:
        val = get_input(prompt, str(default) if default else "")
        return int(val) if val else default
    except ValueError:
        print("⚠️ Valore non valido, uso default")
        return default


def get_bool(prompt: str, default: bool = False) -> bool:
    """Input booleano"""
    default_str = "y" if default else "n"
    val = get_input(f"{prompt} (y/n)", default_str).lower()
    return val in ['s', 'si', 'sì', 'y', 'yes', '1', 'true']


def get_choice(prompt: str, options: list, default: int = 0) -> int:
    """Selezione da lista opzioni"""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i}] {opt}")

    try:
        val = get_input("Scelta", str(default))
        choice = int(val)
        if 0 <= choice < len(options):
            return choice
    except ValueError:
        pass

    return default


# =============================================================================
# FOLLOW-UP VISIT HANDLER v2.0
# =============================================================================

def handle_follow_up_visit():
    """
    Gestisce l'inserimento di una visita di follow-up clinicamente completa.

    Sezioni:
    1. Data e Timing
    2. Stato Clinico (ECOG, Peso, Eventi Avversi)
    3. Esami Ematici (LDH, NLR, Albumina, CEA)
    4. Imaging (RECIST)
    5. Genetica (opzionale)
    6. Terapia (compliance, cambi)
    7. Note cliniche
    """

    print(f"\n{'=' * 60}")
    print(f"📅 VISITA DI FOLLOW-UP - Follow-Up Module {FOLLOW_UP_VERSION}")
    print(f"{'=' * 60}")

    # 1. Seleziona paziente
    patient_id = input("\nID Paziente esistente: ").strip()

    timeline = load_patient_timeline(patient_id, DATA_DIR)
    if not timeline:
        print(f"❌ Paziente {patient_id} non trovato in {DATA_DIR}.")
        input("\nPremere INVIO...")
        return

    print(f"\n✅ Paziente caricato: {patient_id}")
    print(f"   Terapia attuale: {timeline.baseline_data.get('current_therapy', 'N/A')}")
    print(f"   Visite precedenti: {len(timeline.visits)}")

    # Determina dati precedenti per calcolo delta
    if timeline.visits:
        last = timeline.visits[-1]
        print(f"   Ultima visita: {last.date} (Week {last.week_on_therapy})")

        # Estrai dati precedenti (gestisce sia v1 che v2)
        if hasattr(last, 'therapy_info') and last.therapy_info:
            prev_therapy = last.therapy_info.current_therapy or last.therapy_info.new_therapy
        elif hasattr(last, 'new_therapy') and last.new_therapy:
            prev_therapy = last.new_therapy
        elif hasattr(last, 'therapy_at_visit'):
            prev_therapy = last.therapy_at_visit
        else:
            prev_therapy = timeline.baseline_data.get('current_therapy', '')

        prev_week = last.week_on_therapy

        # Peso precedente
        if hasattr(last, 'clinical_status') and last.clinical_status:
            prev_weight = last.clinical_status.weight_kg if hasattr(last.clinical_status, 'weight_kg') else 0
        else:
            prev_weight = 0

        # Blood markers precedenti
        if hasattr(last, 'blood_markers'):
            prev_blood = last.blood_markers
        else:
            prev_blood = None
    else:
        prev_therapy = timeline.baseline_data.get('current_therapy', '')
        prev_week = 0
        prev_weight = 0
        prev_blood = None

    visit_num = len(timeline.visits) + 1
    print(f"\n📋 Questa sarà la VISITA {visit_num}")
    if prev_therapy:
        print(f"   Terapia in corso: {prev_therapy}")

    # =========================================================================
    # SEZIONE 1: DATA E TIMING
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("📅 SEZIONE 1: DATA E TIMING")
    print(f"{'─' * 40}")

    today = datetime.now().strftime('%Y-%m-%d')
    visit_date = get_input("Data visita (YYYY-MM-DD)", today)

    suggested_week = prev_week + 4  # Assumiamo visite ogni 4 settimane
    week_on_therapy = get_int("Settimana in terapia", suggested_week)

    # =========================================================================
    # SEZIONE 2: STATO CLINICO
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("🩺 SEZIONE 2: STATO CLINICO")
    print(f"{'─' * 40}")

    # ECOG
    ecog_options = [
        "0 - Fully active",
        "1 - Restricted but ambulatory",
        "2 - Ambulatory, capable of self-care",
        "3 - Limited self-care, confined to bed/chair >50%",
        "4 - Completely disabled"
    ]

    # Trova ECOG precedente
    if timeline.visits:
        last = timeline.visits[-1]
        if hasattr(last, 'clinical_status') and last.clinical_status:
            prev_ecog = last.clinical_status.ecog_ps
        elif hasattr(last, 'ecog_ps'):
            prev_ecog = last.ecog_ps
        else:
            prev_ecog = 1
    else:
        prev_ecog = timeline.baseline_data.get('ecog_ps', 1)

    ecog_ps = get_choice("ECOG Performance Status:", ecog_options, prev_ecog)

    # Peso
    weight_kg = get_float("Peso (kg)", prev_weight if prev_weight > 0 else 0)

    # Eventi Avversi
    print("\n⚠️ EVENTI AVVERSI (CTCAE v5.0)")
    adverse_events = []

    if FOLLOW_UP_VERSION == "v2.0" and get_bool("Ci sono eventi avversi da segnalare?", False):
        while True:
            print("\n  Nuovo evento avverso:")
            ae_term = get_input("  Termine (es: Fatigue, Rash, Diarrhea) [vuoto per finire]")
            if not ae_term:
                break

            grade_options = [
                "None - Nessuno",
                "G1 - Lieve",
                "G2 - Moderato",
                "G3 - Severo",
                "G4 - Life-threatening"
            ]
            grade_idx = get_choice("  Grado:", grade_options, 1)
            grade = list(AdverseEventGrade)[grade_idx]

            action_options = [
                "Nessuna azione",
                "Dose ridotta",
                "Farmaco sospeso temporaneamente",
                "Farmaco discontinuato",
                "Terapia di supporto aggiunta"
            ]
            action_idx = get_choice("  Azione intrapresa:", action_options, 0)

            resolved = get_bool("  Risolto?", False)

            adverse_events.append(AdverseEvent(
                term=ae_term,
                grade=grade,
                onset_date=visit_date,
                resolved=resolved,
                action_taken=action_options[action_idx]
            ))

            if not get_bool("  Aggiungere altro evento avverso?", False):
                break

    # =========================================================================
    # SEZIONE 3: ESAMI EMATICI
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("🩸 SEZIONE 3: ESAMI EMATICI")
    print(f"{'─' * 40}")

    # Valori precedenti per default
    if prev_blood:
        if hasattr(prev_blood, 'ldh'):
            default_ldh = prev_blood.ldh or 250
            default_neut = prev_blood.neutrophils or 4500
            default_lymph = prev_blood.lymphocytes or 1500
            default_alb = getattr(prev_blood, 'albumin', 0) or 0
        elif isinstance(prev_blood, dict):
            default_ldh = prev_blood.get('ldh') or 250
            default_neut = prev_blood.get('neutrophils') or 4500
            default_lymph = prev_blood.get('lymphocytes') or 1500
            default_alb = prev_blood.get('albumin') or 0
        else:
            default_ldh, default_neut, default_lymph, default_alb = 250, 4500, 1500, 0
    else:
        bl_blood = timeline.baseline_data.get('blood_markers', {}) or {}
        default_ldh = bl_blood.get('ldh') or 250
        default_neut = bl_blood.get('neutrophils') or 4500
        default_lymph = bl_blood.get('lymphocytes') or 1500
        default_alb = bl_blood.get('albumin') or 0

    ldh = get_float("LDH (U/L)", default_ldh)
    neutrophils = get_float("Neutrofili (cells/µL)", default_neut)
    lymphocytes = get_float("Linfociti (cells/µL)", default_lymph)

    # NLR calcolato automaticamente
    nlr = round(neutrophils / lymphocytes, 2) if lymphocytes > 0 else 0
    print(f"   NLR calcolato: {nlr}")

    # Albumina e CEA (nuovi in v2)
    albumin = get_float("Albumina (g/dL) [0=skip]", default_alb)
    cea = get_float("CEA (ng/mL) [0=skip]", 0)

    # =========================================================================
    # SEZIONE 4: IMAGING
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("📷 SEZIONE 4: IMAGING")
    print(f"{'─' * 40}")

    imaging_data = None
    if get_bool("È stato eseguito imaging (CT/PET)?", True):
        imaging_date = get_input("Data imaging", visit_date)

        recist_options = [
            "CR - Complete Response (scomparsa lesioni)",
            "PR - Partial Response (riduzione ≥30%)",
            "SD - Stable Disease (né PR né PD)",
            "PD - Progressive Disease (aumento ≥20% o nuove lesioni)",
            "NE - Not Evaluable"
        ]
        recist_idx = get_choice("Risposta RECIST 1.1:", recist_options, 2)
        recist_map = {0: 'CR', 1: 'PR', 2: 'SD', 3: 'PD', 4: 'NE'}
        response = recist_map[recist_idx]

        tumor_change = get_float("Variazione % massa tumorale (es: -30)", 0)

        new_lesions = get_bool("Nuove lesioni?", False)
        new_lesion_sites = []
        if new_lesions:
            sites = get_input("Sedi nuove lesioni (separate da virgola)")
            new_lesion_sites = [s.strip() for s in sites.split(',') if s.strip()]

        imaging_notes = get_input("Note imaging [opzionale]", "")

        imaging_data = {
            "date": imaging_date,
            "response": response,
            "tumor_change_percent": tumor_change,
            "new_lesions": new_lesions,
            "new_lesion_sites": new_lesion_sites,
            "notes": imaging_notes
        }

    # =========================================================================
    # SEZIONE 5: GENETICA (opzionale) - CON MENU NUMERICI E VAF
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("🧬 SEZIONE 5: GENETICA (per CHRONOS tracking)")
    print(f"{'─' * 40}")

    # Liste opzioni per menu numerici
    GENE_STATUS_OPTIONS = ["wt (Wild-Type)", "mutated", "unknown"]
    KRAS_OPTIONS = ["wt", "G12C", "G12D", "G12V", "G13D", "Other", "unknown"]
    EGFR_OPTIONS = ["wt", "L858R", "Exon 19 del", "T790M", "L858R + T790M", "C797S", "Exon 20 ins", "unknown"]

    genetics_data = None
    if get_bool("È stata eseguita biopsia liquida/ctDNA?", False):
        source_options = ["ctDNA (biopsia liquida)", "Tissue rebiopsy", "Altro"]
        source_idx = get_choice("Fonte:", source_options, 0)
        source = ["ctDNA", "tissue_rebiopsy", "other"][source_idx]

        genetics_data = {
            "source": source,
            "date": visit_date,
            # Status (verranno popolati sotto)
            "tp53_status": "unknown",
            "kras_mutation": "unknown",
            "egfr_status": "unknown",
            "met_status": "unknown",
            "met_cn": 0,
            "stk11_status": "unknown",
            "keap1_status": "unknown",
            # Resistenza
            "t790m_detected": False,
            "c797s_detected": False,
            "met_amplification_acquired": False,
            "new_mutations": []
            # VAF verranno aggiunti sotto se presenti
        }

        # Mutazioni di resistenza comuni
        print("\n  --- Mutazioni di resistenza ---")
        genetics_data["t790m_detected"] = get_bool("  T790M rilevata?", False)
        genetics_data["c797s_detected"] = get_bool("  C797S rilevata?", False)
        genetics_data["met_amplification_acquired"] = get_bool("  MET amplificazione acquisita?", False)

        if genetics_data["met_amplification_acquired"]:
            genetics_data["met_cn"] = get_float("  MET Copy Number", 5.0)
            met_vaf = get_float("  MET VAF % [0=sconosciuto]", 0)
            if met_vaf > 0:
                genetics_data["met_vaf"] = met_vaf

        # Status mutazioni CON MENU NUMERICI E VAF
        print("\n  --- Status mutazioni (da ctDNA) ---")

        # TP53
        tp53_idx = get_choice("  TP53 status:", GENE_STATUS_OPTIONS, 2)
        genetics_data["tp53_status"] = GENE_STATUS_OPTIONS[tp53_idx].split(' ')[0]  # "wt", "mutated", "unknown"
        if tp53_idx == 1:  # mutated
            tp53_vaf = get_float("    → TP53 VAF % [0=sconosciuto]", 0)
            if tp53_vaf > 0:
                genetics_data["tp53_vaf"] = tp53_vaf

        # KRAS
        kras_idx = get_choice("  KRAS mutation:", KRAS_OPTIONS, 6)
        genetics_data["kras_mutation"] = KRAS_OPTIONS[kras_idx]
        if kras_idx not in [0, 6]:  # non wt e non unknown
            kras_vaf = get_float("    → KRAS VAF % [0=sconosciuto]", 0)
            if kras_vaf > 0:
                genetics_data["kras_vaf"] = kras_vaf

        # EGFR
        egfr_idx = get_choice("  EGFR status:", EGFR_OPTIONS, 7)
        genetics_data["egfr_status"] = EGFR_OPTIONS[egfr_idx]
        if egfr_idx not in [0, 7]:  # non wt e non unknown
            egfr_vaf = get_float("    → EGFR VAF % [0=sconosciuto]", 0)
            if egfr_vaf > 0:
                genetics_data["egfr_vaf"] = egfr_vaf

        # STK11
        stk11_idx = get_choice("  STK11 status:", GENE_STATUS_OPTIONS, 2)
        genetics_data["stk11_status"] = GENE_STATUS_OPTIONS[stk11_idx].split(' ')[0]
        if stk11_idx == 1:  # mutated
            stk11_vaf = get_float("    → STK11 VAF % [0=sconosciuto]", 0)
            if stk11_vaf > 0:
                genetics_data["stk11_vaf"] = stk11_vaf

        # KEAP1
        keap1_idx = get_choice("  KEAP1 status:", GENE_STATUS_OPTIONS, 2)
        genetics_data["keap1_status"] = GENE_STATUS_OPTIONS[keap1_idx].split(' ')[0]
        if keap1_idx == 1:  # mutated
            keap1_vaf = get_float("    → KEAP1 VAF % [0=sconosciuto]", 0)
            if keap1_vaf > 0:
                genetics_data["keap1_vaf"] = keap1_vaf

        # PIK3CA (opzionale)
        if get_bool("  PIK3CA mutato?", False):
            genetics_data["pik3ca_status"] = "mutated"
            pik3ca_vaf = get_float("    → PIK3CA VAF % [0=sconosciuto]", 0)
            if pik3ca_vaf > 0:
                genetics_data["pik3ca_vaf"] = pik3ca_vaf

        # BRAF (opzionale)
        if get_bool("  BRAF mutato?", False):
            braf_options = ["V600E", "Other"]
            braf_idx = get_choice("    BRAF mutation:", braf_options, 0)
            genetics_data["braf_status"] = braf_options[braf_idx]
            braf_vaf = get_float("    → BRAF VAF % [0=sconosciuto]", 0)
            if braf_vaf > 0:
                genetics_data["braf_vaf"] = braf_vaf

        # RB1 (importante per SCLC transformation)
        rb1_idx = get_choice("  RB1 status:", GENE_STATUS_OPTIONS, 2)
        genetics_data["rb1_status"] = GENE_STATUS_OPTIONS[rb1_idx].split(' ')[0]
        if rb1_idx == 1:  # mutated
            rb1_vaf = get_float("    → RB1 VAF % [0=sconosciuto]", 0)
            if rb1_vaf > 0:
                genetics_data["rb1_vaf"] = rb1_vaf

        # Altre mutazioni
        other_mutations = get_input("  Altre nuove mutazioni (separate da virgola)", "")
        if other_mutations:
            genetics_data["new_mutations"] = [m.strip() for m in other_mutations.split(',') if m.strip()]

    # =========================================================================
    # SEZIONE 6: TERAPIA
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("💊 SEZIONE 6: TERAPIA")
    print(f"{'─' * 40}")

    print(f"   Terapia attuale: {prev_therapy}")

    therapy_changed = get_bool("La terapia è stata modificata?", False)
    new_therapy = ""
    change_reason = "NONE"

    if therapy_changed:
        new_therapy = get_input("Nuova terapia")

        reason_options = [
            "Progressione di malattia",
            "Tossicità inaccettabile",
            "Scelta del paziente",
            "Decisione medica",
            "Da protocollo",
            "Trattamento completato",
            "Altro"
        ]
        reason_idx = get_choice("Motivo del cambio:", reason_options, 0)
        reason_map = {
            0: "PROGRESSION", 1: "TOXICITY", 2: "PATIENT_CHOICE",
            3: "PHYSICIAN_CHOICE", 4: "PROTOCOL", 5: "COMPLETED", 6: "OTHER"
        }
        change_reason = reason_map[reason_idx]

    # Dose ridotta
    dose_reduced = get_bool("Dose ridotta?", False)
    dose_reduction_detail = ""
    if dose_reduced:
        dose_reduction_detail = get_input("Dettaglio riduzione (es: 80mg -> 40mg)")

    # Compliance
    compliance_options = [
        "100% - Compliance completa",
        "80-99% - Alta compliance",
        "50-79% - Compliance moderata",
        "<50% - Bassa compliance",
        "Non noto"
    ]
    compliance_idx = get_choice("Compliance terapeutica:", compliance_options, 0)
    compliance_map = {0: "FULL", 1: "HIGH", 2: "MODERATE", 3: "LOW", 4: "UNKNOWN"}
    compliance = compliance_map[compliance_idx]

    # =========================================================================
    # SEZIONE 7: NOTE CLINICHE
    # =========================================================================
    print(f"\n{'─' * 40}")
    print("📝 SEZIONE 7: NOTE CLINICHE")
    print(f"{'─' * 40}")

    notes = get_input("Note cliniche [opzionale]", "")

    # =========================================================================
    # CREAZIONE VISITA (formato compatibile con entrambe le versioni)
    # =========================================================================

    visit_data = {
        "visit_id": f"V{visit_num}",
        "date": visit_date,
        "week_on_therapy": week_on_therapy,
        "therapy_at_visit": prev_therapy,
        "ecog_ps": ecog_ps,
        "clinical_status": {
            "ecog_ps": ecog_ps,
            "weight_kg": weight_kg,
            "weight_change_kg": round(weight_kg - prev_weight, 1) if prev_weight > 0 and weight_kg > 0 else 0
        },
        "blood_markers": {
            "ldh": ldh,
            "neutrophils": neutrophils,
            "lymphocytes": lymphocytes,
            "nlr": nlr,
            "albumin": albumin,
            "cea": cea
        },
        "imaging": imaging_data,
        "genetics": genetics_data,
        "therapy_info": {
            "current_therapy": new_therapy if therapy_changed else prev_therapy,
            "therapy_changed": therapy_changed,
            "new_therapy": new_therapy,
            "change_reason": change_reason,
            "dose_reduced": dose_reduced,
            "dose_reduction_detail": dose_reduction_detail,
            "compliance": compliance
        },
        "adverse_events": [ae.to_dict() for ae in adverse_events] if adverse_events else [],
        "notes": notes,
        # Legacy fields per compatibilità
        "therapy_changed": therapy_changed,
        "new_therapy": new_therapy
    }

    # === LLM ENRICHMENT ===
    if LLM_AVAILABLE and notes:
        print("🤖 Analisi LLM delle note cliniche...")
        visit_data = enrich_visit_data(visit_data)
        if visit_data.get("llm_notes_summary"):
            print(f"   📋 {visit_data['llm_notes_summary'][:100]}...")
    # === FINE LLM ===

    # Legacy fields per compatibilità
    visit_data["therapy_changed"] = therapy_changed
    visit_data["new_therapy"] = new_therapy

    # Calcola deltas
    deltas = {}
    if default_ldh is not None and default_ldh > 0:
        deltas['ldh_change'] = round(ldh - default_ldh, 1)
        pct = ((ldh - default_ldh) / default_ldh) * 100
        if pct > 10:
            deltas['ldh_trend'] = "↑ RISING"
        elif pct < -10:
            deltas['ldh_trend'] = "↓ FALLING"
        else:
            deltas['ldh_trend'] = "→ STABLE"

    deltas['ecog_change'] = ecog_ps - prev_ecog
    deltas['weeks_elapsed'] = week_on_therapy - prev_week

    if imaging_data:
        deltas['imaging_response'] = imaging_data['response']
        if imaging_data.get('new_lesions'):
            deltas['new_lesions'] = imaging_data.get('new_lesion_sites', [])

    if genetics_data and genetics_data.get('new_mutations'):
        deltas['new_mutations'] = genetics_data['new_mutations']

    visit_data['deltas'] = deltas

    # =========================================================================
    # RIEPILOGO
    # =========================================================================

    print(f"\n{'=' * 60}")
    print("📊 RIEPILOGO VISITA")
    print(f"{'=' * 60}")
    print(f"Data: {visit_date} | Settimana: {week_on_therapy}")
    print(f"ECOG: {ecog_ps} | Peso: {weight_kg} kg")
    print(f"LDH: {ldh} U/L | NLR: {nlr}")
    if imaging_data:
        print(f"RECIST: {imaging_data['response']} ({tumor_change:+.0f}%)")
    if therapy_changed:
        print(f"Terapia: CAMBIATA -> {new_therapy}")
        print(f"Motivo: {change_reason}")
    print(f"Compliance: {compliance}")

    if adverse_events:
        print(f"\nEventi avversi: {len(adverse_events)}")
        for ae in adverse_events:
            print(f"  - {ae.term}: {ae.grade.name}")

    if deltas:
        print(f"\nDelta vs precedente:")
        if 'ldh_change' in deltas:
            print(f"  LDH: {deltas['ldh_change']:+.0f} ({deltas.get('ldh_trend', '')})")
        if 'ecog_change' in deltas:
            print(f"  ECOG: {deltas['ecog_change']:+d}")

    # =========================================================================
    # SALVATAGGIO
    # =========================================================================

    if not get_bool("\n✅ Confermi il salvataggio?", True):
        print("❌ Visita annullata")
        input("\nPremere INVIO...")
        return

    # Aggiungi visita (usa il metodo appropriato)
    import json
    json_path = DATA_DIR / f"{patient_id}.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'visits' not in data:
        data['visits'] = []

    data['visits'].append(visit_data)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Visita V{visit_num} salvata con successo!")

    # Ricarica timeline per analisi
    timeline = load_patient_timeline(patient_id, DATA_DIR)
    if timeline:
        print_timeline_summary(timeline)

    # =========================================================================
    # RIGENERA PDF
    # =========================================================================

    regen = input("\n📄 Rigenerare report PDF con nuovi dati? (y/n) [y]: ").strip().lower()
    if regen != 'n':
        try:
            # Prova prima tools/, poi scripts/
            try:
                from tools.generate_final_report import generate_pdf
            except ImportError:
                from scripts.generate_final_report import generate_pdf
            generate_pdf(patient_id)
            print("✅ Report PDF rigenerato")
        except Exception as e:
            print(f"⚠️ Errore generazione PDF: {e}")

    input("\nPremere INVIO...")


def main_menu():
    check_environment()

    while True:
        header()
        print("\nSeleziona Modulo Operativo:")
        print(f"  [{C_GREEN}1{C_RESET}] 🏥 Nuovo Paziente (Hybrid Data Entry)")
        print(f"  [{C_GREEN}2{C_RESET}] 📄 Rigenera Report PDF (Dual-Core)")
        print(f"  [{C_GREEN}3{C_RESET}] 🖥️ Dashboard Grafica (Streamlit)")

        fu_status = f"{C_CYAN}[{FOLLOW_UP_VERSION}]{C_RESET}" if FOLLOW_UP_AVAILABLE else f"{C_RED}[N/A]{C_RESET}"
        print(f"  [{C_GREEN}4{C_RESET}] 📅 Follow-Up Visit (Clinical Complete) {fu_status}")

        print("-" * 60)
        print(f"  [{C_RED}0{C_RESET}] Esci")

        choice = input(f"\n{C_YELLOW}SENTINEL > {C_RESET}").strip()

        if choice == '1':
            try:
                from scripts.new_patient import interactive_registration
                interactive_registration()
            except ImportError:
                print(f"{C_RED}❌ Modulo new_patient non trovato.{C_RESET}")
            input("\nPremere INVIO...")

        elif choice == '2':
            p_id = input("ID Paziente: ").strip()
            try:
                try:
                    from tools.generate_final_report import generate_pdf
                except ImportError:
                    from scripts.generate_final_report import generate_pdf
                generate_pdf(p_id)
            except Exception as e:
                print(f"{C_RED}❌ Errore: {e}{C_RESET}")
            input("\nPremere INVIO...")

        elif choice == '3':
            print(f"\n{C_CYAN}Avvio Dashboard v12.0...{C_RESET}")
            dashboard_path = SCRIPTS_DIR / "dashboard.py"
            if dashboard_path.exists():
                subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
            else:
                print(f"{C_RED}❌ Dashboard non trovata: {dashboard_path}{C_RESET}")
                input("\nPremere INVIO...")

        elif choice == '4':
            if not FOLLOW_UP_AVAILABLE:
                print(f"{C_RED}❌ Modulo Follow-Up non trovato.{C_RESET}")
                print("   Assicurati che src/follow_up_v2.py o src/follow_up.py esista.")
                input("\nPremere INVIO...")
                continue
            handle_follow_up_visit()

        elif choice == '0':
            print(f"\n{C_CYAN}Arrivederci!{C_RESET}")
            sys.exit()


if __name__ == "__main__":
    main_menu()