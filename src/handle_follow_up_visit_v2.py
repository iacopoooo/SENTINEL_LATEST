"""
SENTINEL TRIAL - FOLLOW-UP CLI HANDLER v2.0
============================================
Interfaccia CLI per inserimento visite di follow-up clinicamente complete.

Da integrare in main.py sostituendo handle_follow_up_visit()
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# Importa dal modulo follow_up_v2
from follow_up_v2 import (
    Visit, BloodMarkers, ImagingResult, GeneticSnapshot,
    TherapyInfo, ClinicalStatus, AdverseEvent,
    RECISTResponse, AdverseEventGrade, TherapyChangeReason, ComplianceLevel,
    PatientTimeline, load_patient_timeline, save_patient_timeline,
    print_timeline_summary
)


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
    default_str = "s" if default else "n"
    val = get_input(f"{prompt} (s/n)", default_str).lower()
    return val in ['s', 'si', 'sì', 'y', 'yes', '1', 'true']


def get_choice(prompt: str, options: List[str], default: int = 0) -> int:
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
# FOLLOW-UP VISIT HANDLER
# =============================================================================

def handle_follow_up_visit(patient_id: str, data_dir: Path):
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
    
    print(f"\n{'='*60}")
    print(f"📅 VISITA DI FOLLOW-UP - Patient {patient_id}")
    print(f"{'='*60}")
    
    # Carica dati esistenti
    json_path = data_dir / f"{patient_id}.json"
    if not json_path.exists():
        print(f"❌ Paziente {patient_id} non trovato")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    baseline = data.get('baseline', {})
    visits = data.get('visits', [])
    
    # Determina dati precedenti per calcolo delta
    if visits:
        prev_data = visits[-1]
        prev_therapy = prev_data.get('therapy_info', {}).get('current_therapy') or \
                       prev_data.get('new_therapy') or \
                       prev_data.get('therapy_at_visit') or \
                       baseline.get('current_therapy', '')
        prev_week = prev_data.get('week_on_therapy', 0)
        prev_weight = prev_data.get('clinical_status', {}).get('weight_kg', 0)
    else:
        prev_data = baseline
        prev_therapy = baseline.get('current_therapy', '')
        prev_week = 0
        prev_weight = 0
    
    visit_num = len(visits) + 1
    print(f"\n📋 Questa sarà la VISITA {visit_num}")
    print(f"   Terapia attuale: {prev_therapy}")
    
    # =========================================================================
    # SEZIONE 1: DATA E TIMING
    # =========================================================================
    print(f"\n{'─'*40}")
    print("📅 SEZIONE 1: DATA E TIMING")
    print(f"{'─'*40}")
    
    today = datetime.now().strftime('%Y-%m-%d')
    visit_date = get_input("Data visita (YYYY-MM-DD)", today)
    
    suggested_week = prev_week + 4  # Assumiamo visite ogni 4 settimane
    week_on_therapy = get_int("Settimana in terapia", suggested_week)
    
    # =========================================================================
    # SEZIONE 2: STATO CLINICO
    # =========================================================================
    print(f"\n{'─'*40}")
    print("🩺 SEZIONE 2: STATO CLINICO")
    print(f"{'─'*40}")
    
    # ECOG
    ecog_options = [
        "0 - Fully active",
        "1 - Restricted but ambulatory",
        "2 - Ambulatory, capable of self-care",
        "3 - Limited self-care, confined to bed/chair >50%",
        "4 - Completely disabled"
    ]
    prev_ecog = prev_data.get('clinical_status', {}).get('ecog_ps') or prev_data.get('ecog_ps', 1)
    ecog_ps = get_choice("ECOG Performance Status:", ecog_options, prev_ecog)
    
    # Peso
    weight_kg = get_float("Peso (kg)", prev_weight)
    
    # Eventi Avversi
    print("\n⚠️ EVENTI AVVERSI (CTCAE v5.0)")
    adverse_events = []
    
    if get_bool("Ci sono eventi avversi da segnalare?", False):
        while True:
            print("\n  Nuovo evento avverso:")
            ae_term = get_input("  Termine (es: Fatigue, Rash, Diarrhea)")
            if not ae_term:
                break
            
            grade_options = [
                "None - Nessuno",
                "G1 - Lieve",
                "G2 - Moderato", 
                "G3 - Severo",
                "G4 - Life-threatening",
                "G5 - Morte"
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
    
    clinical_status = ClinicalStatus(
        ecog_ps=ecog_ps,
        weight_kg=weight_kg,
        weight_change_kg=round(weight_kg - prev_weight, 1) if prev_weight > 0 else 0
    )
    
    # =========================================================================
    # SEZIONE 3: ESAMI EMATICI
    # =========================================================================
    print(f"\n{'─'*40}")
    print("🩸 SEZIONE 3: ESAMI EMATICI")
    print(f"{'─'*40}")
    
    prev_blood = prev_data.get('blood_markers', {})
    
    ldh = get_float("LDH (U/L)", prev_blood.get('ldh', 250))
    neutrophils = get_float("Neutrofili (cells/µL)", prev_blood.get('neutrophils', 4500))
    lymphocytes = get_float("Linfociti (cells/µL)", prev_blood.get('lymphocytes', 1500))
    
    # NLR calcolato automaticamente
    nlr = round(neutrophils / lymphocytes, 2) if lymphocytes > 0 else 0
    print(f"   NLR calcolato: {nlr}")
    
    # Albumina e CEA (nuovi)
    albumin = get_float("Albumina (g/dL) [opzionale, 0=skip]", 0)
    cea = get_float("CEA (ng/mL) [opzionale, 0=skip]", 0)
    
    blood_markers = BloodMarkers(
        ldh=ldh,
        neutrophils=neutrophils,
        lymphocytes=lymphocytes,
        nlr=nlr,
        albumin=albumin,
        cea=cea
    )
    
    # =========================================================================
    # SEZIONE 4: IMAGING
    # =========================================================================
    print(f"\n{'─'*40}")
    print("📷 SEZIONE 4: IMAGING")
    print(f"{'─'*40}")
    
    imaging = None
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
        recist = list(RECISTResponse)[recist_idx]
        
        tumor_change = get_float("Variazione % massa tumorale (es: -30 per riduzione)", 0)
        
        new_lesions = get_bool("Nuove lesioni?", False)
        new_lesion_sites = []
        if new_lesions:
            sites = get_input("Sedi nuove lesioni (separate da virgola)")
            new_lesion_sites = [s.strip() for s in sites.split(',') if s.strip()]
        
        imaging_notes = get_input("Note imaging [opzionale]", "")
        
        imaging = ImagingResult(
            date=imaging_date,
            response=recist,
            tumor_change_percent=tumor_change,
            new_lesions=new_lesions,
            new_lesion_sites=new_lesion_sites,
            notes=imaging_notes
        )
    
    # =========================================================================
    # SEZIONE 5: GENETICA (opzionale)
    # =========================================================================
    print(f"\n{'─'*40}")
    print("🧬 SEZIONE 5: GENETICA")
    print(f"{'─'*40}")
    
    genetics = None
    if get_bool("È stata eseguita biopsia liquida/ctDNA?", False):
        source_options = ["ctDNA (biopsia liquida)", "Tissue rebiopsy", "Altro"]
        source_idx = get_choice("Fonte:", source_options, 0)
        source = ["ctDNA", "tissue_rebiopsy", "other"][source_idx]
        
        genetics = GeneticSnapshot(source=source, date=visit_date)
        
        # Mutazioni di resistenza comuni
        print("\n  Mutazioni di resistenza rilevate:")
        genetics.t790m_detected = get_bool("  T790M rilevata?", False)
        genetics.c797s_detected = get_bool("  C797S rilevata?", False)
        genetics.met_amplification_acquired = get_bool("  MET amplificazione acquisita?", False)
        
        if genetics.met_amplification_acquired:
            genetics.met_cn = get_float("  MET Copy Number", 5.0)
        
        # Altre mutazioni
        other_mutations = get_input("  Altre nuove mutazioni (separate da virgola)", "")
        if other_mutations:
            genetics.new_mutations = [m.strip() for m in other_mutations.split(',') if m.strip()]
        
        # VAF tracking
        if get_bool("  Inserire valori VAF?", False):
            while True:
                gene = get_input("    Gene (es: EGFR, TP53) [vuoto per finire]")
                if not gene:
                    break
                vaf = get_float(f"    VAF % per {gene}", 0)
                if vaf > 0:
                    genetics.vaf_values[gene] = vaf
    
    # =========================================================================
    # SEZIONE 6: TERAPIA
    # =========================================================================
    print(f"\n{'─'*40}")
    print("💊 SEZIONE 6: TERAPIA")
    print(f"{'─'*40}")
    
    print(f"   Terapia attuale: {prev_therapy}")
    
    therapy_changed = get_bool("La terapia è stata modificata?", False)
    new_therapy = ""
    change_reason = TherapyChangeReason.NONE
    change_reason_detail = ""
    
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
        change_reason = list(TherapyChangeReason)[reason_idx + 1]  # +1 perché NONE è 0
        
        if change_reason == TherapyChangeReason.OTHER:
            change_reason_detail = get_input("Specifica motivo")
    
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
    compliance = list(ComplianceLevel)[compliance_idx]
    
    therapy_info = TherapyInfo(
        current_therapy=new_therapy if therapy_changed else prev_therapy,
        therapy_changed=therapy_changed,
        new_therapy=new_therapy,
        change_reason=change_reason,
        change_reason_detail=change_reason_detail,
        dose_reduced=dose_reduced,
        dose_reduction_detail=dose_reduction_detail,
        compliance=compliance
    )
    
    # =========================================================================
    # SEZIONE 7: NOTE CLINICHE
    # =========================================================================
    print(f"\n{'─'*40}")
    print("📝 SEZIONE 7: NOTE CLINICHE")
    print(f"{'─'*40}")
    
    notes = get_input("Note cliniche [opzionale]", "")
    
    # =========================================================================
    # CREAZIONE VISITA
    # =========================================================================
    
    visit = Visit(
        visit_id=f"V{visit_num}",
        date=visit_date,
        week_on_therapy=week_on_therapy,
        clinical_status=clinical_status,
        adverse_events=adverse_events,
        blood_markers=blood_markers,
        imaging=imaging,
        genetics=genetics,
        therapy_info=therapy_info,
        notes=notes
    )
    
    # Calcola deltas
    if visits:
        prev_blood_obj = BloodMarkers.from_dict(prev_data.get('blood_markers', {}))
        prev_clinical_obj = ClinicalStatus.from_dict(
            prev_data.get('clinical_status', {'ecog_ps': prev_data.get('ecog_ps', 1)})
        )
    else:
        prev_blood_obj = BloodMarkers.from_dict(baseline.get('blood_markers', {}))
        prev_clinical_obj = ClinicalStatus(ecog_ps=baseline.get('ecog_ps', 1))
    
    # Calcola deltas manualmente qui (oppure usa PatientTimeline)
    deltas = {}
    if prev_blood_obj.ldh > 0:
        deltas['ldh_change'] = round(blood_markers.ldh - prev_blood_obj.ldh, 1)
        pct = ((blood_markers.ldh - prev_blood_obj.ldh) / prev_blood_obj.ldh) * 100
        if pct > 10:
            deltas['ldh_trend'] = "↑ RISING"
        elif pct < -10:
            deltas['ldh_trend'] = "↓ FALLING"
        else:
            deltas['ldh_trend'] = "→ STABLE"
    
    deltas['ecog_change'] = clinical_status.ecog_ps - prev_clinical_obj.ecog_ps
    deltas['weeks_elapsed'] = week_on_therapy - prev_week
    
    if clinical_status.weight_kg > 0 and prev_clinical_obj.weight_kg > 0:
        deltas['weight_change'] = round(clinical_status.weight_kg - prev_clinical_obj.weight_kg, 1)
    
    if imaging:
        deltas['imaging_response'] = imaging.response.name
        if imaging.new_lesions:
            deltas['new_lesions'] = imaging.new_lesion_sites
    
    if genetics and genetics.new_mutations:
        deltas['new_mutations'] = genetics.new_mutations
    
    visit.deltas = deltas
    
    # =========================================================================
    # SALVATAGGIO
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("📊 RIEPILOGO VISITA")
    print(f"{'='*60}")
    print(f"Data: {visit_date} | Settimana: {week_on_therapy}")
    print(f"ECOG: {ecog_ps} | Peso: {weight_kg} kg")
    print(f"LDH: {ldh} U/L | NLR: {nlr}")
    if imaging:
        print(f"RECIST: {imaging.response.name} ({tumor_change:+.0f}%)")
    if therapy_changed:
        print(f"Terapia: CAMBIATA -> {new_therapy}")
    print(f"Compliance: {compliance.value}")
    
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
    
    if not get_bool("\n✅ Confermi il salvataggio?", True):
        print("❌ Visita annullata")
        return
    
    # Aggiungi visita
    visits.append(visit.to_dict())
    data['visits'] = visits
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Visita V{visit_num} salvata con successo!")
    
    # =========================================================================
    # ANALISI TIMELINE
    # =========================================================================
    
    timeline = PatientTimeline(patient_id)
    timeline.baseline_data = baseline
    for v_data in visits:
        timeline.visits.append(Visit.from_dict(v_data))
    
    print_timeline_summary(timeline)
    
    # Check resistenza
    alerts = timeline.detect_resistance_patterns()
    if alerts:
        print("\n⚠️ ATTENZIONE - Pattern di resistenza rilevati!")
        for alert in alerts:
            print(f"  [{alert.urgency.upper()}] {alert.pattern.value}")
            print(f"    → {alert.recommendation}")
    
    # Rigenera PDF?
    if get_bool("\n📄 Vuoi rigenerare il report PDF?", True):
        try:
            from generate_final_report import generate_pdf
            generate_pdf(patient_id)
            print("✅ Report PDF rigenerato")
        except Exception as e:
            print(f"⚠️ Errore generazione PDF: {e}")


# =============================================================================
# MAIN (per test)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        patient_id = input("Patient ID: ")
    
    # Assumiamo la directory standard
    data_dir = Path(__file__).parent.parent / 'data' / 'patients'
    
    handle_follow_up_visit(patient_id, data_dir)
