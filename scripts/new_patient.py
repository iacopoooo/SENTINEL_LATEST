#!/usr/bin/env python3
"""
SENTINEL Trial - Registra Nuovo Paziente (v18.1 FIXED)
====================================================================
FIX APPLICATI:
- Blood markers resi OPZIONALI (premere INVIO per skippare)
- Gestione corretta di None/null nei JSON
- NLR calcolato solo se dati disponibili
"""

import sys
import os
import json
import random
from datetime import datetime
from pathlib import Path

# SETUP PATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import Parser
try:
    from src.vcf_parser import SentinelGenomics

    VCF_ACTIVE = True
except ImportError:
    VCF_ACTIVE = False

DATA_DIR = BASE_DIR / 'data' / 'patients'
GENOMICS_DIR = BASE_DIR / 'data' / 'genomics'
BIOPSY_DIR = BASE_DIR / 'data' / 'biopsies'

for d in [DATA_DIR, GENOMICS_DIR, BIOPSY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LISTE COMPLETE (v18.0 EXPANDED)
# ============================================================================

VALID_SEX = ['M', 'F']
VALID_SMOKING = ['Never', 'Former', 'Current']
VALID_STAGE = ['III', 'IVA', 'IVB']
VALID_HISTOLOGY = [
    'Adenocarcinoma', 'Squamous', 'Large Cell', 'Small Cell (SCLC)',
    'Adenosquamous', 'Carcinoid', 'NOS (Not Otherwise Specified)'
]
VALID_GENE_STATUS = ['wt', 'mutated', 'unknown']
VALID_KRAS = ['wt', 'G12C', 'G12D', 'G12V', 'G13D', 'Other']

# EXPANDED: Tutte le terapie per VETO system
VALID_THERAPIES = [
    # EGFR Inhibitors
    'Osimertinib (EGFR 3rd-gen)',
    'Gefitinib (EGFR 1st-gen)',
    'Erlotinib (EGFR 1st-gen)',
    'Afatinib (EGFR 2nd-gen)',
    'Dacomitinib (EGFR 2nd-gen)',

    # KRAS Inhibitors
    'Sotorasib (KRAS G12C)',
    'Adagrasib (KRAS G12C)',

    # ALK Inhibitors
    'Alectinib (ALK)',
    'Crizotinib (ALK/MET)',
    'Ceritinib (ALK)',
    'Brigatinib (ALK)',
    'Lorlatinib (ALK 3rd-gen)',

    # MET Inhibitors
    'Capmatinib (MET)',
    'Tepotinib (MET)',

    # BRAF/MEK
    'Dabrafenib + Trametinib (BRAF V600E)',

    # HER2
    'Trastuzumab Deruxtecan (HER2)',

    # Immunotherapy
    'Pembrolizumab (PD-1 Inhibitor)',
    'Nivolumab (PD-1 Inhibitor)',
    'Atezolizumab (PD-L1 Inhibitor)',
    'Durvalumab (PD-L1 Inhibitor)',

    # Chemotherapy
    'Platinum + Pemetrexed',
    'Platinum + Etoposide (SCLC)',
    'Docetaxel',

    # Combinations
    'Chemo + Immunotherapy',
    'Amivantamab + Lazertinib (EGFR)',

    # Other
    'Clinical Trial',
    'Best Supportive Care',
    'None/Naive'
]

# PGx Genotypes (Farmacogenomica)
VALID_PGX_GENOTYPES = {
    'CYP2D6': ['*1/*1 (Normal)', '*1/*4 (Intermediate)', '*4/*4 (Poor)', '*1/*2 (Normal)', 'unknown'],
    'CYP2C19': ['*1/*1 (Normal)', '*1/*2 (Intermediate)', '*2/*2 (Poor)', '*17/*17 (Ultra-rapid)', 'unknown'],
    'CYP3A4': ['*1/*1 (Normal)', '*1/*22 (Intermediate)', '*22/*22 (Poor)', 'unknown'],
    'DPYD': ['*1/*1 (Normal)', '*1/*2A (Intermediate - 5-FU risk)', '*2A/*2A (Poor - 5-FU contraindicated)', 'unknown'],
    'UGT1A1': ['*1/*1 (Normal)', '*1/*28 (Gilbert - Irinotecan risk)', '*28/*28 (Poor - Irinotecan reduce)', 'unknown'],
    'TPMT': ['*1/*1 (Normal)', '*1/*3A (Intermediate)', '*3A/*3A (Poor - 6-MP contraindicated)', 'unknown'],
}


# ============================================================================
# HELPER INPUT FUNCTIONS
# ============================================================================

def show_menu(title, options):
    """Mostra menu a tendina e ritorna la scelta"""
    print(f"\n{title}")
    print("-" * 40)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")

    while True:
        try:
            choice = input(f"Scegli [1-{len(options)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            print(f"⚠️  Scegli un numero tra 1 e {len(options)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Input invalido")


def input_number(prompt, min_val=0, max_val=100000, allow_empty=False, default=None):
    """
    Input numerico con validazione.

    Args:
        prompt: Testo del prompt
        min_val: Valore minimo accettato
        max_val: Valore massimo accettato
        allow_empty: Se True, permette di premere INVIO per skippare
        default: Valore di default se allow_empty=True e input vuoto

    Returns:
        float o None (se allow_empty e input vuoto)
    """
    while True:
        try:
            user_input = input(f"{prompt}: ").strip()

            # Gestione campo vuoto
            if not user_input:
                if allow_empty:
                    return default  # Può essere None
                else:
                    print("⚠️  Campo obbligatorio")
                    continue

            val = float(user_input.replace(',', '.'))
            if min_val <= val <= max_val:
                return val
            print(f"⚠️  Valore fuori range ({min_val}-{max_val})")
        except ValueError:
            print("⚠️  Inserisci un numero valido")
        except KeyboardInterrupt:
            print("\n❌ Operazione annullata")
            sys.exit(0)


def check_veto_warnings(therapy, genetics):
    """
    Controlla incompatibilità terapia-genetica (VETO preview)
    Ritorna lista di warning
    """
    warnings = []

    # VETO 1: Osimertinib senza EGFR
    if "Osimertinib" in therapy:
        egfr = genetics.get('egfr_status', 'wt').lower()
        if egfr in ['wt', 'unknown', 'wild-type', 'wildtype']:
            warnings.append(
                "⚠️  VETO: Osimertinib richiede mutazione EGFR!\n"
                "   → Terapia inappropriata: nessun target molecolare"
            )

    # VETO 2: Gefitinib/Erlotinib con T790M
    if "Gefitinib" in therapy or "Erlotinib" in therapy:
        egfr = genetics.get('egfr_status', '').lower()
        if "t790m" in egfr:
            warnings.append(
                "⚠️  VETO: Gefitinib/Erlotinib inefficaci vs T790M!\n"
                "   → Resistenza nota: necessario switch a Osimertinib 3rd-gen"
            )

    # VETO 3: Sotorasib/Adagrasib senza G12C
    if "Sotorasib" in therapy or "Adagrasib" in therapy:
        kras = genetics.get('kras_mutation', '').lower()
        if "g12c" not in kras:
            warnings.append(
                "⚠️  VETO: Sotorasib/Adagrasib specifici per KRAS G12C!\n"
                "   → Terapia inappropriata: mutazione G12C non presente"
            )

    # VETO 4: ALK inhibitors senza ALK+
    alk_drugs = ["Alectinib", "Crizotinib", "Ceritinib", "Brigatinib", "Lorlatinib"]
    if any(drug in therapy for drug in alk_drugs):
        alk = genetics.get('alk_status', '').lower()
        if 'rearrangement' not in alk and 'alk+' not in alk:
            warnings.append(
                "⚠️  VETO: ALK inhibitor richiede ALK rearrangement!\n"
                "   → Terapia inappropriata: nessuna fusione ALK rilevata"
            )

    # VETO 5: MET inhibitors senza MET amplification/exon14
    if "Capmatinib" in therapy or "Tepotinib" in therapy:
        met = genetics.get('met_status', '').lower()
        if 'amplification' not in met and 'exon 14' not in met:
            warnings.append(
                "⚠️  WARNING: MET inhibitor senza MET alterazione\n"
                "   → Verificare amplificazione o exon 14 skipping"
            )

    return warnings


# ============================================================================
# INTERACTIVE REGISTRATION
# ============================================================================

def interactive_registration():
    """Registrazione interattiva completa di un nuovo paziente"""

    print("\n" + "🧬" * 35)
    print("SENTINEL v18.1 FIXED - NUOVO PAZIENTE")
    print("🧬" * 35)

    # ========================================================================
    # 1. ID PAZIENTE & VCF CHECK
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 1: IDENTIFICAZIONE PAZIENTE")
    print("=" * 70)

    patient_id = input("\nID Paziente (es. SENT-001): ").strip()
    if not patient_id:
        print("❌ ID paziente obbligatorio")
        return

    # Check VCF esistente
    vcf_data = {}
    if VCF_ACTIVE and (GENOMICS_DIR / f"{patient_id}.vcf").exists():
        print("⚡ VCF Trovato! Importazione automatica...")
        try:
            parser = SentinelGenomics(patient_id)
            vcf_dict = parser.parse_vcf(str(GENOMICS_DIR / f"{patient_id}.vcf"))
            vcf_data = {
                'tp53': vcf_dict.get('tp53_status'),
                'kras': vcf_dict.get('kras_mutation'),
                'stk11': vcf_dict.get('stk11_status'),
                'keap1': vcf_dict.get('keap1_status')
            }
            print(f"   ✅ Importati: TP53={vcf_data.get('tp53')}, KRAS={vcf_data.get('kras')}")
        except Exception as e:
            print(f"   ⚠️ Errore parsing VCF: {e}")

    # ========================================================================
    # 2. DATI DEMOGRAFICI E CLINICI
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 2: DATI DEMOGRAFICI E CLINICI")
    print("=" * 70)

    age = int(input_number("Età", 18, 110))
    sex = show_menu("Sesso", VALID_SEX)
    smoking = show_menu("Storia di Fumo", VALID_SMOKING)
    ecog = int(input_number("ECOG Performance Status (0-4)", 0, 4))
    stage = show_menu("Stadio TNM", VALID_STAGE)
    histology = show_menu("Istologia", VALID_HISTOLOGY)

    # ========================================================================
    # 3. PROFILO GENOMICO
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 3: PROFILO GENOMICO (Hybrid Engine Ready)")
    print("=" * 70)

    # TP53
    # TP53
    tp53 = vcf_data.get('tp53') or show_menu("Status TP53", VALID_GENE_STATUS)
    tp53_vaf = None
    if tp53 and tp53.lower() in ['mutated', 'mut', 'loss']:
        tp53_vaf = input_number("  → TP53 VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # KRAS
    kras = vcf_data.get('kras')
    if not kras:
        kras_opt = show_menu("Mutazione KRAS", VALID_KRAS)
        if kras_opt == 'Other':
            kras = input("Specifica KRAS: ").strip() or 'Other'
        else:
            kras = kras_opt
    kras_vaf = None
    if kras and kras.lower() not in ['wt', 'unknown', '']:
        kras_vaf = input_number("  → KRAS VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # EGFR (Importante per VETO)
    egfr_options = [
        'wt (Wild-Type)',
        'Exon 19 deletion (sensitizing)',
        'L858R (sensitizing)',
        'T790M (resistance)',
        'Exon 19 del + T790M',
        'L858R + T790M',
        'C797S (3rd-gen resistance)',
        'Exon 20 insertion',
        'Other sensitizing',
        'unknown'
    ]
    egfr_choice = show_menu("Status EGFR", egfr_options)
    egfr = egfr_choice.split(' ')[0] if '(' in egfr_choice else egfr_choice
    egfr_vaf = None
    if egfr and egfr.lower() not in ['wt', 'unknown', '']:
        egfr_vaf = input_number("  → EGFR VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # MET
    met_options = ['wt', 'Amplification (low)', 'Amplification (high)', 'Exon 14 skipping', 'unknown']
    met = show_menu("Status MET", met_options)
    met_cn = None
    if met and 'amplification' in met.lower():
        met_cn = input_number("  → MET Copy Number [Invio=default 5]", 1, 50, allow_empty=True, default=5.0)

    # BRAF
    braf_options = ['wt', 'V600E', 'Other mutation', 'unknown']
    braf_choice = show_menu("Status BRAF", braf_options)
    if braf_choice == 'Other mutation':
        braf = input("Specifica BRAF: ").strip() or 'Other'
    else:
        braf = braf_choice
    braf_vaf = None
    if braf and braf.lower() not in ['wt', 'unknown', '']:
        braf_vaf = input_number("  → BRAF VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # HER2
    her2 = show_menu("HER2 Status", ['wt', 'Amplification', 'Mutation', 'unknown'])
    her2_vaf = None
    if her2 and her2.lower() not in ['wt', 'unknown', '']:
        her2_vaf = input_number("  → HER2 VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # ALK
    alk = show_menu("ALK Status", ['wt', 'Rearrangement (ALK+)', 'unknown'])
    alk_vaf = None
    if alk and 'rearrangement' in alk.lower():
        alk_vaf = input_number("  → ALK VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # STK11 & KEAP1
    stk11 = vcf_data.get('stk11') or show_menu("STK11 Status (Immuno-Resistance)", VALID_GENE_STATUS)
    stk11_vaf = None
    if stk11 and stk11.lower() in ['mutated', 'mut', 'loss']:
        stk11_vaf = input_number("  → STK11 VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    keap1 = vcf_data.get('keap1') or show_menu("KEAP1 Status (Metabolic)", VALID_GENE_STATUS)
    keap1_vaf = None
    if keap1 and keap1.lower() in ['mutated', 'mut', 'loss']:
        keap1_vaf = input_number("  → KEAP1 VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # RB1 (importante per SCLC transformation)
    rb1 = show_menu("RB1 Status (SCLC transformation risk)", VALID_GENE_STATUS)
    rb1_vaf = None
    if rb1 and rb1.lower() in ['mutated', 'mut', 'loss']:
        rb1_vaf = input_number("  → RB1 VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # PIK3CA (nuovo)
    pik3ca = show_menu("PIK3CA Status", VALID_GENE_STATUS)
    pik3ca_vaf = None
    if pik3ca and pik3ca.lower() in ['mutated', 'mut']:
        pik3ca_vaf = input_number("  → PIK3CA VAF % [Invio=sconosciuto]", 0, 100, allow_empty=True, default=None)

    # KRAS

    # TMB & PD-L1
    print("\n--- Biomarkers Immunoterapia ---")
    print("(Premi INVIO per skippare se non disponibile)")
    tmb = input_number("TMB Score (mut/Mb) [Invio=skip]", 0, 500, allow_empty=True, default=None)
    pdl1 = input_number("PD-L1 Expression % [Invio=skip]", 0, 100, allow_empty=True, default=None)

    # ========================================================================
    # 3b. FARMACOGENOMICA (PGx Profile)
    # ========================================================================

    print("\n--- Farmacogenomica (PGx) ---")
    print("(Premi INVIO per skippare - default: unknown)")

    pgx_profile = {}

    collect_pgx = input("\nHai dati farmacogenomici? [y/N]: ").strip().lower()
    if collect_pgx == 'y':
        for gene, options in VALID_PGX_GENOTYPES.items():
            choice = show_menu(f"Genotipo {gene}", options)
            genotype = choice.split(' ')[0]  # Estrae solo il genotipo (es. "*1/*4")
            if genotype != 'unknown':
                pgx_profile[gene] = genotype

        # Warning per genotipi critici
        if pgx_profile.get('DPYD') in ['*1/*2A', '*2A/*2A']:
            print("\n   ⚠️  DPYD variante rilevata: RISCHIO TOSSICITÀ 5-FU/Capecitabina!")
        if pgx_profile.get('UGT1A1') in ['*1/*28', '*28/*28']:
            print("\n   ⚠️  UGT1A1*28 rilevato: Ridurre dose Irinotecan!")
        if pgx_profile.get('CYP2D6') in ['*4/*4']:
            print("\n   ⚠️  CYP2D6 Poor Metabolizer: Attenzione a Tamoxifene, Codeina!")

    # ========================================================================
    # 4. EMATO-ONCOLOGIA (Blood Work) - CAMPI OPZIONALI
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 4: EMATO-ONCOLOGIA (Blood Markers)")
    print("=" * 70)
    print("(Premi INVIO per saltare i campi non disponibili)\n")

    ldh = input_number("LDH (U/L) [Soglia Warburg > 350, Invio=skip]", 0, 5000, allow_empty=True, default=None)
    neutrophils = input_number("Neutrofili Assoluti (es. 4500) [Invio=skip]", 0, 50000, allow_empty=True, default=None)
    lymphocytes = input_number("Linfociti Assoluti (es. 1500) [Invio=skip]", 0, 50000, allow_empty=True, default=None)
    albumin = input_number("Albumina (g/dL) [Invio=skip]", 0, 10, allow_empty=True, default=None)
    cea = input_number("CEA (ng/mL) [Invio=skip]", 0, 10000, allow_empty=True, default=None)

    # Calcola NLR solo se abbiamo entrambi i valori
    if neutrophils and lymphocytes and lymphocytes > 0:
        nlr = round(neutrophils / lymphocytes, 1)
        print(f"\n   → NLR Calcolato: {nlr}")
        if nlr > 5:
            print(f"   ⚠️  NLR elevato (>5): possibile immunosoppressione")
        if nlr > 10:
            print(f"   ⚠️  NLR molto elevato (>10): grave immunosoppressione")
    else:
        nlr = None
        print("\n   ℹ️  NLR non calcolabile (dati mancanti)")

    # Warning LDH
    if ldh and ldh > 350:
        print(f"\n   🐘 LDH ELEVATO ({ldh} U/L): Elephant Protocol verrà attivato!")

    # ========================================================================
    # 5. TERAPIA CORRENTE
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 5: TERAPIA CORRENTE")
    print("=" * 70)

    therapy_choice = show_menu("Terapia Attuale", VALID_THERAPIES)

    if therapy_choice == "None/Naive":
        therapy = "Naive"
    else:
        therapy = therapy_choice

    # ========================================================================
    # 6. VETO PREVIEW
    # ========================================================================

    genetics_dict = {
        # Status
        'tp53_status': tp53,
        'kras_mutation': kras,
        'egfr_status': egfr,
        'met_status': met,
        'braf_status': braf,
        'her2_status': her2,
        'alk_status': alk,
        'stk11_status': stk11,
        'keap1_status': keap1,
        'rb1_status': rb1,
        'pik3ca_status': pik3ca,
        # VAF (per Clonal Tracker)
        'tp53_vaf': tp53_vaf,
        'kras_vaf': kras_vaf,
        'egfr_vaf': egfr_vaf,
        'met_cn': met_cn,
        'braf_vaf': braf_vaf,
        'her2_vaf': her2_vaf,
        'alk_vaf': alk_vaf,
        'stk11_vaf': stk11_vaf,
        'keap1_vaf': keap1_vaf,
        'rb1_vaf': rb1_vaf,
        'pik3ca_vaf': pik3ca_vaf
    }

    # Rimuovi campi None per JSON più pulito
    genetics_dict = {k: v for k, v in genetics_dict.items() if v is not None}

    veto_warnings = check_veto_warnings(therapy, genetics_dict)

    if veto_warnings:
        print("\n" + "🚨" * 35)
        print("INCOMPATIBILITÀ TERAPIA-GENETICA RILEVATA")
        print("🚨" * 35)
        for warning in veto_warnings:
            print(f"\n{warning}")
        print("\n" + "🚨" * 35)

        confirm = input("\n⚠️  Terapia incompatibile rilevata. Continuare comunque? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("\n❌ Registrazione annullata.")
            return

    # ========================================================================
    # 7. BIOPSIA (Vision AI)
    # ========================================================================

    print("\n" + "=" * 70)
    print("SEZIONE 6: DIGITAL PATHOLOGY (Vision AI)")
    print("=" * 70)

    biopsy_files = list(BIOPSY_DIR.glob("*.jpg")) + list(BIOPSY_DIR.glob("*.png"))
    biopsy_path = None

    if biopsy_files:
        print("\nBiopsie disponibili:")
        for i, f in enumerate(biopsy_files, 1):
            print(f"  {i}. {f.name}")

        sel = input("\nSeleziona numero (o Invio per nessuna): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(biopsy_files):
            biopsy_path = str(biopsy_files[int(sel) - 1])
            print(f"✅ Biopsia selezionata: {biopsy_files[int(sel) - 1].name}")
            if sel.isdigit() and 1 <= int(sel) <= len(biopsy_files):
                biopsy_path = str(biopsy_files[int(sel) - 1])
                print(f"✅ Biopsia selezionata: {biopsy_files[int(sel) - 1].name}")

                # === NUOVO BLOCCO PATHOLOGIST ===
                print("\n📋 PATHOLOGIST ASSESSMENT (required for Vision AI)")
                print("   Leave blank to skip Vision AI analysis")

                visual_risk = input("   Visual Risk Score (0-100): ").strip()

                if visual_risk:
                    from vision_ai_net import enter_pathologist_assessment

                    chaos_score = input("   Chaos/Atypia Score (0-10): ").strip() or "5"
                    cellularity = input("   Cellularity (Low/Medium/High): ").strip() or "Medium"
                    classification = input("   Classification: ").strip() or "CARCINOMA NOS"
                    mitosis_count = input("   Mitosis Count (per 10 HPF): ").strip() or "5"
                    assessor = input("   Pathologist Name: ").strip() or "Unknown"

                    enter_pathologist_assessment(
                        image_path=biopsy_path,
                        visual_risk=float(visual_risk),
                        chaos_score=float(chaos_score),
                        cellularity=cellularity,
                        classification=classification,
                        mitosis_count=int(mitosis_count),
                        assessor_name=assessor
                    )
                    print("   ✅ Assessment recorded")
                # === FINE BLOCCO ===

            else:
                print("ℹ️  Nessuna biopsia disponibile in data/biopsies/")
    else:
        print("ℹ️  Nessuna biopsia disponibile in data/biopsies/")

    # ========================================================================
    # 8. RIEPILOGO & SALVATAGGIO
    # ========================================================================

    print("\n" + "=" * 70)
    print("RIEPILOGO PAZIENTE")
    print("=" * 70)
    print(f"ID: {patient_id}")
    print(f"Demografia: {age}y {sex}, ECOG {ecog}, {smoking}, Stage {stage}")
    print(f"Istologia: {histology}")
    print(f"\nGenetics:")
    print(f"  TP53={tp53}" + (f" (VAF:{tp53_vaf}%)" if tp53_vaf else ""))
    print(f"  KRAS={kras}" + (f" (VAF:{kras_vaf}%)" if kras_vaf else ""))
    print(f"  EGFR={egfr}" + (f" (VAF:{egfr_vaf}%)" if egfr_vaf else ""))
    print(f"  MET={met}" + (f" (CN:{met_cn})" if met_cn else ""))
    print(f"  BRAF={braf}, HER2={her2}, ALK={alk}")
    print(f"  STK11={stk11}, KEAP1={keap1}, RB1={rb1}, PIK3CA={pik3ca}")
    print(f"  MET={met}, BRAF={braf}, HER2={her2}, ALK={alk}")
    print(f"  STK11={stk11}, KEAP1={keap1}, RB1={rb1}")
    print(f"\nBiomarkers:")
    print(f"  TMB={tmb if tmb else 'N/A'} mut/Mb, PD-L1={pdl1 if pdl1 else 'N/A'}%")
    if pgx_profile:
        print(f"\nFarmacogenomica (PGx):")
        for gene, genotype in pgx_profile.items():
            print(f"  {gene}: {genotype}")
    print(f"\nBlood:")
    print(f"  LDH={ldh if ldh else 'N/A'} U/L, NLR={nlr if nlr else 'N/A'}")
    print(f"  Neutrophils={neutrophils if neutrophils else 'N/A'}, Lymphocytes={lymphocytes if lymphocytes else 'N/A'}")
    print(f"\nTerapia: {therapy}")
    if biopsy_path:
        print(f"Biopsia: {Path(biopsy_path).name}")
    print("=" * 70)

    confirm = input("\n✅ Confermi i dati? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("❌ Registrazione annullata")
        return

    # Costruzione JSON (None diventa null automaticamente)
    patient_data = {
        "baseline": {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "smoking_status": smoking,
            "ecog_ps": ecog,
            "stage": stage,
            "histology": histology,

            "genetics": genetics_dict,

            "biomarkers": {
                "tmb_score": tmb,  # Può essere None
                "pdl1_percent": pdl1  # Può essere None
            },

            "pgx_profile": pgx_profile if pgx_profile else None,

            "blood_markers": {
                "ldh": ldh,  # Può essere None
                "neutrophils": neutrophils,  # Può essere None
                "lymphocytes": lymphocytes,  # Può essere None
                "nlr": nlr,  # Può essere None
                "albumin": albumin,  # Può essere None
                "cea": cea  # Può essere None
            },

            "current_therapy": therapy,

            "physics_simulation": {
                "binding_energy": -9.0  # Placeholder
            },

            "biopsy_image_path": biopsy_path
        }
    }

    # Salvataggio
    # Salvataggio
    out_path = DATA_DIR / f"{patient_id}.json"
    with open(out_path, 'w') as f:
        json.dump(patient_data, f, indent=2)

    print(f"\n✅ Dati salvati: {out_path}")

    # =========================================================================
    # AI CLINICAL SYNTHESIS (Generazione automatica sommario)
    # =========================================================================
    try:
        from src.clinical_notes_llm import is_available as llm_available, summarize_notes, GOOGLE_API_KEY, MODEL_ID, \
            _call_gemini_safe
        import google.generativeai as genai

        if llm_available():
            print("\n🤖 Generazione AI Clinical Synthesis...")

            # Costruisci prompt per sintesi iniziale
            synthesis_prompt = f"""Sei un oncologo esperto. Analizza i dati di questo NUOVO paziente oncologico e fornisci una sintesi clinica iniziale.

    DATI PAZIENTE:
    - ID: {patient_id}
    - Età/Sesso: {age}/{sex}
    - Istologia: {histology}, Stadio: {stage}
    - Fumo: {smoking}
    - ECOG: {ecog}

    PROFILO GENOMICO:
    - TP53: {tp53} {f'(VAF: {tp53_vaf}%)' if tp53_vaf else ''}
    - KRAS: {kras} {f'(VAF: {kras_vaf}%)' if kras_vaf else ''}
    - EGFR: {egfr}
    - STK11: {stk11} {f'(VAF: {stk11_vaf}%)' if stk11_vaf else ''}
    - KEAP1: {keap1} {f'(VAF: {keap1_vaf}%)' if keap1_vaf else ''}
    - MET: {met}
    - BRAF: {braf}
    - ALK: {alk}
    - RB1: {rb1}
    - PIK3CA: {pik3ca}
    - TMB: {tmb if tmb else 'N/A'} mut/Mb
    - PD-L1: {pdl1 if pdl1 else 'N/A'}%

    ATTENZIONE: Valuta sempre il TMB e il PD-L1 per giustificare il successo o il fallimento dell'Immunoterapia. Un TMB alto (>10) o un PD-L1 alto spiegano una forte risposta immunitaria.

    EMATO-ONCOLOGIA:
    - LDH: {ldh if ldh else 'N/A'} U/L {'(ELEVATO - Warburg)' if ldh and ldh > 350 else ''}
    - NLR: {nlr if nlr else 'N/A'} {'(ELEVATO)' if nlr and nlr > 5 else ''}
    - Albumina: {albumin if albumin else 'N/A'} g/dL

    TERAPIA: {therapy}

    FARMACOGENOMICA (PGx):
    {json.dumps(pgx_profile, indent=2) if pgx_profile else 'Non disponibile'}

    FORNISCI UNA SINTESI IN ITALIANO (max 200 parole) CON:
    1. QUADRO CLINICO INIZIALE (caratteristiche principali)
    2. PROFILO DI RISCHIO (basato su genetica e biomarkers)
    3. CONSIDERAZIONI TERAPEUTICHE INIZIALI
    4. PUNTI DI ATTENZIONE (cosa monitorare)

    Sii conciso. Usa linguaggio medico. NON fare prescrizioni."""

            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(MODEL_ID)
            response = _call_gemini_safe(model, synthesis_prompt)

            if response and response.text:
                ai_synthesis = response.text.strip()

                # Salva nel JSON del paziente
                patient_data['ai_initial_synthesis'] = {
                    'text': ai_synthesis,
                    'generated_at': datetime.now().isoformat(),
                    'model': MODEL_ID
                }

                with open(out_path, 'w') as f:
                    json.dump(patient_data, f, indent=2)

                print("\n" + "=" * 60)
                print("🧠 AI CLINICAL SYNTHESIS")
                print("=" * 60)
                print(ai_synthesis[:500] + "..." if len(ai_synthesis) > 500 else ai_synthesis)
                print("=" * 60)
                print("✅ Sintesi AI salvata nel JSON del paziente")
            else:
                print("⚠️ AI Synthesis: nessuna risposta dal modello")

    except ImportError as e:
        print(f"ℹ️ AI Synthesis non disponibile: {e}")
    except Exception as e:
        print(f"⚠️ Errore AI Synthesis: {e}")

    # ========================================================================
    # 9. GENERAZIONE REPORT PDF
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERAZIONE REPORT SENTINEL v18.1")
    print("=" * 70)

    try:
        try:
            import generate_final_report as gr
        except ImportError:
            import scripts.generate_final_report as gr

        gr.generate_pdf(patient_id)

    except Exception as e:
        print(f"⚠️  Errore generazione report: {e}")
        print("   Il file JSON è stato salvato correttamente.")
        print(f"   Puoi generare il report manualmente con:")
        print(f"   python scripts/generate_final_report.py {patient_id}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        interactive_registration()
    except KeyboardInterrupt:
        print("\n\n❌ Operazione annullata dall'utente")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Errore critico: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
