"""
FRIZIONE ZERO - Upload Referti NGS + JSON
==========================================
Pagina Streamlit per caricamento:
- PDF NGS (parsing automatico con OCR/LLM)
- JSON diretto (import/test pazienti)
- Pazienti test pre-configurati

Posizionare in: scripts/pages/01_upload_referti.py
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

DATA_DIR = BASE_DIR / 'data' / 'patients'
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Import Frizione Zero (opzionale per JSON)
try:
    from frizione_zero import IngestionPipeline, IngestionStatus, ValidationEngine
    FRIZIONE_AVAILABLE = True
except ImportError as e:
    FRIZIONE_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Config
st.set_page_config(page_title="SENTINEL - Upload", page_icon="📄", layout="wide")

# CSS
st.markdown("""
<style>
    .upload-box { border: 2px dashed #4CAF50; border-radius: 10px; padding: 30px; text-align: center; }
    .success-box { background: #E8F5E9; border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; }
    .error-box { background: #FFEBEE; border: 2px solid #D32F2F; border-radius: 10px; padding: 20px; }
    .warning-box { background: #FFF3E0; border: 2px solid #FF9800; border-radius: 10px; padding: 20px; }
    .info-box { background: #E3F2FD; border: 2px solid #2196F3; border-radius: 10px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📄 SENTINEL - Upload Pazienti")
st.markdown("### Caricamento PDF (Frizione Zero) o JSON diretto")

# =============================================================================
# TABS PRINCIPALI
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "📋 Upload JSON", "🧪 Pazienti Test"])

# =============================================================================
# TAB 1: UPLOAD PDF (Frizione Zero)
# =============================================================================
with tab1:
    st.markdown("### 📄 Carica Referto NGS (PDF)")
    
    if not FRIZIONE_AVAILABLE:
        st.warning(f"⚠️ Modulo Frizione Zero non disponibile: {IMPORT_ERROR}")
        st.info("L'upload PDF richiede il modulo Frizione Zero. Usa il tab 'Upload JSON' per importare direttamente.")
    else:
        # Inizializza pipeline
        @st.cache_resource
        def get_pipeline():
            return IngestionPipeline(data_dir=DATA_DIR)
        
        pipeline = get_pipeline()
        
        uploaded_file = st.file_uploader(
            "Seleziona PDF",
            type=['pdf'],
            help="Carica un referto NGS in formato PDF",
            key="pdf_uploader"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_id = st.text_input(
                "Patient ID (opzionale)",
                placeholder="es. PT-001",
                help="Se vuoto, verrà generato automaticamente",
                key="pdf_patient_id"
            )
        
        with col2:
            merge_existing = st.checkbox(
                "Merge con paziente esistente",
                value=True,
                help="Se il paziente esiste, aggiorna i dati invece di sovrascrivere",
                key="pdf_merge"
            )
        
        if uploaded_file:
            st.markdown("---")
            st.markdown(f"**File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 PROCESSA PDF", type="primary", use_container_width=True):
                temp_path = UPLOAD_DIR / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("🔄 Elaborazione in corso..."):
                    result = pipeline.process_pdf(
                        temp_path,
                        patient_id=custom_id if custom_id else None,
                        merge_existing=merge_existing
                    )
                
                if result.status == IngestionStatus.SUCCESS:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Importazione Completata!</h3>
                        <p><b>Patient ID:</b> {result.patient_id}</p>
                        <p><b>Confidence:</b> {result.extraction_confidence * 100:.0f}%</p>
                        <p><b>Tempo:</b> {result.processing_time_ms} ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📋 Dati Estratti", expanded=True):
                        st.json(result.patient_data)
                else:
                    st.error(f"❌ Importazione fallita: {', '.join(result.errors)}")

# =============================================================================
# TAB 2: UPLOAD JSON DIRETTO
# =============================================================================
with tab2:
    st.markdown("### 📋 Carica JSON Paziente")
    st.markdown("Importa direttamente un file JSON nel formato SENTINEL.")
    
    uploaded_json = st.file_uploader(
        "Seleziona JSON",
        type=['json'],
        help="Carica un file JSON paziente",
        key="json_uploader"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        override_id = st.text_input(
            "Override Patient ID (opzionale)",
            placeholder="Lascia vuoto per usare ID dal JSON",
            help="Se specificato, sovrascrive il patient_id nel JSON",
            key="json_override_id"
        )
    
    with col2:
        overwrite_existing = st.checkbox(
            "Sovrascrivi se esiste",
            value=False,
            help="Se il paziente esiste già, sovrascrivilo",
            key="json_overwrite"
        )
    
    if uploaded_json:
        try:
            # Leggi JSON
            json_content = uploaded_json.read().decode('utf-8')
            patient_data = json.loads(json_content)
            
            # Determina patient_id
            if "baseline" in patient_data:
                original_id = patient_data["baseline"].get("patient_id", "UNKNOWN")
            else:
                original_id = patient_data.get("patient_id", "UNKNOWN")
            
            final_id = override_id if override_id else original_id
            
            st.markdown("---")
            st.markdown(f"**File:** {uploaded_json.name}")
            st.markdown(f"**Patient ID:** `{final_id}`")
            
            # Preview
            with st.expander("👁️ Anteprima JSON", expanded=False):
                st.json(patient_data)
            
            # Statistiche
            col_a, col_b, col_c = st.columns(3)
            
            baseline = patient_data.get("baseline", patient_data)
            visits = patient_data.get("visits", [])
            genetics = baseline.get("genetics", {})
            
            col_a.metric("Visite", len(visits))
            col_b.metric("Geni", len(genetics))
            col_c.metric("Terapia", baseline.get("current_therapy", "N/A")[:20])
            
            # Check se esiste
            target_path = DATA_DIR / f"{final_id}.json"
            if target_path.exists() and not overwrite_existing:
                st.warning(f"⚠️ Paziente `{final_id}` esiste già! Attiva 'Sovrascrivi se esiste' per procedere.")
                can_save = False
            else:
                can_save = True
            
            if can_save and st.button("💾 SALVA PAZIENTE", type="primary", use_container_width=True):
                # Aggiorna patient_id se override
                if override_id:
                    if "baseline" in patient_data:
                        patient_data["baseline"]["patient_id"] = override_id
                    else:
                        patient_data["patient_id"] = override_id
                
                # Aggiungi metadata
                if "baseline" in patient_data:
                    patient_data["baseline"]["imported_at"] = datetime.now().isoformat()
                    patient_data["baseline"]["source"] = "json_upload"
                
                # Salva
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(patient_data, f, indent=2, ensure_ascii=False)
                
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Paziente Salvato!</h3>
                    <p><b>Patient ID:</b> {final_id}</p>
                    <p><b>Visite:</b> {len(visits)}</p>
                    <p><b>Path:</b> {target_path}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
        
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON non valido: {e}")
        except Exception as e:
            st.error(f"❌ Errore: {e}")

# =============================================================================
# TAB 3: PAZIENTI TEST PRE-CONFIGURATI
# =============================================================================
with tab3:
    st.markdown("### 🧪 Crea Pazienti di Test")
    st.markdown("Genera pazienti pre-configurati per testare Prophet e Oracle.")
    
    st.markdown("---")
    
    # Test Patient 1: Progressione (Prophet CRITICAL)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 TEST_PROGRESSION")
        st.markdown("Paziente con **progressione rapida** → Prophet CRITICAL")
        st.markdown("""
        - 5 visite su 16 settimane
        - LDH: 220 → 420 (+90%)
        - VAF: 0.5% → 5.5%
        - Trend: accelerazione
        """)
        
        if st.button("➕ Crea TEST_PROGRESSION", key="create_prog"):
            test_progression = {
                "baseline": {
                    "patient_id": "TEST_PROGRESSION",
                    "age": 62,
                    "ecog_ps": 1,
                    "stage": "IIIB",
                    "current_therapy": "Pembrolizumab",
                    "genetics": {"tp53_status": "mutated", "kras_mutation": "G12C"},
                    "blood_markers": {"ldh": 220},
                    "biomarkers": {"tmb_score": 14, "pd_l1_tps": 60}
                },
                "visits": [
                    {"week_on_therapy": 0, "blood_markers": {"ldh": 220, "crp": 2.0, "neutrophils": 5000, "lymphocytes": 1800}, "genetics": {"tp53_vaf": 0.5, "kras_vaf": 0.3}},
                    {"week_on_therapy": 4, "blood_markers": {"ldh": 260, "crp": 2.8, "neutrophils": 5800, "lymphocytes": 1600}, "genetics": {"tp53_vaf": 1.2, "kras_vaf": 0.8}},
                    {"week_on_therapy": 8, "blood_markers": {"ldh": 310, "crp": 3.8, "neutrophils": 6800, "lymphocytes": 1400}, "genetics": {"tp53_vaf": 2.5, "kras_vaf": 1.6}},
                    {"week_on_therapy": 12, "blood_markers": {"ldh": 370, "crp": 5.2, "neutrophils": 7800, "lymphocytes": 1200}, "genetics": {"tp53_vaf": 4.0, "kras_vaf": 2.8}},
                    {"week_on_therapy": 16, "blood_markers": {"ldh": 420, "crp": 6.5, "neutrophils": 9000, "lymphocytes": 1000}, "genetics": {"tp53_vaf": 5.5, "kras_vaf": 4.0}}
                ]
            }
            
            path = DATA_DIR / "TEST_PROGRESSION.json"
            with open(path, 'w') as f:
                json.dump(test_progression, f, indent=2)
            st.success(f"✅ Creato: {path}")
    
    with col2:
        st.markdown("#### 🟢 TEST_RESPONDER")
        st.markdown("Paziente con **risposta alla terapia** → Prophet LOW")
        st.markdown("""
        - 4 visite su 12 settimane
        - LDH: 200 → 165 (-18%)
        - VAF: 2.0% → 0.2%
        - Trend: discesa
        """)
        
        if st.button("➕ Crea TEST_RESPONDER", key="create_resp"):
            test_responder = {
                "baseline": {
                    "patient_id": "TEST_RESPONDER",
                    "age": 55,
                    "ecog_ps": 0,
                    "stage": "IIA",
                    "current_therapy": "Osimertinib",
                    "genetics": {"egfr_status": "mutated", "egfr_mutation": "L858R"},
                    "blood_markers": {"ldh": 200},
                    "biomarkers": {"tmb_score": 8}
                },
                "visits": [
                    {"week_on_therapy": 0, "blood_markers": {"ldh": 200, "crp": 1.5, "neutrophils": 4800}, "genetics": {"egfr_vaf": 2.0}},
                    {"week_on_therapy": 4, "blood_markers": {"ldh": 185, "crp": 1.2, "neutrophils": 4600}, "genetics": {"egfr_vaf": 1.0}},
                    {"week_on_therapy": 8, "blood_markers": {"ldh": 172, "crp": 0.9, "neutrophils": 4400}, "genetics": {"egfr_vaf": 0.4}},
                    {"week_on_therapy": 12, "blood_markers": {"ldh": 165, "crp": 0.7, "neutrophils": 4200}, "genetics": {"egfr_vaf": 0.2}}
                ]
            }
            
            path = DATA_DIR / "TEST_RESPONDER.json"
            with open(path, 'w') as f:
                json.dump(test_responder, f, indent=2)
            st.success(f"✅ Creato: {path}")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🔮 TEST_ORACLE_DRIFT")
        st.markdown("Soggetto sano con **drift metabolico** → Oracle Alert")
        st.markdown("""
        - 6 visite su 2.5 anni
        - Glicemia: 84 → 106 mg/dL
        - Peso: 82 → 75 kg
        - Pattern: divergente
        """)
        
        if st.button("➕ Crea TEST_ORACLE_DRIFT", key="create_oracle"):
            test_oracle = {
                "baseline": {
                    "patient_id": "TEST_ORACLE_DRIFT",
                    "age": 58,
                    "screening": True
                },
                "visits": [
                    {"date": "2022-01", "blood": {"glucose": 84, "ldh": 185, "crp": 1.2, "neutrophils": 4200, "lymphocytes": 1800, "albumin": 4.5}, "clinical": {"weight": 82.0}},
                    {"date": "2022-07", "blood": {"glucose": 88, "ldh": 192, "crp": 1.5, "neutrophils": 4500, "lymphocytes": 1700, "albumin": 4.4}, "clinical": {"weight": 80.5}},
                    {"date": "2023-01", "blood": {"glucose": 92, "ldh": 200, "crp": 1.9, "neutrophils": 4900, "lymphocytes": 1550, "albumin": 4.3}, "clinical": {"weight": 79.0}},
                    {"date": "2023-07", "blood": {"glucose": 96, "ldh": 210, "crp": 2.4, "neutrophils": 5400, "lymphocytes": 1400, "albumin": 4.1}, "clinical": {"weight": 77.2}},
                    {"date": "2024-01", "blood": {"glucose": 101, "ldh": 222, "crp": 3.0, "neutrophils": 5900, "lymphocytes": 1250, "albumin": 3.9}, "clinical": {"weight": 76.0}},
                    {"date": "2024-07", "blood": {"glucose": 106, "ldh": 238, "crp": 3.8, "neutrophils": 6500, "lymphocytes": 1100, "albumin": 3.7}, "clinical": {"weight": 75.0}}
                ]
            }
            
            path = DATA_DIR / "TEST_ORACLE_DRIFT.json"
            with open(path, 'w') as f:
                json.dump(test_oracle, f, indent=2)
            st.success(f"✅ Creato: {path}")
    
    with col4:
        st.markdown("#### 🐘 TEST_ELEPHANT")
        st.markdown("Paziente con **LDH alto** → Protocollo Elefante")
        st.markdown("""
        - LDH > 350 U/L
        - Metabolismo Warburg attivo
        - Trigger: Metformina
        """)
        
        if st.button("➕ Crea TEST_ELEPHANT", key="create_elephant"):
            test_elephant = {
                "baseline": {
                    "patient_id": "TEST_ELEPHANT",
                    "age": 68,
                    "ecog_ps": 2,
                    "stage": "IV",
                    "current_therapy": "Carboplatin + Pemetrexed",
                    "genetics": {"tp53_status": "mutated", "stk11_status": "mutated"},
                    "blood_markers": {"ldh": 480},
                    "biomarkers": {"tmb_score": 6, "pd_l1_tps": 10}
                },
                "visits": [
                    {"week_on_therapy": 0, "blood_markers": {"ldh": 380, "crp": 8.0, "neutrophils": 8500}, "genetics": {"tp53_vaf": 8.0}},
                    {"week_on_therapy": 3, "blood_markers": {"ldh": 420, "crp": 9.5, "neutrophils": 9200}, "genetics": {"tp53_vaf": 10.5}},
                    {"week_on_therapy": 6, "blood_markers": {"ldh": 480, "crp": 12.0, "neutrophils": 10500}, "genetics": {"tp53_vaf": 14.0}}
                ]
            }
            
            path = DATA_DIR / "TEST_ELEPHANT.json"
            with open(path, 'w') as f:
                json.dump(test_elephant, f, indent=2)
            st.success(f"✅ Creato: {path}")
    
    st.markdown("---")
    
    # Crea tutti
    if st.button("🚀 CREA TUTTI I PAZIENTI TEST", type="primary", use_container_width=True):
        # Progression
        test_progression = {
            "baseline": {"patient_id": "TEST_PROGRESSION", "age": 62, "ecog_ps": 1, "stage": "IIIB", "current_therapy": "Pembrolizumab", "genetics": {"tp53_status": "mutated", "kras_mutation": "G12C"}, "blood_markers": {"ldh": 220}, "biomarkers": {"tmb_score": 14}},
            "visits": [
                {"week_on_therapy": 0, "blood_markers": {"ldh": 220, "crp": 2.0, "neutrophils": 5000, "lymphocytes": 1800}, "genetics": {"tp53_vaf": 0.5}},
                {"week_on_therapy": 4, "blood_markers": {"ldh": 260, "crp": 2.8, "neutrophils": 5800, "lymphocytes": 1600}, "genetics": {"tp53_vaf": 1.2}},
                {"week_on_therapy": 8, "blood_markers": {"ldh": 310, "crp": 3.8, "neutrophils": 6800, "lymphocytes": 1400}, "genetics": {"tp53_vaf": 2.5}},
                {"week_on_therapy": 12, "blood_markers": {"ldh": 370, "crp": 5.2, "neutrophils": 7800, "lymphocytes": 1200}, "genetics": {"tp53_vaf": 4.0}},
                {"week_on_therapy": 16, "blood_markers": {"ldh": 420, "crp": 6.5, "neutrophils": 9000, "lymphocytes": 1000}, "genetics": {"tp53_vaf": 5.5}}
            ]
        }
        
        # Responder
        test_responder = {
            "baseline": {"patient_id": "TEST_RESPONDER", "age": 55, "ecog_ps": 0, "stage": "IIA", "current_therapy": "Osimertinib", "genetics": {"egfr_status": "mutated"}, "blood_markers": {"ldh": 200}},
            "visits": [
                {"week_on_therapy": 0, "blood_markers": {"ldh": 200, "crp": 1.5}, "genetics": {"egfr_vaf": 2.0}},
                {"week_on_therapy": 4, "blood_markers": {"ldh": 185, "crp": 1.2}, "genetics": {"egfr_vaf": 1.0}},
                {"week_on_therapy": 8, "blood_markers": {"ldh": 172, "crp": 0.9}, "genetics": {"egfr_vaf": 0.4}},
                {"week_on_therapy": 12, "blood_markers": {"ldh": 165, "crp": 0.7}, "genetics": {"egfr_vaf": 0.2}}
            ]
        }
        
        # Oracle
        test_oracle = {
            "baseline": {"patient_id": "TEST_ORACLE_DRIFT", "age": 58, "screening": True},
            "visits": [
                {"date": "2022-01", "blood": {"glucose": 84, "ldh": 185, "crp": 1.2, "neutrophils": 4200, "lymphocytes": 1800, "albumin": 4.5}, "clinical": {"weight": 82.0}},
                {"date": "2022-07", "blood": {"glucose": 88, "ldh": 192, "crp": 1.5, "neutrophils": 4500, "lymphocytes": 1700, "albumin": 4.4}, "clinical": {"weight": 80.5}},
                {"date": "2023-01", "blood": {"glucose": 92, "ldh": 200, "crp": 1.9, "neutrophils": 4900, "lymphocytes": 1550, "albumin": 4.3}, "clinical": {"weight": 79.0}},
                {"date": "2023-07", "blood": {"glucose": 96, "ldh": 210, "crp": 2.4, "neutrophils": 5400, "lymphocytes": 1400, "albumin": 4.1}, "clinical": {"weight": 77.2}},
                {"date": "2024-01", "blood": {"glucose": 101, "ldh": 222, "crp": 3.0, "neutrophils": 5900, "lymphocytes": 1250, "albumin": 3.9}, "clinical": {"weight": 76.0}},
                {"date": "2024-07", "blood": {"glucose": 106, "ldh": 238, "crp": 3.8, "neutrophils": 6500, "lymphocytes": 1100, "albumin": 3.7}, "clinical": {"weight": 75.0}}
            ]
        }
        
        # Elephant
        test_elephant = {
            "baseline": {"patient_id": "TEST_ELEPHANT", "age": 68, "ecog_ps": 2, "stage": "IV", "current_therapy": "Carboplatin", "genetics": {"tp53_status": "mutated", "stk11_status": "mutated"}, "blood_markers": {"ldh": 480}},
            "visits": [
                {"week_on_therapy": 0, "blood_markers": {"ldh": 380, "crp": 8.0}, "genetics": {"tp53_vaf": 8.0}},
                {"week_on_therapy": 3, "blood_markers": {"ldh": 420, "crp": 9.5}, "genetics": {"tp53_vaf": 10.5}},
                {"week_on_therapy": 6, "blood_markers": {"ldh": 480, "crp": 12.0}, "genetics": {"tp53_vaf": 14.0}}
            ]
        }
        
        # PGx - Paziente con varianti farmacogenomiche (FOLFOX con DPYD mutato)
        test_pgx = {
            "baseline": {
                "patient_id": "TEST_PGX_TOXICITY",
                "age": 58,
                "ecog_ps": 1,
                "stage": "IIIA",
                "current_therapy": "FOLFOX (5-FU + Oxaliplatino + Leucovorin)",
                "genetics": {
                    "tp53_status": "wild-type",
                    "kras_status": "wild-type",
                    "dpyd_status": "*1/*2A",
                    "ugt1a1_status": "*28/*28",
                    "cyp2d6_status": "Poor Metabolizer",
                    "tpmt_status": "*1/*1",
                    "g6pd_status": "Normal"
                },
                "blood_markers": {"ldh": 210},
                "biomarkers": {"tmb_score": 5}
            },
            "visits": [
                {"week_on_therapy": 0, "blood_markers": {"ldh": 210, "neutrophils": 5500}, "genetics": {}},
                {"week_on_therapy": 3, "blood_markers": {"ldh": 220, "neutrophils": 4200}, "genetics": {}}
            ]
        }
        
        # Salva tutti
        for name, data in [("TEST_PROGRESSION", test_progression), ("TEST_RESPONDER", test_responder), ("TEST_ORACLE_DRIFT", test_oracle), ("TEST_ELEPHANT", test_elephant), ("TEST_PGX_TOXICITY", test_pgx)]:
            path = DATA_DIR / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        st.success("✅ Creati 5 pazienti di test!")
        st.balloons()
        
        st.markdown("""
        ### 🎯 Come Testarli
        
        **⏱️ TEMPORAL ENGINE:**
        
        | Paziente | Engine | Risultato Atteso |
        |----------|--------|------------------|
        | TEST_PROGRESSION | Prophet | 🔴 CRITICAL |
        | TEST_RESPONDER | Prophet | 🟢 LOW |
        | TEST_ORACLE_DRIFT | Oracle | 🔮 Metabolic Drift |
        | TEST_ELEPHANT | Prophet | 🐘 Elephant Protocol |
        
        **💊 FARMACOGENOMICA:**
        
        | Paziente | Alert Atteso |
        |----------|--------------|
        | TEST_PGX_TOXICITY | 🚨 DPYD *2A → ridurre 5-FU 50% |
        |                    | ⚠️ UGT1A1 *28/*28 → ridurre Irinotecano |
        """)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 📊 Database")
    
    # Lista pazienti
    patients = list(DATA_DIR.glob("*.json"))
    st.metric("Pazienti Totali", len(patients))
    
    if patients:
        with st.expander("📋 Lista Pazienti"):
            for p in sorted(patients):
                st.caption(f"• {p.stem}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Formati Supportati")
    st.caption("""
    **PDF** (Frizione Zero):
    - Referti NGS
    - Estrazione automatica con OCR/LLM
    
    **JSON** (Diretto):
    - Formato SENTINEL standard
    - Import pazienti esistenti
    - Test e debug
    """)

# Footer
st.markdown("---")
st.caption(f"SENTINEL Upload v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
