"""
FARMACOGENOMICA - SENTINEL Dashboard Page
==========================================
Pagina Streamlit per analisi farmacogenomica.
Visualizza alert PGx e raccomandazioni CPIC/DPWG.

Posizionare in: pages/02_farmacogenomica.py
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path

# Setup path - scripts/pages/ → risali 2 livelli per SENTINEL_TRIAL/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

# Import moduli farmacogenomica
try:
    from farmacogenomica.pharmgkb_database import PharmGKBDatabase
    from farmacogenomica.pgx_extractor import PGxExtractor
    from farmacogenomica.metabolizer_classifier import MetabolizerClassifier, MetabolizerPhenotype
    from farmacogenomica.drug_interaction import DrugInteractionEngine
    from farmacogenomica.pgx_recommender import PGxRecommender
    from farmacogenomica.pgx_alert_engine import PGxAlertEngine, AlertPriority
    PGX_AVAILABLE = True
except ImportError as e:
    PGX_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Config
st.set_page_config(page_title="SENTINEL - Farmacogenomica", page_icon="💊", layout="wide")

# CSS
st.markdown("""
<style>
    .alert-critical { border-left: 4px solid #B71C1C; background: rgba(183,28,28,0.1); padding: 15px; margin: 10px 0; border-radius: 4px; }
    .alert-high { border-left: 4px solid #D32F2F; background: rgba(211,47,47,0.1); padding: 15px; margin: 10px 0; border-radius: 4px; }
    .alert-moderate { border-left: 4px solid #FF9800; background: rgba(255,152,0,0.1); padding: 15px; margin: 10px 0; border-radius: 4px; }
    .alert-low { border-left: 4px solid #2196F3; background: rgba(33,150,243,0.1); padding: 15px; margin: 10px 0; border-radius: 4px; }
    .gene-chip { display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 15px; font-size: 13px; font-weight: bold; }
    .chip-pm { background: #D32F2F; color: white; }
    .chip-im { background: #FF9800; color: white; }
    .chip-nm { background: #4CAF50; color: white; }
    .chip-unknown { background: #757575; color: white; }
    .status-safe { background: #4CAF50; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .status-warning { background: #FF9800; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    .status-danger { background: #D32F2F; color: white; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("💊 SENTINEL - Farmacogenomica")
st.markdown("### Predizione Tossicità e Raccomandazioni CPIC/DPWG")

if not PGX_AVAILABLE:
    st.error(f"❌ Modulo Farmacogenomica non disponibile: {IMPORT_ERROR}")
    st.stop()

# Inizializza componenti
@st.cache_resource
def init_pgx_engine():
    return PGxAlertEngine()

alert_engine = init_pgx_engine()

# Session state
if 'pgx_patient_data' not in st.session_state:
    st.session_state.pgx_patient_data = None
if 'pgx_summary' not in st.session_state:
    st.session_state.pgx_summary = None

# Sidebar - Caricamento paziente
st.sidebar.header("📂 Carica Paziente")

# Opzione 1: Da file JSON
uploaded_file = st.sidebar.file_uploader("Carica JSON paziente", type=['json'])

# Opzione 2: Da database esistente
DATA_DIR = BASE_DIR / 'data' / 'patients'
if DATA_DIR.exists():
    json_files = list(DATA_DIR.glob("*.json"))
    if json_files:
        patient_options = ["-- Seleziona --"] + [f.stem for f in json_files]
        selected_patient = st.sidebar.selectbox("Oppure seleziona paziente:", patient_options)
        
        if selected_patient != "-- Seleziona --":
            with open(DATA_DIR / f"{selected_patient}.json", 'r') as f:
                st.session_state.pgx_patient_data = json.load(f)

if uploaded_file:
    st.session_state.pgx_patient_data = json.load(uploaded_file)

# Opzione 3: Input manuale terapia e PGx
st.sidebar.markdown("---")
st.sidebar.header("✏️ Input Manuale")

manual_therapy = st.sidebar.text_input("Terapia", placeholder="es. FOLFOX, 5-FU + Irinotecano")

st.sidebar.markdown("**Varianti PGx note:**")
manual_dpyd = st.sidebar.selectbox("DPYD", ["Non testato", "*1/*1 (Normal)", "*1/*2A (IM)", "*2A/*2A (PM)"])
manual_ugt1a1 = st.sidebar.selectbox("UGT1A1", ["Non testato", "*1/*1 (Normal)", "*1/*28 (IM)", "*28/*28 (PM)"])
manual_cyp2d6 = st.sidebar.selectbox("CYP2D6", ["Non testato", "Normal", "Intermediate", "Poor"])
manual_tpmt = st.sidebar.selectbox("TPMT", ["Non testato", "*1/*1 (Normal)", "*1/*3A (IM)", "*3A/*3A (PM)"])
manual_g6pd = st.sidebar.selectbox("G6PD", ["Non testato", "Normal", "Deficient"])

if st.sidebar.button("🔬 Analizza Input Manuale"):
    # Costruisci paziente da input manuale
    genetics = {}
    if manual_dpyd != "Non testato":
        genetics['dpyd_status'] = manual_dpyd.split(" ")[0]
    if manual_ugt1a1 != "Non testato":
        genetics['ugt1a1_status'] = manual_ugt1a1.split(" ")[0]
    if manual_cyp2d6 != "Non testato":
        genetics['cyp2d6_status'] = manual_cyp2d6
    if manual_tpmt != "Non testato":
        genetics['tpmt_status'] = manual_tpmt.split(" ")[0]
    if manual_g6pd != "Non testato":
        genetics['g6pd_status'] = manual_g6pd
    
    st.session_state.pgx_patient_data = {
        'baseline': {
            'patient_id': 'MANUAL-001',
            'current_therapy': manual_therapy,
            'genetics': genetics
        }
    }

# Main content
if st.session_state.pgx_patient_data:
    patient_data = st.session_state.pgx_patient_data
    baseline = patient_data.get('baseline', patient_data)

    # === FIX: Converti pgx_profile in formato genetics per PGxExtractor ===
    pgx_profile = baseline.get('pgx_profile', {})
    if pgx_profile:
        genetics = baseline.get('genetics', {})
        # Mappa pgx_profile → genetics con chiavi _status
        for gene, genotype in pgx_profile.items():
            if genotype:
                key = f"{gene.lower()}_status"
                genetics[key] = genotype
        baseline['genetics'] = genetics
        patient_data['baseline'] = baseline
    # === FINE FIX ===
    
    # Header paziente
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"### 👤 Paziente: {baseline.get('patient_id', 'N/A')}")
    with col2:
        therapy = baseline.get('current_therapy', 'Non specificata')
        st.markdown(f"### 💊 Terapia: {therapy}")
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state.pgx_summary = None
            st.rerun()
    
    st.markdown("---")
    
    # Analisi
    if st.session_state.pgx_summary is None:
        with st.spinner("🔬 Analisi farmacogenomica in corso..."):
            st.session_state.pgx_summary = alert_engine.analyze_patient_therapy(patient_data)
    
    summary = st.session_state.pgx_summary
    
    # Status Box
    if summary.has_contraindications:
        st.markdown("""
        <div class="status-danger">
            <h2>🚨 CONTROINDICAZIONI FARMACOGENOMICHE</h2>
            <p>Uno o più farmaci sono controindicati per questo paziente</p>
        </div>
        """, unsafe_allow_html=True)
    elif not summary.therapy_safe:
        st.markdown("""
        <div class="status-warning">
            <h2>⚠️ RICHIEDE REVISIONE TERAPIA</h2>
            <p>Alert ad alta priorità richiedono attenzione</p>
        </div>
        """, unsafe_allow_html=True)
    elif summary.total_alerts > 0:
        st.markdown("""
        <div class="status-warning">
            <h2>⚡ ALERT PRESENTI</h2>
            <p>Considerare le raccomandazioni PGx</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-safe">
            <h2>✅ NESSUN ALERT FARMACOGENOMICO</h2>
            <p>Nessuna interazione PGx critica rilevata</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Metriche
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Alert Totali", summary.total_alerts)
    col2.metric("🚨 Critici", summary.critical_count, delta="STOP" if summary.critical_count > 0 else None, delta_color="inverse")
    col3.metric("⚠️ Alti", summary.high_count)
    col4.metric("⚡ Moderati", summary.moderate_count)
    col5.metric("ℹ️ Info", summary.low_count)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Alert", "🧬 Profilo PGx", "📚 Database"])
    
    with tab1:
        if summary.alerts:
            for alert in summary.alerts:
                # Alert card
                css_class = f"alert-{alert.priority.name.lower()}"
                
                with st.container():
                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="display: flex; justify-content: space-between;">
                            <strong style="font-size: 16px;">{alert.priority.icon} {alert.title}</strong>
                            <span style="background: {alert.priority.color}; color: white; 
                                        padding: 2px 10px; border-radius: 10px; font-size: 12px;">
                                {alert.priority.name}
                            </span>
                        </div>
                        <p style="margin: 10px 0;"><strong>{alert.drug}</strong>: {alert.message}</p>
                        <p style="color: #FFD54F;"><strong>📋 Azione:</strong> {alert.action_required}</p>
                        <small style="color: #888;">{alert.gene} | {alert.evidence_level.value} | {alert.source}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Raccomandazione dettagliata in expander
                    if alert.recommendation:
                        with st.expander(f"📖 Dettagli CPIC per {alert.drug}"):
                            rec = alert.recommendation
                            st.write(f"**Azione:** {rec.action_detail}")
                            if rec.dose_adjustment:
                                st.write(f"**Dosaggio:** {rec.dose_adjustment}")
                            if rec.alternatives:
                                st.write(f"**Alternative:** {', '.join(rec.alternatives)}")
                            if rec.monitoring:
                                st.write(f"**Monitoraggio:** {', '.join(rec.monitoring)}")
                            st.caption(f"Fonte: {rec.guideline_source} | Evidenza: {rec.evidence_level.value}")
                            if rec.black_box_warning:
                                st.error("⬛ BLACK BOX WARNING FDA")
        else:
            st.success("✅ Nessun alert farmacogenomico per la terapia attuale")
        
        # Warning geni non testati
        if summary.genes_not_tested:
            st.warning(f"⚠️ **Geni PGx non testati:** {', '.join(summary.genes_not_tested)}")
            st.caption("Considera test farmacogenomico prima di terapie con fluoropirimidine, irinotecano, tamoxifene, tiopurine o rasburicase.")
    
    with tab2:
        st.markdown("### Profilo Farmacogenomico")
        
        # Estrai e mostra fenotipi
        extractor = PGxExtractor()
        classifier = MetabolizerClassifier()
        extraction = extractor.extract_from_sentinel(patient_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧬 Geni Testati")
            if extraction.variants_found:
                for variant in extraction.variants_found:
                    genotype = variant.genotype or variant.variant
                    result = classifier.classify(variant.gene, genotype)
                    
                    # Chip colorato
                    if result.phenotype == MetabolizerPhenotype.POOR:
                        chip_class = "chip-pm"
                    elif result.phenotype == MetabolizerPhenotype.INTERMEDIATE:
                        chip_class = "chip-im"
                    elif result.phenotype == MetabolizerPhenotype.NORMAL:
                        chip_class = "chip-nm"
                    else:
                        chip_class = "chip-unknown"
                    
                    st.markdown(f"""
                    <span class="gene-chip {chip_class}">{variant.gene}: {result.phenotype.abbreviation}</span>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"Genotipo: {genotype}")
                    if result.activity_score is not None:
                        st.caption(f"Activity Score: {result.activity_score}")
            else:
                st.info("Nessuna variante PGx trovata nei dati")
        
        with col2:
            st.markdown("#### ⚠️ Geni Non Testati")
            if extraction.genes_not_tested:
                for gene in extraction.genes_not_tested:
                    st.markdown(f'<span class="gene-chip chip-unknown">{gene}: ?</span>', unsafe_allow_html=True)
            else:
                st.success("Tutti i geni PGx critici sono stati testati")
        
        # Raccomandazioni test
        if extraction.recommendations:
            st.markdown("---")
            st.markdown("#### 📋 Test Raccomandati")
            for rec in extraction.recommendations:
                st.warning(rec)
    
    with tab3:
        st.markdown("### 📚 Database PharmGKB")
        
        db = PharmGKBDatabase()
        
        # Ricerca
        search_query = st.text_input("🔍 Cerca gene o farmaco", placeholder="es. DPYD, 5-FU, tamoxifene")
        
        if search_query:
            results = db.search(search_query)
            
            if results['interactions']:
                st.markdown(f"**{len(results['interactions'])} interazioni trovate:**")
                for inter in results['interactions'][:10]:
                    with st.expander(f"{inter.gene} + {inter.drug}: {inter.phenotype}"):
                        st.write(f"**Effetto:** {inter.effect}")
                        st.write(f"**Raccomandazione:** {inter.recommendation}")
                        st.write(f"**Tox Risk:** {inter.toxicity_risk}% | **Efficacy Risk:** {inter.efficacy_risk}%")
                        st.write(f"**Evidenza:** {inter.evidence_level.value} | **Fonte:** {inter.source}")
                        if inter.is_life_threatening:
                            st.error("⚠️ POTENZIALMENTE FATALE")
            else:
                st.info("Nessun risultato trovato")
        
        # Stats database
        st.markdown("---")
        st.markdown("#### 📊 Statistiche Database")
        col1, col2, col3 = st.columns(3)
        col1.metric("Geni", len(db.genes))
        col2.metric("Farmaci", len(db.drugs))
        col3.metric("Interazioni", sum(len(v) for v in db.interactions.values()))
        
        st.markdown("**Geni critici (life-threatening):**")
        critical_genes = db.get_critical_genes()
        st.write(", ".join(critical_genes))

else:
    # No patient loaded
    st.info("👈 Carica un paziente dalla sidebar o inserisci i dati manualmente")
    
    # Demo
    st.markdown("---")
    st.markdown("### 🎯 Demo Rapida")
    
    if st.button("▶️ Esegui Demo (Paziente con DPYD *2A + FOLFOX)"):
        demo_patient = {
            'baseline': {
                'patient_id': 'DEMO-PGX-001',
                'current_therapy': 'FOLFOX (5-FU + Oxaliplatino)',
                'genetics': {
                    'dpyd_status': '*1/*2A heterozygous',
                    'ugt1a1_status': '*1/*28',
                    'tp53_status': 'mutated'
                }
            }
        }
        st.session_state.pgx_patient_data = demo_patient
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    SENTINEL Farmacogenomica v1.0 | Database: PharmGKB + CPIC/DPWG<br>
    ⚠️ Solo per uso di ricerca - Le raccomandazioni devono essere validate da un farmacologo clinico
</div>
""", unsafe_allow_html=True)
