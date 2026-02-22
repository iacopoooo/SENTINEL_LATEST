"""
SENTINEL CLINICAL SAFETY — Dashboard per il Medico
====================================================
Il medico incolla gli esami del sangue → parser li traduce in JSON →
ClinicalSafetyEngine analizza → banner colorati con alert salvavita.

Pagina Streamlit: scripts/pages/04_clinical_safety.py
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

# Import Parser
from lab_parser import (
    parse_lab_text, build_patient_data,
    REFERENCE_RANGES, DISPLAY_NAMES
)

# Import Safety Engine
try:
    from safety_alerts import ClinicalSafetyEngine, ClinicalAlert, AlertSeverity, AlertCategory
    SAFETY_AVAILABLE = True
except ImportError as e:
    SAFETY_AVAILABLE = False
    SAFETY_ERROR = str(e)

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="SENTINEL - Safety Check",
    page_icon="🚨",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    /* Alert banners */
    .alert-critical {
        background: linear-gradient(135deg, #B71C1C, #D32F2F);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(211, 47, 47, 0.4);
        animation: pulse-red 2s infinite;
    }
    .alert-high {
        background: linear-gradient(135deg, #E65100, #FF9800);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.3);
    }
    .alert-moderate {
        background: linear-gradient(135deg, #F57F17, #FFCA28);
        color: #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(255, 202, 40, 0.3);
    }
    .alert-low {
        background: linear-gradient(135deg, #1B5E20, #4CAF50);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    .alert-title {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .alert-message {
        font-size: 1.0em;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    .alert-actions {
        font-size: 0.9em;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Pulse animation for critical */
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 12px rgba(211, 47, 47, 0.4); }
        50% { box-shadow: 0 4px 24px rgba(211, 47, 47, 0.8); }
        100% { box-shadow: 0 4px 12px rgba(211, 47, 47, 0.4); }
    }
    
    /* All clear box */
    .all-clear {
        background: linear-gradient(135deg, #1B5E20, #388E3C);
        color: white;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        font-size: 1.5em;
        box-shadow: 0 4px 12px rgba(56, 142, 60, 0.4);
    }
    
    /* Header */
    .sentinel-header {
        background: linear-gradient(135deg, #0D47A1, #1565C0);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.4);
    }
    .sentinel-header h1 { margin: 0; font-size: 2em; }
    .sentinel-header p { margin: 5px 0 0 0; opacity: 0.8; }
    
    /* Stats */
    .stat-number { font-size: 2.5em; font-weight: bold; }
    .stat-label { font-size: 0.8em; opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="sentinel-header">
    <h1>🏥 SENTINEL — Clinical Safety Check</h1>
    <p>Incolla gli esami del sangue • Premi Analizza • Ricevi alert salvavita in tempo reale</p>
</div>
""", unsafe_allow_html=True)

if not SAFETY_AVAILABLE:
    st.error(f"❌ Modulo Safety non disponibile: {SAFETY_ERROR}")
    st.stop()

# =============================================================================
# LAYOUT: INPUT + RESULTS
# =============================================================================

col_input, col_results = st.columns([1, 2])

# ========================= LEFT COLUMN: INPUT =========================
with col_input:
    st.markdown("### 📋 Dati Paziente")
    
    with st.expander("⚙️ Contesto Clinico (opzionale)", expanded=False):
        patient_id = st.text_input("Patient ID", value="QUICK_CHECK", key="cs_pid")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Età", min_value=1, max_value=120, value=65, key="cs_age")
        with col_b:
            sex = st.selectbox("Sesso", ["M", "F"], key="cs_sex")
        therapy = st.text_input("Terapia attuale", placeholder="es. Osimertinib", key="cs_therapy")
        histology = st.text_input("Istologia", value="NSCLC", key="cs_hist")
    
    st.markdown("### 📝 Esami del Sangue")
    st.caption("Incolla qui sotto i risultati degli esami. Il sistema riconosce automaticamente i valori.")
    
    if st.button("📌 Carica Esempio Critico", use_container_width=True):
        st.session_state['lab_text'] = """Neutrofili: 300
Temperatura: 39.2
Potassio: 6.8 mEq/L
Emoglobina: 6.2 g/dL
Piastrine: 15000
LDH: 520 U/L
Creatinina: 3.5 mg/dL
Sodio: 128 mEq/L
Albumina: 2.8 g/dL"""
    
    lab_text = st.text_area(
        "Risultati esami",
        value=st.session_state.get('lab_text', ''),
        height=300,
        placeholder="""Esempio:
Neutrofili: 300
LDH: 480 U/L
Potassio: 6.8
Emoglobina: 6.2 g/dL
Temperatura: 39.2
Piastrine: 15000
Creatinina: 3.5""",
        key="lab_text_area"
    )
    
    if lab_text:
        st.session_state['lab_text'] = lab_text
    
    analyze_clicked = st.button(
        "🔬 ANALIZZA RISCHIO PAZIENTE",
        type="primary",
        use_container_width=True,
        disabled=not lab_text.strip()
    )

# ========================= SIDEBAR: PARSED VALUES =========================
with st.sidebar:
    st.markdown("### 🔍 Valori Riconosciuti")
    
    if lab_text.strip():
        parsed = parse_lab_text(lab_text)
        
        if parsed:
            for lab, value in sorted(parsed.items()):
                display = DISPLAY_NAMES.get(lab, lab)
                ref = REFERENCE_RANGES.get(lab)
                
                if ref:
                    low, high, unit = ref
                    in_range = low <= value <= high
                    icon = "✅" if in_range else "🔴"
                    st.markdown(f"{icon} **{display}**: {value} {unit}  \n"
                                f"<small style='opacity:0.6'>Range: {low}-{high}</small>",
                                unsafe_allow_html=True)
                else:
                    st.markdown(f"📊 **{display}**: {value}")
            
            st.markdown("---")
            st.metric("Valori trovati", len(parsed))
        else:
            st.warning("⚠️ Nessun valore riconosciuto nel testo.")
            st.caption("Prova con formato: `Neutrofili: 300` o `LDH 480`")
    else:
        st.info("💡 Incolla gli esami nel riquadro a sinistra per vedere l'anteprima.")
    
    st.markdown("---")
    st.markdown("### 📖 Formati Supportati")
    st.caption("""
    **Label: Valore**
    `Neutrofili: 300`
    `LDH: 480 U/L`
    
    **Abbreviazione = Valore**
    `K+ = 6.8`
    `HB 6.2`
    
    **Tabellare**
    `PLT    15000`
    `NEU    300`
    """)

# ========================= RIGHT COLUMN: RESULTS =========================
with col_results:
    if analyze_clicked and lab_text.strip():
        parsed = parse_lab_text(lab_text)
        
        if not parsed:
            st.error("❌ Nessun valore riconosciuto. Controlla il formato del testo.")
        else:
            patient_data = build_patient_data(
                labs=parsed,
                patient_id=patient_id,
                age=age,
                sex=sex,
                therapy=therapy,
                histology=histology
            )
            
            with st.spinner("🔬 Analisi clinica in corso..."):
                engine = ClinicalSafetyEngine()
                alerts = engine.run_full_safety_check(patient_data)
            
            n_critical = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            n_high = sum(1 for a in alerts if a.severity == AlertSeverity.HIGH)
            n_moderate = sum(1 for a in alerts if a.severity == AlertSeverity.MODERATE)
            n_low = sum(1 for a in alerts if a.severity == AlertSeverity.LOW)
            
            # Stats bar
            st.markdown(f"""
            <div style="display: flex; gap: 12px; margin-bottom: 24px;">
                <div style="flex:1; background:#D32F2F; color:white; border-radius:12px; padding:16px; text-align:center;">
                    <div class="stat-number">{n_critical}</div>
                    <div class="stat-label">CRITICAL</div>
                </div>
                <div style="flex:1; background:#FF9800; color:white; border-radius:12px; padding:16px; text-align:center;">
                    <div class="stat-number">{n_high}</div>
                    <div class="stat-label">HIGH</div>
                </div>
                <div style="flex:1; background:#FFCA28; color:#333; border-radius:12px; padding:16px; text-align:center;">
                    <div class="stat-number">{n_moderate}</div>
                    <div class="stat-label">MODERATE</div>
                </div>
                <div style="flex:1; background:#4CAF50; color:white; border-radius:12px; padding:16px; text-align:center;">
                    <div class="stat-number">{n_low}</div>
                    <div class="stat-label">LOW</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if not alerts:
                st.markdown("""
                <div class="all-clear">
                    ✅ NESSUN ALERT — Tutti i valori nella norma
                </div>
                """, unsafe_allow_html=True)
            else:
                for alert in alerts:
                    severity_class = {
                        AlertSeverity.CRITICAL: "alert-critical",
                        AlertSeverity.HIGH: "alert-high",
                        AlertSeverity.MODERATE: "alert-moderate",
                        AlertSeverity.LOW: "alert-low",
                    }.get(alert.severity, "alert-moderate")
                    
                    actions_html = ""
                    if alert.recommended_actions:
                        actions_list = "".join(
                            f"<li>{action}</li>" for action in alert.recommended_actions
                        )
                        actions_html = f"""
                        <div class="alert-actions">
                            <strong>Azioni Raccomandate:</strong>
                            <ul style="margin:4px 0; padding-left:20px;">{actions_list}</ul>
                        </div>
                        """
                    
                    immediate = ""
                    if alert.requires_immediate_action:
                        immediate = " ⚡ AZIONE IMMEDIATA RICHIESTA"
                    
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <div class="alert-title">{alert.title}{immediate}</div>
                        <div class="alert-message">{alert.message}</div>
                        {actions_html}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.caption(f"🕐 Analisi completata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Download
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                report_data = {
                    "patient_id": patient_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "parsed_labs": parsed,
                    "alerts": [
                        {
                            "category": a.category.value,
                            "severity": a.severity.value,
                            "title": a.title,
                            "message": a.message,
                            "recommended_actions": a.recommended_actions,
                            "requires_immediate_action": a.requires_immediate_action
                        }
                        for a in alerts
                    ],
                    "summary": {
                        "total_alerts": len(alerts),
                        "critical": n_critical,
                        "high": n_high,
                        "moderate": n_moderate,
                        "low": n_low
                    }
                }
                
                st.download_button(
                    "📥 Scarica Referto JSON",
                    data=json.dumps(report_data, indent=2, ensure_ascii=False),
                    file_name=f"safety_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_dl2:
                st.download_button(
                    "📥 Scarica Patient Data JSON",
                    data=json.dumps(patient_data, indent=2, ensure_ascii=False),
                    file_name=f"patient_data_{patient_id}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    elif not analyze_clicked:
        st.markdown("### 🎯 Come Funziona")
        
        st.markdown("""
        <div style="background:#1E1E1E; border-radius:12px; padding:24px; margin-bottom:16px;">
            <h4 style="margin-top:0;">1️⃣ Incolla gli Esami</h4>
            <p>Copia i risultati degli esami dal gestionale e incollali nel riquadro a sinistra.</p>
        </div>
        <div style="background:#1E1E1E; border-radius:12px; padding:24px; margin-bottom:16px;">
            <h4 style="margin-top:0;">2️⃣ Verifica i Valori</h4>
            <p>Nella sidebar appare l'anteprima dei valori riconosciuti con semafori rosso/verde.</p>
        </div>
        <div style="background:#1E1E1E; border-radius:12px; padding:24px; margin-bottom:16px;">
            <h4 style="margin-top:0;">3️⃣ Premi Analizza</h4>
            <p>Il motore SENTINEL analizza i dati e genera alert visivi in tempo reale.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Prova il bottone **📌 Carica Esempio Critico** per vedere un caso con alert multipli.")

# Footer
st.markdown("---")
st.caption(f"SENTINEL Clinical Safety v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | Engine: ClinicalSafetyEngine (7 moduli)")
