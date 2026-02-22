#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL TEMPORAL ENGINE - Dedicated Page
==========================================
Pagina dedicata per analisi temporale avanzata.
Posizionare in: pages/03_temporal_engine.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime

# Setup path - IMPORTANTE: scripts/pages/ → risali 2 livelli per arrivare a SENTINEL_TRIAL/
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/pages → scripts → SENTINEL_TRIAL
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

DATA_DIR = BASE_DIR / 'data' / 'patients'

# Import Temporal Engine
try:
    from temporal_engine import (
        TemporalEngine,
        analyze_patient_temporal,
        run_prophet_only,
        run_oracle_only,
        PatientPhase,
    )
    from temporal_engine.prophet_v4 import Urgency, ConfidenceLevel
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    TEMPORAL_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Config
st.set_page_config(
    page_title="SENTINEL Temporal Engine",
    page_icon="⏱️",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .temporal-header {
        background: linear-gradient(135deg, #7B1FA2, #1976D2);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        color: white;
    }
    .engine-card {
        border: 2px solid;
        border-radius: 12px;
        padding: 20px;
        background: #1E1E1E;
        margin-bottom: 15px;
    }
    .prophet-card { border-color: #FF9800; }
    .oracle-card { border-color: #9C27B0; }
    .signal-box {
        background: #2D2D2D;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .urgency-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
    }
    .urgency-critical { background: #D32F2F; color: white; }
    .urgency-high { background: #FF9800; color: white; }
    .urgency-medium { background: #FFC107; color: black; }
    .urgency-low { background: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="temporal-header">
    <h1 style="margin: 0;">⏱️ SENTINEL TEMPORAL ENGINE</h1>
    <p style="margin: 10px 0 0 0; font-size: 18px;">
        Navigazione Temporale della Malattia | Prophet + Oracle
    </p>
</div>
""", unsafe_allow_html=True)

if not TEMPORAL_AVAILABLE:
    st.error(f"❌ Temporal Engine non disponibile: {IMPORT_ERROR}")
    st.info("""
    **Per installare:**
    1. Scarica `SENTINEL_TEMPORAL_ENGINE.zip`
    2. Estrai in `src/temporal_engine/`
    3. Riavvia l'applicazione
    """)
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Configurazione")

# Selezione modalità
mode = st.sidebar.radio(
    "Modalità Analisi:",
    ["🔄 Auto-Detect", "🏎️ Solo Prophet", "🔮 Solo Oracle"],
    help="Auto-Detect rileva automaticamente la fase del paziente"
)

# VAF Threshold
vaf_threshold = st.sidebar.slider(
    "Soglia VAF (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="Soglia per considerare VAF clinicamente significativo"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Guida")
st.sidebar.markdown("""
**🏎️ PROPHET** (Pazienti in Trattamento)
- Calcola velocità e accelerazione
- Predice fuga 4-12 settimane prima
- Urgency: LOW → CRITICAL

**🔮 ORACLE** (Screening/Remissione)
- Cerca drift sub-clinici
- Lead time 18-36 mesi
- Bayesian evidence fusion
""")

# =============================================================================
# CARICAMENTO DATI
# =============================================================================

@st.cache_data(ttl=10)
def load_patients():
    if not DATA_DIR.exists():
        return []
    
    patients = []
    for f in DATA_DIR.glob("*.json"):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                base = data.get('baseline', data)
                patients.append({
                    'id': base.get('patient_id'),
                    'therapy': base.get('current_therapy', 'N/A'),
                    'has_visits': len(data.get('visits', [])) > 0,
                    'n_visits': len(data.get('visits', [])),
                    'path': str(f)
                })
        except:
            continue
    return patients

patients = load_patients()

if not patients:
    st.warning("⚠️ Nessun paziente nel database.")
    st.stop()

# =============================================================================
# SELEZIONE PAZIENTE
# =============================================================================

st.markdown("### 📋 Seleziona Paziente")

# Filtra pazienti con visite se necessario
if mode == "🏎️ Solo Prophet":
    available = [p for p in patients if p['has_visits']]
    if not available:
        st.warning("Nessun paziente ha visite per Prophet.")
        st.stop()
else:
    available = patients

# Dataframe per selezione
patients_df = pd.DataFrame(available)
patients_df = patients_df.rename(columns={
    'id': 'ID', 
    'therapy': 'Terapia',
    'has_visits': 'Ha Visite',
    'n_visits': 'N. Visite'
})

col1, col2 = st.columns([2, 1])

with col1:
    selected_id = st.selectbox(
        "Paziente:",
        [p['id'] for p in available],
        format_func=lambda x: f"{x} ({[p['n_visits'] for p in available if p['id']==x][0]} visite)"
    )

with col2:
    run_analysis = st.button("🚀 ANALIZZA", type="primary", use_container_width=True)

# =============================================================================
# ANALISI
# =============================================================================

# Debug info
with st.expander("🔍 Debug Info", expanded=False):
    st.write(f"**BASE_DIR:** {BASE_DIR}")
    st.write(f"**DATA_DIR:** {DATA_DIR}")
    st.write(f"**DATA_DIR exists:** {DATA_DIR.exists()}")
    st.write(f"**Pazienti trovati:** {len(patients)}")
    st.write(f"**Paziente selezionato:** {selected_id}")
    if available:
        sel_patient = [p for p in available if p['id'] == selected_id]
        if sel_patient:
            st.write(f"**Path paziente:** {sel_patient[0]['path']}")
            st.write(f"**N. Visite:** {sel_patient[0]['n_visits']}")

if run_analysis and selected_id:
    # Carica dati paziente
    patient_path = [p['path'] for p in available if p['id'] == selected_id][0]
    with open(patient_path, 'r') as f:
        patient_data = json.load(f)
    
    st.markdown("---")
    
    visits = patient_data.get('visits', [])
    if len(visits) <= 1 and mode in ["🔄 Auto-Detect", "🏎️ Solo Prophet"]:
        st.warning("⚠️ **ATTENZIONE:** Prophet ha necessità di almeno due visite per una corretta stima temporale. Con una sola visita a disposizione (o solo la baseline), il motore matematico restituirà velocità 0.0 (STABLE) per mancanza di storico.")

    with st.spinner("⏱️ Analisi temporale in corso..."):
        
        # =================================================================
        # AUTO-DETECT MODE
        # =================================================================
        if mode == "🔄 Auto-Detect":
            result = analyze_patient_temporal(patient_data)
            
            # Header risultato
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                urgency_class = result.overall_risk_level.lower()
                st.markdown(f"""
                <div style="text-align: center;">
                    <span class="urgency-badge urgency-{urgency_class}">
                        {result.overall_risk_level}
                    </span>
                    <p style="margin-top: 10px; color: #888;">Overall Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                phase_display = result.phase.value.replace("_", " ").title()
                st.metric("Fase Rilevata", phase_display)
            
            with col3:
                st.metric("Engines Usati", ", ".join(result.engines_used))
            
            with col4:
                st.metric("Prossimo Check", result.next_assessment)
            
            # Primary Concern
            st.markdown(f"""
            <div style="background: #2D2D2D; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #7B1FA2; margin: 20px 0;">
                <h4 style="margin: 0; color: #CE93D8;">🎯 PRIMARY CONCERN</h4>
                <p style="margin: 10px 0 0 0; font-size: 18px;">{result.primary_concern}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prophet Results
            if result.prophet_signals:
                st.markdown("### 🏎️ PROPHET - Segnali Dinamici")
                
                sig_cols = st.columns(4)
                for i, (key, sig) in enumerate(result.prophet_signals.items()):
                    with sig_cols[i]:
                        vel_color = "#D32F2F" if sig.velocity_norm > 5 else "#4CAF50" if sig.velocity_norm < 0 else "#FFC107"
                        accel_color = "#D32F2F" if sig.acceleration_norm > 2 else "#4CAF50"
                        
                        st.markdown(f"""
                        <div class="signal-box">
                            <h4 style="color: #FF9800; margin: 0;">{key.upper()}</h4>
                            <h2 style="margin: 10px 0;">{sig.current_value}</h2>
                            <p style="color: {vel_color}; margin: 5px 0;">
                                ⚡ {sig.velocity_norm:+.1f}%/wk
                            </p>
                            <p style="color: {accel_color}; margin: 5px 0; font-size: 12px;">
                                📈 {sig.acceleration_norm:+.1f}%/wk²
                            </p>
                            <p style="color: #666; font-size: 11px; margin: 5px 0 0 0;">
                                Conf: {sig.confidence.level.name} | R²: {sig.r_squared}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Fusion Result
                if result.prophet_fusion:
                    fusion = result.prophet_fusion
                    urgency_colors = {
                        "CRITICAL": "#D32F2F",
                        "HIGH": "#FF9800",
                        "MEDIUM": "#FFC107",
                        "LOW": "#4CAF50"
                    }
                    urg_color = urgency_colors.get(fusion.urgency.value, "#9E9E9E")
                    
                    st.markdown(f"""
                    <div class="engine-card prophet-card">
                        <h3 style="color: #FF9800;">🏎️ PROPHET FUSION</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0;">{fusion.archetype}</h4>
                                <p style="color: #888; margin: 5px 0 0 0;">{fusion.action}</p>
                            </div>
                            <span class="urgency-badge" style="background: {urg_color};">
                                {fusion.urgency.value}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if fusion.flags:
                        with st.expander("🚩 Flags"):
                            for flag in fusion.flags:
                                st.markdown(f"- {flag}")
            
            # Oracle Results
            if result.oracle_alerts:
                st.markdown("### 🔮 ORACLE - Alert Pre-Clinici")
                
                for alert in result.oracle_alerts:
                    prob_color = "#D32F2F" if alert.probability > 50 else "#FF9800" if alert.probability > 25 else "#4CAF50"
                    
                    st.markdown(f"""
                    <div class="engine-card oracle-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="color: #9C27B0; margin: 0;">{alert.risk_type}</h4>
                            <span style="background: {prob_color}; color: white; 
                                        padding: 5px 15px; border-radius: 15px;">
                                {alert.probability}%
                            </span>
                        </div>
                        <p style="margin: 10px 0;">{alert.summary}</p>
                        <p style="color: #888; font-size: 13px;">
                            Lead Time: {alert.lead_time} | 
                            Confidence: {alert.confidence.level.value}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommended Actions
            if result.recommended_actions:
                st.markdown("### 📋 Azioni Raccomandate")
                for i, action in enumerate(result.recommended_actions[:7], 1):
                    st.markdown(f"{i}. {action}")
        
        # =================================================================
        # SOLO PROPHET
        # =================================================================
        elif mode == "🏎️ Solo Prophet":
            visits = patient_data.get('visits', [])
            if not visits:
                st.error("Nessuna visita disponibile per questo paziente.")
            else:
                signals, fusion = run_prophet_only(visits)
                
                st.markdown("### 🏎️ PROPHET ANALYSIS")
                
                # Urgency principale
                urg = fusion.urgency.value
                urg_colors = {"CRITICAL": "#D32F2F", "HIGH": "#FF9800", "MEDIUM": "#FFC107", "LOW": "#4CAF50"}
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 30px; 
                                background: {urg_colors.get(urg, '#666')}; border-radius: 15px;">
                        <h1 style="margin: 0; color: white;">{urg}</h1>
                        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Urgency Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 20px; background: #2D2D2D; border-radius: 15px;">
                        <h3 style="margin: 0;">{fusion.archetype}</h3>
                        <p style="margin: 10px 0 0 0;">{fusion.action}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Signals detail
                st.markdown("#### 📊 Signal Details")
                
                for key, sig in signals.items():
                    with st.expander(f"{key.upper()} - Current: {sig.current_value}"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Velocity", f"{sig.velocity_norm:+.2f}%/wk")
                        col_b.metric("Acceleration", f"{sig.acceleration_norm:+.2f}%/wk²")
                        col_c.metric("Confidence", sig.confidence.level.name)
                        
                        st.write(f"**Forecast 3m:** {sig.forecast_3m}")
                        st.write(f"**R²:** {sig.r_squared}")
                        if sig.outlier_flag:
                            st.warning("⚠️ Ultimo punto potrebbe essere outlier")
                
                # Timeline plot
                st.markdown("#### 📈 Timeline")
                
                weeks = [v.get('week_on_therapy', i*4) for i, v in enumerate(visits)]
                ldh_vals = [v.get('blood_markers', {}).get('ldh') for v in visits]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weeks, y=ldh_vals,
                    mode='lines+markers',
                    name='LDH',
                    line=dict(color='#FF9800', width=3)
                ))
                
                # Add forecast
                if signals['ldh'].forecast_3m:
                    last_week = weeks[-1] if weeks else 0
                    forecast_week = last_week + 12
                    fig.add_trace(go.Scatter(
                        x=[last_week, forecast_week],
                        y=[ldh_vals[-1] if ldh_vals else 0, signals['ldh'].forecast_point],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#FF9800', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=[forecast_week, forecast_week],
                        y=list(signals['ldh'].forecast_3m),
                        mode='markers',
                        name='95% PI',
                        marker=dict(size=10, symbol='line-ns-open', color='#FF9800')
                    ))
                
                fig.update_layout(
                    title="LDH Trend + Forecast",
                    xaxis_title="Settimane",
                    yaxis_title="LDH (U/L)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # SOLO ORACLE
        # =================================================================
        elif mode == "🔮 Solo Oracle":
            # Prepara storia per Oracle
            visits = patient_data.get('visits', [])
            oracle_history = []
            
            for i, v in enumerate(visits):
                entry = {
                    "date": v.get('date') or f"2024-{(i+1):02d}",
                    "blood": v.get('blood_markers', v.get('blood', {})),
                    "clinical": v.get('clinical', {})
                }
                oracle_history.append(entry)
            
            if len(oracle_history) < 3:
                st.warning("Oracle richiede almeno 3 visite con storia temporale.")
            else:
                alerts = run_oracle_only(
                    oracle_history,
                    patient_id=selected_id
                )
                
                st.markdown("### 🔮 ORACLE ANALYSIS")
                
                if not alerts:
                    st.success("✅ Nessun pattern sub-clinico rilevato.")
                    st.info("Oracle cerca drift lenti su 18+ mesi. Potrebbe servire più storia.")
                else:
                    for alert in alerts:
                        st.markdown(f"""
                        <div class="engine-card oracle-card">
                            <h3 style="color: #9C27B0;">{alert.risk_type}</h3>
                            <div style="display: flex; gap: 20px; margin: 15px 0;">
                                <div>
                                    <h2 style="margin: 0;">{alert.probability}%</h2>
                                    <small>Probability</small>
                                </div>
                                <div>
                                    <h4 style="margin: 0;">{alert.confidence.level.value}</h4>
                                    <small>Confidence</small>
                                </div>
                                <div>
                                    <h4 style="margin: 0;">{alert.lead_time}</h4>
                                    <small>Lead Time</small>
                                </div>
                            </div>
                            <p>{alert.summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📋 Azioni Raccomandate"):
                            for action in alert.recommended_actions:
                                st.markdown(f"• {action}")
                        
                        with st.expander("🔬 Evidenze"):
                            for ev in alert.signal_sources:
                                st.markdown(f"- **{ev.key}**: LR={ev.weight_lr}, score={ev.score:.2f}")
                                st.caption(ev.details)
    
    # =================================================================
    # EXPORT
    # =================================================================
    st.markdown("---")
    
    with st.expander("📄 Export Risultati"):
        if mode == "🔄 Auto-Detect":
            st.json(result.to_dict())
        elif mode == "🏎️ Solo Prophet":
            export_data = {
                "urgency": fusion.urgency.value,
                "archetype": fusion.archetype,
                "action": fusion.action,
                "signals": {k: {"value": v.current_value, "velocity": v.velocity_norm} 
                           for k, v in signals.items()}
            }
            st.json(export_data)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"⏱️ SENTINEL Temporal Engine | Prophet v4 + Oracle v3 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
