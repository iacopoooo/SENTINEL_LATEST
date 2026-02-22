#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL CONTROL TOWER v14.0
==============================
Dashboard principale con integrazione:
- Dual-Core Engine (Tank + Ferrari)
- Temporal Engine (Prophet + Oracle)
- Farmacogenomica
- Frizione Zero
- Genetics Converter (NUOVO v14)

Miglioramenti v14:
- Integrazione Genetics Converter
- Visualizzazione dual-format (flat + noise_variants)
- Test conversione in tempo reale
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURAZIONE PATH
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'src'))

DATA_DIR = BASE_DIR / 'data' / 'patients'

# =============================================================================
# IMPORT MODULI SENTINEL
# =============================================================================

# Motore Ibrido (obbligatorio)
try:
    from src.sentinel_engine import analyze_patient_risk

    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    st.error("⚠️ ERRORE: Non trovo 'src/sentinel_engine.py'")

# Temporal Engine (opzionale)
try:
    from src.temporal_engine import (
        TemporalEngine,
        analyze_patient_temporal,
        PatientPhase,
        Urgency
    )

    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

# Farmacogenomica (opzionale)
try:
    from src.farmacogenomica import PGxAlertEngine

    PGX_AVAILABLE = True
except ImportError:
    PGX_AVAILABLE = False

# Genetics Converter (NUOVO v14)
try:
    from src.genetics_converter import GeneticsConverter

    CONVERTER_AVAILABLE = True
except ImportError:
    try:
        from genetics_converter import GeneticsConverter

        CONVERTER_AVAILABLE = True
    except ImportError:
        CONVERTER_AVAILABLE = False

# =============================================================================
# CONFIGURAZIONE UI
# =============================================================================
st.set_page_config(
    page_title="SENTINEL CONTROL TOWER",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# CSS Migliorato
st.markdown("""
<style>
    /* Status boxes */
    .status-box { 
        padding: 20px; 
        border-radius: 12px; 
        color: white; 
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .critical { background: linear-gradient(135deg, #D32F2F, #B71C1C); }
    .stable { background: linear-gradient(135deg, #388E3C, #1B5E20); }
    .elephant { background: linear-gradient(135deg, #F57C00, #E65100); }
    .temporal-alert { background: linear-gradient(135deg, #7B1FA2, #4A148C); }

    /* Recommendation boxes */
    .rec-box { 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 6px solid; 
        margin-bottom: 15px; 
        font-weight: 500;
    }
    .rec-elephant { background-color: #FFF3E0; border-color: #FF9800; color: #E65100; }
    .rec-critical { background-color: #FFEBEE; border-color: #D32F2F; color: #B71C1C; }
    .rec-warning { background-color: #FFF8E1; border-color: #FFC107; color: #FF6F00; }
    .rec-stable { background-color: #E8F5E9; border-color: #4CAF50; color: #1B5E20; }
    .rec-temporal { background-color: #F3E5F5; border-color: #9C27B0; color: #4A148C; }

    /* Engine boxes */
    .engine-box {
        border: 2px solid;
        border-radius: 12px;
        padding: 20px;
        background-color: #1E1E1E;
        margin-bottom: 15px;
    }

    /* Urgency badges */
    .urgency-critical { background: #D32F2F; color: white; padding: 5px 15px; border-radius: 20px; }
    .urgency-high { background: #FF9800; color: white; padding: 5px 15px; border-radius: 20px; }
    .urgency-medium { background: #FFC107; color: black; padding: 5px 15px; border-radius: 20px; }
    .urgency-low { background: #4CAF50; color: white; padding: 5px 15px; border-radius: 20px; }

    /* Cards */
    .metric-card {
        background: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    /* Genetics Converter Box */
    .genetics-box {
        background: linear-gradient(135deg, #1A237E, #283593);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .genetics-format {
        background: #1E1E1E;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 12px;
    }

    /* Table styling */
    .stDataFrame { cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if 'selected_patient_id' not in st.session_state:
    st.session_state['selected_patient_id'] = None


# =============================================================================
# FUNZIONI UTILITY
# =============================================================================

def get_urgency_color(urgency_str: str) -> str:
    """Ritorna colore CSS per urgency"""
    colors = {
        "CRITICAL": "#D32F2F",
        "HIGH": "#FF9800",
        "MEDIUM": "#FFC107",
        "LOW": "#4CAF50"
    }
    return colors.get(urgency_str, "#9E9E9E")


def get_urgency_emoji(urgency_str: str) -> str:
    """Ritorna emoji per urgency"""
    emojis = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    return emojis.get(urgency_str, "⚪")


def simulate_vision_metrics(image_path):
    """Simula metriche Vision AI"""
    if not image_path:
        return None
    filename = os.path.basename(str(image_path)).lower()
    if "tumor" in filename:
        return {
            "visual_risk": round(random.uniform(85.0, 99.0), 1),
            "chaos": 2,
            "cellularity": 90,
            "class": "HIGH GRADE"
        }
    elif "healthy" in filename:
        return {
            "visual_risk": round(random.uniform(5.0, 20.0), 1),
            "chaos": 0,
            "cellularity": 20,
            "class": "LOW GRADE"
        }
    return {
        "visual_risk": round(random.uniform(40.0, 60.0), 1),
        "chaos": 0,
        "cellularity": 50,
        "class": "INDETERMINATE"
    }


@st.cache_data(ttl=5)
def load_patients_data():
    """Carica tutti i pazienti dal database"""
    if not DATA_DIR.exists():
        return pd.DataFrame()

    json_files = list(DATA_DIR.glob("*.json"))
    data_list = []

    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                base = data.get('baseline', data)
                genetics = base.get('genetics', {})
                blood = base.get('blood_markers', {})
                biomarkers = base.get('biomarkers', {})

                ldh = float(blood.get('ldh', 200))
                tmb = float(biomarkers.get('tmb_score', base.get('tmb', 0)))

                # Calcolo rischio rapido per ordinamento
                risk = 0
                if ldh > 350: risk += 50
                if genetics.get('tp53_status') == 'mutated': risk += 20
                if tmb > 15: risk -= 30
                risk = max(0, min(risk, 100))

                patient_info = {
                    "ID": base.get('patient_id'),
                    "Age": base.get('age'),
                    "ECOG": base.get('ecog_ps', 0),
                    "Stage": base.get('stage', 'I'),
                    "Therapy": base.get('current_therapy', 'N/A'),
                    "TP53": genetics.get('tp53_status'),
                    "KRAS": genetics.get('kras_mutation'),
                    "TMB": tmb,
                    "LDH": ldh,
                    "Protocol": "ELEPHANT 🐘" if ldh > 350 else "Standard",
                    "Base Risk": risk,
                    "Biopsy Path": base.get('biopsy_image_path'),
                    "Has Visits": len(data.get('visits', [])) > 0
                }
                data_list.append(patient_info)
        except Exception as e:
            continue

    return pd.DataFrame(data_list)


def load_patient_json(patient_id: str) -> dict:
    """Carica JSON completo di un paziente"""
    json_path = DATA_DIR / f"{patient_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


# =============================================================================
# COMPONENTI UI RIUTILIZZABILI
# =============================================================================

def render_dual_core_analysis(analysis: dict):
    """Renderizza la sezione Dual-Core (Tank + Ferrari)"""
    st.markdown("### 🧠 DUAL-CORE AI ANALYSIS")

    col_a, col_b = st.columns(2)

    # Tank Engine
    with col_a:
        tank_score = analysis.get('tank_score', 'N/A')
        tank_reasons = analysis.get('tank_reasons', [])

        st.markdown(f"""
        <div class="engine-box" style="border-color: #4CAF50;">
            <h3 style="color: #4CAF50; margin-top: 0;">🛡️ TANK ENGINE (Clinical Rules)</h3>
            <h1 style="font-size: 48px; margin: 10px 0;">{tank_score}%</h1>
            <p><b>Risk Score Deterministico</b></p>
            <hr style="border-color: #333;">
            <p style="font-size: 13px; color: #BBB;">
                <b>DRIVER RILEVATI:</b><br>
                {', '.join(tank_reasons) if tank_reasons else "Nessun driver maggiore."}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Ferrari Engine
    with col_b:
        ferrari_score = analysis.get('ferrari_score')
        ferrari_details = analysis.get('ferrari_details', 'Dati insufficienti')
        match_status = analysis.get('match_status', 'GREEN')

        border_colors = {
            "RED": "#D32F2F",
            "ORANGE": "#FF9800",
            "YELLOW": "#FFEB3B",
            "GREEN": "#4CAF50"
        }
        border_color = border_colors.get(match_status, "#4CAF50")
        score_disp = f"{ferrari_score}%" if ferrari_score is not None else "N/A"

        st.markdown(f"""
        <div class="engine-box" style="border-color: {border_color};">
            <h3 style="color: {border_color}; margin-top: 0;">🏎️ FERRARI ENGINE (Bayesian)</h3>
            <h1 style="font-size: 48px; margin: 10px 0;">{score_disp}</h1>
            <p><b>Probabilità Inferenziale</b></p>
            <hr style="border-color: #333;">
            <p style="font-size: 13px; color: #BBB;">
                <b>INFERENZA:</b><br>
                {ferrari_details}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Consensus
    consensus_msg = analysis.get('consensus', 'Nessun dato')
    if match_status in ["RED", "ORANGE"]:
        st.error(f"🛑 **CRITICAL DIVERGENCE**: {consensus_msg}")
    elif match_status == "YELLOW":
        st.warning(f"⚠️ **MODERATE DIVERGENCE**: {consensus_msg}")
    else:
        st.success(f"✅ **AI CONSENSUS**: {consensus_msg}")


def render_temporal_analysis(temporal_result):
    """Renderizza la sezione Temporal Engine"""
    st.markdown("### ⏱️ TEMPORAL ENGINE ANALYSIS")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Overall Risk
    with col1:
        risk_level = temporal_result.overall_risk_level
        emoji = get_urgency_emoji(risk_level)
        color = get_urgency_color(risk_level)

        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: {color}; 
                    border-radius: 12px; color: white;">
            <h2 style="margin: 0;">{emoji} {risk_level}</h2>
            <p style="margin: 5px 0 0 0;">Overall Temporal Risk</p>
        </div>
        """, unsafe_allow_html=True)

    # Phase
    with col2:
        phase = temporal_result.phase.value.replace("_", " ").title()
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #2D2D2D; 
                    border-radius: 12px; border: 2px solid #7B1FA2;">
            <h3 style="margin: 0; color: #CE93D8;">📊 {phase}</h3>
            <p style="margin: 5px 0 0 0; color: #888;">Patient Phase</p>
        </div>
        """, unsafe_allow_html=True)

    # Next Assessment
    with col3:
        next_assess = temporal_result.next_assessment
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #2D2D2D; 
                    border-radius: 12px; border: 2px solid #1976D2;">
            <h4 style="margin: 0; color: #64B5F6;">📅 {next_assess}</h4>
            <p style="margin: 5px 0 0 0; color: #888;">Next Assessment</p>
        </div>
        """, unsafe_allow_html=True)

    # Primary Concern
    st.markdown(f"""
    <div class="rec-box rec-temporal">
        <b>🎯 PRIMARY CONCERN:</b> {temporal_result.primary_concern}
    </div>
    """, unsafe_allow_html=True)

    # Prophet Signals (se disponibili)
    if temporal_result.prophet_signals:
        st.markdown("#### 📈 Prophet Signals (Velocity & Acceleration)")

        sig_cols = st.columns(4)
        signals = temporal_result.prophet_signals

        for i, (key, sig) in enumerate(signals.items()):
            with sig_cols[i % 4]:
                vel_color = "#D32F2F" if sig.velocity_norm > 5 else "#4CAF50"
                st.markdown(f"""
                <div style="background: #2D2D2D; padding: 15px; border-radius: 8px; text-align: center;">
                    <b style="color: #888;">{key.upper()}</b><br>
                    <span style="font-size: 24px;">{sig.current_value}</span><br>
                    <span style="color: {vel_color};">⚡ {sig.velocity_norm}%/wk</span><br>
                    <span style="font-size: 11px; color: #666;">
                        Conf: {sig.confidence.level.name}
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # Prophet Fusion (se disponibile)
    if temporal_result.prophet_fusion:
        fusion = temporal_result.prophet_fusion
        urgency_color = get_urgency_color(fusion.urgency.value)

        with st.expander("🏎️ Prophet Fusion Details", expanded=False):
            st.markdown(f"""
            **Archetype:** {fusion.archetype}  
            **Urgency:** <span style="color: {urgency_color}; font-weight: bold;">
                {fusion.urgency.value}</span>  
            **Action:** {fusion.action}
            """, unsafe_allow_html=True)

            if fusion.flags:
                st.markdown("**Flags:**")
                for flag in fusion.flags:
                    st.markdown(f"- {flag}")

    # Oracle Alerts (se disponibili)
    if temporal_result.oracle_alerts:
        with st.expander(f"🔮 Oracle Alerts ({len(temporal_result.oracle_alerts)})", expanded=False):
            for alert in temporal_result.oracle_alerts:
                st.markdown(f"""
                **{alert.risk_type}**  
                Probability: {alert.probability}% | Confidence: {alert.confidence.level.value}  
                Lead Time: {alert.lead_time}  
                > {alert.summary}
                """)

    # Recommended Actions
    if temporal_result.recommended_actions:
        with st.expander("📋 Recommended Actions", expanded=True):
            for action in temporal_result.recommended_actions[:5]:
                st.markdown(f"• {action}")


def render_genetics_converter(raw_data: dict):
    """
    Renderizza la sezione Genetics Converter (NUOVO v14)
    Mostra entrambi i formati: flat (CHRONOS) e noise_variants (Oracle)
    """
    if not CONVERTER_AVAILABLE:
        st.warning("⚠️ Genetics Converter non disponibile. Posiziona `genetics_converter.py` in `src/`")
        return

    st.markdown("### 🧬 GENETICS CONVERTER")
    st.markdown("""
    <div class="genetics-box">
        <p style="color: white; margin: 0;">
            <b>Dual-Format View:</b> Visualizza i dati genetici in entrambi i formati usati da SENTINEL
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Estrai genetics dal paziente
    base = raw_data.get('baseline', raw_data)
    genetics_flat = base.get('genetics', {})

    # Formato FLAT (per CHRONOS)
    with col1:
        st.markdown("#### 📊 Formato FLAT (CHRONOS)")
        st.caption("Usato da: Clonal Tracker, Report PDF")

        if genetics_flat:
            # Mostra solo campi VAF
            vaf_data = {k: v for k, v in genetics_flat.items() if '_vaf' in k or '_status' in k or '_mutation' in k}
            if vaf_data:
                st.json(vaf_data)
            else:
                st.info("Nessun dato VAF nel formato flat")
        else:
            st.warning("Nessun dato genetics")

    # Formato NOISE_VARIANTS (per Oracle)
    with col2:
        st.markdown("#### 🔮 Formato NOISE_VARIANTS (Oracle)")
        st.caption("Usato da: Oracle Early Detection, Temporal Engine")

        # Converti usando GeneticsConverter
        try:
            noise_variants = GeneticsConverter.get_unified_variants(raw_data)

            if noise_variants:
                st.json(noise_variants)

                # Statistiche
                st.markdown(f"""
                <div style="background: #2D2D2D; padding: 10px; border-radius: 8px; margin-top: 10px;">
                    <b style="color: #4CAF50;">✅ {len(noise_variants)} varianti rilevate</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Nessuna variante con VAF > 0")

        except Exception as e:
            st.error(f"Errore conversione: {e}")

    # Sezione visite (se disponibili)
    visits = raw_data.get('visits', [])
    if visits:
        with st.expander(f"📅 Varianti per Visita ({len(visits)} visite)", expanded=False):
            try:
                by_visit = GeneticsConverter.get_variants_by_visit(raw_data)

                if by_visit:
                    for v in by_visit:
                        date = v.get('date', 'N/A')
                        variants = v.get('noise_variants', [])

                        if variants:
                            st.markdown(f"**{date}** - {len(variants)} varianti")
                            for var in variants:
                                gene = var.get('gene', '?')
                                vaf = var.get('vaf', 0)
                                vaf_pct = vaf * 100 if vaf <= 1 else vaf
                                st.markdown(f"  - `{gene}`: **{vaf_pct:.1f}%**")
                        else:
                            st.markdown(f"**{date}** - Nessuna variante")
                else:
                    st.info("Nessun dato NGS nelle visite")

            except Exception as e:
                st.error(f"Errore: {e}")

    # Test conversione manuale
    with st.expander("🧪 Test Conversione Manuale", expanded=False):
        st.markdown("Inserisci dati in un formato e vedi la conversione nell'altro")

        test_col1, test_col2 = st.columns(2)

        with test_col1:
            st.markdown("**Input (Flat)**")
            test_input = st.text_area(
                "JSON flat",
                value='{"tp53_vaf": 28.0, "kras_vaf": 18.0}',
                height=100,
                key="test_flat_input"
            )

            if st.button("Converti → Noise Variants", key="btn_to_noise"):
                try:
                    input_data = json.loads(test_input)
                    result = GeneticsConverter.flat_to_noise_variants(input_data)
                    st.success("Conversione riuscita!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Errore: {e}")

        with test_col2:
            st.markdown("**Input (Noise Variants)**")
            test_input2 = st.text_area(
                "JSON noise_variants",
                value='[{"gene": "TP53", "vaf": 0.28}]',
                height=100,
                key="test_noise_input"
            )

            if st.button("Converti → Flat", key="btn_to_flat"):
                try:
                    input_data = json.loads(test_input2)
                    result = GeneticsConverter.noise_variants_to_flat(input_data)
                    st.success("Conversione riuscita!")
                    st.json(result)
                except Exception as e:
                    st.error(f"Errore: {e}")


def render_therapeutic_plan(p: dict, total_risk: int, analysis: dict, vision_data: dict):
    """Renderizza il piano terapeutico"""
    st.markdown("### 🤖 SENTINEL THERAPEUTIC PLAN")

    vis_risk = vision_data.get('visual_risk', 0) if vision_data else 0

    # 1. ELEPHANT Protocol
    if "ELEPHANT" in str(p.get('Protocol', '')):
        st.markdown(f"""
        <div class="rec-box rec-elephant">
            🐘 <b>ELEPHANT PROTOCOL ACTIVATED</b><br>
            LDH ({p['LDH']} U/L) sopra soglia 350. Attività metabolica Warburg.<br>
            <b>PIANO:</b> 1. 💊 Metformina (Target Metabolico) 2. 🔄 Switch Immunoterapia
        </div>
        """, unsafe_allow_html=True)

    # 2. Vision Override
    elif total_risk < 50 and vis_risk > 80:
        st.markdown(f"""
        <div class="rec-box rec-warning">
            👁️ <b>DISCREPANZA GENETICO-VISIVA</b><br>
            Profilo genetico pulito MA AI Vision rileva alto rischio ({vis_risk}%).<br>
            <b>PIANO:</b> 1. Ripetere Biopsia Liquida 2. PET/CT immediata
        </div>
        """, unsafe_allow_html=True)

    # 3. Treatment Failure
    elif total_risk >= 60:
        st.markdown(f"""
        <div class="rec-box rec-critical">
            🛑 <b>FALLIMENTO TERAPEUTICO</b><br>
            Genetica avversa o Progressione Clinica confermata.<br>
            <b>PIANO:</b> Chemio di Salvataggio / Clinical Trial
        </div>
        """, unsafe_allow_html=True)

    # 4. Standard Monitoring
    else:
        st.markdown(f"""
        <div class="rec-box rec-stable">
            ✅ <b>MONITORAGGIO STANDARD</b><br>
            Il paziente risponde bene alla terapia (Responder).<br>
            <b>PIANO:</b> Mantenere {p.get('Therapy', 'terapia attuale')}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# HEADER
# =============================================================================
st.title("🧬 SENTINEL CONTROL TOWER")

# Version & Status Bar
col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
with col_v1:
    st.caption(f"**v14.0** | Dual-Core + Temporal + Converter")
with col_v2:
    st.caption(f"🧠 Engine: {'✅' if ENGINE_AVAILABLE else '❌'}")
with col_v3:
    st.caption(f"⏱️ Temporal: {'✅' if TEMPORAL_AVAILABLE else '❌'}")
with col_v4:
    st.caption(f"💊 PGx: {'✅' if PGX_AVAILABLE else '❌'}")
with col_v5:
    st.caption(f"🧬 Converter: {'✅' if CONVERTER_AVAILABLE else '❌'}")

# =============================================================================
# LOAD DATA
# =============================================================================
df = load_patients_data()

if df.empty:
    st.warning("⚠️ Nessun paziente nel database.")
    st.info(f"Aggiungi file JSON in: `{DATA_DIR}`")
    st.stop()

# =============================================================================
# TABS PRINCIPALI
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 OVERVIEW GLOBALE",
    "🔎 SCHEDA PAZIENTE",
    "⏱️ TEMPORAL ENGINE" if TEMPORAL_AVAILABLE else "⏱️ TEMPORAL (N/A)"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab1:
    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Pazienti Totali", len(df))

    n_elephant = len(df[df['Protocol'].str.contains("ELEPHANT", na=False)])
    c2.metric("Protocollo Elefante", n_elephant,
              delta="Metabolic Action" if n_elephant > 0 else "None",
              delta_color="inverse")

    n_critical = len(df[df['Base Risk'] > 60])
    c3.metric("Alto Rischio", n_critical,
              delta="Critical" if n_critical > 0 else "OK",
              delta_color="inverse" if n_critical > 0 else "normal")

    avg_ldh = round(df['LDH'].mean(), 1) if not df['LDH'].empty else 0
    c4.metric("LDH Medio", f"{avg_ldh} U/L")

    st.markdown("---")

    # Grafici
    g1, g2, g3 = st.columns(3)

    # 1. Metabolismo
    with g1:
        fig = px.scatter(
            df, x="LDH", y="Base Risk",
            color="Protocol", size="ECOG",
            title="🔥 Metabolismo (LDH vs Risk)",
            color_discrete_map={
                "ELEPHANT 🐘": "#FF5722",
                "Standard": "#4CAF50"
            }
        )
        fig.add_vline(x=350, line_dash="dash", line_color="orange",
                      annotation_text="Soglia 350")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # 2. TMB
    with g2:
        df_sorted = df.sort_values(by="TMB", ascending=False)
        fig2 = px.bar(
            df_sorted, x="ID", y="TMB",
            title="🛡️ Immuno-Response (TMB)",
            color="TMB",
            color_continuous_scale="Viridis"
        )
        fig2.add_hline(y=15, line_dash="dash", line_color="#00E676",
                       annotation_text="Rescue > 15")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # 3. KRAS Distribution
    with g3:
        kras_data = df['KRAS'].fillna('WT / N.D.').replace(['None', ''], 'WT / N.D.')
        kras_counts = kras_data.value_counts().reset_index()
        kras_counts.columns = ['Mutation', 'Count']

        color_map = {
            'G12C': '#EF5350', 'G12D': '#EF5350', 'G12V': '#EF5350',
            'WT / N.D.': '#42A5F5', 'Other': '#FFA726', 'wt': '#42A5F5'
        }

        fig3 = px.pie(
            kras_counts, values='Count', names='Mutation', hole=0.4,
            title="🧬 Varianti KRAS",
            color='Mutation',
            color_discrete_map=color_map
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # Lista Pazienti
    st.subheader("📋 Lista Pazienti")

    display_cols = ['ID', 'Age', 'Protocol', 'Base Risk', 'LDH', 'TMB', 'KRAS', 'Has Visits']
    display_df = df[display_cols].sort_values(by='Base Risk', ascending=False)

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event.selection.rows:
        idx = event.selection.rows[0]
        st.session_state['selected_patient_id'] = display_df.iloc[idx]['ID']

# =============================================================================
# TAB 2: SCHEDA PAZIENTE
# =============================================================================
with tab2:
    patient_ids = df['ID'].tolist()

    # Determina indice corrente
    curr_idx = 0
    if st.session_state['selected_patient_id'] in patient_ids:
        curr_idx = patient_ids.index(st.session_state['selected_patient_id'])

    selected_id = st.selectbox("Seleziona Paziente:", patient_ids, index=curr_idx)
    st.session_state['selected_patient_id'] = selected_id

    # Carica dati paziente
    p = df[df['ID'] == selected_id].iloc[0].to_dict()
    raw_data = load_patient_json(selected_id)
    vision_data = simulate_vision_metrics(p.get('Biopsy Path'))

    if not raw_data:
        st.error("❌ File JSON non trovato.")
        st.stop()

    # =================================================================
    # ANALISI DUAL-CORE
    # =================================================================
    if ENGINE_AVAILABLE:
        analysis = analyze_patient_risk(raw_data)
        total_risk = analysis.get('final_risk', p['Base Risk'])

        # Vision Override
        vis_score = vision_data.get('visual_risk', 0) if vision_data else 0
        if vis_score > 80 and total_risk < 40:
            total_risk = int(vis_score)
            analysis.setdefault('tank_reasons', []).append(
                f"⚠️ VISION OVERRIDE (High Grade: {vis_score}%)"
            )
            p['Protocol'] = "AI VISION ALERT"
    else:
        analysis = {}
        total_risk = p['Base Risk']

    # Header Status Box
    if "ELEPHANT" in str(p.get('Protocol', '')):
        color_class = "elephant"
    elif total_risk > 60:
        color_class = "critical"
    else:
        color_class = "stable"

    st.markdown(f"""
    <div class="status-box {color_class}">
        <h2 style="margin: 0;">{p['ID']}</h2>
        <h4 style="margin: 5px 0 0 0;">
            Protocollo: {p.get('Protocol', 'Standard')} | 
            Integrated Risk: {total_risk}/100
        </h4>
    </div>
    """, unsafe_allow_html=True)

    # Dual-Core Analysis
    if analysis:
        render_dual_core_analysis(analysis)

    st.markdown("---")

    # =================================================================
    # TEMPORAL ENGINE (se disponibile e ha visite)
    # =================================================================
    if TEMPORAL_AVAILABLE and raw_data.get('visits'):
        with st.expander("⏱️ TEMPORAL ENGINE ANALYSIS", expanded=True):
            try:
                temporal_result = analyze_patient_temporal(raw_data)
                render_temporal_analysis(temporal_result)
            except Exception as e:
                st.warning(f"Temporal Engine error: {e}")
    elif not TEMPORAL_AVAILABLE:
        st.info("ℹ️ Temporal Engine non disponibile. Installa `src/temporal_engine/`")
    elif not raw_data.get('visits'):
        st.info("ℹ️ Nessuna visita disponibile per analisi temporale.")

    st.markdown("---")

    # =================================================================
    # GENETICS CONVERTER (NUOVO v14)
    # =================================================================
    with st.expander("🧬 GENETICS CONVERTER - Dual Format View", expanded=False):
        render_genetics_converter(raw_data)

    st.markdown("---")

    # Piano Terapeutico
    render_therapeutic_plan(p, total_risk, analysis, vision_data)

    st.markdown("---")

    # Footer: Info Paziente
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("👤 CLINICA")
        st.write(f"**Età:** {p.get('Age')} | **ECOG:** {p.get('ECOG')}")
        st.write(f"**Stage:** {p.get('Stage')}")
        st.write(f"**Terapia:** `{p.get('Therapy')}`")

    with c2:
        st.warning("🧬 GENETICA")
        st.write(f"**TP53:** {p.get('TP53')}")
        st.write(f"**KRAS:** {p.get('KRAS')}")
        st.write(f"**LDH:** {p.get('LDH')} U/L")
        st.write(f"**TMB:** {p.get('TMB')} mut/Mb")

    with c3:
        st.error("🔬 VISIONE (AI Pathology)")
        img_path = p.get('Biopsy Path')
        if img_path and os.path.exists(str(img_path)):
            st.image(img_path, use_container_width=True)
        else:
            st.markdown("⚠️ *Immagine non disponibile*")

        if vision_data:
            m1, m2 = st.columns(2)
            m1.metric("Cellularity", f"{vision_data['cellularity']}%")
            m2.metric("Chaos", f"{vision_data['chaos']}")

            risk_val = vision_data['visual_risk']
            st.metric(
                "Visual Risk", f"{risk_val}/100",
                delta="High Risk" if risk_val > 80 else "Low Risk",
                delta_color="inverse" if risk_val > 80 else "normal"
            )

# =============================================================================
# TAB 3: TEMPORAL ENGINE DEDICATO
# =============================================================================
with tab3:
    if not TEMPORAL_AVAILABLE:
        st.warning("⏱️ Temporal Engine non installato.")
        st.info("""
        Per attivare il Temporal Engine:
        1. Scarica `SENTINEL_TEMPORAL_ENGINE.zip`
        2. Estrai in `src/temporal_engine/`
        3. Riavvia la dashboard
        """)
        st.stop()

    st.markdown("### ⏱️ TEMPORAL ENGINE - Navigazione Temporale della Malattia")

    st.markdown("""
    <div style="background: linear-gradient(90deg, #7B1FA2, #1976D2); 
                padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h4 style="color: white; margin: 0;">
            🏎️ PROPHET (Pazienti in Trattamento) + 🔮 ORACLE (Screening/Remissione)
        </h4>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
            Prophet predice la fuga 4-12 settimane prima. Oracle intercetta segnali 18-36 mesi prima.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Selezione paziente per analisi temporale
    patients_with_visits = df[df['Has Visits'] == True]['ID'].tolist()

    if not patients_with_visits:
        st.warning("⚠️ Nessun paziente ha visite longitudinali per l'analisi temporale.")
        st.info("Aggiungi `visits` ai file JSON dei pazienti.")
    else:
        selected_temporal = st.selectbox(
            "Seleziona Paziente per Analisi Temporale:",
            patients_with_visits,
            key="temporal_patient"
        )

        if st.button("🚀 ESEGUI ANALISI TEMPORALE", type="primary"):
            raw_data = load_patient_json(selected_temporal)

            if raw_data:
                with st.spinner("⏱️ Analisi in corso..."):
                    try:
                        result = analyze_patient_temporal(raw_data)

                        # Risultato principale
                        st.markdown("---")
                        render_temporal_analysis(result)

                        # Timeline Grafico
                        if result.prophet_signals and result.prophet_signals.get('ldh'):
                            st.markdown("### 📈 Trend Temporale")

                            visits = raw_data.get('visits', [])
                            if visits:
                                # Prepara dati per grafico
                                weeks = []
                                ldh_vals = []
                                vaf_vals = []

                                for v in visits:
                                    w = v.get('week_on_therapy')
                                    if w is not None:
                                        weeks.append(w)
                                        ldh_vals.append(
                                            v.get('blood_markers', {}).get('ldh')
                                        )
                                        # Cerca max VAF (con check per None)
                                        genetics = v.get('genetics') or {}
                                        if isinstance(genetics, dict) and genetics:
                                            vaf_list = [val for k, val in genetics.items()
                                                        if '_vaf' in str(k) and isinstance(val, (int, float))]
                                            max_vaf = max(vaf_list, default=None) if vaf_list else None
                                        else:
                                            max_vaf = None
                                        vaf_vals.append(max_vaf)

                                if weeks:
                                    fig = go.Figure()

                                    # LDH
                                    fig.add_trace(go.Scatter(
                                        x=weeks, y=ldh_vals,
                                        name='LDH (U/L)',
                                        mode='lines+markers',
                                        line=dict(color='#FF9800', width=3)
                                    ))

                                    # VAF
                                    if any(v is not None for v in vaf_vals):
                                        fig.add_trace(go.Scatter(
                                            x=weeks, y=vaf_vals,
                                            name='Max VAF (%)',
                                            mode='lines+markers',
                                            line=dict(color='#E91E63', width=3),
                                            yaxis='y2'
                                        ))

                                    fig.update_layout(
                                        title="Andamento LDH e VAF nel Tempo",
                                        xaxis_title="Settimane in Terapia",
                                        yaxis_title="LDH (U/L)",
                                        yaxis2=dict(
                                            title="VAF (%)",
                                            overlaying='y',
                                            side='right'
                                        ),
                                        height=400,
                                        hovermode='x unified'
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                        # Export JSON
                        with st.expander("📄 Export Risultati (JSON)"):
                            st.json(result.to_dict())

                    except Exception as e:
                        st.error(f"Errore nell'analisi: {e}")
                        import traceback

                        st.code(traceback.format_exc())

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🛠️ SENTINEL TOOLS")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    st.markdown("### 📊 Quick Stats")
    st.metric("Pazienti", len(df))
    st.metric("Con Visite", len(df[df['Has Visits'] == True]))

    st.markdown("---")

    st.markdown("### 🔗 Moduli")
    st.caption(f"Engine: {'✅' if ENGINE_AVAILABLE else '❌'}")
    st.caption(f"Temporal: {'✅' if TEMPORAL_AVAILABLE else '❌'}")
    st.caption(f"PGx: {'✅' if PGX_AVAILABLE else '❌'}")
    st.caption(f"Converter: {'✅' if CONVERTER_AVAILABLE else '❌'}")

    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")