"""
SENTINEL PROMETHEUS — Digital Twin & Previsione a 5 Anni
=========================================================
Il medico seleziona un paziente dal database → PROMETHEUS
lo proietta nel "latent space" clinico → cluster di rischio →
previsione temporale → terapie correttive suggerite.

Architettura:
  - Risk Engine: scoring multi-dimensionale (genetica + biomarker + lifestyle)
  - Clustering: K-Means su feature space clinico → gruppi di rischio
  - Timeline: proiezione a 5 anni con fattori acceleranti/protettivi
  - Corrections: terapie mirate basate sui driver di rischio

Pagina Streamlit: scripts/pages/05_prometheus_prediction.py
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# Import PROMETHEUS modules
try:
    from prometheus.feature_engineering import extract_patient_features, engineer_features
    from prometheus.oracle_bridge import check_patient_rules
    from digital_twin import DigitalTwin, DigitalTwinResult
    PROMETHEUS_AVAILABLE = True
except ImportError as e:
    PROMETHEUS_AVAILABLE = False
    PROMETHEUS_ERROR = str(e)


from prometheus.risk_engine import compute_risk_score


# =============================================================================
# LOAD PATIENT DATABASE
# =============================================================================

def load_patients_list() -> List[Dict]:
    """Carica la lista dei pazienti dal database."""
    patients_dir = BASE_DIR / "data" / "patients"
    if not patients_dir.exists():
        return []

    patients = []
    for fpath in sorted(patients_dir.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            base = data.get("baseline", data)
            pid = base.get("patient_id", fpath.stem)
            patients.append({
                "id": pid,
                "file": str(fpath),
                "age": base.get("age", "?"),
                "sex": base.get("sex", "?"),
                "therapy": base.get("current_therapy") or base.get("therapy", "N/A"),
                "stage": base.get("stage", "?"),
                "histology": base.get("histology", "N/A"),
                "data": data,
            })
        except Exception:
            continue
    return patients


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="SENTINEL - PROMETHEUS Digital Twin",
    page_icon="🔮",
    layout="wide",
)

# CSS
st.markdown("""
<style>
    /* Header */
    .prometheus-header {
        background: linear-gradient(135deg, #4A148C, #7B1FA2);
        color: white;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(74, 20, 140, 0.4);
    }
    .prometheus-header h1 { margin: 0; font-size: 2em; }
    .prometheus-header p { margin: 5px 0 0 0; opacity: 0.8; }

    /* Risk gauge */
    .risk-gauge {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .risk-score { font-size: 4em; font-weight: 900; }
    .risk-label { font-size: 1.1em; margin-top: 4px; opacity: 0.9; }

    /* Cluster card */
    .cluster-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 5px solid;
    }

    /* Factor cards */
    .factor-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .factor-name { font-weight: 600; }
    .factor-weight {
        background: rgba(255,255,255,0.1);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.85em;
    }

    /* Correction card */
    .correction-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .correction-alta { background: #2D1117; border-color: #F44336; }
    .correction-media { background: #2D2517; border-color: #FF9800; }
    .correction-bassa { background: #172D17; border-color: #4CAF50; }

    .correction-target {
        font-weight: bold;
        font-size: 1.05em;
        margin-bottom: 6px;
    }
    .correction-action {
        opacity: 0.85;
        font-size: 0.95em;
    }

    /* Protection card */
    .protection-card {
        background: #0D2818;
        border-left: 4px solid #4CAF50;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }

    /* Epistatic warning */
    .epistatic-card {
        background: linear-gradient(135deg, #311B92, #4527A0);
        color: white;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(69, 39, 160, 0.4);
    }

    /* Timeline bar */
    .timeline-bar {
        display: flex;
        gap: 2px;
        margin: 10px 0;
    }
    .timeline-segment {
        height: 28px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75em;
        font-weight: 600;
        color: white;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .stat-card {
        flex: 1;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-big { font-size: 2.2em; font-weight: 800; }
    .stat-desc { font-size: 0.8em; opacity: 0.7; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="prometheus-header">
    <h1>🔮 PROMETHEUS — Digital Twin & Previsione</h1>
    <p>Seleziona un paziente • Analisi multi-dimensionale • Previsione a 5 anni • Terapie correttive</p>
</div>
""", unsafe_allow_html=True)

if not PROMETHEUS_AVAILABLE:
    st.error(f"❌ Modulo PROMETHEUS non disponibile: {PROMETHEUS_ERROR}")
    st.stop()

# =============================================================================
# PATIENT SELECTION
# =============================================================================

patients = load_patients_list()

if not patients:
    st.warning("⚠ Nessun paziente trovato in `data/patients/`. Carica pazienti prima di usare PROMETHEUS.")
    st.stop()

# Sidebar: patient selector
with st.sidebar:
    st.markdown("### 🏥 Seleziona Paziente")

    patient_options = {f"{p['id']} ({p['age']}y, {p['sex']})": i for i, p in enumerate(patients)}
    selected_label = st.selectbox("Paziente", list(patient_options.keys()))
    selected_idx = patient_options[selected_label]
    patient = patients[selected_idx]

    st.markdown("---")
    st.markdown("### 📋 Info Paziente")
    st.markdown(f"**ID:** {patient['id']}")
    st.markdown(f"**Età:** {patient['age']}")
    st.markdown(f"**Sesso:** {patient['sex']}")
    st.markdown(f"**Terapia:** {patient['therapy']}")
    st.markdown(f"**Stadio:** {patient['stage']}")
    st.markdown(f"**Istologia:** {patient['histology']}")

    n_visits = len(patient["data"].get("visits", []))
    st.markdown(f"**Visite:** {n_visits}")

    st.markdown("---")
    st.markdown("### 📊 Database")
    st.metric("Pazienti totali", len(patients))

# =============================================================================
# ANALYSIS
# =============================================================================

analyze = st.button("🔮 ANALIZZA DIGITAL TWIN", type="primary", use_container_width=True)

if analyze:
    with st.spinner("🔬 Analisi PROMETHEUS in corso..."):
            selected_data = patient["data"]
            # Estrai features grezze
            features = extract_patient_features(selected_data)
            
            # Calcolo Score
            result = compute_risk_score(features, selected_data)

    # ========================= RISK GAUGE + CLUSTER =========================
    col_gauge, col_cluster = st.columns([1, 2])

    with col_gauge:
        score = result["risk_score"]
        cluster = result["cluster"]

        # Color based on score
        if score < 25:
            gauge_bg = "linear-gradient(135deg, #1B5E20, #388E3C)"
        elif score < 50:
            gauge_bg = "linear-gradient(135deg, #E65100, #FF9800)"
        elif score < 75:
            gauge_bg = "linear-gradient(135deg, #BF360C, #FF5722)"
        else:
            gauge_bg = "linear-gradient(135deg, #B71C1C, #D32F2F)"

        st.markdown(f"""
        <div class="risk-gauge" style="background: {gauge_bg}; color: white;">
            <div class="risk-score">{score}</div>
            <div class="risk-label">Risk Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

        # Cluster assignment
        st.markdown(f"""
        <div class="cluster-card" style="background: {cluster['color']}22; border-color: {cluster['color']};">
            <div style="font-size:1.3em; font-weight:bold;">{cluster['emoji']} {cluster['name']}</div>
            <div style="margin-top:6px; opacity:0.85;">{cluster['desc']}</div>
            <div style="margin-top:8px;"><strong>Monitoraggio:</strong> {cluster['monitoring']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cluster:
        # Stats row
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card" style="background: #D32F2F33;">
                <div class="stat-big" style="color:#F44336;">{result['n_active_factors']}</div>
                <div class="stat-desc">Fattori di Rischio</div>
            </div>
            <div class="stat-card" style="background: #4CAF5033;">
                <div class="stat-big" style="color:#4CAF50;">{len(result['protections'])}</div>
                <div class="stat-desc">Fattori Protettivi</div>
            </div>
            <div class="stat-card" style="background: #2196F333;">
                <div class="stat-big" style="color:#2196F3;">{len(result['corrections'])}</div>
                <div class="stat-desc">Interventi Suggeriti</div>
            </div>
            <div class="stat-card" style="background: #7B1FA233;">
                <div class="stat-big" style="color:#CE93D8;">{len(result['epistatic_warnings'])}</div>
                <div class="stat-desc">Regole Epistatiche</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Timeline Adattiva
        if score > 75:
            st.markdown("#### 🚨 Proiezione a Breve Termine (Paziente Critico)")
        else:
            st.markdown("#### 📈 Proiezione a 5 Anni (senza intervento)")

        timeline = result["timeline"]
        if timeline:
            # Visual timeline bar
            segments_html = ""
            for point in timeline:
                r = point["risk"]
                if r < 25:
                    seg_color = "#4CAF50"
                elif r < 50:
                    seg_color = "#FF9800"
                elif r < 75:
                    seg_color = "#FF5722"
                else:
                    seg_color = "#D32F2F"
                segments_html += f'<div class="timeline-segment" style="flex:1; background:{seg_color};">{point["label"]}</div>'

            st.markdown(f'<div class="timeline-bar">{segments_html}</div>', unsafe_allow_html=True)

            # Chart
            chart_data = pd.DataFrame({
                "Mese": [p["month"] for p in timeline],
                "Rischio (%)": [p["risk"] for p in timeline],
            })
            st.line_chart(chart_data.set_index("Mese"), height=200, use_container_width=True)

            last_risk = timeline[-1]["risk"]
            if score > 75:
                st.caption(f"⚠ Traiettoria critica: rischio al mese {timeline[-1]['month']} = **{last_risk:.0f}%**. "
                           f"Intervento immediato necessario per la sopravvivenza.")
            else:
                st.caption(f"⚠ Senza intervento: rischio a 5 anni = **{last_risk:.0f}%**. "
                           f"Con correzioni mirate il rischio può essere ridotto significativamente.")

    # ========================= EPISTATIC WARNINGS =========================
    if result["epistatic_warnings"]:
        st.markdown("---")
        st.markdown("### ⚡ Regole Epistatiche (PROMETHEUS)")
        st.caption("Interazioni genetica × biomarker scoperte dall'analisi popolazione")

        for ev in result["epistatic_warnings"]:
            st.markdown(f"""
            <div class="epistatic-card">
                <div style="font-weight:bold; font-size:1.1em;">🧬 {ev['key']}</div>
                <div style="margin-top:6px;">{ev['details']}</div>
                <div style="margin-top:8px; font-size:0.85em; opacity:0.7;">
                    LR: {ev['weight_lr']} | Score: {ev['score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================= RISK FACTORS & PROTECTIONS =========================
    st.markdown("---")
    col_risks, col_protect = st.columns(2)

    with col_risks:
        st.markdown("### 🔴 Fattori di Rischio Attivi")
        if result["risk_factors"]:
            for rf in result["risk_factors"]:
                category_icon = {
                    "genetica": "🧬", "farmacogenomica": "💊",
                    "biomarker": "🩸", "lifestyle": "🏃"
                }.get(rf["category"], "📊")

                st.markdown(f"""
                <div class="factor-card">
                    <div class="factor-name">{category_icon} {rf['label']}</div>
                    <div class="factor-weight" style="color: {'#F44336' if rf['weight'] >= 10 else '#FF9800'};">
                        +{rf['weight']} pts
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Nessun fattore di rischio attivo rilevato")

    with col_protect:
        st.markdown("### 🟢 Fattori Protettivi")
        if result["protections"]:
            for pf in result["protections"]:
                st.markdown(f"""
                <div class="protection-card">
                    <div style="font-weight:bold;">✅ {pf['label']}</div>
                    <div style="margin-top:4px; font-size:0.9em; opacity:0.8;">{pf['benefit']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ Nessun fattore protettivo specifico identificato")

    # ========================= CORRECTIONS =========================
    st.markdown("---")
    st.markdown("### 💊 Piano di Correzione Terapeutica")
    st.caption("Interventi mirati per ridurre il rischio — ordinati per priorità")

    if result["corrections"]:
        for corr in result["corrections"]:
            css_class = f"correction-{corr['priority'].lower()}"
            priority_icon = "🔴" if corr["priority"] == "ALTA" else ("🟡" if corr["priority"] == "MEDIA" else "🟢")

            st.markdown(f"""
            <div class="correction-card {css_class}">
                <div class="correction-target">{priority_icon} {corr['target']} <span style="opacity:0.5; font-size:0.8em;">({corr['priority']})</span></div>
                <div class="correction-action">→ {corr['action']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Nessun intervento correttivo necessario")

    # ========================= RISK BREAKDOWN =========================
    st.markdown("---")
    st.markdown("### 📊 Risk Breakdown per Categoria")

    cat_scores = result["category_scores"]
    total_cat = max(1, sum(cat_scores.values()))

    num_cols = max(1, len(cat_scores))
    cat_cols = st.columns(num_cols)
    cat_meta = {
        "genetica": ("🧬", "#E91E63"),
        "farmacogenomica": ("💊", "#9C27B0"),
        "biomarker": ("🩸", "#FF5722"),
        "lifestyle": ("🏃", "#2196F3"),
        "VETO TERAPEUTICO": ("🛑", "#D32F2F"),
    }

    for i, (cat, score) in enumerate(cat_scores.items()):
        icon, color = cat_meta.get(cat, ("📊", "#9E9E9E"))
        pct = int(score / total_cat * 100) if total_cat > 0 else 0
        with cat_cols[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:16px; background:{color}22; border-radius:12px;">
                <div style="font-size:2em;">{icon}</div>
                <div style="font-size:1.8em; font-weight:800; color:{color};">{score}</div>
                <div style="font-size:0.8em; opacity:0.7;">{cat.upper()}</div>
                <div style="margin-top:6px; background:#333; border-radius:10px; height:8px;">
                    <div style="width:{pct}%; background:{color}; height:8px; border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================= DOWNLOAD =========================
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    report = {
        "patient_id": patient["id"],
        "timestamp": datetime.now().isoformat(),
        "prometheus_version": "2.0",
        "risk_score": result["risk_score"],
        "cluster": result["cluster"]["name"],
        "monitoring": result["cluster"]["monitoring"],
        "risk_factors": result["risk_factors"],
        "protections": result["protections"],
        "corrections": result["corrections"],
        "category_scores": result["category_scores"],
        "timeline": result["timeline"],
        "epistatic_warnings": [
            {"key": e["key"], "details": e["details"]}
            for e in result["epistatic_warnings"]
        ],
    }

    with col_dl1:
        st.download_button(
            "📥 Scarica Report PROMETHEUS (JSON)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"prometheus_{patient['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        # Text report
        text_lines = [
            f"PROMETHEUS DIGITAL TWIN REPORT",
            f"{'=' * 50}",
            f"Paziente: {patient['id']}",
            f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"RISK SCORE: {result['risk_score']}/100",
            f"CLUSTER: {result['cluster']['name']}",
            f"MONITORAGGIO: {result['cluster']['monitoring']}",
            f"",
            f"FATTORI DI RISCHIO ({result['n_active_factors']}):",
        ]
        for rf in result["risk_factors"]:
            text_lines.append(f"  • {rf['label']} (+{rf['weight']})")
        text_lines.append("")
        text_lines.append(f"CORREZIONI SUGGERITE:")
        for c in result["corrections"]:
            text_lines.append(f"  [{c['priority']}] {c['target']}: {c['action']}")

        st.download_button(
            "📥 Scarica Report Testo",
            data="\n".join(text_lines),
            file_name=f"prometheus_{patient['id']}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.caption(f"🕐 Analisi completata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
               f"PROMETHEUS v2.0 | Paziente: {patient['id']}")

elif not analyze:
    # Welcome screen
    st.markdown("### 🎯 Come Funziona PROMETHEUS")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div style="background:#1E1E2E; border-radius:12px; padding:24px; height:200px;">
            <h4 style="margin-top:0;">1️⃣ Seleziona Paziente</h4>
            <p>Scegli un paziente dal database nella sidebar. PROMETHEUS legge i dati genetici,
            biomarker e clinici.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#1E1E2E; border-radius:12px; padding:24px; height:200px;">
            <h4 style="margin-top:0;">2️⃣ Analisi Multi-Dimensionale</h4>
            <p>Il motore calcola un Risk Score su 4 assi: genetica, farmacogenomica,
            biomarker e lifestyle. Assegna un cluster di rischio.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div style="background:#1E1E2E; border-radius:12px; padding:24px; height:200px;">
            <h4 style="margin-top:0;">3️⃣ Piano Correttivo</h4>
            <p>PROMETHEUS identifica i fattori modificabili e suggerisce terapie mirate
            per ridurre il rischio a 5 anni.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"💡 **{len(patients)} pazienti** nel database. Seleziona un paziente dalla sidebar e premi **🔮 ANALIZZA DIGITAL TWIN**.")

# Footer
st.markdown("---")
st.caption(f"SENTINEL PROMETHEUS v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
           f"Digital Twin Engine | {len(patients)} pazienti nel database")
