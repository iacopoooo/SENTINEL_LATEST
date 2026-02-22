"""
SENTINEL TRIAL - INTEGRATED ONCOLOGY REPORT GENERATOR
======================================================
Versione: v18.0 LONGITUDINAL FULL
Features:
- Report COMPLETO per ogni visita (in ordine cronologico)
- Baseline come prima sezione
- Ogni visita successiva come report completo
- Longitudinal Analysis come pagina finale
"""

import json
import os
import sys
from fpdf import FPDF
from pathlib import Path
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import copy
from typing import Dict, List, Optional
# SETUP PATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'src'))  # Per importare clinical_notes_llm etc.
from src.sentinel_utils import safe_float, interpret_pdl1



# IMPORTS
try:
    from src.follow_up import (
        PatientTimeline,
        load_patient_timeline,
        get_current_patient_state
    )
    FOLLOW_UP_AVAILABLE = True
except ImportError:
    FOLLOW_UP_AVAILABLE = False
    print("⚠️ Follow-up module not available")

try:
    from src.digital_twin import ElephantProtocol, DigitalTwin
    DIGITAL_TWIN_AVAILABLE = True
except ImportError:
    DIGITAL_TWIN_AVAILABLE = False

try:
    from src.sentinel_engine import analyze_patient_risk
    print("✅ Engine v18 LONGITUDINAL agganciato.")
except ImportError as e:
    print(f"❌ ERRORE: {e}")
    sys.exit(1)

# Import Elephant Protocol v2.0
try:
    from elephant_protocol import generate_elephant_protocol, ElephantProtocolResult
    ELEPHANT_V2_AVAILABLE = True
except ImportError:
    ELEPHANT_V2_AVAILABLE = False
    print("⚠️ Elephant Protocol v2.0 not available, using legacy")

# === ADVANCED FEATURES ===

# Import Predictive Timeline
try:
    from src.predictive_timeline import generate_predictive_timeline, PredictiveTimelineResult
    PREDICTIVE_TIMELINE_AVAILABLE = True
except ImportError:
    PREDICTIVE_TIMELINE_AVAILABLE = False
    print("⚠️ Predictive Timeline not available")

# Import Clonal Evolution Tracker
try:
    from src.clonal_tracker import analyze_clonal_evolution, ClonalArchitecture, CloneStatus, ClinicalUrgency
    CLONAL_TRACKER_AVAILABLE = True
except ImportError:
    CLONAL_TRACKER_AVAILABLE = False
    print("⚠️ Clonal Evolution Tracker not available")

# Import Adaptive Therapy Optimizer
try:
    from src.adaptive_therapy import generate_adaptive_protocol, AdaptiveProtocol, TherapyPhase, ResponseStatus
    ADAPTIVE_THERAPY_AVAILABLE = True
except ImportError:
    ADAPTIVE_THERAPY_AVAILABLE = False
    print("⚠️ Adaptive Therapy Optimizer not available")

# Import Synthetic Lethality Finder
try:
    from src.synthetic_lethality import find_synthetic_lethality, SyntheticLethalityResult, EvidenceLevel, TumorType
    SYNTHETIC_LETHALITY_AVAILABLE = True
except ImportError:
    SYNTHETIC_LETHALITY_AVAILABLE = False
    print("⚠️ Synthetic Lethality Finder not available")

# Import SENTINEL CHRONOS
try:
    from src.sentinel_chronos import SentinelChronos, generate_chronos_from_clonal_data
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("⚠️ SENTINEL CHRONOS not available")

# Import PROMETHEUS Risk Engine
try:
    from prometheus.feature_engineering import extract_patient_features
    from prometheus.risk_engine import compute_risk_score
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️ PROMETHEUS not available")

DATA_DIR = BASE_DIR / 'data' / 'patients'
OUTPUT_DIR = BASE_DIR / 'reports'
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# UTILITIES
# =============================================================================

def safe_text(text):
    """Pulisce il testo per FPDF"""
    if text is None:
        return "N/A"
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "⚠️": "[!]", "✅": "[OK]", "❌": "[X]", "📈": "[^]", "📉": "[v]",
        "🔴": "[!]", "🟢": "[OK]", "🟡": "[~]", "🐘": "[E]", "💊": "[Rx]",
        "📅": "[D]", "→": "->", "←": "<-", "↑": "^", "↓": "v", "≥": ">="
    }
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    return text.encode('latin-1', 'replace').decode('latin-1')


def pdl1_status(val):
    try:
        v = float(val)
    except:
        return "Unknown"
    if v < 1.0:
        return "Negative (<1%)"
    elif v < 50.0:
        return "Low/Intermediate (1-49%)"
    else:
        return "High Expression (>=50%)"


def format_date(date_str):
    if not date_str:
        return "N/A"
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%d/%m/%Y')
    except:
        return str(date_str)


# =============================================================================
# PDF CLASS
# =============================================================================

class SentinelReport(FPDF):
    def __init__(self):
        super().__init__()
        self.current_visit_label = ""

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, safe_text('SENTINEL TRIAL - INTEGRATED ONCOLOGY REPORT'), 0, 1, 'C')
        self.set_font('Arial', 'I', 9)
        self.set_text_color(100)
        self.cell(0, 5, safe_text('Genomics | Immuno | Blood | Physics | Vision | Neural Net'), 0, 1, 'C')
        self.set_draw_color(0, 51, 102)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 10, safe_text(f'Page {self.page_no()} - SENTINEL v18.0 LONGITUDINAL - {ts}'), 0, 0, 'C')

    def section_title(self, title, color=(240, 240, 240)):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(*color)
        self.set_text_color(0)
        self.cell(0, 10, safe_text(f'  {title}'), 0, 1, 'L', 1)
        self.ln(2)

    def section_body(self, body):
        self.set_font('Arial', '', 11)
        self.set_text_color(0)
        self.multi_cell(0, 6, safe_text(body))
        self.ln(3)

    def visit_separator(self, visit_label, date, is_current=False):
        """Separatore grande tra visite"""
        self.add_page()

        if is_current:
            self.set_fill_color(0, 102, 0)  # Verde per visita attuale
        else:
            self.set_fill_color(0, 51, 102)  # Blu per visite storiche

        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 18)
        self.cell(0, 20, safe_text(f'  {visit_label}'), 0, 1, 'L', 1)

        self.set_font('Arial', 'I', 12)
        self.cell(0, 10, safe_text(f'  Report Date: {format_date(date)}'), 0, 1, 'L', 1)

        self.set_text_color(0)
        self.ln(10)


# =============================================================================
# GRAPH GENERATORS
# =============================================================================

def generate_pfs_curve(risk_score, pfs_soc, pfs_sentinel, elephant_active, veto_active, patient_id):
    """Genera curva Kaplan-Meier per PFS con indicatore mediana"""
    months = np.linspace(0, 24, 100)
    ln2 = np.log(2)

    lambda_soc = ln2 / max(pfs_soc, 0.5)
    surv_soc = 100 * np.exp(-lambda_soc * months)

    has_benefit = elephant_active and not veto_active and pfs_sentinel > pfs_soc
    surv_sentinel = None
    if has_benefit:
        lambda_sentinel = ln2 / pfs_sentinel
        surv_sentinel = 100 * np.exp(-lambda_sentinel * months)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    if veto_active:
        ax.plot(months, surv_soc, 'r--', label='Current Therapy (MISMATCHED)', linewidth=2.5, alpha=0.8)
        ax.fill_between(months, 0, surv_soc, alpha=0.1, color='red')
    else:
        ax.plot(months, surv_soc, 'b-', label=f'Standard of Care (PFS: {pfs_soc:.1f}m)', linewidth=2.5)

    if surv_sentinel is not None:
        ax.plot(months, surv_sentinel, 'g-', label=f'SENTINEL Protocol (PFS: {pfs_sentinel:.1f}m)', linewidth=2.5)
        ax.fill_between(months, surv_soc, surv_sentinel, alpha=0.2, color='green')

    # Linea 50% mediana
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(23.5, 52, '50%', fontsize=9, color='gray', ha='right')

    # === INDICATORE BLU MEDIANA PFS ===
    # Punto blu sulla curva dove incrocia il 50%
    pfs_soc_capped = min(pfs_soc, 24)  # Cap a 24 mesi per visualizzazione
    if pfs_soc_capped <= 24:
        ax.plot(pfs_soc_capped, 50, 'bo', markersize=12, zorder=5)  # Punto blu grande
        # Annotazione con il valore PFS
        ax.annotate(f'{pfs_soc:.1f}m',
                    xy=(pfs_soc_capped, 50),
                    xytext=(pfs_soc_capped + 1.5, 58),
                    fontsize=10, fontweight='bold', color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Se c'è beneficio SENTINEL, mostra anche quel punto
    if surv_sentinel is not None and pfs_sentinel <= 24:
        ax.plot(pfs_sentinel, 50, 'go', markersize=12, zorder=5)
        ax.annotate(f'{pfs_sentinel:.1f}m',
                    xy=(pfs_sentinel, 50),
                    xytext=(pfs_sentinel + 1.5, 42),
                    fontsize=10, fontweight='bold', color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # Box con Risk Score in basso a sinistra
    ax.text(0.5, 8, f'Risk: {risk_score}/100', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Time (Months)', fontweight='bold')
    ax.set_ylabel('Progression-Free Survival (%)', fontweight='bold')
    ax.set_title(f'Digital Twin Projection - {patient_id}\nRisk Score: {risk_score}/100', fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return temp_file.name


def generate_longitudinal_chart(baseline_data, visits):
    """Genera grafico longitudinale LDH + ECOG"""
    weeks = [0]
    # LDH baseline: prova dal baseline, se 0 o mancante usa la prima visita
    _bl_ldh = float((baseline_data.get('blood_markers') or {}).get('ldh') or 0)
    if _bl_ldh == 0 and visits:
        _bl_ldh = float((visits[0].get('blood_markers') or {}).get('ldh') or 0)
    ldh_values = [_bl_ldh]
    ecog_values = [int(baseline_data.get('ecog_ps', 2))]
    labels = ['BL']

    for i, v in enumerate(visits):
        weeks.append(v.get('week_on_therapy', 0))
        blood = v.get('blood_markers', {})
        ldh_values.append(float(blood.get('ldh') or ldh_values[-1]))
        ecog_values.append(int(v.get('ecog_ps', ecog_values[-1])))
        labels.append(f'V{i+1}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig.patch.set_facecolor('white')

    # LDH
    ax1.plot(weeks, ldh_values, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=350, color='red', linestyle='--', alpha=0.7, label='Elephant Threshold (350)')
    ax1.fill_between(weeks, 0, ldh_values, alpha=0.2, color='blue')
    for i, (w, ldh, lbl) in enumerate(zip(weeks, ldh_values, labels)):
        ax1.annotate(lbl, (w, ldh), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax1.set_xlabel('Week on Therapy', fontweight='bold')
    ax1.set_ylabel('LDH (U/L)', fontweight='bold')
    ax1.set_title('LDH Evolution', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ECOG
    colors = ['green' if e <= 1 else 'orange' if e == 2 else 'red' for e in ecog_values]
    bars = ax2.bar(range(len(weeks)), ecog_values, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(weeks)))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Visit', fontweight='bold')
    ax2.set_ylabel('ECOG PS', fontweight='bold')
    ax2.set_title('Performance Status Evolution', fontweight='bold')
    ax2.set_ylim(0, 4)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return temp_file.name


# =============================================================================
# COMPLETE VISIT REPORT GENERATOR
# =============================================================================

def draw_predictive_timeline(pdf, patient_data: Dict, current_risk: float, months_on_therapy: int = 0):
    """Disegna la sezione Predictive Resistance Timeline"""

    if not PREDICTIVE_TIMELINE_AVAILABLE:
        return

    try:
        timeline = generate_predictive_timeline(patient_data, current_risk, months_on_therapy)
    except Exception as e:
        print(f"⚠️ Predictive Timeline error: {e}")
        return

    pdf.add_page()

    # Header
    pdf.set_fill_color(75, 0, 130)  # Indigo
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, safe_text("  PREDICTIVE RESISTANCE TIMELINE"), 0, 1, 'L', 1)
    pdf.set_text_color(0)
    pdf.ln(5)

    # Current Status Box
    pdf.set_fill_color(240, 240, 255)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text(f"  Current Overall Risk: {timeline.current_risk:.0f}%"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, safe_text(f"  Model Confidence: {timeline.model_confidence}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"  Monitoring Intensity: {timeline.monitoring_intensity}"), 0, 1)
    pdf.ln(3)

    # === PROJECTION TABLE ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("12-MONTH RISK PROJECTION:"), 0, 1)

    # Table header
    pdf.set_fill_color(200, 200, 230)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(25, 6, "Month", 1, 0, 'C', 1)
    pdf.cell(35, 6, "Risk %", 1, 0, 'C', 1)
    pdf.cell(50, 6, "95% CI", 1, 0, 'C', 1)
    pdf.cell(80, 6, "Visual", 1, 1, 'C', 1)

    # Table rows
    pdf.set_font('Arial', '', 9)
    for proj in timeline.monthly_projections:
        # Color based on risk
        if proj.risk_percent >= 70:
            pdf.set_fill_color(255, 200, 200)
        elif proj.risk_percent >= 50:
            pdf.set_fill_color(255, 230, 200)
        elif proj.risk_percent >= 30:
            pdf.set_fill_color(255, 255, 200)
        else:
            pdf.set_fill_color(200, 255, 200)

        pdf.cell(25, 5, f"+{proj.month}", 1, 0, 'C', 1)
        pdf.cell(35, 5, f"{proj.risk_percent:.1f}%", 1, 0, 'C', 1)
        pdf.cell(50, 5, f"{proj.confidence_low:.0f}% - {proj.confidence_high:.0f}%", 1, 0, 'C', 1)

        # Visual bar
        bar_width = int(proj.risk_percent * 0.75)
        bar_text = "#" * max(1, bar_width // 3)
        pdf.cell(80, 5, bar_text, 1, 1, 'L', 1)

    pdf.ln(5)

    # === THRESHOLD ALERTS ===
    if timeline.alerts:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, safe_text("THRESHOLD ALERTS:"), 0, 1)

        pdf.set_font('Arial', '', 9)
        for alert in timeline.alerts:
            if alert.threshold_percent >= 70:
                pdf.set_text_color(200, 0, 0)
                icon = "[!!!]"
            elif alert.threshold_percent >= 50:
                pdf.set_text_color(255, 140, 0)
                icon = "[!!]"
            else:
                pdf.set_text_color(0, 100, 0)
                icon = "[!]"

            pdf.cell(0, 5, safe_text(
                f"  {icon} {alert.threshold_percent}% risk expected at month {alert.expected_month:.1f} "
                f"(range: {alert.confidence_range[0]:.1f}-{alert.confidence_range[1]:.1f})"
            ), 0, 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"      Action: {alert.recommended_action}"), 0, 1)
            pdf.set_text_color(0)

        pdf.ln(3)

    # === RISK FACTORS ===
    if timeline.accelerating_factors or timeline.protective_factors:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, safe_text("MODIFYING FACTORS:"), 0, 1)

        if timeline.accelerating_factors:
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 5, safe_text("  Accelerating (faster resistance):"), 0, 1)
            pdf.set_font('Arial', '', 9)
            for factor in timeline.accelerating_factors:
                pdf.cell(0, 4, safe_text(f"    - {factor}"), 0, 1)
            pdf.set_text_color(0)

        if timeline.protective_factors:
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 5, safe_text("  Protective (slower resistance):"), 0, 1)
            pdf.set_font('Arial', '', 9)
            for factor in timeline.protective_factors:
                pdf.cell(0, 4, safe_text(f"    - {factor}"), 0, 1)
            pdf.set_text_color(0)

        pdf.ln(3)

    # === MONITORING RECOMMENDATIONS ===
    pdf.set_fill_color(230, 255, 230)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("  RECOMMENDED MONITORING SCHEDULE"), 0, 1, 'L', 1)

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, safe_text(f"  Next ctDNA: Week {timeline.next_ctdna_week}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"  Next Imaging: Week {timeline.next_imaging_week}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"  Intensity: {timeline.monitoring_intensity}"), 0, 1)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(100, 100, 100)
    # Determina tipo tumore per nota disclaimer
    _base = patient_data.get('baseline', patient_data)
    _histology = str(_base.get('histology', '')).lower()
    _cancer_label = 'Solid Tumor (generic)'
    _cancer_map = {
        'adenocarcinoma': 'Adenocarcinoma', 'squamous': 'Squamous Cell',
        'large cell': 'Large Cell Lung Cancer', 'urothelial': 'Urothelial Carcinoma',
        'bladder': 'Urothelial Carcinoma', 'gastroesophageal': 'Gastroesophageal Adenocarcinoma',
        'gastric': 'Gastric Cancer', 'breast': 'Breast Cancer',
        'colorectal': 'Colorectal Cancer', 'melanoma': 'Melanoma',
        'pancreatic': 'Pancreatic Cancer', 'renal': 'Renal Cell Carcinoma',
        'nsclc': 'NSCLC',
    }
    for _key, _label in _cancer_map.items():
        if _key in _histology:
            _cancer_label = _label
            break
    pdf.multi_cell(0, 3, safe_text(
        f"Note: Predictions based on Weibull survival model calibrated to {_cancer_label} literature. "
        "Individual patient trajectories may vary. Use as decision support, not definitive prognosis."
    ))
    pdf.set_text_color(0)


def draw_clonal_evolution(pdf, patient_data: Dict, visits: List[Dict]):
    """Disegna la sezione Clonal Evolution Tracker"""

    if not CLONAL_TRACKER_AVAILABLE:
        return

    if not visits:
        return  # Serve almeno una visita per tracking

    try:
        clonal = analyze_clonal_evolution(patient_data, visits)
    except Exception as e:
        print(f"⚠️ Clonal Evolution error: {e}")
        return

    pdf.add_page()

    # Header
    pdf.set_fill_color(0, 100, 100)  # Teal
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, safe_text("  CLONAL EVOLUTION TRACKER"), 0, 1, 'L', 1)
    pdf.set_text_color(0)
    pdf.ln(3)

    # === STATUS SUMMARY ===
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(95, 7, safe_text(f"  Dominant Clone: {clonal.dominant_clone or 'None'}"), 0, 0, 'L', 1)
    pdf.cell(95, 7, safe_text(f"  VAF: {clonal.dominant_vaf:.1f}%"), 0, 1, 'L', 1)

    pdf.set_font('Arial', '', 9)
    pdf.cell(63, 6, safe_text(f"  Tumor Burden (sum VAF): {clonal.total_tumor_burden:.1f}%"), 0, 0, 'L', 1)
    pdf.cell(63, 6, safe_text(f"  Clonal Diversity: {clonal.clonal_diversity:.2f}"), 0, 0, 'L', 1)
    pdf.cell(64, 6, safe_text(f"  Next ctDNA: {clonal.next_ctdna_weeks} weeks"), 0, 1, 'L', 1)
    pdf.ln(3)

    # === FLAGS ===
    if clonal.transformation_risk or clonal.polyclonal_resistance:
        pdf.set_fill_color(255, 200, 200)
        pdf.set_font('Arial', 'B', 10)
        if clonal.transformation_risk:
            pdf.set_text_color(180, 0, 0)
            pdf.cell(0, 6, safe_text("  [!!!] TRANSFORMATION RISK: TP53+RB1 detected"), 0, 1, 'L', 1)
        if clonal.polyclonal_resistance:
            pdf.set_text_color(200, 100, 0)
            pdf.cell(0, 6, safe_text("  [!!] POLYCLONAL RESISTANCE: Multiple expanding clones"), 0, 1, 'L', 1)
        pdf.set_text_color(0)
        pdf.ln(2)

    # === PRIMARY CONCERN ===
    pdf.set_fill_color(255, 255, 220)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text(f"  Primary Concern: {clonal.primary_concern}"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 6, safe_text(f"  Recommended Action: {clonal.recommended_action}"), 0, 1)
    pdf.ln(3)

    # === CLONAL ARCHITECTURE TABLE ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("CLONAL ARCHITECTURE:"), 0, 1)

    # Table header
    pdf.set_fill_color(200, 220, 220)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(45, 6, "Clone/Mutation", 1, 0, 'C', 1)
    pdf.cell(20, 6, "Current", 1, 0, 'C', 1)
    pdf.cell(20, 6, "Baseline", 1, 0, 'C', 1)
    pdf.cell(25, 6, "Trend", 1, 0, 'C', 1)
    pdf.cell(25, 6, "Status", 1, 0, 'C', 1)
    pdf.cell(25, 6, "Urgency", 1, 0, 'C', 1)
    pdf.cell(30, 6, "Target", 1, 1, 'C', 1)

    # Table rows
    pdf.set_font('Arial', '', 8)
    for clone in sorted(clonal.clones, key=lambda c: -(c.current_vaf or 0))[:10]:  # Top 10
        # Row color based on urgency
        if clone.clinical_urgency == ClinicalUrgency.CRITICAL:
            pdf.set_fill_color(255, 180, 180)
        elif clone.clinical_urgency == ClinicalUrgency.HIGH:
            pdf.set_fill_color(255, 220, 180)
        elif clone.clinical_urgency == ClinicalUrgency.FAVORABLE:
            pdf.set_fill_color(200, 255, 200)
        else:
            pdf.set_fill_color(255, 255, 255)

        # Mutation name (truncate if too long)
        mut_name = clone.mutation[:20] + "..." if len(clone.mutation) > 20 else clone.mutation
        pdf.cell(45, 5, safe_text(mut_name), 1, 0, 'L', 1)

        # Current VAF
        pdf.cell(20, 5, f"{clone.current_vaf:.1f}%" if clone.current_vaf else "N/A", 1, 0, 'C', 1)
        # Baseline VAF
        baseline_str = f"{clone.baseline_vaf:.1f}%" if clone.baseline_vaf else "NEW"
        pdf.cell(20, 5, baseline_str, 1, 0, 'C', 1)

        # Trend
        if clone.trend_percent is None:
            trend_str = "N/A"
        elif clone.trend_percent == 999 or clone.trend_percent == float('inf'):
            trend_str = "NEW"
        elif clone.trend_percent > 0:
            trend_str = f"+{clone.trend_percent:.0f}%"
        else:
            trend_str = f"{clone.trend_percent:.0f}%"
        pdf.cell(25, 5, trend_str, 1, 0, 'C', 1)

        # Status
        status_short = clone.status.value[:8]
        pdf.cell(25, 5, status_short, 1, 0, 'C', 1)

        # Urgency
        urgency_str = clone.clinical_urgency.value[:8]
        pdf.cell(25, 5, urgency_str, 1, 0, 'C', 1)

        # Target therapy
        target_str = clone.target_therapy[:12] + "..." if clone.target_therapy and len(clone.target_therapy) > 12 else (
                    clone.target_therapy or "-")
        pdf.cell(30, 5, safe_text(target_str), 1, 1, 'L', 1)

    pdf.ln(3)

    # === VAF EVOLUTION VISUAL ===
    if clonal.clones:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, safe_text("VAF EVOLUTION (Visual):"), 0, 1)

        # Calcola numero di timepoints (baseline + visite)
        num_visits = len(visits) if 'visits' in dir() else 4
        num_timepoints = min(5, num_visits + 1)  # Max 5 colonne (BL + 4 visite)

        # Header con sfondo
        pdf.set_fill_color(230, 240, 250)
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(38, 5, "Clone", 1, 0, 'C', 1)
        pdf.cell(12, 5, "BL", 1, 0, 'C', 1)
        for i in range(num_timepoints - 1):
            pdf.cell(12, 5, f"V{i + 1}", 1, 0, 'C', 1)
        pdf.cell(14, 5, "Delta", 1, 0, 'C', 1)
        pdf.cell(45, 5, "Trend", 1, 0, 'C', 1)
        pdf.cell(25, 5, "Action", 1, 1, 'C', 1)

        # Rows - ordina per urgenza clinica poi VAF
        sorted_clones = sorted(clonal.clones,
                               key=lambda c: (
                                   0 if c.clinical_urgency.value == 'CRITICAL' else
                                   1 if c.clinical_urgency.value == 'HIGH' else
                                   2 if c.status.value == 'EXPANDING' else
                                   3 if c.status.value == 'EMERGING' else
                                   4,
                                   -(c.current_vaf or 0)
                               ))

        pdf.set_font('Courier', '', 7)
        for clone in sorted_clones[:8]:  # Max 8 cloni
            # Row color based on status
            if clone.status == CloneStatus.EXPANDING or clone.clinical_urgency.value == 'CRITICAL':
                pdf.set_fill_color(255, 220, 220)
            elif clone.status == CloneStatus.EMERGING:
                pdf.set_fill_color(255, 250, 220)
            elif clone.status == CloneStatus.DECLINING or clone.status == CloneStatus.CLEARED:
                pdf.set_fill_color(220, 255, 220)
            else:
                pdf.set_fill_color(255, 255, 255)

            # Clone name (truncate)
            name_short = clone.mutation[:16] + ".." if len(clone.mutation) > 16 else clone.mutation
            pdf.cell(38, 4, safe_text(name_short), 1, 0, 'L', 1)

            # VAF values per timepoint
            for i in range(num_timepoints):
                if i < len(clone.vaf_history):
                    vaf = clone.vaf_history[i][1]
                    if vaf is None:
                        pdf.set_text_color(150, 150, 150)
                        vaf_str = "?"
                    elif vaf >= 30:
                        pdf.set_text_color(200, 0, 0)  # Rosso per alto
                        vaf_str = f"{vaf:.0f}"
                    elif vaf >= 10:
                        pdf.set_text_color(200, 100, 0)  # Arancione
                        vaf_str = f"{vaf:.0f}"
                    elif vaf > 0.5:
                        pdf.set_text_color(0, 0, 0)
                        vaf_str = f"{vaf:.0f}"
                    else:
                        pdf.set_text_color(150, 150, 150)
                        vaf_str = "-"
                else:
                    pdf.set_text_color(150, 150, 150)
                    vaf_str = "-"

                pdf.cell(12, 4, vaf_str, 1, 0, 'C', 1)
                pdf.set_text_color(0)

            # Pad remaining columns if needed
            for _ in range(num_timepoints, 5):
                pdf.set_text_color(150, 150, 150)
                pdf.cell(12, 4, "-", 1, 0, 'C', 1)
                pdf.set_text_color(0)

            # Delta (change from baseline)
            if clone.trend_percent is None:
                delta_str = "N/A"
                pdf.set_text_color(100, 100, 100)
            elif clone.trend_percent == 999 or clone.trend_percent == float('inf'):
                delta_str = "NEW"
                pdf.set_text_color(255, 140, 0)
            elif clone.trend_percent > 50:
                delta_str = f"+{clone.trend_percent:.0f}%"
                pdf.set_text_color(200, 0, 0)
            elif clone.trend_percent > 0:
                delta_str = f"+{clone.trend_percent:.0f}%"
                pdf.set_text_color(200, 100, 0)
            elif clone.trend_percent < -30:
                delta_str = f"{clone.trend_percent:.0f}%"
                pdf.set_text_color(0, 150, 0)
            elif clone.trend_percent < 0:
                delta_str = f"{clone.trend_percent:.0f}%"
                pdf.set_text_color(0, 100, 0)
            else:
                delta_str = "0%"
                pdf.set_text_color(100, 100, 100)

            pdf.cell(14, 4, delta_str, 1, 0, 'C', 1)
            pdf.set_text_color(0)

            # Visual trend (sparkline-style)
            if len(clone.vaf_history) >= 2:
                # Costruisci mini-grafico ASCII
                # FILTRA i None prima di calcolare!
                vafs = [v for _, v in clone.vaf_history[-4:] if v is not None]
                if len(vafs) >= 2:
                    trend_chars = []
                    for i in range(1, len(vafs)):
                        diff = vafs[i] - vafs[i - 1]
                        if diff > 5:
                            trend_chars.append("^")
                        elif diff > 0:
                            trend_chars.append("/")
                        elif diff < -5:
                            trend_chars.append("v")
                        elif diff < 0:
                            trend_chars.append("\\")
                        else:
                            trend_chars.append("-")

                    trend_visual = "".join(trend_chars)

                    # Aggiungi label
                    if clone.status == CloneStatus.EXPANDING:
                        trend_visual += " RISING!"
                        pdf.set_text_color(200, 0, 0)
                    elif clone.status == CloneStatus.DECLINING:
                        trend_visual += " falling"
                        pdf.set_text_color(0, 150, 0)
                    elif clone.status == CloneStatus.CLEARED:
                        trend_visual += " CLEAR"
                        pdf.set_text_color(0, 150, 0)
                    elif clone.status == CloneStatus.EMERGING:
                        trend_visual += " NEW!"
                        pdf.set_text_color(255, 140, 0)
                    else:
                        trend_visual += " stable"
                        pdf.set_text_color(100, 100, 100)
                else:
                    trend_visual = "---"
                    pdf.set_text_color(100, 100, 100)
            else:
                trend_visual = "---"
                pdf.set_text_color(100, 100, 100)

            pdf.cell(45, 4, safe_text(trend_visual), 1, 0, 'L', 1)
            pdf.set_text_color(0)

            # Action column
            if clone.clinical_urgency.value == 'CRITICAL':
                action = "SWITCH!"
                pdf.set_text_color(255, 255, 255)
                pdf.set_fill_color(200, 0, 0)
            elif clone.clinical_urgency.value == 'HIGH':
                action = "Monitor+"
                pdf.set_text_color(200, 0, 0)
                pdf.set_fill_color(255, 220, 220)
            elif clone.status == CloneStatus.DECLINING:
                action = "Continue"
                pdf.set_text_color(0, 128, 0)
            elif clone.status == CloneStatus.EMERGING:
                action = "Watch"
                pdf.set_text_color(255, 140, 0)
            else:
                action = "-"
                pdf.set_text_color(100, 100, 100)

            pdf.cell(25, 4, action, 1, 1, 'C', 1)
            pdf.set_text_color(0)
            pdf.set_fill_color(255, 255, 255)

        pdf.ln(2)

        # === LEGEND ===
        pdf.set_font('Arial', 'I', 6)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 3,
                 safe_text("Legend: ^ rising | v falling | - stable | VAF colors: Red >30% | Orange 10-30% | Gray <1%"),
                 0, 1)
        pdf.set_text_color(0)

    pdf.ln(3)

    # === EMERGING CLONES ALERT ===
    if clonal.emerging_clones:
        pdf.set_fill_color(255, 240, 200)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(200, 100, 0)
        pdf.cell(0, 7, safe_text(f"  [!] EMERGING CLONES DETECTED ({len(clonal.emerging_clones)}):"), 0, 1, 'L', 1)
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0)

        for clone in clonal.emerging_clones[:3]:
            action_str = f" -> {clone.target_therapy}" if clone.target_therapy else " -> Monitor"
            vaf_str = f"{clone.current_vaf:.1f}%" if clone.current_vaf else "N/A"
            pdf.cell(0, 5, safe_text(f"    - {clone.mutation} (VAF {vaf_str}){action_str}"), 0, 1)
    # === DECLINING CLONES (Good news) ===
    if clonal.declining_clones:
        pdf.set_fill_color(220, 255, 220)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 7, safe_text(f"  [OK] RESPONDING CLONES ({len(clonal.declining_clones)}):"), 0, 1, 'L', 1)
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0)

        for clone in clonal.declining_clones[:3]:
            trend_str = "N/A" if clone.trend_percent is None else f"{clone.trend_percent:.0f}%"
            pdf.cell(0, 5, safe_text(f"    - {clone.mutation}: {trend_str} (declining)"), 0, 1)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 3, safe_text(
        "Note: Clonal evolution tracking based on ctDNA/NGS data. VAF values may not reflect true tumor burden. "
        "Emerging clones require confirmation in subsequent samples before clinical action."
    ))
    pdf.set_text_color(0)


def draw_chronos_chart(pdf, patient_data: Dict, visits: List[Dict], clonal_architecture, output_dir: str = None):
    """Genera e inserisce il grafico CHRONOS nel PDF"""

    if not CHRONOS_AVAILABLE:
        return

    if not clonal_architecture or not clonal_architecture.clones:
        return

        # ═══════════════════════════════════════════════════════════════
        # CHECK 1: Verifica che almeno un clone abbia VAF > 0
        # ═══════════════════════════════════════════════════════════════
        has_vaf_data = any(
            (clone.current_vaf and clone.current_vaf > 0) or
            (clone.baseline_vaf and clone.baseline_vaf > 0)
            for clone in clonal_architecture.clones
        )

        if not has_vaf_data:
            print("ℹ️  CHRONOS skipped: nessun dato VAF disponibile")
            return

        # ═══════════════════════════════════════════════════════════════
        # CHECK 2: Verifica che i VAF vengano da dati REALI nel JSON
        # (non inventati dal clonal_tracker)
        # ═══════════════════════════════════════════════════════════════
        baseline = patient_data.get('baseline', {})
        genetics = baseline.get('genetics', {})

        # Cerca se nel JSON ci sono VAF reali (non solo status)
        has_real_vaf_in_json = False
        for gene, value in genetics.items():
            if isinstance(value, dict) and 'vaf' in value:
                # Formato: {"TP53": {"status": "mutated", "vaf": 35.0}}
                has_real_vaf_in_json = True
                break
            elif isinstance(value, (int, float)) and value > 0:
                # Formato: {"TP53_vaf": 35.0}
                if 'vaf' in gene.lower():
                    has_real_vaf_in_json = True
                    break

        # Controlla anche nelle visite
        if not has_real_vaf_in_json:
            for visit in (patient_data.get('visits', []) or []):
                visit_genetics = visit.get('genetics', {})
                if isinstance(visit_genetics, dict):
                    for key, val in visit_genetics.items():
                        if 'vaf' in key.lower() and isinstance(val, (int, float)) and val > 0:
                            has_real_vaf_in_json = True
                            break
                if has_real_vaf_in_json:
                    break

        if not has_real_vaf_in_json:
            print("ℹ️  CHRONOS skipped: paziente senza dati VAF reali (solo status mutazionale)")
            return

        # ═══════════════════════════════════════════════════════════════
        # CHECK 3: Servono almeno 2 timepoints per grafico temporale
        # ═══════════════════════════════════════════════════════════════
        if not patient_data.get('visits') or len(patient_data.get('visits', [])) < 1:
            print("ℹ️  CHRONOS skipped: serve almeno 1 visita")
            return

    # ═══════════════════════════════════════════════════════════════

    try:
        # Prepara dati cloni
        clones_data = []
        for clone in clonal_architecture.clones:
            timepoints = []
            values = []

            for i, (date, vaf) in enumerate(clone.vaf_history):
                if vaf is None:
                    continue  # Salta timepoints senza VAF
                if i == 0:
                    week = 0
                elif i <= len(visits):
                    week = visits[i - 1].get('week_on_therapy', i * 8)
                else:
                    week = i * 8

                timepoints.append(float(week))
                values.append(float(vaf))

            if values and max(values) >= 3:  # Solo cloni con VAF >= 3%
                clones_data.append({
                    'name': clone.mutation,
                    'timepoints': timepoints,
                    'values': values,
                    'is_emerging': clone.status.value == 'EMERGING'
                })

        if not clones_data:
            return

        # Calcola settimana corrente
        current_week = float(visits[-1].get('week_on_therapy', len(visits) * 8)) if visits else 0

        # Crea annotazioni
        annotations = []
        for i, visit in enumerate(visits):
            imaging = visit.get('imaging', {})
            response = imaging.get('response', '')
            blood = visit.get('blood_markers', {})
            ldh = blood.get('ldh', 0)

            label_parts = [f"V{i + 1}"]
            if response:
                label_parts.append(response)
            if ldh and ldh > 0:
                label_parts.append(f"LDH {ldh:.0f}")

            if response == 'PD':
                alert_level = 'critical'
            elif ldh and ldh > 400:
                alert_level = 'warning'
            else:
                alert_level = 'info'

            week = visit.get('week_on_therapy', (i + 1) * 8)

            annotations.append({
                'week': week,
                'label': '\n'.join(label_parts),
                'y_position': 0.25 + (i % 3) * 0.15,
                'alert_level': alert_level
            })

        # Genera chart
        base = patient_data.get('baseline', patient_data)
        patient_id = patient_data.get('baseline', patient_data).get('patient_id', 'unknown')
        # Output path
        if output_dir:
            chart_path = os.path.join(output_dir, f'chronos_{patient_id}.png')
        else:
            chart_path = f'/tmp/chronos_{patient_id}.png'

        chronos = SentinelChronos(style='dark')
        chronos.generate_chart(
            patient_id=patient_id,
            clones_data=clones_data,
            current_week=current_week,
            annotations=annotations,
            predict_weeks=8,
            pd_threshold=3.0,
            output_path=chart_path,
            figsize=(12, 6)
        )

        # Inserisci nel PDF
        pdf.add_page()

        # Header
        pdf.set_fill_color(20, 20, 40)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 12, safe_text("  SENTINEL CHRONOS - CLONAL EVOLUTION TIMELINE"), 0, 1, 'L', 1)
        pdf.set_text_color(0)
        pdf.ln(3)

        # Inserisci immagine
        if os.path.exists(chart_path):
            # Calcola dimensioni per fit nella pagina
            img_width = 190  # mm
            img_height = 95  # mm (aspect ratio ~2:1)

            x_pos = (210 - img_width) / 2  # Centra
            pdf.image(chart_path, x=x_pos, y=pdf.get_y(), w=img_width, h=img_height)
            pdf.ln(img_height + 5)

        # Legenda testuale
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 6, safe_text("INTERPRETAZIONE:"), 0, 1)
        pdf.set_font('Arial', '', 8)

        # Conta tipi di cloni
        sensitive_count = sum(
            1 for c in clones_data if 'L858R' in c['name'] or 'Exon 19' in c['name'] or 'Ex19' in c['name'])
        resistant_count = len(clones_data) - sensitive_count

        pdf.set_text_color(46, 204, 113)  # Verde
        pdf.cell(0, 4, safe_text(f"  Verde: Clone sensibile (risponde alla terapia) - {sensitive_count} tracked"), 0, 1)
        pdf.set_text_color(231, 76, 60)  # Rosso
        pdf.cell(0, 4, safe_text(f"  Rosso: Clone resistente bypass (MET, HER2, PIK3CA)"), 0, 1)
        pdf.set_text_color(241, 196, 15)  # Giallo
        pdf.cell(0, 4, safe_text(f"  Giallo: Clone resistente on-target (C797S, T790M) - {resistant_count} tracked"), 0,
                 1)
        pdf.set_text_color(0)

        pdf.ln(2)
        pdf.set_font('Arial', 'I', 7)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 3, safe_text(
            "L'area a destra della linea 'OGGI' rappresenta la predizione dell'evoluzione clonale basata sui trend osservati. "
            "La linea rossa orizzontale indica la soglia di progressione clinica (PD)."
        ))
        pdf.set_text_color(0)

    except Exception as e:
        print(f"⚠️ CHRONOS chart error: {e}")
        import traceback
        traceback.print_exc()


def draw_adaptive_therapy(pdf, patient_data: Dict, visits: List[Dict], ai_result: Dict):
    """Disegna la sezione Adaptive Therapy Optimizer"""

    if not ADAPTIVE_THERAPY_AVAILABLE:
        return

    # Calcola parametri dai dati
    base = patient_data.get('baseline', patient_data)

    # Tumor change (da imaging se disponibile)
    tumor_change = -40  # Default
    weeks_stable = 12
    baseline_burden = 100
    current_burden = 60

    if visits:
        last_visit = visits[-1]
        if last_visit.get('imaging'):
            tumor_change = float(last_visit['imaging'].get('tumor_change_percent', -40) or -40)
        weeks_stable = int(last_visit.get('week_on_therapy', 12) or 12)

    # PFS dal digital twin
    baseline_pfs = ai_result.get('digital_twin', {}).get('pfs_soc', 18)

    try:
        protocol = generate_adaptive_protocol(
            patient_data, tumor_change, weeks_stable,
            baseline_burden, current_burden, baseline_pfs
        )
    except Exception as e:
        print(f"⚠️ Adaptive Therapy error: {e}")
        return

    pdf.add_page()

    # Header
    pdf.set_fill_color(128, 0, 128)  # Purple
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, safe_text("  ADAPTIVE THERAPY OPTIMIZER"), 0, 1, 'L', 1)
    pdf.set_text_color(0)
    pdf.ln(3)

    # === ELIGIBILITY STATUS ===
    if protocol.eligible_for_adaptive:
        pdf.set_fill_color(200, 255, 200)
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, safe_text("  [OK] ELIGIBLE FOR ADAPTIVE THERAPY"), 0, 1, 'L', 1)
    else:
        pdf.set_fill_color(255, 220, 220)
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 8, safe_text("  [X] NOT ELIGIBLE FOR ADAPTIVE THERAPY"), 0, 1, 'L', 1)
        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0)
        pdf.cell(0, 6, safe_text(f"  Reason: {protocol.ineligibility_reason}"), 0, 1)

    pdf.set_text_color(0)
    pdf.ln(2)

    # === CURRENT STATUS ===
    pdf.set_fill_color(240, 240, 255)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("  CURRENT STATUS"), 0, 1, 'L', 1)

    pdf.set_font('Arial', '', 9)
    pdf.cell(63, 5, safe_text(f"  Therapy: {protocol.current_therapy[:25]}..."), 0, 0)
    pdf.cell(63, 5, safe_text(f"  Response: {protocol.response_status.value}"), 0, 0)
    pdf.cell(64, 5, safe_text(f"  Phase: {protocol.current_phase.value}"), 0, 1)

    pdf.cell(63, 5, safe_text(f"  Tumor Change: {tumor_change:+.0f}%"), 0, 0)
    pdf.cell(63, 5, safe_text(f"  Stable: {weeks_stable} weeks"), 0, 0)
    pdf.cell(64, 5, safe_text(f"  Next Assessment: {protocol.next_assessment_weeks}w"), 0, 1)
    pdf.ln(3)

    # === PFS BENEFIT PROJECTION ===
    pdf.set_fill_color(255, 250, 230)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("  PROJECTED PFS BENEFIT"), 0, 1, 'L', 1)

    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, safe_text(f"  Standard Therapy PFS: {protocol.estimated_pfs_standard:.1f} months"), 0, 1)

    if protocol.eligible_for_adaptive:
        pdf.set_text_color(0, 128, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text(
            f"  Adaptive Therapy PFS: {protocol.estimated_pfs_adaptive:.1f} months (+{protocol.benefit_months:.1f}m)"),
                 0, 1)
        pdf.set_text_color(0)
    else:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, safe_text(f"  Adaptive Therapy: Not applicable"), 0, 1)

    pdf.ln(2)

    # === THRESHOLDS ===
    if protocol.eligible_for_adaptive:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 7, safe_text("ADAPTIVE THRESHOLDS:"), 0, 1)

        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, safe_text(f"  Drug Holiday Safe When: Tumor burden < {protocol.holiday_threshold:.0f}mm"), 0, 1)
        pdf.cell(0, 5, safe_text(f"  Resume Therapy When: Tumor burden > {protocol.resume_threshold:.0f}mm"), 0, 1)
        pdf.ln(3)

    # === CYCLE PROTOCOL ===
    if protocol.cycles:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, safe_text("ADAPTIVE CYCLE PROTOCOL:"), 0, 1)

        for i, cycle in enumerate(protocol.cycles[:4]):  # Max 4 cycles
            # Phase header
            if cycle.phase == TherapyPhase.DRUG_HOLIDAY:
                pdf.set_fill_color(200, 255, 200)
                phase_icon = "[HOLIDAY]"
            elif cycle.phase == TherapyPhase.INTENSIFICATION:
                pdf.set_fill_color(255, 200, 200)
                phase_icon = "[INTENSE]"
            elif cycle.phase == TherapyPhase.METRONOMIC:
                pdf.set_fill_color(230, 230, 255)
                phase_icon = "[METRO]"
            else:
                pdf.set_fill_color(255, 255, 230)
                phase_icon = "[TREAT]"

            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 6,
                     safe_text(f"  CYCLE {i + 1}: {cycle.phase.value} {phase_icon} - {cycle.duration_weeks} weeks"), 0,
                     1, 'L', 1)

            pdf.set_font('Arial', '', 8)
            pdf.cell(0, 4, safe_text(f"    Drug: {cycle.drug} | Dose: {cycle.dose}"), 0, 1)
            pdf.cell(0, 4, safe_text(f"    Rationale: {cycle.rationale[:70]}..."), 0, 1)

            # Monitoring
            pdf.set_font('Arial', 'I', 7)
            pdf.set_text_color(100, 100, 100)
            monitoring_str = " | ".join(cycle.monitoring[:2])
            pdf.cell(0, 3, safe_text(f"    Monitoring: {monitoring_str}"), 0, 1)

            # Triggers
            pdf.set_text_color(0, 100, 150)
            trigger_str = cycle.transition_triggers[0] if cycle.transition_triggers else "Per protocol"
            pdf.cell(0, 3, safe_text(f"    Transition: {trigger_str}"), 0, 1)
            pdf.set_text_color(0)

            pdf.ln(1)

    # === IMMEDIATE ACTION ===
    pdf.ln(2)
    pdf.set_fill_color(230, 255, 230)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("  SCENARIO EXPLORATION (Requires MDT Review)"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 9)
    # Wrap recommendation as exploratory scenario, not prescription
    action_text = protocol.immediate_action
    # Rimuovi frasi troppo prescrittive
    action_text = action_text.replace("Excellent response!", "Deep response observed.")
    action_text = action_text.replace("Consider initiating drug holiday protocol.",
                                      "Drug holiday could be explored under trial-like monitoring conditions.")
    action_text = action_text.replace("Consider initiating", "Could explore")

    if "Scenario" not in action_text and "scenario" not in action_text:
        action_text = f"Scenario: {action_text}"
    if "MDT" not in action_text and "investigational" not in action_text:
        action_text = f"{action_text} (investigational - requires MDT discussion)"
    pdf.multi_cell(0, 5, safe_text(f"  {action_text}"))

    # === RATIONALE BOX ===
    pdf.ln(3)
    pdf.set_fill_color(245, 245, 255)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 6, safe_text("  SCIENTIFIC RATIONALE"), 0, 1, 'L', 1)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 4, safe_text(
        "  Adaptive therapy exploits evolutionary dynamics: resistant clones carry a 'fitness cost' "
        "when drug pressure is removed. Drug holidays allow sensitive clones to re-expand and "
        "outcompete resistant ones, maintaining tumor heterogeneity and delaying resistance. "
        "Studies show 20-40% PFS improvement in selected patients (Gatenby et al., Nat Rev Cancer 2019)."
    ))

    # Disclaimer
    pdf.ln(3)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 3, safe_text(
        "Note: Adaptive therapy is investigational for most tumor types. Implementation requires close monitoring "
        "and multidisciplinary team oversight. Patient selection is critical for safety and efficacy."
    ))
    pdf.set_text_color(0)


def draw_synthetic_lethality(pdf, patient_data: Dict):
    """Disegna la sezione Synthetic Lethality Finder"""

    if not SYNTHETIC_LETHALITY_AVAILABLE:
        return

    try:
        result = find_synthetic_lethality(patient_data, TumorType.NSCLC)
    except Exception as e:
        print(f"⚠️ Synthetic Lethality error: {e}")
        return

    # Skip se nessuna opportunità
    if result.total_opportunities == 0:
        return

    pdf.add_page()

    # Header
    pdf.set_fill_color(139, 69, 19)  # Brown/Bronze
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, safe_text("  SYNTHETIC LETHALITY FINDER"), 0, 1, 'L', 1)
    pdf.set_text_color(0)
    pdf.ln(3)

    # === SUMMARY BOX ===
    pdf.set_fill_color(255, 250, 240)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text(f"  {result.total_opportunities} Synthetic Lethality Opportunities Identified"), 0, 1, 'L',
             1)

    pdf.set_font('Arial', '', 9)
    pdf.cell(63, 5, safe_text(f"  FDA Approved: {result.fda_approved_count}"), 0, 0)
    pdf.cell(63, 5, safe_text(f"  High Confidence (>=70%): {result.high_confidence_count}"), 0, 0)
    pdf.cell(64, 5, safe_text(f"  Clinical Trials: {result.total_opportunities - result.fda_approved_count}"), 0, 1)
    pdf.ln(2)

    # === DETECTED ALTERATIONS ===
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("DETECTED GENETIC ALTERATIONS:"), 0, 1)
    pdf.set_font('Arial', '', 9)
    for alt in result.detected_alterations[:6]:
        pdf.cell(0, 5, safe_text(f"  - {alt}"), 0, 1)
    pdf.ln(2)

    # === TOP RECOMMENDATION ===
    if result.top_recommendation:
        top = result.top_recommendation

        if top.evidence_level == EvidenceLevel.FDA_APPROVED:
            pdf.set_fill_color(200, 255, 200)
            status_icon = "[FDA APPROVED]"
        elif top.evidence_level in [EvidenceLevel.PHASE_3, EvidenceLevel.PHASE_2]:
            pdf.set_fill_color(255, 255, 200)
            status_icon = f"[{top.evidence_level.value}]"
        else:
            pdf.set_fill_color(255, 230, 200)
            status_icon = f"[{top.evidence_level.value}]"

        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, safe_text(f"  TOP RECOMMENDATION {status_icon}"), 0, 1, 'L', 1)

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text(f"  {top.tumor_alteration} -> {top.inhibitor_drug}"), 0, 1)

        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, safe_text(f"  Target: {top.synthetic_partner}"), 0, 1)
        pdf.cell(0, 5, safe_text(f"  Dose: {top.drug_dose}"), 0, 1)
        pdf.cell(0, 5, safe_text(f"  Confidence: {top.confidence}%"), 0, 1)

        if top.response_rate:
            pdf.cell(0, 5, safe_text(f"  Expected Response Rate: {top.response_rate * 100:.0f}%"), 0, 1)

        if top.key_trial:
            pdf.set_text_color(0, 100, 150)
            pdf.cell(0, 5, safe_text(f"  Key Trial: {top.key_trial}"), 0, 1)
            pdf.set_text_color(0)

        # Mechanism
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, safe_text(f"  Mechanism: {top.mechanism[:200]}..."))
        pdf.set_text_color(0)

        # Biomarker required
        if top.biomarker_required:
            pdf.ln(1)
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(0, 4, safe_text(f"  Biomarker Required: {top.biomarker_required}"), 0, 1)

        pdf.ln(2)

    # === ALL OPPORTUNITIES TABLE ===
    if len(result.opportunities) > 1:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 7, safe_text("ALL SYNTHETIC LETHALITY OPPORTUNITIES:"), 0, 1)

        # Table header
        pdf.set_fill_color(220, 200, 180)
        pdf.set_font('Arial', 'B', 7)
        pdf.cell(40, 5, "Alteration", 1, 0, 'C', 1)
        pdf.cell(25, 5, "Target", 1, 0, 'C', 1)
        pdf.cell(35, 5, "Drug", 1, 0, 'C', 1)
        pdf.cell(25, 5, "Evidence", 1, 0, 'C', 1)
        pdf.cell(20, 5, "Conf.", 1, 0, 'C', 1)
        pdf.cell(20, 5, "ORR", 1, 0, 'C', 1)
        pdf.cell(25, 5, "Action", 1, 1, 'C', 1)

        # Table rows
        pdf.set_font('Arial', '', 7)
        for opp in result.opportunities[:8]:  # Max 8
            # Color by evidence level
            if opp.evidence_level == EvidenceLevel.FDA_APPROVED:
                pdf.set_fill_color(220, 255, 220)
            elif opp.evidence_level in [EvidenceLevel.PHASE_3, EvidenceLevel.PHASE_2]:
                pdf.set_fill_color(255, 255, 220)
            else:
                pdf.set_fill_color(255, 240, 220)

            # Truncate long strings
            alt_short = opp.tumor_alteration[:18] + ".." if len(opp.tumor_alteration) > 18 else opp.tumor_alteration
            target_short = opp.synthetic_partner[:12] if len(opp.synthetic_partner) > 12 else opp.synthetic_partner
            drug_short = opp.inhibitor_drug[:16] + ".." if len(opp.inhibitor_drug) > 16 else opp.inhibitor_drug

            pdf.cell(40, 4, safe_text(alt_short), 1, 0, 'L', 1)
            pdf.cell(25, 4, safe_text(target_short), 1, 0, 'C', 1)
            pdf.cell(35, 4, safe_text(drug_short), 1, 0, 'L', 1)
            pdf.cell(25, 4, opp.evidence_level.value[:10], 1, 0, 'C', 1)
            pdf.cell(20, 4, f"{opp.confidence}%", 1, 0, 'C', 1)

            orr_str = f"{opp.response_rate * 100:.0f}%" if opp.response_rate else "N/A"
            pdf.cell(20, 4, orr_str, 1, 0, 'C', 1)

            action = "Rx" if opp.evidence_level == EvidenceLevel.FDA_APPROVED else "Trial"
            pdf.cell(25, 4, action, 1, 1, 'C', 1)

        pdf.ln(3)

    # === ACTIONABILITY SUMMARY ===
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("  CLINICAL ACTIONABILITY"), 0, 1, 'L', 1)

    pdf.set_font('Arial', '', 9)
    if result.immediate_actionable:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 5, safe_text("  [OK] FDA-approved options available - can prescribe now"), 0, 1)
    else:
        pdf.set_text_color(255, 140, 0)
        pdf.cell(0, 5, safe_text("  [!] No FDA-approved options - clinical trial enrollment recommended"), 0, 1)

    if result.clinical_trial_recommended:
        pdf.set_text_color(0, 100, 150)
        pdf.cell(0, 5, safe_text("  [>] Additional opportunities available through clinical trials"), 0, 1)

    pdf.set_text_color(0)

    # === MONITORING FOR TOP REC ===
    if result.top_recommendation and result.top_recommendation.monitoring:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 6, safe_text("MONITORING REQUIREMENTS (if initiating top recommendation):"), 0, 1)
        pdf.set_font('Arial', '', 8)
        for mon in result.top_recommendation.monitoring[:3]:
            pdf.cell(0, 4, safe_text(f"  - {mon}"), 0, 1)

    # === CONTRAINDICATIONS ===
    if result.top_recommendation and result.top_recommendation.contraindications:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 6, safe_text("CONTRAINDICATIONS:"), 0, 1)
        pdf.set_font('Arial', '', 8)
        for contra in result.top_recommendation.contraindications[:3]:
            pdf.cell(0, 4, safe_text(f"  - {contra}"), 0, 1)
        pdf.set_text_color(0)

    # Disclaimer
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 3, safe_text(
        "Note: Synthetic lethality opportunities are based on genetic alterations detected in patient profile. "
        "Clinical application requires confirmation of biomarker status and consideration of individual patient factors. "
        "Many options are investigational - verify current trial availability at clinicaltrials.gov."
    ))
    pdf.set_text_color(0)

def generate_complete_visit_report(pdf, visit_data, ai_result, visit_label, visit_date,
                                    is_baseline=False, previous_data=None):
    """
    Genera un report COMPLETO per una singola visita.
    Questo è equivalente al vecchio PDF di 5 pagine.
    """
    base = visit_data
    gen = base.get('genetics', {})
    blood = base.get('blood_markers', {})
    bio = base.get('biomarkers', {})

    tank_score = ai_result.get('tank_score', 0)
    ferrari_score = ai_result.get('ferrari_score', 50)
    match_status = ai_result.get('match_status', 'GREEN')
    veto_active = ai_result.get('veto_active', False)
    ldh = float(blood.get('ldh') or 0)
    is_elephant = ldh > 350

    # =========================================================================
    # PAGINA 1: Header + AI Scores + Explainability
    # =========================================================================

    # Header anagrafico
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(100, 8, safe_text(f"PATIENT ID: {base.get('patient_id', 'Unknown')}"), 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, safe_text(f"Therapy: {base.get('current_therapy', 'N/A')}"), 0, 1, 'R')

    pdf.set_font('Arial', '', 10)
    pdf.cell(60, 6, safe_text(f"Visit Date: {format_date(visit_date)}"), 0)
    pdf.cell(60, 6, safe_text(f"Age/Sex: {base.get('age', 'N/A')}/{base.get('sex', 'N/A')}"), 0)
    pdf.cell(0, 6, safe_text(f"ECOG PS: {base.get('ecog_ps', 'N/A')}"), 0, 1)

    pdf.cell(60, 6, safe_text(f"Histology: {base.get('histology', 'N/A')}"), 0)
    pdf.cell(60, 6, safe_text(f"Stage: {base.get('stage', 'N/A')}"), 0)
    pdf.cell(0, 6, safe_text(f"Smoking: {base.get('smoking_status', 'N/A')}"), 0, 1)
    pdf.ln(5)

    # === DELTA BOX (solo per visite non-baseline) ===
    if not is_baseline and previous_data:
        draw_delta_box(pdf, base, previous_data)

    # === BANNER DIVERGENZA ===
    if match_status == "RED":
        bg, txt = (220, 20, 60), (255, 255, 255)
        msg = "CRITICAL DIVERGENCE (Review Required)"
    elif match_status == "ORANGE":
        bg, txt = (255, 140, 0), (255, 255, 255)
        msg = "HIGH DIVERGENCE"
    else:
        bg, txt = (34, 139, 34), (255, 255, 255)
        msg = "AI CONSENSUS REACHED"

    pdf.set_fill_color(*bg)
    pdf.set_text_color(*txt)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 12, safe_text(msg), 0, 1, 'C', 1)

    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(95, 8, safe_text(f"CLINICAL RULES (Tank): {tank_score}/100"), 1, 0, 'C')
    pdf.cell(95, 8, safe_text(f"BIOLOGICAL AI (Ferrari): {ferrari_score}%"), 1, 1, 'C')
    pdf.ln(4)

    # === EXPLAINABILITY ===
    # Pre-calc: visita compatta = stessa genetica e terapia del precedente
    _pre_compact = (not is_baseline and previous_data
                    and gen == previous_data.get('genetics', {})
                    and str(base.get('current_therapy', '')).lower() == str(previous_data.get('current_therapy', '')).lower())

    if _pre_compact:
        # Versione compatta: solo i punteggi, senza breakdown completo
        explainability = ai_result.get('explainability', {})
        overall_risk_data = next(
            (e for e in explainability.get('ferrari_breakdown', []) if e.get('evidence') == '_OVERALL_RISK'), None)
        overall_risk = overall_risk_data.get('probability', 0) if overall_risk_data else 0

        pdf.set_fill_color(245, 245, 255)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 7, safe_text(f"  Overall Resistance Risk: {overall_risk:.0f}% | "
                                  f"Tank: {tank_score}/100 | Ferrari: {ferrari_score}%"), 0, 1, 'L', 1)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, safe_text("  (Full breakdown unchanged - see baseline/previous visit)"), 0, 1)
        pdf.set_text_color(0)
    else:
        draw_explainability_section(pdf, ai_result, tank_score, ferrari_score)

    # === VETO ===
    if veto_active:
        draw_veto_section(pdf, ai_result)

    # === PRIMARY DRIVERS ===
    drivers = ai_result.get('active_genes', [])
    summary_parts = []
    if drivers:
        summary_parts.append(", ".join(drivers))
    if ldh > 350:
        summary_parts.append(f"High LDH ({ldh} U/L)")
    if is_elephant:
        summary_parts.append("[!] ELEPHANT Protocol")
    if not _pre_compact:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text("Primary Drivers Detected:"), 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, safe_text(", ".join(summary_parts) if summary_parts else "No Major Drivers"), 0, 'C')
    else:
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, safe_text(f"Drivers: {', '.join(summary_parts) if summary_parts else 'None'}"), 0, 1)

    # =========================================================================
    # PAGINA 2: Clinical Sections
    # =========================================================================
    # Helper: controlla se la genetica è cambiata rispetto alla visita precedente
    _genetics_changed = True
    _therapy_changed = True
    if not is_baseline and previous_data:
        prev_gen = previous_data.get('genetics', {})
        _genetics_changed = (gen != prev_gen)
        prev_therapy = str(previous_data.get('current_therapy', '')).lower()
        curr_therapy = str(base.get('current_therapy', '')).lower()
        _therapy_changed = (prev_therapy != curr_therapy)

    # Per visite compatte: non aggiungere una nuova pagina (le sezioni cliniche sono brevi)
    _is_compact_visit = (not is_baseline and not _genetics_changed and not _therapy_changed)
    if not _is_compact_visit:
        pdf.add_page()
    else:
        pdf.ln(5)

    # 1. GENOMICS
    pdf.section_title("1. GENOMIC & IMMUNE PROFILE (NGS/IHC)", (230, 230, 250))
    pdl1_val = bio.get('pdl1_percent', 0)
    tmb_value = bio.get('tmb_score') or ai_result.get('tmb', 0)

    if not is_baseline and not _genetics_changed:
        # Genetica invariata → versione compatta
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, safe_text("Genomic profile unchanged from previous visit."), 0, 1)
        pdf.set_text_color(0)
    else:
        gen_text = f"TP53 Status: {gen.get('tp53_status', 'wt')}\n"
        gen_text += f"KRAS Status: {gen.get('kras_mutation', 'wt')}\n"
        gen_text += f"HER2 Status: {gen.get('her2_status', 'wt')}\n"
        gen_text += f"STK11: {gen.get('stk11_status', 'wt')} | KEAP1: {gen.get('keap1_status', 'wt')}\n"
        gen_text += f"EGFR: {gen.get('egfr_status', 'wt')} | MET: {gen.get('met_status', 'wt')}\n"
        gen_text += f"TMB Score: {tmb_value} mut/Mb\n"
        gen_text += f"PD-L1 Expression (TPS): {pdl1_val}% --> {pdl1_status(pdl1_val)}"
        pdf.section_body(gen_text)

    # 1.1 PHARMACOGENOMICS (PGx)
    pgx = base.get('pgx_profile', {})
    if pgx:
        pdf.section_title("1.1 PHARMACOGENOMICS (Safety)", (240, 230, 240))
        pgx_text = ""
        # Priority genes order
        priority = ['DPYD', 'UGT1A1', 'TPMT', 'CYP2D6', 'CYP2C19', 'CYP3A4']
        
        for gene in priority:
            if gene in pgx:
                val = pgx[gene]
                alert = ""
                # Simple alert logic for visualization
                if gene == 'DPYD' and ('*2A' in val or '*13' in val):
                    alert = " --> [!!!] CRITICAL RISK (5-FU CONTRAINDICATED)"
                elif gene == 'UGT1A1' and '*28/*28' in val:
                    alert = " --> [!] RISK (Irinotecan Dose Reduction)"
                elif gene == 'TPMT' and ('*3A' in val or '*3C' in val):
                    alert = " --> [!] RISK (Thiopurine Toxicity)"
                
                pgx_text += f"{gene}: {val}{alert}\n"
        
        # Others
        for gene, val in pgx.items():
            if gene not in priority:
                pgx_text += f"{gene}: {val}\n"
                
        pdf.section_body(pgx_text)

    # 2. HEMATO-ONCOLOGY
    pdf.section_title("2. HEMATO-ONCOLOGY (Metabolic & Immune)", (255, 228, 196))
    warburg_tag = "--> [!] HIGH (Warburg Effect)" if ldh > 350 else "--> Normal"
    neut = float(blood.get('neutrophils') or 0)
    lymph = float(blood.get('lymphocytes') or 0)
    # NLR: ricalcolato da neutrofili/linfociti reali (mai default a 1)
    if neut > 0 and lymph > 0:
        nlr = round(neut / lymph, 1)
    else:
        nlr = 0
    # Validazione: NLR > 50 clinicamente impossibile
    if nlr > 50:
        nlr = 0

    hem_text = f"LDH Level: {ldh} U/L {warburg_tag}\n"
    hem_text += f"NLR Ratio: {nlr} (Neut/Lymph)\n"
    hem_text += f"Neutrophils: {neut} | Lymphocytes: {lymph}"
    pdf.section_body(hem_text)

    # 3. DIGITAL PATHOLOGY
    pdf.section_title("3. DIGITAL PATHOLOGY (AI VISION)", (255, 192, 203))
    vision_data = ai_result.get('vision_data')
    if vision_data and vision_data.get('visual_risk'):
        pdf.section_body(f"Biopsy Analysis: {vision_data.get('class', 'Unknown')} (Risk: {vision_data.get('visual_risk', 0)}%)")
    else:
        pdf.section_body("No biopsy image attached.")

    # 4. PHYSICS VALIDATION (AlphaFold + Vina Integration)
    pdf.section_title("4. PHYSICS VALIDATION (AlphaFold/Vina)", (176, 224, 230))
    # Se non è baseline e terapia+genetica non cambiate → versione compatta
    _skip_physics_details = (not is_baseline and not _therapy_changed and not _genetics_changed)
    if _skip_physics_details:
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, safe_text("Same therapy and molecular target as previous visit. See baseline assessment."), 0, 1)
        pdf.set_text_color(0)

    if not _skip_physics_details:
        # Try to get real AlphaFold validation
        physics_result = None
        current_therapy = str(base.get('current_therapy', '')).lower()
        is_immunotherapy = False
        is_bsc = False
        ici_target = None

        # Check for immunotherapy or BSC first
        if any(x in current_therapy for x in ['pembrolizumab', 'nivolumab', 'keytruda', 'opdivo']):
            is_immunotherapy = True
            ici_target = 'PD-1'
        elif any(x in current_therapy for x in ['atezolizumab', 'tecentriq']):
            is_immunotherapy = True
            ici_target = 'PD-L1'
        elif any(x in current_therapy for x in ['durvalumab', 'imfinzi']):
            is_immunotherapy = True
            ici_target = 'PD-L1'
        elif any(x in current_therapy for x in ['ipilimumab', 'yervoy']):
            is_immunotherapy = True
            ici_target = 'CTLA-4'
        elif any(x in current_therapy for x in ['supportive', 'bsc', 'palliative', 'best supportive']):
            is_bsc = True

        # Only try molecular docking for small molecule drugs
        if not is_immunotherapy and not is_bsc:
            try:
                from alphafold_integration import validate_therapy, is_available as alphafold_available

                if alphafold_available():
                    genetics = base.get('genetics', {})

                    # Determine target gene and mutation
                    target_gene = None
                    target_mutation = None

                    # EGFR
                    egfr = str(genetics.get('egfr_status', '')).upper()
                    if egfr and egfr not in ['WT', 'WILD-TYPE', 'NEGATIVE', '']:
                        target_gene = 'EGFR'
                        target_mutation = egfr.replace('EGFR', '').strip().split()[0]

                    # KRAS
                    kras = str(genetics.get('kras_mutation', '')).upper()
                    if kras and kras not in ['WT', 'WILD-TYPE', 'NEGATIVE', '']:
                        target_gene = 'KRAS'
                        target_mutation = kras

                    # BRAF
                    braf = str(genetics.get('braf_status', '')).upper()
                    if braf and braf not in ['WT', 'WILD-TYPE', 'NEGATIVE', '']:
                        target_gene = 'BRAF'
                        target_mutation = braf

                    # MET amplification
                    met_cn = float(genetics.get('met_cn', 0) or 0)
                    if met_cn >= 6:
                        target_gene = 'MET'
                        target_mutation = 'amplification'

                    # Determine drug from therapy
                    drug = None
                    if 'osimertinib' in current_therapy:
                        drug = 'osimertinib'
                    elif 'gefitinib' in current_therapy:
                        drug = 'gefitinib'
                    elif 'erlotinib' in current_therapy:
                        drug = 'erlotinib'
                    elif 'sotorasib' in current_therapy:
                        drug = 'sotorasib'
                    elif 'adagrasib' in current_therapy:
                        drug = 'adagrasib'
                    elif 'dabrafenib' in current_therapy:
                        drug = 'dabrafenib'
                    elif 'trametinib' in current_therapy:
                        drug = 'trametinib'
                    elif 'capmatinib' in current_therapy:
                        drug = 'capmatinib'
                    elif 'crizotinib' in current_therapy:
                        drug = 'crizotinib'
                    elif 'tepotinib' in current_therapy:
                        drug = 'tepotinib'
                    elif 'alectinib' in current_therapy:
                        drug = 'alectinib'
                    elif 'lorlatinib' in current_therapy:
                        drug = 'lorlatinib'

                    # Run validation if we have all components
                    if target_gene and target_mutation and drug:
                        physics_result = validate_therapy(target_gene, target_mutation, drug)
            except Exception as e:
                print(f"⚠️ AlphaFold validation error: {e}")

        # Display results
        pdf.set_font('Arial', '', 11)

        if physics_result:
            # Real AlphaFold + Vina results
            pdf.cell(0, 6, safe_text(f"Target Protein: {physics_result.target_gene} (AlphaFold DB v6)"), 0, 1)
            pdf.cell(0, 6, safe_text(f"Mutation Analyzed: {physics_result.mutation}"), 0, 1)
            pdf.cell(0, 6, safe_text(f"Drug: {physics_result.drug}"), 0, 1)

            # pLDDT at mutation site
            if physics_result.plddt_at_mutation:
                plddt = physics_result.plddt_at_mutation
                if plddt >= 90:
                    plddt_qual = "Very High"
                elif plddt >= 70:
                    plddt_qual = "High"
                elif plddt >= 50:
                    plddt_qual = "Medium"
                else:
                    plddt_qual = "Low"
                pdf.cell(0, 6, safe_text(f"Structure Confidence (pLDDT): {plddt:.1f} ({plddt_qual})"), 0, 1)

            # Binding energy
            pdf.cell(0, 6, safe_text(f"Predicted Binding Energy: {physics_result.binding_affinity} kcal/mol"), 0, 1)

            # Affinity classification with color - DEGRADE if pLDDT < 70
            binding_display = physics_result.binding_quality
            confidence_note = ""

            if physics_result.plddt_at_mutation and physics_result.plddt_at_mutation < 70:
                # Low structure confidence - degrade binding classification
                confidence_note = " (LOW CONFIDENCE - pLDDT < 70)"
                if physics_result.binding_quality == 'STRONG':
                    binding_display = "TENTATIVE-STRONG"
                    pdf.set_text_color(200, 150, 0)  # Orange instead of green
                else:
                    pdf.set_text_color(200, 150, 0)
            elif physics_result.binding_quality == 'STRONG':
                pdf.set_text_color(0, 150, 0)
            elif physics_result.binding_quality == 'WEAK':
                pdf.set_text_color(200, 0, 0)
            else:
                pdf.set_text_color(200, 150, 0)

            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, safe_text(f"Binding Classification: {binding_display}{confidence_note}"), 0, 1)
            pdf.set_text_color(0)

            # Recommendation
            pdf.set_font('Arial', 'I', 10)
            if 'APPROPRIATE' in physics_result.therapy_recommendation:
                pdf.set_text_color(0, 100, 0)
                pdf.cell(0, 6, safe_text(f"-> {physics_result.therapy_recommendation}"), 0, 1)
            elif 'RESISTANCE' in physics_result.therapy_recommendation:
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 6, safe_text(f"[!] {physics_result.therapy_recommendation}"), 0, 1)
            else:
                pdf.cell(0, 6, safe_text(f"-> {physics_result.therapy_recommendation}"), 0, 1)
            pdf.set_text_color(0)

        elif is_immunotherapy:
            # Immunotherapy - docking not applicable
            pdf.cell(0, 6, safe_text(f"Target: {ici_target} (Immune Checkpoint)"), 0, 1)
            pdf.cell(0, 6, safe_text(f"Drug Class: Immune Checkpoint Inhibitor (Antibody)"), 0, 1)
            pdf.ln(2)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, safe_text("Molecular docking not applicable for antibody-based therapies."), 0, 1)
            pdf.cell(0, 5, safe_text("Efficacy depends on PD-L1 expression, TMB, and tumor microenvironment."), 0, 1)
            pdf.set_text_color(0)

        elif is_bsc:
            # Best Supportive Care
            pdf.cell(0, 6, safe_text(f"Status: Best Supportive Care"), 0, 1)
            pdf.ln(2)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, safe_text("No active antineoplastic therapy."), 0, 1)
            pdf.cell(0, 5, safe_text("Molecular docking analysis not applicable."), 0, 1)
            pdf.set_text_color(0)

        else:
            # Unknown therapy or no actionable target
            pdf.cell(0, 6, safe_text(f"Target Protein Structure: AlphaFold DB v6"), 0, 1)
            pdf.cell(0, 6, safe_text(f"Drug-Target Binding Energy: Not computed"), 0, 1)
            pdf.cell(0, 6, safe_text(f"Affinity Classification: N/A"), 0, 1)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(100, 100, 100)

            # =========================================================================
            # PHYSICS VALIDATION - Context-Aware Messages
            # =========================================================================

            def get_physics_message(genetics: dict, diagnosis: str) -> dict:
                """
                Genera messaggio Physics context-aware basato sul profilo molecolare.

                Returns:
                    dict con 'main_message', 'rationale', 'alternatives'
                """

                # Estrai mutazioni
                tp53 = genetics.get('tp53_status', 'wt')
                egfr = genetics.get('egfr_status', 'wt')
                kras = genetics.get('kras_mutation', 'wt')
                alk = genetics.get('alk_status', 'wt')
                braf = genetics.get('braf_status', genetics.get('braf_mutation', 'wt'))
                her2 = genetics.get('her2_status', 'wt')
                met = genetics.get('met_status', 'wt')

                # Check se è wild-type per tutto
                wt_values = ['wt', 'WT', 'wild-type', 'Wild-Type', '', None, 'negative']

                has_tp53 = tp53 not in wt_values
                has_egfr = egfr not in wt_values
                has_kras = kras not in wt_values
                has_alk = alk not in wt_values
                has_braf = braf not in wt_values
                has_her2 = her2 not in wt_values
                has_met = met not in wt_values and 'amp' in str(met).lower()

                # === CASO 1: EGFR mutato (druggable) ===
                if has_egfr:
                    variant = egfr if egfr not in ['mutated', 'positive'] else 'activating mutation'
                    return {
                        'main_message': f"EGFR {variant} - DRUGGABLE TARGET",
                        'target': 'EGFR kinase domain',
                        'rationale': "EGFR tyrosine kinase inhibitors (TKIs) bind the ATP pocket, blocking oncogenic signaling.",
                        'docking_available': True,
                        'alternatives': [
                            "1st/2nd gen TKI: Erlotinib, Gefitinib, Afatinib",
                            "3rd gen TKI: Osimertinib (preferred if T790M or 1L)",
                            "4th gen TKI: Under development for C797S"
                        ]
                    }

                # === CASO 2: ALK/ROS1 (druggable) ===
                if has_alk:
                    return {
                        'main_message': "ALK fusion - DRUGGABLE TARGET",
                        'target': 'ALK kinase domain',
                        'rationale': "ALK inhibitors competitively bind the kinase domain, blocking constitutive activation.",
                        'docking_available': True,
                        'alternatives': [
                            "1st gen: Crizotinib",
                            "2nd gen: Alectinib, Brigatinib, Ceritinib",
                            "3rd gen: Lorlatinib"
                        ]
                    }

                # === CASO 3: KRAS G12C (druggable) ===
                if has_kras and 'G12C' in str(kras).upper():
                    return {
                        'main_message': "KRAS G12C - DRUGGABLE TARGET",
                        'target': 'KRAS switch II pocket',
                        'rationale': "Covalent inhibitors trap KRAS in inactive GDP-bound state via C12 residue.",
                        'docking_available': True,
                        'alternatives': [
                            "Sotorasib (AMG 510) - FDA approved",
                            "Adagrasib (MRTX849) - FDA approved",
                            "Combination strategies under investigation"
                        ]
                    }

                # === CASO 4: BRAF V600E (druggable) ===
                if has_braf and 'V600' in str(braf).upper():
                    return {
                        'main_message': "BRAF V600E - DRUGGABLE TARGET",
                        'target': 'BRAF kinase domain',
                        'rationale': "BRAF inhibitors block constitutively active kinase. Combine with MEK inhibitor to prevent resistance.",
                        'docking_available': True,
                        'alternatives': [
                            "Dabrafenib + Trametinib (preferred)",
                            "Vemurafenib + Cobimetinib",
                            "Encorafenib + Binimetinib"
                        ]
                    }

                # === CASO 5: HER2 (druggable) ===
                if has_her2:
                    return {
                        'main_message': "HER2 alteration - DRUGGABLE TARGET",
                        'target': 'HER2/ERBB2 receptor',
                        'rationale': "HER2-directed therapies include antibodies, ADCs, and TKIs.",
                        'docking_available': True,
                        'alternatives': [
                            "Trastuzumab + Pertuzumab",
                            "T-DXd (Trastuzumab deruxtecan) - ADC",
                            "Tucatinib, Neratinib, Lapatinib - TKIs"
                        ]
                    }

                # === CASO 6: MET amplification (druggable) ===
                if has_met:
                    return {
                        'main_message': "MET amplification - DRUGGABLE TARGET",
                        'target': 'MET receptor tyrosine kinase',
                        'rationale': "MET inhibitors block HGF-independent activation from gene amplification.",
                        'docking_available': True,
                        'alternatives': [
                            "Capmatinib - FDA approved",
                            "Tepotinib - FDA approved",
                            "Savolitinib - in development"
                        ]
                    }

                # === CASO 7: TP53 mutato (NOT druggable) ===
                if has_tp53:
                    return {
                        'main_message': "TP53 mutation - NOT DIRECTLY DRUGGABLE",
                        'target': 'TP53 (tumor suppressor)',
                        'rationale': "TP53 is a tumor suppressor that functions through loss-of-function. "
                                     "Unlike oncogenic kinases, there is no active site to inhibit. "
                                     "Therapeutic strategies focus on synthetic lethality or p53 reactivation.",
                        'docking_available': False,
                        'alternatives': [
                            "Synthetic Lethality: WEE1 inhibitors (Adavosertib)",
                            "Synthetic Lethality: ATR inhibitors (Ceralasertib)",
                            "p53 Reactivation: APR-246/Eprenetapopt (investigational)",
                            "Immunotherapy: TP53 mutations may increase neoantigen load"
                        ],
                        'cross_reference': "See SYNTHETIC LETHALITY section for actionable options."
                    }

                # === CASO 8: KRAS non-G12C (NOT druggable yet) ===
                if has_kras:
                    variant = kras if kras not in ['mutated', 'positive'] else 'mutation'
                    return {
                        'main_message': f"KRAS {variant} - LIMITED DRUGGABILITY",
                        'target': 'KRAS',
                        'rationale': f"KRAS {variant} lacks the cysteine residue required for covalent inhibitors. "
                                     "Direct targeting is challenging; focus on downstream pathway inhibition.",
                        'docking_available': False,
                        'alternatives': [
                            "MEK inhibitors: Trametinib, Cobimetinib",
                            "SHP2 inhibitors: Under investigation",
                            "SOS1 inhibitors: Under investigation",
                            "Pan-KRAS inhibitors: In early trials"
                        ]
                    }

                # === CASO 9: Immunotherapy patient (no molecular target) ===
                diagnosis_lower = str(diagnosis).lower()
                immuno_histologies = ['urothelial', 'bladder', 'melanoma', 'msi-h', 'tmb-h']

                if any(h in diagnosis_lower for h in immuno_histologies):
                    return {
                        'main_message': "Immunotherapy-responsive histology",
                        'target': 'Immune checkpoint (PD-1/PD-L1/CTLA-4)',
                        'rationale': "Physics-based docking not applicable for immunotherapy. "
                                     "Therapeutic effect is mediated by immune system reactivation, not direct drug-target binding.",
                        'docking_available': False,
                        'alternatives': [
                            "PD-1 inhibitors: Pembrolizumab, Nivolumab",
                            "PD-L1 inhibitors: Atezolizumab, Durvalumab, Avelumab",
                            "CTLA-4 inhibitors: Ipilimumab (combination)"
                        ],
                        'note': "Response correlates with PD-L1, TMB, and MSI status rather than structural drug-target interactions."
                    }

                # === CASO 10: Wild-type / No actionable targets ===
                return {
                    'main_message': "No actionable molecular targets identified",
                    'target': 'None',
                    'rationale': "Current molecular profile does not reveal druggable oncogenic drivers. "
                                 "Consider comprehensive genomic profiling (CGP) if not already performed.",
                    'docking_available': False,
                    'alternatives': [
                        "Chemotherapy based on histology",
                        "Clinical trial enrollment",
                        "Repeat NGS if disease progresses",
                        "Liquid biopsy for emerging mutations"
                    ]
                }

            # =========================================================
            # CHIAMATA FUNZIONE E RENDERING NEL PDF
            # =========================================================
            genetics = base.get('genetics', {})
            diagnosis = base.get('diagnosis', base.get('histology', ''))
            physics_msg = get_physics_message(genetics, diagnosis)

            # Main message con colore appropriato
            pdf.set_font('Arial', 'B', 10)
            if physics_msg.get('docking_available'):
                pdf.set_text_color(0, 128, 0)  # Verde = druggable
            else:
                pdf.set_text_color(180, 100, 0)  # Arancione = not druggable
            pdf.cell(0, 6, safe_text(physics_msg['main_message']), 0, 1)

            # Target
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)
            pdf.cell(0, 5, safe_text(f"Target: {physics_msg['target']}"), 0, 1)

            # Rationale
            pdf.ln(2)
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 5, safe_text(physics_msg['rationale']))

            # Alternatives
            if physics_msg.get('alternatives'):
                pdf.ln(2)
                pdf.set_text_color(0)
                pdf.set_font('Arial', 'B', 9)
                pdf.cell(0, 5, safe_text("Therapeutic Strategies:"), 0, 1)
                pdf.set_font('Arial', '', 9)
                for alt in physics_msg['alternatives']:
                    pdf.cell(0, 5, safe_text(f"  - {alt}"), 0, 1)

            # Cross-reference (per TP53 -> Synthetic Lethality)
            if physics_msg.get('cross_reference'):
                pdf.ln(2)
                pdf.set_font('Arial', 'BI', 9)
                pdf.set_text_color(0, 100, 150)
                pdf.cell(0, 5, safe_text(physics_msg['cross_reference']), 0, 1)

            pdf.set_text_color(0)
        # === 4b. ALPHAGENOME SPLICING ANALYSIS (if relevant) ===

        # === 4b. ALPHAGENOME SPLICING ANALYSIS (if relevant) ===
        try:
            from alphagenome_integration import AlphaGenomeAnalyzer, is_available as alphagenome_available

            if alphagenome_available():
                genetics = base.get('genetics', {})
                notes = str(base.get('notes', '')).lower()
                splicing_analyses = []

                # === Check for MET exon 14 skipping ===
                met_status = str(genetics.get('met_status', '')).lower()
                met_mutation = str(genetics.get('met_mutation', '')).lower()
                met_cn = float(genetics.get('met_cn', 0) or 0)
                has_met_exon14 = any(x in met_status + met_mutation for x in ['exon14', 'exon 14', 'ex14', 'skip'])
                has_met_splicing = any(x in notes for x in ['met exon 14', 'met splicing', 'met skip'])

                if has_met_exon14 or has_met_splicing or met_cn >= 6:
                    splicing_analyses.append({
                        'gene': 'MET',
                        'type': 'exon14',
                        'chromosome': 'chr7',
                        'position': 116771994,
                        'ref': 'G',
                        'alt': 'A'
                    })

                # === Check for EGFR exon 19 deletion ===
                egfr_status = str(genetics.get('egfr_status', '')).upper()
                egfr_mutation = str(genetics.get('egfr_mutation', '')).upper()
                egfr_combined = egfr_status + egfr_mutation

                has_egfr_ex19 = any(
                    x in egfr_combined for x in ['EXON19', 'EXON 19', 'EX19', 'DEL19', '19DEL', 'E746', 'L747'])
                has_egfr_ex19_notes = any(x in notes for x in ['exon 19', 'exon19', 'del19'])

                if has_egfr_ex19 or has_egfr_ex19_notes:
                    splicing_analyses.append({
                        'gene': 'EGFR',
                        'type': 'exon19',
                        'chromosome': 'chr7',
                        'position': 55242470,
                        'ref': 'G',
                        'alt': 'A'
                    })

                # === Check for EGFR exon 20 insertion ===
                has_egfr_ex20 = any(x in egfr_combined for x in
                                    ['EXON20', 'EXON 20', 'EX20', 'INS20', '20INS', 'A767', 'S768', 'V769', 'D770'])
                has_egfr_ex20_notes = any(x in notes for x in ['exon 20', 'exon20', 'ins20'])

                if has_egfr_ex20 or has_egfr_ex20_notes:
                    splicing_analyses.append({
                        'gene': 'EGFR',
                        'type': 'exon20',
                        'chromosome': 'chr7',
                        'position': 55249000,
                        'ref': 'C',
                        'alt': 'T'
                    })

                # === Run AlphaGenome analysis if we have targets ===
                if splicing_analyses:
                    ag = AlphaGenomeAnalyzer()
                    if ag.is_available():
                        pdf.ln(3)
                        pdf.set_fill_color(240, 248, 255)
                        pdf.set_font('Arial', 'B', 10)
                        pdf.set_text_color(0, 70, 140)
                        pdf.cell(0, 7, safe_text("  [DeepMind AlphaGenome] Splicing Analysis"), 0, 1, 'L', 1)
                        pdf.set_text_color(0)

                        for analysis in splicing_analyses:
                            gene = analysis['gene']
                            analysis_type = analysis['type']

                            # Run appropriate analysis
                            if gene == 'MET' and analysis_type == 'exon14':
                                splice_result = ag.analyze_met_exon14()
                                region_name = "MET Exon 14 Splice Region"
                            else:
                                splice_result = ag.analyze_splicing_variant(
                                    gene=analysis['gene'],
                                    chromosome=analysis['chromosome'],
                                    position=analysis['position'],
                                    ref=analysis['ref'],
                                    alt=analysis['alt']
                                )
                                if analysis_type == 'exon19':
                                    region_name = "EGFR Exon 19 Deletion Region"
                                elif analysis_type == 'exon20':
                                    region_name = "EGFR Exon 20 Insertion Region"
                                else:
                                    region_name = f"{gene} Splice Region"

                            if splice_result and not splice_result.error:
                                pdf.set_font('Arial', 'B', 9)
                                pdf.set_text_color(0, 50, 100)
                                pdf.cell(0, 6, safe_text(f"  {region_name}"), 0, 1)
                                pdf.set_text_color(0)

                                pdf.set_font('Arial', '', 9)

                                # Affected exon
                                if splice_result.affected_exon:
                                    pdf.cell(0, 5, safe_text(f"    Affected Exon: {splice_result.affected_exon}"), 0, 1)

                                # Splice site scores
                                if splice_result.ref_max_score:
                                    score_quality = "Strong" if splice_result.ref_max_score > 0.8 else (
                                        "Moderate" if splice_result.ref_max_score > 0.5 else "Weak")
                                    site_info = f" ({splice_result.splice_site_type})" if splice_result.splice_site_type else ""
                                    pdf.cell(0, 5, safe_text(
                                        f"    Splice Site Strength: {splice_result.ref_max_score:.3f} - {score_quality}{site_info}"),
                                             0, 1)

                                # Delta splice score if significant
                                if splice_result.delta_splice_score and abs(splice_result.delta_splice_score) > 0.01:
                                    if splice_result.delta_splice_score < -0.1:
                                        pdf.set_text_color(200, 0, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Splice Score Change: {splice_result.delta_splice_score:.3f} (DISRUPTED)"),
                                                 0, 1)
                                    elif splice_result.delta_splice_score > 0.1:
                                        pdf.set_text_color(200, 100, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Splice Score Change: {splice_result.delta_splice_score:+.3f} (ENHANCED)"),
                                                 0, 1)
                                    else:
                                        pdf.cell(0, 5, safe_text(
                                            f"    Splice Score Change: {splice_result.delta_splice_score:+.3f}"), 0, 1)
                                    pdf.set_text_color(0)

                                # Expression change
                                if splice_result.expression_change:
                                    if splice_result.expression_change < 0.5:
                                        pdf.set_text_color(200, 0, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Predicted Expression: {splice_result.expression_change:.2f}x (SEVERELY REDUCED)"),
                                                 0, 1)
                                    elif splice_result.expression_change < 0.7:
                                        pdf.set_text_color(200, 100, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Predicted Expression: {splice_result.expression_change:.2f}x (REDUCED)"),
                                                 0, 1)
                                    elif splice_result.expression_change > 2.0:
                                        pdf.set_text_color(200, 0, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Predicted Expression: {splice_result.expression_change:.2f}x (OVEREXPRESSED)"),
                                                 0, 1)
                                    elif splice_result.expression_change > 1.5:
                                        pdf.set_text_color(200, 100, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Predicted Expression: {splice_result.expression_change:.2f}x (INCREASED)"),
                                                 0, 1)
                                    else:
                                        pdf.set_text_color(0, 100, 0)
                                        pdf.cell(0, 5, safe_text(
                                            f"    Predicted Expression: {splice_result.expression_change:.2f}x (stable)"),
                                                 0, 1)
                                    pdf.set_text_color(0)

                                # Clinical relevance - highlight if actionable
                                if 'ACTIONABLE' in splice_result.clinical_relevance.upper():
                                    pdf.set_font('Arial', 'B', 9)
                                    pdf.set_text_color(0, 120, 0)
                                    pdf.cell(0, 6, safe_text(f"    [!] {splice_result.clinical_relevance}"), 0, 1)
                                    pdf.set_text_color(0)
                                else:
                                    pdf.set_font('Arial', 'I', 8)
                                    pdf.set_text_color(80, 80, 80)
                                    pdf.cell(0, 5, safe_text(f"    {splice_result.clinical_relevance}"), 0, 1)
                                    pdf.set_text_color(0)

                                pdf.ln(1)

                        # Footer
                        pdf.set_font('Arial', 'I', 7)
                        pdf.set_text_color(100, 100, 100)
                        pdf.cell(0, 4, safe_text(
                            "    Model: DeepMind AlphaGenome | Tissue: Lung (UBERON:0002048) | Resolution: 1bp"), 0, 1)
                        pdf.set_text_color(0)
                        pdf.ln(2)

        except Exception as e:
            print(f"⚠️ AlphaGenome PDF integration error: {e}")

    # =========================================================================
    # PAGINA 3: ELEPHANT PROTOCOL (se attivo)
    # =========================================================================

    # --- PAGINA PROTOCOLLO ELEFANTE (v2.0 Dynamic) ---
    if is_elephant:
        # Per visite non-baseline con stessa genetica: versione compatta
        if not is_baseline and not _genetics_changed:
            # Elephant Protocol già mostrato nel baseline, solo update metabolico
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 11)
            pdf.set_fill_color(255, 140, 0)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, safe_text("  [!] ELEPHANT PROTOCOL - UPDATE"), 0, 1, 'L', 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, safe_text(f"  LDH: {ldh:.0f} U/L (see baseline for full protocol)"), 0, 1)
        else:
            pdf.add_page()
            # Determina se TP53 è mutato
            tp53_status = (base.get('genetics') or {}).get('tp53_status', '').lower()
            tp53_mut = tp53_status in ['mutated', 'mut', 'loss']
            if ELEPHANT_V2_AVAILABLE:
                # Genera protocollo personalizzato
                elephant_result = generate_elephant_protocol({'baseline': visit_data}, ai_result)
                _draw_elephant_protocol_v2(pdf, elephant_result, ai_result)
            else:
                # Fallback al protocollo legacy
                _draw_elephant_protocol_legacy(pdf, base, ldh, tp53_mut, ai_result)
            pdf.set_text_color(0)

    # =========================================================================
    # PAGINA 4: DIGITAL TWIN
    # =========================================================================
    # Per visite follow-up compatte (stessa terapia e genetica): versione condensata
    _compact_twin = (not is_baseline and not _therapy_changed and not _genetics_changed)
    if not _compact_twin:
        pdf.add_page()
    else:
        pdf.ln(5)
    pdf.section_title("5. DIGITAL TWIN SIMULATION (PROGNOSIS)", (0, 102, 204))
    # Nella sezione DIGITAL TWIN, dopo pdf.section_title():

    twin = ai_result.get('digital_twin', {})
    model_source = twin.get('model_source', 'FORMULA')

    # Mostra la fonte del modello
    if model_source == 'ML_500K':
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(0, 128, 0)  # Verde
        pdf.cell(0, 5, safe_text("Model: ML-trained on 501,661 real patients (C-index: 0.818)"), 0, 1)
        pdf.set_text_color(0, 0, 0)
    elif model_source == 'VETO_OVERRIDE':
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(255, 0, 0)  # Rosso
        pdf.cell(0, 5, safe_text("⚠️ VETO ACTIVE - Therapy mismatch detected"), 0, 1)
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)  # Grigio
        pdf.cell(0, 5, safe_text("Model: Formula-based estimation"), 0, 1)
        pdf.set_text_color(0, 0, 0)

    # Se disponibile, mostra anche il death risk ML
    if 'ml_death_risk' in twin:
        pdf.set_font("Helvetica", "", 9)
        risk = twin['ml_death_risk']
        risk_color = (255, 0, 0) if risk > 70 else (255, 165, 0) if risk > 40 else (0, 128, 0)
        pdf.set_text_color(*risk_color)
        pdf.cell(0, 5, safe_text(f"ML Death Risk: {risk}%"), 0, 1)
        pdf.set_text_color(0, 0, 0)
    twin = ai_result.get('digital_twin', {})
    if twin:
        if _compact_twin:
            # Versione compatta: solo stats, senza grafico PFS (risparmia ~1 pagina)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, safe_text(f"Median PFS (SOC): {twin.get('pfs_soc', 'N/A')} months"), ln=True)
            delta = twin.get('delta', 0)
            if delta > 0:
                pdf.set_text_color(0, 150, 0)
                pdf.cell(0, 6, safe_text(f"SENTINEL Protocol: +{delta} months benefit"), ln=True)
            elif veto_active:
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 6, safe_text("WARNING: Therapy Mismatch - PFS compromised"), ln=True)
            pdf.set_text_color(0)
            pdf.set_font("Courier", "", 9)
            pdf.cell(0, 5, safe_text(f"Dynamics: {twin.get('dynamics', 'N/A')} | Forecast: {twin.get('forecast', 'N/A')}"), ln=True)
        else:
            # Versione completa con grafico PFS
            # PFS Graph
            try:
                graph_path = generate_pfs_curve(
                    risk_score=max(tank_score, ferrari_score),
                    pfs_soc=twin.get('pfs_soc', 12),
                    pfs_sentinel=twin.get('pfs_sentinel', 12),
                    elephant_active=is_elephant,
                    veto_active=veto_active,
                    patient_id=base.get('patient_id', 'Unknown')
                )
                pdf.image(graph_path, x=15, y=pdf.get_y() + 5, w=180)
                pdf.ln(100)
                os.unlink(graph_path)
            except Exception as e:
                print(f"⚠️ PFS graph error: {e}")
                pdf.ln(10)

            # Stats
            pdf.set_fill_color(240, 248, 255)
            pdf.rect(10, pdf.get_y(), 190, 35, 'F')
            pdf.set_xy(15, pdf.get_y() + 5)

            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, safe_text(f"Median PFS (SOC): {twin.get('pfs_soc', 'N/A')} Months"), ln=True)

            delta = twin.get('delta', 0)
            if delta > 0:
                pdf.set_text_color(0, 150, 0)
                pdf.set_x(15)
                pdf.cell(0, 7, safe_text(f"SENTINEL Protocol: +{delta} months benefit"), ln=True)
            elif veto_active:
                pdf.set_text_color(200, 0, 0)
                pdf.set_x(15)
                pdf.cell(0, 7, safe_text("WARNING: Therapy Mismatch - PFS compromised"), ln=True)
            else:
                pdf.set_text_color(0, 102, 204)
                pdf.set_x(15)
                pdf.cell(0, 7, safe_text("Optimal therapy - No intervention needed"), ln=True)

            pdf.set_text_color(0)
            pdf.ln(5)
            pdf.set_font("Courier", "", 10)
            pdf.set_x(15)
            pdf.cell(0, 6, safe_text(f"Dynamics: {twin.get('dynamics', 'N/A')}"), ln=True)
            pdf.set_x(15)
            pdf.cell(0, 6, safe_text(f"Forecast: {twin.get('forecast', 'N/A')}"), ln=True)

    # =========================================================================
    # PAGINA 5: PREDICTIVE TIMELINE (se disponibile)
    # =========================================================================
    if PREDICTIVE_TIMELINE_AVAILABLE:
        # Calcola mesi in terapia
        weeks_on_therapy = visit_data.get('week_on_therapy', 0) if not is_baseline else 0
        months_on_therapy = weeks_on_therapy // 4

        # Estrai overall risk da ai_result
        overall_risk_data = None
        ferrari_breakdown = ai_result.get('explainability', {}).get('ferrari_breakdown', [])
        for ev in ferrari_breakdown:
            if ev.get('evidence') == '_OVERALL_RISK':
                overall_risk_data = ev
                break

        current_overall_risk = overall_risk_data.get('probability', 50) if overall_risk_data else 50

        # Per visite compatte: versione condensata del timeline
        _compact_timeline = (not is_baseline and not _therapy_changed and not _genetics_changed)
        if _compact_timeline:
            # Genera il timeline per i dati ma mostra solo un riassunto
            try:
                timeline = generate_predictive_timeline(visit_data, current_overall_risk, months_on_therapy)
                pdf.ln(5)
                pdf.set_fill_color(230, 230, 255)
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, safe_text("  PREDICTIVE TIMELINE UPDATE"), 0, 1, 'L', 1)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, safe_text(
                    f"  Current Risk: {timeline.current_risk:.0f}% | "
                    f"Confidence: {timeline.model_confidence} | "
                    f"Monitoring: {timeline.monitoring_intensity}"
                ), 0, 1)
                # Mostra solo gli alert critici (>= 50%)
                critical_alerts = [a for a in timeline.alerts if a.threshold_percent >= 50]
                if critical_alerts:
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(200, 0, 0)
                    for alert in critical_alerts:
                        pdf.cell(0, 5, safe_text(
                            f"  [!] {alert.threshold_percent}% risk at month {alert.expected_month:.1f}"
                        ), 0, 1)
                    pdf.set_text_color(0)
                pdf.set_font('Arial', '', 9)
                pdf.cell(0, 5, safe_text(
                    f"  Next ctDNA: Week {timeline.next_ctdna_week} | "
                    f"Next Imaging: Week {timeline.next_imaging_week}"
                ), 0, 1)
            except Exception as e:
                print(f"⚠️ Compact Predictive Timeline error: {e}")
        else:
            draw_predictive_timeline(pdf, visit_data, current_overall_risk, months_on_therapy)

def draw_delta_box(pdf, current_data, previous_data):
    """Box che mostra il delta rispetto alla visita precedente"""
    curr_blood = current_data.get('blood_markers', {})
    prev_blood = previous_data.get('blood_markers', {})

    ldh_curr = float(curr_blood.get('ldh') or 0)
    ldh_prev = float(prev_blood.get('ldh') or ldh_curr)
    ldh_change = ldh_curr - ldh_prev

    ecog_curr = int(current_data.get('ecog_ps', 1))
    ecog_prev = int(previous_data.get('ecog_ps', ecog_curr))
    ecog_change = ecog_curr - ecog_prev

    therapy_curr = current_data.get('current_therapy', '')
    therapy_prev = previous_data.get('current_therapy', therapy_curr)
    therapy_changed = therapy_curr != therapy_prev

    # Box
    pdf.set_fill_color(255, 250, 230)
    pdf.set_draw_color(255, 200, 100)
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 18, 'DF')

    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(180, 100, 0)
    pdf.cell(0, 5, safe_text("[DELTA] Changes since previous visit:"), ln=True)

    pdf.set_font("Courier", "", 9)
    pdf.set_x(15)

    # LDH change
    if ldh_prev > 0:
        ldh_pct = ((ldh_curr - ldh_prev) / ldh_prev) * 100
        if ldh_change < 0:
            pdf.set_text_color(0, 128, 0)
            symbol = "[-]"
        elif ldh_change > 0:
            pdf.set_text_color(200, 0, 0)
            symbol = "[+]"
        else:
            pdf.set_text_color(0, 0, 200)
            symbol = "[=]"
        str_ldh = f"LDH: {ldh_change:+.0f} ({ldh_pct:+.1f}%) {symbol}"
    else:
        str_ldh = f"LDH: {ldh_curr:.0f}"
        pdf.set_text_color(0)

    pdf.cell(60, 5, safe_text(str_ldh), ln=0)

    # ECOG change
    if ecog_change < 0:
        pdf.set_text_color(0, 128, 0)
        str_ecog = f"ECOG: {ecog_change:+d} (Improved)"
    elif ecog_change > 0:
        pdf.set_text_color(200, 0, 0)
        str_ecog = f"ECOG: {ecog_change:+d} (Worse)"
    else:
        pdf.set_text_color(0, 0, 200)
        str_ecog = "ECOG: Stable"

    pdf.cell(60, 5, safe_text(str_ecog), ln=0)

    # Therapy change
    if therapy_changed:
        pdf.set_text_color(255, 140, 0)
        str_ther = "Therapy: CHANGED"
    else:
        pdf.set_text_color(0, 128, 0)
        str_ther = "Therapy: Same"

    pdf.cell(70, 5, safe_text(str_ther), ln=True)

    pdf.set_text_color(0)
    pdf.set_line_width(0.2)
    pdf.ln(8)


def draw_explainability_section(pdf, ai_result, tank_score, ferrari_score):
    """Sezione Explainability con Overall Resistance Risk"""
    explainability = ai_result.get('explainability', {})
    tank_breakdown = explainability.get('tank_breakdown', [])
    ferrari_breakdown = explainability.get('ferrari_breakdown', [])
    synergies = explainability.get('synergies', [])

    if not tank_breakdown and not ferrari_breakdown:
        return

    pdf.set_fill_color(245, 245, 255)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("  WHY THIS SCORE? (Explainability Report)"), 0, 1, 'L', 1)
    pdf.ln(2)

    # === TANK BREAKDOWN ===
    if tank_breakdown:
        pdf.set_font('Arial', 'B', 10)

        # Controlla se c'è stato cap
        cap_info = next((c for c in tank_breakdown if c.get('factor') == '_CAP_APPLIED'), None)
        if cap_info:
            raw_score = int(cap_info.get('value', '').replace('Raw score: ', ''))
            pdf.cell(0, 6, safe_text(f"Tank Score Breakdown ({tank_score}/100, capped from {raw_score}):"), 0, 1)
        else:
            pdf.cell(0, 6, safe_text(f"Tank Score Breakdown ({tank_score}/100):"), 0, 1)

        pdf.set_font('Arial', '', 9)
        for contrib in tank_breakdown[:8]:
            if contrib.get('factor') == '_CAP_APPLIED':
                continue
            factor = contrib.get('factor', 'Unknown')
            weight = contrib.get('weight', 0)
            category = contrib.get('category', '')
            if category == 'resistance':
                pdf.set_text_color(200, 0, 0)
            elif category == 'metabolic':
                pdf.set_text_color(255, 140, 0)
            else:
                pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 5, safe_text(f"    * {factor}: +{weight} pts"), 0, 1)
        pdf.set_text_color(0)

        if cap_info:
            pdf.set_font('Arial', 'I', 7)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"    Raw total: {raw_score} pts -> capped to 100 (maximum)"), 0, 1)
            pdf.set_text_color(0)
        pdf.ln(2)

    # === FERRARI BREAKDOWN ===
    if ferrari_breakdown:
        ferrari_mech = explainability.get('ferrari_mechanism', 'Unknown')
        ferrari_conf = explainability.get('ferrari_confidence', 'N/A')

        # Estrai Overall Risk se presente
        overall_risk_data = next((e for e in ferrari_breakdown if e.get('evidence') == '_OVERALL_RISK'), None)
        overall_risk = overall_risk_data.get('probability', 0) if overall_risk_data else 0
        num_evidences = overall_risk_data.get('num_evidences', 0) if overall_risk_data else 0

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text(f"Ferrari Score Breakdown ({ferrari_score}%):"), 0, 1)
        pdf.set_font('Arial', '', 9)

        # =======================================================================
        # FIX DEFINITIVO v2: Meccanismi che richiedono evidenza diretta
        # SEMPRE mostra come "theoretical" per C797S/T790M/MET se non c'è
        # evidenza DIRETTA nel paziente (ctDNA positivo, amplificazione confermata)
        # =======================================================================
        mechanisms_requiring_evidence = {
            'C797S_mutation': ['c797s', 'ctdna_c797s_confirmed', 'ctdna_c797s_trace'],
            'T790M_acquired': ['t790m', 'ctdna_t790m_confirmed', 'ctdna_t790m_trace'],
            'MET_amplification': ['met_amp', 'met amplification', 'met_cn_high'],
            'HER2_amplification': ['her2_amp', 'her2 amplification', 'her2_cn_high'],
            'SCLC_transformation': ['sclc', 'rb1_loss', 'neuroendocrine', 'small_cell'],
            'EMT_phenotype': ['emt', 'vimentin_high', 'e-cadherin_low'],
        }

        # Conta evidenze REALI (non priors/baseline) nel breakdown
        real_evidence_count = 0
        has_patient_evidence = False

        for ev in ferrari_breakdown:
            ev_name = str(ev.get('evidence', '')).lower()

            # Skip meta-evidenze e priors generici
            if ev_name.startswith('_') or 'prior' in ev_name:
                continue

            # Skip evidenze baseline che non sono specifiche per resistenza
            if ev_name in ['egfr_mutated', 'baseline', 'therapy_duration']:
                continue

            # Una vera evidenza di RESISTENZA deve avere uno di questi pattern:
            # - ctDNA con mutazione specifica (c797s, t790m)
            # - Amplificazione genica confermata (met_cn >= 5)
            # - Trasformazione istologica
            resistance_keywords = ['c797s', 't790m', 'met_amp', 'met_cn', 'her2_amp',
                                   'sclc', 'rb1_loss', 'transformation', 'resistance']

            is_resistance_evidence = any(kw in ev_name for kw in resistance_keywords)

            if is_resistance_evidence:
                # Verifica che abbia un likelihood_ratio significativo (> 5 = impatto reale)
                lr = ev.get('likelihood_ratio', 1.0)
                if lr > 5.0:
                    real_evidence_count += 1

                    # Controlla se supporta specificamente il meccanismo primario
                    if ferrari_mech in mechanisms_requiring_evidence:
                        evidence_keywords = mechanisms_requiring_evidence[ferrari_mech]
                        for keyword in evidence_keywords:
                            if keyword in ev_name:
                                has_patient_evidence = True
                                break

        # Se il meccanismo richiede evidenza e non ne abbiamo, è SEMPRE theoretical
        if ferrari_mech in mechanisms_requiring_evidence and not has_patient_evidence:
            pdf.cell(0, 5, safe_text(f"    Primary Risk (theoretical): {ferrari_mech}"), 0, 1)
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 4, safe_text(f"    (Population prior - no patient-specific evidence detected)"), 0, 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)
        else:
            pdf.cell(0, 5, safe_text(f"    Primary Mechanism: {ferrari_mech}"), 0, 1)

        pdf.cell(0, 5, safe_text(f"    Confidence: {ferrari_conf}"), 0, 1)

        # === OVERALL RESISTANCE RISK ===
        if overall_risk > 0:
            pdf.ln(2)
            if overall_risk >= 80:
                pdf.set_fill_color(255, 200, 200)
                risk_label = "CRITICAL"
                pdf.set_text_color(180, 0, 0)
            elif overall_risk >= 60:
                pdf.set_fill_color(255, 230, 200)
                risk_label = "HIGH"
                pdf.set_text_color(200, 100, 0)
            else:
                pdf.set_fill_color(255, 255, 200)
                risk_label = "MODERATE"
                pdf.set_text_color(150, 150, 0)

            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 7, safe_text(f"OVERALL RESISTANCE RISK: {overall_risk:.0f}% [{risk_label}]"), 0, 1, 'L', 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)

            # FIX: Mostra "priors-only" se non ci sono evidenze reali di resistenza
            if real_evidence_count > 0:
                pdf.cell(0, 5, safe_text(
                    f"    Probability of ANY resistance mechanism being active ({real_evidence_count} molecular signals detected)"),
                         0, 1)
            else:
                pdf.cell(0, 5, safe_text(
                    "    Resistance risk based on population priors (no patient-specific resistance signals detected)"),
                         0, 1)

        # === INDIVIDUAL MECHANISM PROBABILITIES ===
        # Rinomina sezione se sono solo priors
        pdf.set_font('Arial', 'B', 9)
        if real_evidence_count > 0:
            pdf.cell(0, 5, safe_text("    Individual Mechanism Probabilities:"), 0, 1)
        else:
            pdf.cell(0, 5, safe_text("    Top Prior Mechanisms (population-based, not detected in patient):"), 0, 1)
        pdf.set_font('Arial', '', 9)

        displayed_count = 0
        for ev in ferrari_breakdown[:6]:
            evidence = ev.get('evidence', 'Unknown')

            # Salta il marcatore _OVERALL_RISK
            if evidence == '_OVERALL_RISK':
                continue

            # Salta AlphaMissense dalla sezione resistance (è driver pathogenicity, non resistance)
            if 'alphamissense' in evidence.lower() or 'pathogenicity' in evidence.lower():
                continue

            prob = ev.get('probability', 0)

            if prob >= 50:
                label = "[LIKELY]"
                pdf.set_text_color(200, 0, 0)
            elif prob >= 20:
                label = "[Possible]"
                pdf.set_text_color(255, 140, 0)
            else:
                label = "[Low]"
                pdf.set_text_color(100, 100, 100)

            pdf.cell(0, 4, safe_text(f"      - {evidence}: {prob}% {label}"), 0, 1)
            pdf.set_text_color(0)
            displayed_count += 1

        # Se nessun meccanismo mostrato (tutti filtrati)
        if displayed_count == 0:
            pdf.set_text_color(100, 100, 100)
            pdf.set_font('Arial', 'I', 8)
            pdf.cell(0, 4, safe_text("      No resistance mechanisms detected in this patient"), 0, 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)

        # Nota esplicativa
        pdf.set_font('Arial', 'I', 7)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, safe_text("      * Probabilities are independent - mechanisms may co-occur"), 0, 1)
        pdf.set_text_color(0)
        pdf.ln(2)

        # === ALPHAMISSENSE PATHOGENICITY ===
        alphamissense_data = next((e for e in ferrari_breakdown if e.get('evidence') == 'AlphaMissense_Pathogenicity'),
                                  None)
        if alphamissense_data:
            pdf.ln(2)
            pdf.set_fill_color(230, 245, 255)
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(0, 100, 150)

            boost_pct = alphamissense_data.get('probability', 0)
            risk_level = alphamissense_data.get('risk_level', 'MEDIUM')
            pdf.cell(0, 6, safe_text(f"    [DeepMind] AlphaMissense Pathogenicity: +{boost_pct:.0f}% [{risk_level}]"),
                     0, 1, 'L', 1)

            # Show individual mutations
            details = alphamissense_data.get('details', [])
            if details:
                pdf.set_font('Arial', '', 8)
                pdf.set_text_color(0, 80, 120)
                for mut in details:
                    gene = mut.get('gene', '')
                    variant = mut.get('variant', '')
                    pathogenicity = mut.get('pathogenicity', 0)
                    am_class = mut.get('class', '')

                    if am_class == 'likely_pathogenic':
                        emoji = "●"  # Red dot
                        pdf.set_text_color(180, 0, 0)
                    elif am_class == 'likely_benign':
                        emoji = "○"  # Green
                        pdf.set_text_color(0, 150, 0)
                    else:
                        emoji = "◐"  # Yellow/ambiguous
                        pdf.set_text_color(200, 150, 0)

                    pdf.cell(0, 4, safe_text(
                        f"      {emoji} {gene} {variant}: {pathogenicity:.3f} ({am_class.replace('_', ' ')})"), 0, 1)

                pdf.set_text_color(0)

            pdf.set_font('Arial', 'I', 7)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text("      Source: DeepMind AlphaMissense (71M variants)"), 0, 1)
            pdf.set_text_color(0)
            pdf.ln(2)

    # === SYNERGIES ===
    if synergies:
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 6, safe_text("[!] SYNERGIES DETECTED:"), 0, 1)
        pdf.set_font('Arial', '', 9)
        for syn in synergies:
            pdf.cell(0, 5, safe_text(f"    * {syn.get('pair', '')}: {syn.get('effect', '')}"), 0, 1)
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(100)
            pdf.cell(0, 4, safe_text(f"      Clinical: {syn.get('clinical_impact', '')}"), 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.set_text_color(200, 0, 0)
        pdf.set_text_color(0)
    pdf.ln(3)


def draw_veto_section(pdf, ai_result):
    """Sezione VETO"""
    veto_reason = ai_result.get('veto_reason', 'Therapy incompatibility')
    veto_recommendation = ai_result.get('veto_recommendation', 'Review therapy')

    pdf.set_fill_color(220, 20, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, safe_text("  [!] THERAPY VETO ACTIVE"), 1, 1, 'L', 1)

    pdf.set_fill_color(255, 240, 240)
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, safe_text(f"  Reason: {veto_reason}"), 1, 1, 'L', 1)

    pdf.set_font('Arial', '', 9)
    for line in veto_recommendation.split('\n'):
        if line.strip():
            pdf.cell(0, 6, safe_text(f"  {line.strip()}"), 1, 1, 'L')
    pdf.ln(3)


def draw_patient_journey(pdf, baseline_data, visits):
    """
    Sezione Patient Journey (Longitudinal Analysis)
    Include: Therapy Timeline, LDH Evolution, Response Summary, Visit History Table
    """
    pdf.add_page()

    # Header
    pdf.set_fill_color(255, 255, 200)  # Giallo chiaro come nell'immagine
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, safe_text("  6. PATIENT JOURNEY (Longitudinal Analysis)"), 0, 1, 'L', 1)
    pdf.ln(5)

    # === THERAPY TIMELINE ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, safe_text("THERAPY TIMELINE:"), 0, 1)

    therapy_baseline = baseline_data.get('current_therapy', 'Unknown')

    # Box per timeline
    pdf.set_fill_color(245, 245, 245)
    y_start = pdf.get_y()
    pdf.rect(10, y_start, 190, 20, 'F')

    pdf.set_xy(12, y_start + 2)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, safe_text(f"Started: {therapy_baseline}"), 0, 1)

    # Terapia attuale
    if visits:
        current_therapy = therapy_baseline
        for v in visits:
            # Check multiple possible locations for therapy change
            therapy_changed = False
            new_ther = None

            # Check therapy_info dict (nuovo formato)
            if v.get('therapy_info'):
                ti = v['therapy_info']
                if ti.get('therapy_changed') and ti.get('new_therapy'):
                    therapy_changed = True
                    new_ther = ti['new_therapy']

            # Check legacy format
            if not therapy_changed and v.get('therapy_changed') and v.get('new_therapy'):
                therapy_changed = True
                new_ther = v['new_therapy']

            if therapy_changed and new_ther:
                current_therapy = new_ther

        if current_therapy != therapy_baseline:
            pdf.set_x(12)
            pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 6, safe_text(f"Current: {current_therapy} [SWITCHED]"), 0, 1)
        else:
            pdf.set_x(12)
            pdf.set_text_color(0, 100, 0)
            pdf.cell(0, 6, safe_text(f"Current: {current_therapy} [MAINTAINED]"), 0, 1)

    pdf.set_text_color(0)
    pdf.ln(8)

    # === LDH EVOLUTION ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, safe_text("LDH EVOLUTION:"), 0, 1)

    ldh_baseline = float((baseline_data.get('blood_markers') or {}).get('ldh', 0))
    # Fallback: se baseline LDH è 0, usa prima visita
    if ldh_baseline == 0 and visits:
        ldh_baseline = float((visits[0].get('blood_markers') or {}).get('ldh', 0))
    if visits:
        ldh_current = float((visits[-1].get('blood_markers') or {}).get('ldh', ldh_baseline))
    else:
        ldh_current = ldh_baseline

    pdf.set_font('Arial', '', 10)
    pdf.cell(60, 6, safe_text(f"Baseline: {ldh_baseline:.0f} U/L"), 0, 0)
    pdf.cell(20, 6, safe_text("-->"), 0, 0, 'C')

    # Colore basato su trend
    if ldh_baseline > 0:
        ldh_pct = ((ldh_current - ldh_baseline) / ldh_baseline) * 100
        if ldh_pct < -10:
            pdf.set_text_color(0, 128, 0)  # Verde
            trend_symbol = "v FALLING"
        elif ldh_pct > 10:
            pdf.set_text_color(200, 0, 0)  # Rosso
            trend_symbol = "^ RISING"
        else:
            pdf.set_text_color(0, 0, 200)  # Blu
            trend_symbol = "? STABLE"
    else:
        ldh_pct = 0
        trend_symbol = "? N/A"

    pdf.cell(60, 6, safe_text(f"Current: {ldh_current:.0f} U/L"), 0, 1)

    pdf.set_x(12)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, safe_text(f"? Change: {ldh_pct:+.1f}%"), 0, 1)
    pdf.set_text_color(0)
    pdf.ln(5)

    # === RESPONSE SUMMARY TABLE ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, safe_text("RESPONSE SUMMARY:"), 0, 1)

    # Determina response trajectory
    responses = []
    for v in visits:
        if v.get('imaging') and (v.get('imaging') or {}).get('response'):
            responses.append(v['imaging']['response'])

    if responses:
        last_response = responses[-1]
        if last_response == 'CR':
            response_text = "Complete Response Achieved"
        elif last_response == 'PR':
            response_text = "Partial Response Maintained"
        elif last_response == 'SD':
            response_text = "Stable Disease"
        elif last_response == 'PD':
            response_text = "Progressive Disease"
        else:
            response_text = "Not Evaluable"
    else:
        response_text = "No imaging data"

    weeks_on_therapy = visits[-1].get('week_on_therapy', 0) if visits else 0

    # Tabella summary
    pdf.set_font('Arial', '', 10)
    pdf.set_fill_color(240, 240, 240)

    # Riga 1
    pdf.cell(95, 7, safe_text(f"  Total Visits: {len(visits)}"), 1, 0, 'L', 1)
    pdf.cell(95, 7, safe_text(f"  Weeks on Therapy: {weeks_on_therapy}"), 1, 1, 'L', 1)

    # Riga 2
    pdf.cell(95, 7, safe_text(f"  LDH Trend: {trend_symbol}"), 1, 0, 'L', 1)
    pdf.cell(95, 7, safe_text(f"  Response: {response_text}"), 1, 1, 'L', 1)

    pdf.ln(8)

    # === VISIT HISTORY TABLE ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, safe_text("VISIT HISTORY:"), 0, 1)

    if visits:
        # Header tabella
        pdf.set_font('Arial', 'B', 9)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(18, 7, "Visit", 1, 0, 'C', 1)
        pdf.cell(28, 7, "Date", 1, 0, 'C', 1)
        pdf.cell(18, 7, "Week", 1, 0, 'C', 1)
        pdf.cell(22, 7, "LDH", 1, 0, 'C', 1)
        pdf.cell(22, 7, "RECIST", 1, 0, 'C', 1)
        pdf.cell(82, 7, "Key Findings", 1, 1, 'C', 1)

        # Righe visite (ultime 6)
        pdf.set_font('Arial', '', 9)
        pdf.set_fill_color(255, 255, 255)

        for v in visits[-6:]:  # Max 6 visite
            visit_id = v.get('visit_id', '?')
            date = v.get('date', 'N/A')
            # Week: prova week_on_therapy, poi week_offset, poi calcola dalla data
            week = v.get('week_on_therapy')
            if week is None or week == '?':
                week = v.get('week_offset', '?')
            ldh = (v.get('blood_markers') or {}).get('ldh', 'N/A')

            # RECIST
            if v.get('imaging') and (v.get('imaging') or {}).get('response'):
                recist = v['imaging']['response']
            else:
                recist = 'NE'

            # Key findings - migliorato
            findings = []

            # Therapy change
            therapy_changed = False
            new_ther = None
            if v.get('therapy_info') and v['therapy_info'].get('therapy_changed'):
                therapy_changed = True
                new_ther = v['therapy_info'].get('new_therapy', '')
            elif v.get('therapy_changed') and v.get('new_therapy'):
                therapy_changed = True
                new_ther = v.get('new_therapy', '')

            if therapy_changed and new_ther:
                # Abbrevia nome terapia
                short_therapy = new_ther[:20] + "..." if len(new_ther) > 20 else new_ther
                findings.append(f"-> {short_therapy}")

            # Mutazioni
            if v.get('genetics'):
                gen = v['genetics']
                muts = gen.get('new_mutations', [])
                if muts:
                    # Abbrevia mutazioni
                    mut_str = ", ".join([m[:15] for m in muts[:2]])
                    findings.append(f"Mut: {mut_str}")
                elif gen.get('met_cn') and float(gen.get('met_cn', 0)) > 5:
                    findings.append(f"MET CN={gen['met_cn']}")

            # Nuove lesioni
            if (v.get('imaging') or {}).get('new_lesions'):
                sites = (v.get('imaging') or {}).get('new_lesion_sites', [])
                if sites:
                    findings.append(f"New: {', '.join(sites[:2])}")

            key_findings = "; ".join(findings) if findings else "-"

            pdf.cell(18, 6, safe_text(str(visit_id)), 1, 0, 'C')
            pdf.cell(28, 6, safe_text(str(date)), 1, 0, 'C')
            pdf.cell(18, 6, safe_text(str(week)), 1, 0, 'C')
            pdf.cell(22, 6, safe_text(str(ldh)), 1, 0, 'C')
            pdf.cell(22, 6, safe_text(str(recist)), 1, 0, 'C')
            pdf.cell(82, 6, safe_text(str(key_findings)[:40]), 1, 1, 'L')
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 6, safe_text("No follow-up visits recorded"), 0, 1)

    pdf.ln(8)

    # === TIMELINE RECOMMENDATIONS ===
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, safe_text("TIMELINE RECOMMENDATIONS:"), 0, 1)

    pdf.set_font('Arial', '', 10)

    # Genera raccomandazioni basate sui dati
    recommendations = []

    if ldh_baseline > 0 and ldh_pct > 20:
        recommendations.append("Consider ctDNA for resistance mutations")
    if ldh_baseline > 0 and ldh_pct < -30:
        recommendations.append("Excellent metabolic response - continue current therapy")
    if not recommendations:
        recommendations.append("Continue current therapy with standard monitoring")

    for rec in recommendations:
        pdf.cell(0, 6, safe_text(f"  * {rec}"), 0, 1)


def draw_longitudinal_analysis(pdf, baseline_data, visits):
    """Pagina finale con grafici longitudinali + Patient Journey"""

    # Prima pagina: Patient Journey con tabella
    draw_patient_journey(pdf, baseline_data, visits)

    # Seconda pagina: Grafici longitudinali
    pdf.add_page()

    # Header speciale
    pdf.set_fill_color(144, 238, 144)
    pdf.set_text_color(0)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, safe_text("  LONGITUDINAL ANALYSIS"), 0, 1, 'L', 1)
    pdf.ln(10)

    # Grafico
    try:
        graph_path = generate_longitudinal_chart(baseline_data, visits)
        if graph_path:
            pdf.image(graph_path, x=10, y=pdf.get_y(), w=190)
            pdf.ln(85)
            os.unlink(graph_path)
    except Exception as e:
        print(f"⚠️ Longitudinal chart error: {e}")

    # Therapy Timeline
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("THERAPY TIMELINE:"), 0, 1)
    pdf.set_font('Arial', '', 10)

    therapy_baseline = baseline_data.get('current_therapy', 'Unknown')
    pdf.cell(0, 6, safe_text(f"  W0 (Baseline): {therapy_baseline}"), 0, 1)

    for v in visits:
        if v.get('therapy_changed'):
            week = v.get('week_on_therapy') or v.get('week_offset', '?')
            new_ther = v.get('new_therapy', v.get('current_therapy', 'Unknown'))
            pdf.set_text_color(255, 140, 0)
            pdf.cell(0, 6, safe_text(f"  W{week}: SWITCHED to {new_ther}"), 0, 1)
            pdf.set_text_color(0)

    pdf.ln(5)

    # Summary Statistics
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("SUMMARY STATISTICS:"), 0, 1)
    pdf.set_font('Arial', '', 10)

    ldh_baseline = float((baseline_data.get('blood_markers') or {}).get('ldh', 0))
    # Fallback: se baseline LDH è 0, usa prima visita
    if ldh_baseline == 0 and visits:
        ldh_baseline = float((visits[0].get('blood_markers') or {}).get('ldh', 0))
    if visits:
        ldh_current = float((visits[-1].get('blood_markers') or {}).get('ldh', ldh_baseline))
        weeks = visits[-1].get('week_on_therapy', 0)
    else:
        ldh_current = ldh_baseline
        weeks = 0

    pdf.cell(0, 6, safe_text(f"  Total visits: {len(visits)}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"  Weeks on therapy: {weeks}"), 0, 1)

    if ldh_baseline > 0:
        ldh_pct = ((ldh_current - ldh_baseline) / ldh_baseline) * 100
        pdf.cell(0, 6, safe_text(f"  LDH change: {ldh_baseline:.0f} -> {ldh_current:.0f} ({ldh_pct:+.1f}%)"), 0, 1)


        # === DEFINE patient_data FOR ALL SUBSEQUENT CALLS ===
    patient_data = {'baseline': baseline_data, 'visits': visits}
        # === CLONAL EVOLUTION (dopo Longitudinal) ===
    clonal_result = None
    if CLONAL_TRACKER_AVAILABLE and visits and len(visits) > 0:
        try:
            clonal_result = analyze_clonal_evolution({'baseline': baseline_data}, visits)
            draw_clonal_evolution(pdf, {'baseline': baseline_data}, visits)
        except Exception as e:
            print(f"⚠️ Clonal Evolution error: {e}")
            clonal_result = None

    # === CHRONOS CHART (dopo Clonal Evolution) ===
    if CHRONOS_AVAILABLE and clonal_result is not None and visits and len(visits) > 0:
        try:
            output_dir = str(OUTPUT_DIR) if 'OUTPUT_DIR' in dir() else '/tmp'
            draw_chronos_chart(pdf, {'baseline': baseline_data}, visits, clonal_result, output_dir)
        except Exception as e:
            print(f"⚠️ CHRONOS error: {e}")

    # === PROMETHEUS: STRATEGIC ACTION PLAN (Dopo Chronos, prima di Adaptive) ===
    if PROMETHEUS_AVAILABLE:
        draw_prometheus_plan(pdf, patient_data)

    # === ADAPTIVE THERAPY (dopo Clonal Evolution) ===
    if ADAPTIVE_THERAPY_AVAILABLE and visits:
        draw_adaptive_therapy(pdf, {'baseline': baseline_data}, visits, {})

    # === SYNTHETIC LETHALITY (dopo Adaptive Therapy) ===
    if SYNTHETIC_LETHALITY_AVAILABLE:
        draw_synthetic_lethality(pdf, {'baseline': baseline_data})


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def draw_prometheus_plan(pdf, patient_data: Dict):
    """Disegna la singola pagina PROMETHEUS: STRATEGIC ACTION PLAN"""
    if not PROMETHEUS_AVAILABLE:
        return
        
    try:
        from prometheus.feature_engineering import extract_patient_features
        from prometheus.risk_engine import compute_risk_score
        
        # Le features sono estratte dal current dict "patient_data"
        features = extract_patient_features(patient_data)
        result = compute_risk_score(features, patient_data)
    except Exception as e:
        print(f"⚠️ PROMETHEUS error during PDF generation: {e}")
        return
        
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(74, 20, 140)  # Dark purple
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, safe_text("  PROMETHEUS: STRATEGIC ACTION PLAN"), 0, 1, 'L', 1)
    pdf.set_text_color(0)
    pdf.ln(5)
    
    # Overall Risk Score
    score = result.get('risk_score', 0)
    cluster = result.get('cluster', {})
    cluster_name = cluster.get('name', 'Unknown')
    monitoring = cluster.get('monitoring', 'Standard')
    
    pdf.set_fill_color(240, 240, 245)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, safe_text(f"  Overall Risk Score: {score}/100 - {cluster_name}"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, safe_text(f"  Recommended Monitoring: {monitoring}"), 0, 1)
    pdf.ln(5)
    
    # Risk Breakdown Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, safe_text("Risk Breakdown:"), 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.set_fill_color(245, 245, 245)
    
    cat_scores = result.get('category_scores', {})
    for cat, val in cat_scores.items():
        if val > 0:
            pdf.cell(70, 7, safe_text(f"  {cat.upper()}:"), 0, 0, 'L', 1)
            pdf.cell(0, 7, safe_text(f"{val} pts"), 0, 1, 'L', 1)
    pdf.ln(5)
    
    # Piano di Correzione Terapeutica
    corrections = result.get('corrections', [])
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, safe_text("Piano di Correzione Terapeutica:"), 0, 1)
    
    if not corrections:
        pdf.set_font('Arial', 'I', 11)
        pdf.cell(0, 8, safe_text("  Nessun intervento correttivo necessario al momento."), 0, 1)
    else:
        for corr in corrections:
            pri = corr.get('priority', 'BASSA')
            # Colore basato sulla priorità
            if pri == "ALTA":
                icon = "[!!!] ALTA"
                pdf.set_text_color(200, 0, 0)
            elif pri == "MEDIA":
                icon = "[!!] MEDIA"
                pdf.set_text_color(200, 100, 0)
            else:
                icon = "[!] BASSA"
                pdf.set_text_color(0, 128, 0)
                
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(40, 7, safe_text(f"  {icon}:"), 0, 0)
            pdf.set_text_color(0)
            pdf.cell(0, 7, safe_text(f"{corr.get('target', '')} -> {corr.get('action', '')}"), 0, 1)
            
    pdf.ln(5)

def _draw_elephant_protocol_v2(pdf, elephant_result: 'ElephantProtocolResult', ai_result: Dict):
    """Disegna il Protocollo Elephant v2.0 personalizzato"""

    # Header
    pdf.set_fill_color(255, 69, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, safe_text("[!] PROTOCOL ELEPHANT ACTIVATED"), 1, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.ln(5)

    # Rationale
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("RATIONALE:"), 0, 1)
    pdf.set_font('Arial', '', 10)
    for reason in elephant_result.activation_reasons:
        pdf.cell(0, 5, safe_text(f"  - {reason}"), 0, 1)
    pdf.ln(3)

    # Quantitative Projections Box
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("  QUANTITATIVE PROJECTIONS"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 10)

    sens = elephant_result.metabolic_sensitivity
    if sens >= 70:
        sens_label = "HIGH (Excellent candidate)"
        pdf.set_text_color(0, 128, 0)
    elif sens >= 40:
        sens_label = "MODERATE (Good candidate)"
        pdf.set_text_color(255, 140, 0)
    else:
        sens_label = "LOW (Consider alternatives)"
        pdf.set_text_color(200, 0, 0)

    pdf.cell(0, 6, safe_text(f"  Metabolic Sensitivity: {sens:.0f}% - {sens_label}"), 0, 1)
    pdf.set_text_color(0)

    # Proiezioni tabella
    base_regression = sens * 0.42
    pdf.set_font('Arial', '', 9)

    # Header tabella
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(50, 6, "Phase", 1, 0, 'C', 1)
    pdf.cell(30, 6, "Duration", 1, 0, 'C', 1)
    pdf.cell(55, 6, "Expected Regression", 1, 0, 'C', 1)
    pdf.cell(55, 6, "Cumulative", 1, 1, 'C', 1)

    # Righe
    pdf.set_font('Arial', '', 9)
    pdf.set_fill_color(255, 255, 255)

    p1_min, p1_max = base_regression * 0.5, base_regression
    pdf.cell(50, 5, "PHASE 1: INDUCTION", 1, 0, 'L')
    pdf.cell(30, 5, "4-6 weeks", 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{p1_min:.0f}% to -{p1_max:.0f}%"), 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{p1_min:.0f}% to -{p1_max:.0f}%"), 1, 1, 'C')

    p2_min, p2_max = base_regression * 0.2, base_regression * 0.5
    c2_min, c2_max = p1_min + p2_min, p1_max + p2_max
    pdf.cell(50, 5, "PHASE 2: CONSOLIDATION", 1, 0, 'L')
    pdf.cell(30, 5, "8-12 weeks", 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{p2_min:.0f}% to -{p2_max:.0f}%"), 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{c2_min:.0f}% to -{c2_max:.0f}%"), 1, 1, 'C')

    p3_min, p3_max = 0, base_regression * 0.1
    c3_min, c3_max = c2_min + p3_min, c2_max + p3_max
    pdf.cell(50, 5, "PHASE 3: MAINTENANCE", 1, 0, 'L')
    pdf.cell(30, 5, "Indefinite", 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{p3_min:.0f}% to -{p3_max:.0f}%"), 1, 0, 'C')
    pdf.cell(55, 5, safe_text(f"-{c3_min:.0f}% to -{c3_max:.0f}%"), 1, 1, 'C')

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(50, 6, "TOTAL:", 1, 0, 'L', 1)
    pdf.cell(30, 6, "", 1, 0, 'C', 1)
    pdf.cell(55, 6, "", 1, 0, 'C', 1)
    pdf.cell(55, 6, safe_text(f"-{c3_min:.0f}% to -{c3_max:.0f}%"), 1, 1, 'C', 1)
    pdf.ln(5)

    # INTERVENTION STRATEGY
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("  INTERVENTION STRATEGY"), 0, 1, 'L', 1)

    # VETO note se attivo
    if ai_result.get('veto_active'):
        pdf.set_text_color(200, 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text(f"  [!] VETO: {ai_result.get('veto_reason', '')}"), 0, 1)
        pdf.set_text_color(0)

    # Disegna ogni fase
    for phase in elephant_result.phases:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 6, safe_text(phase.name), 0, 1)
        pdf.set_text_color(0)

        pdf.set_font('Arial', '', 9)
        for therapy in phase.therapies:
            # Icona evidence
            if therapy.evidence.value in ["FDA Approved", "EMA Approved"]:
                ev_icon = "[OK]"
                pdf.set_text_color(0, 128, 0)
            elif "Phase" in therapy.evidence.value:
                ev_icon = "[EXP]"
                pdf.set_text_color(255, 140, 0)
            else:
                ev_icon = "[?]"
                pdf.set_text_color(100, 100, 100)

            pdf.cell(0, 5, safe_text(f"  - {therapy.drug_name} {ev_icon}: {therapy.rationale}"), 0, 1)
            pdf.set_text_color(0)

            if therapy.dose and therapy.dose != therapy.drug_name:
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 4, safe_text(f"    Dose: {therapy.dose}"), 0, 1)
                pdf.set_font('Arial', '', 9)
                pdf.set_text_color(0)

        # Excluded therapies
        if phase.excluded_therapies:
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(150, 0, 0)
            for drug, reason in phase.excluded_therapies:
                pdf.cell(0, 4, safe_text(f"  [EXCLUDED] {drug}: {reason}"), 0, 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)

        # Warnings
        if phase.warnings:
            pdf.set_font('Arial', 'I', 8)
            pdf.set_text_color(255, 140, 0)
            for warning in phase.warnings:
                pdf.cell(0, 4, safe_text(f"  [!] {warning}"), 0, 1)
            pdf.set_text_color(0)
            pdf.set_font('Arial', '', 9)

    # Experimental Options (se presenti)
    if elephant_result.experimental_options:
        pdf.ln(3)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(128, 0, 128)
        pdf.cell(0, 6, safe_text("EXPERIMENTAL OPTIONS (Clinical Trials):"), 0, 1)
        pdf.set_font('Arial', '', 9)
        for exp in elephant_result.experimental_options[:3]:  # Max 3
            pdf.cell(0, 5, safe_text(f"  - {exp.drug_name}: {exp.rationale}"), 0, 1)
        pdf.set_text_color(0)

    # Personalization Summary
    pdf.ln(5)
    pdf.set_fill_color(255, 255, 224)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, safe_text("  PERSONALIZED RECOMMENDATION"), 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 9)

    summary_lines = elephant_result.personalization_summary.split('\n')
    for line in summary_lines:
        if line.strip():
            pdf.cell(0, 5, safe_text(f"  {line}"), 0, 1)

    # Disclaimer finale
    pdf.ln(5)
    pdf.set_draw_color(200, 0, 0)
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 35)
    pdf.set_line_width(0.2)

    pdf.set_font('Arial', 'B', 9)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(0, 5, safe_text("  IMPORTANT NOTICE"), 0, 1)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 4, safe_text(
        "  This Elephant Protocol is suggested based on the patient's metabolic profile, genetic alterations, "
        "and clinical status as analyzed by SENTINEL AI. Implementation requires Multidisciplinary Tumor Board "
        "approval. The final therapeutic decision must be made by the treating oncologist in consultation with "
        "the patient, considering individual circumstances, comorbidities, and patient preferences."
    ))
    pdf.set_text_color(0)


def _get_elephant_triad(base: Dict, ai_result: Dict) -> Dict:
    """
    Genera la Triade Elephant Completa basata sul profilo del paziente.

    Le 3 Armi per trasformare un Uomo in un Elefante:
    1. METABOLIC ATTACK (Warburg Shutdown) - Metformin + Keto
    2. DNA REPAIR COLLAPSE (Synthetic Lethality) - PARP Inhibitors
    3. APOPTOSIS BOOST (p53 Reactivation) - MDM2 Inhibitors

    Returns:
        Dict con le 3 armi e relative indicazioni/controindicazioni
    """
    genetics = base.get('genetics', {})
    blood = base.get('blood_markers', {})

    ecog = int(base.get('ecog_ps', 1))
    age = int(base.get('age', 65))
    ldh = float(blood.get('ldh', 200) or 200)
    albumin = float(blood.get('albumin', 4.0) or 4.0)

    # Status genetici
    tp53 = str(genetics.get('tp53_status', '')).lower()
    tp53_mutated = tp53 in ['mutated', 'mut', 'loss']
    tp53_wildtype = tp53 in ['wt', 'wild-type', 'wildtype', 'none', '']

    brca1 = str(genetics.get('brca1_status', '')).lower()
    brca2 = str(genetics.get('brca2_status', '')).lower()
    brca_mutated = brca1 in ['mutated', 'mut'] or brca2 in ['mutated', 'mut']

    # HRD status (Homologous Recombination Deficiency)
    hrd = str(genetics.get('hrd_status', '')).lower()
    hrd_positive = hrd in ['positive', 'high', 'deficient']

    # ATM/ATR status
    atm = str(genetics.get('atm_status', '')).lower()
    atm_mutated = atm in ['mutated', 'mut', 'loss']

    # RB1 per rischio SCLC transformation
    rb1 = str(genetics.get('rb1_status', '')).lower()
    rb1_mutated = rb1 in ['mutated', 'mut', 'loss']

    triad = {
        "metabolic_attack": {
            "name": "METABOLIC ATTACK (Warburg Shutdown)",
            "icon": "⚡",
            "indicated": True,
            "therapies": [],
            "rationale": "",
            "warnings": []
        },
        "dna_repair_collapse": {
            "name": "DNA REPAIR COLLAPSE (Synthetic Lethality)",
            "icon": "🧬",
            "indicated": False,
            "therapies": [],
            "rationale": "",
            "contraindication": "",
            "warnings": []
        },
        "apoptosis_boost": {
            "name": "APOPTOSIS BOOST (p53 Reactivation)",
            "icon": "💀",
            "indicated": False,
            "therapies": [],
            "rationale": "",
            "contraindication": "",
            "warnings": []
        }
    }

    # =========================================================================
    # ARMA 1: METABOLIC ATTACK (Sempre indicata se LDH > 350)
    # =========================================================================

    triad["metabolic_attack"]["indicated"] = ldh > 350
    triad["metabolic_attack"]["rationale"] = f"LDH {ldh:.0f} U/L indicates Warburg effect - tumor is glucose-dependent"

    # Metformina
    if ecog >= 3:
        triad["metabolic_attack"]["therapies"].append({
            "drug": "Metformin (dose-reduced)",
            "dose": "500mg QD, max 500mg BID",
            "mechanism": "Mitochondrial Complex I inhibition - forces oxidative stress",
            "evidence": "Phase 2",
            "note": "Reduced dose due to ECOG ≥3"
        })
    else:
        triad["metabolic_attack"]["therapies"].append({
            "drug": "Metformin",
            "dose": "500mg BID, titrate to 1000mg BID over 2 weeks",
            "mechanism": "Mitochondrial Complex I inhibition - forces oxidative stress",
            "evidence": "Phase 2"
        })

    # Dieta Ketogenica
    if albumin >= 3.0 and ecog <= 2:
        triad["metabolic_attack"]["therapies"].append({
            "drug": "Ketogenic Diet",
            "dose": "<50g carbs/day, supervised by dietitian",
            "mechanism": "Glucose deprivation - exploits metabolic inflexibility of cancer cells",
            "evidence": "Phase 2"
        })
    else:
        triad["metabolic_attack"]["warnings"].append(
            f"Ketogenic diet contraindicated (Albumin {albumin:.1f}, ECOG {ecog})"
        )

    # 2-DG se disponibile
    if ldh > 500:
        triad["metabolic_attack"]["therapies"].append({
            "drug": "2-Deoxyglucose (2-DG)",
            "dose": "45 mg/kg with radiation (if in trial)",
            "mechanism": "Glycolysis inhibitor - blocks glucose utilization",
            "evidence": "Phase 1 - Experimental",
            "note": "Requires clinical trial enrollment"
        })

    # =========================================================================
    # ARMA 2: DNA REPAIR COLLAPSE (PARP Inhibitors)
    # =========================================================================

    # Indicazioni per PARP inhibitors:
    # - BRCA1/2 mutato (letalità sintetica classica)
    # - HRD+ (BRCAness)
    # - TP53 mutato (instabilità genomica)
    # - ATM mutato (difetto riparazione DNA)

    parp_score = 0
    parp_reasons = []

    if brca_mutated:
        parp_score += 3
        parp_reasons.append("BRCA1/2 mutation - classic synthetic lethality")

    if hrd_positive:
        parp_score += 3
        parp_reasons.append("HRD positive - BRCAness phenotype")

    if tp53_mutated:
        parp_score += 2
        parp_reasons.append("TP53 mutated - genomic instability, impaired DNA damage response")

    if atm_mutated:
        parp_score += 2
        parp_reasons.append("ATM mutated - defective DNA repair pathway")

    if tp53_mutated and rb1_mutated:
        parp_score += 1
        parp_reasons.append("TP53+RB1 co-mutation - high genomic instability")

    if parp_score >= 2:
        triad["dna_repair_collapse"]["indicated"] = True
        triad["dna_repair_collapse"]["rationale"] = " | ".join(parp_reasons)

        # Scegli il PARP inhibitor appropriato
        if brca_mutated or hrd_positive:
            triad["dna_repair_collapse"]["therapies"].append({
                "drug": "Olaparib",
                "dose": "300mg BID",
                "mechanism": "PARP1/2 inhibition - blocks single-strand break repair, causes replication fork collapse",
                "evidence": "FDA Approved (BRCA+/HRD+)",
                "monitoring": "CBC weekly x4, then monthly. Watch for anemia, neutropenia."
            })
            triad["dna_repair_collapse"]["therapies"].append({
                "drug": "Niraparib (alternative)",
                "dose": "200-300mg QD based on weight/platelets",
                "mechanism": "PARP1/2 inhibition with additional PARP trapping",
                "evidence": "FDA Approved (HRD+)",
                "monitoring": "CBC, BP monitoring (hypertension risk)"
            })
        else:
            # Off-label per TP53/ATM mutated
            triad["dna_repair_collapse"]["therapies"].append({
                "drug": "Olaparib",
                "dose": "300mg BID",
                "mechanism": "PARP inhibition exploits existing DNA repair defects",
                "evidence": "Off-label - Phase 2 evidence for TP53mut tumors",
                "note": "Discuss in Tumor Board - emerging evidence supports use"
            })

        # Talazoparib per casi aggressivi
        if parp_score >= 3:
            triad["dna_repair_collapse"]["therapies"].append({
                "drug": "Talazoparib",
                "dose": "1mg QD",
                "mechanism": "Most potent PARP trapper - superior DNA damage",
                "evidence": "FDA Approved (BRCA+ breast)",
                "note": "Consider for aggressive disease, higher toxicity"
            })

        # Warnings
        if ecog >= 2:
            triad["dna_repair_collapse"]["warnings"].append(
                "ECOG ≥2: Monitor closely for myelosuppression"
            )
        if age > 75:
            triad["dna_repair_collapse"]["warnings"].append(
                f"Age {age}: Consider dose reduction (200mg BID)"
            )
    else:
        triad["dna_repair_collapse"]["indicated"] = False
        triad["dna_repair_collapse"][
            "contraindication"] = "No DNA repair defects detected (BRCA wt, HRD-, TP53 wt). PARP inhibitors unlikely to benefit without synthetic lethality context."

    # =========================================================================
    # ARMA 3: APOPTOSIS BOOST (MDM2 Inhibitors)
    # =========================================================================

    # MDM2 inhibitors ONLY work if TP53 is WILD-TYPE (functional but suppressed)
    # If TP53 is mutated, p53 protein is already non-functional, MDM2i won't help

    if tp53_wildtype and not tp53_mutated:
        triad["apoptosis_boost"]["indicated"] = True
        triad["apoptosis_boost"][
            "rationale"] = "TP53 wild-type - p53 is functional but suppressed by MDM2. Releasing p53 will trigger massive apoptosis in tumor cells."

        triad["apoptosis_boost"]["therapies"].append({
            "drug": "Idasanutlin (RG7388)",
            "dose": "150-300mg QD, days 1-5 of 28-day cycle",
            "mechanism": "MDM2 inhibitor - blocks p53 degradation, reactivates tumor suppression",
            "evidence": "Phase 2 - Clinical trials ongoing",
            "note": "Requires trial enrollment. Check clinicaltrials.gov for open studies."
        })

        triad["apoptosis_boost"]["therapies"].append({
            "drug": "Navtemadlin (AMG-232)",
            "dose": "240mg QD, days 1-7 of 21-day cycle",
            "mechanism": "Potent MDM2 inhibitor - stabilizes p53, induces cell cycle arrest and apoptosis",
            "evidence": "Phase 1/2 - Emerging data in solid tumors",
            "note": "Experimental - monitor for GI toxicity, thrombocytopenia"
        })

        # Milademetan come alternativa
        triad["apoptosis_boost"]["therapies"].append({
            "drug": "Milademetan (DS-3032b)",
            "dose": "120mg QD, days 1-21 of 28-day cycle",
            "mechanism": "MDM2 inhibitor with improved tolerability profile",
            "evidence": "Phase 2",
            "note": "Better tolerated, consider for frail patients"
        })

        # Warnings specifici MDM2i
        triad["apoptosis_boost"]["warnings"].append(
            "MDM2 inhibitors are experimental - bone marrow toxicity is common"
        )
        triad["apoptosis_boost"]["warnings"].append(
            "Monitor: CBC twice weekly during cycle 1, then weekly"
        )
        if ecog >= 2:
            triad["apoptosis_boost"]["warnings"].append(
                "ECOG ≥2: Start with lowest dose, careful toxicity monitoring"
            )
    else:
        triad["apoptosis_boost"]["indicated"] = False
        triad["apoptosis_boost"][
            "contraindication"] = "TP53 mutated - p53 protein is non-functional. MDM2 inhibitors cannot reactivate a broken p53. This arm is NOT indicated."

    # =========================================================================
    # TRIAD SUMMARY
    # =========================================================================

    active_arms = sum([
        triad["metabolic_attack"]["indicated"],
        triad["dna_repair_collapse"]["indicated"],
        triad["apoptosis_boost"]["indicated"]
    ])

    triad["summary"] = {
        "active_arms": active_arms,
        "total_arms": 3,
        "strength": "FULL TRIAD" if active_arms == 3 else (
            "DUAL ATTACK" if active_arms == 2 else (
                "SINGLE ARM" if active_arms == 1 else "METABOLIC ONLY"
            )
        ),
        "prognosis_modifier": f"+{active_arms * 15}% containment probability" if active_arms > 1 else "Standard protocol"
    }

    return triad


def _get_intensified_therapies(base: Dict, ai_result: Dict) -> Dict:
    """
    Genera terapie personalizzate per intensified containment
    basate su profilo genetico, PD-L1, ECOG, resistenze rilevate.

    Returns:
        Dict con:
        - priority_therapies: Lista terapie prioritarie
        - metabolic_agents: Agenti metabolici consigliati
        - immunotherapy: Opzioni immunoterapia (o motivo esclusione)
        - fallback: Opzioni di fallback
        - warnings: Avvertenze specifiche paziente
    """
    genetics = base.get('genetics', {})
    blood = base.get('blood_markers', {})
    biomarkers = base.get('biomarkers', {})

    ecog = int(base.get('ecog_ps', 1))
    age = int(base.get('age', 65))
    histology = base.get('histology', 'Adenocarcinoma').lower()
    current_therapy = base.get('current_therapy', '').lower()

    # PD-L1
    pdl1 = float(biomarkers.get('pdl1_percent', 0) or (base.get('biomarkers') or {}).get('pdl1_percent', 0) or 0)

    # Estrai status genetici
    egfr = str(genetics.get('egfr_status', '')).lower()
    kras = str(genetics.get('kras_mutation', '')).lower()
    alk = str(genetics.get('alk_status', '')).lower()
    met = str(genetics.get('met_status', '')).lower()
    met_cn = float(genetics.get('met_cn', 0) or 0)
    braf = str(genetics.get('braf_status', '')).lower()
    her2 = str(genetics.get('her2_status', '')).lower()
    ros1 = str(genetics.get('ros1_status', '')).lower()
    tp53 = str(genetics.get('tp53_status', '')).lower()
    stk11 = str(genetics.get('stk11_status', '')).lower()
    keap1 = str(genetics.get('keap1_status', '')).lower()

    # VETO info
    veto_active = ai_result.get('veto_active', False)
    veto_reason = ai_result.get('veto_reason', '')

    # Risultati
    priority_therapies = []
    metabolic_agents = []
    immunotherapy_option = None
    immunotherapy_excluded_reason = None
    fallback_options = []
    warnings = []

    # === 1. PRIORITY THERAPIES (basate su VETO e mutazioni) ===

    # Se c'è VETO, la correzione è priorità #1
    if veto_active:
        if 't790m' in veto_reason.lower():
            priority_therapies.append({
                "drug": "Osimertinib",
                "dose": "80mg QD",
                "rationale": "T790M resistance - 3rd gen EGFR-TKI required",
                "evidence": "FDA Approved",
                "priority": 1
            })
        elif 'met' in veto_reason.lower() and ('amplification' in veto_reason.lower() or 'amp' in veto_reason.lower()):
            priority_therapies.append({
                "drug": "Capmatinib",
                "dose": "400mg BID",
                "rationale": f"MET amplification (CN={met_cn}) requires MET inhibitor",
                "evidence": "FDA Approved",
                "priority": 1
            })
        elif 'g12c' in veto_reason.lower():
            priority_therapies.append({
                "drug": "Sotorasib",
                "dose": "960mg QD",
                "rationale": "KRAS G12C specific inhibitor",
                "evidence": "FDA Approved",
                "priority": 1
            })

    # Mutazioni actionable non coperte dal VETO
    if 't790m' in egfr and 'osimertinib' not in current_therapy:
        if not any(t['drug'] == 'Osimertinib' for t in priority_therapies):
            priority_therapies.append({
                "drug": "Osimertinib",
                "dose": "80mg QD",
                "rationale": "EGFR T790M detected",
                "evidence": "FDA Approved",
                "priority": 1
            })

    if 'c797s' in egfr:
        priority_therapies.append({
            "drug": "Amivantamab",
            "dose": "1050mg (<80kg) or 1400mg (>80kg) IV weekly x4, then q2w",
            "rationale": "EGFR C797S resistance - bispecific antibody",
            "evidence": "FDA Approved",
            "priority": 1
        })

    if ('amplification' in met or met_cn >= 5) and not any(
            'capmatinib' in t.get('drug', '').lower() for t in priority_therapies):
        priority_therapies.append({
            "drug": "Capmatinib",
            "dose": "400mg BID",
            "rationale": f"MET amplification (CN={met_cn:.1f})",
            "evidence": "FDA Approved",
            "priority": 1
        })

    if 'g12c' in kras and not any('sotorasib' in t.get('drug', '').lower() for t in priority_therapies):
        priority_therapies.append({
            "drug": "Sotorasib or Adagrasib",
            "dose": "Sotorasib 960mg QD or Adagrasib 600mg BID",
            "rationale": "KRAS G12C mutation",
            "evidence": "FDA Approved",
            "priority": 2
        })

    if alk not in ['wt', 'none', '', 'negative']:
        priority_therapies.append({
            "drug": "Alectinib",
            "dose": "600mg BID",
            "rationale": "ALK rearrangement",
            "evidence": "FDA Approved",
            "priority": 1
        })

    if 'v600' in braf:
        priority_therapies.append({
            "drug": "Dabrafenib + Trametinib",
            "dose": "Dabrafenib 150mg BID + Trametinib 2mg QD",
            "rationale": "BRAF V600E mutation",
            "evidence": "FDA Approved",
            "priority": 1
        })

    if ros1 not in ['wt', 'none', '', 'negative']:
        priority_therapies.append({
            "drug": "Entrectinib or Crizotinib",
            "dose": "Entrectinib 600mg QD",
            "rationale": "ROS1 rearrangement",
            "evidence": "FDA Approved",
            "priority": 1
        })

    if her2 not in ['wt', 'none', '', 'negative']:
        priority_therapies.append({
            "drug": "Trastuzumab Deruxtecan",
            "dose": "5.4mg/kg IV q3w",
            "rationale": "HER2 mutation/amplification",
            "evidence": "FDA Approved",
            "priority": 2
        })

    # === 2. METABOLIC AGENTS ===

    # Metformina (quasi sempre)
    met_contraindicated = False
    if ecog >= 3:
        metabolic_agents.append({
            "drug": "Metformin (dose-reduced)",
            "dose": "500mg QD (max 500mg BID due to ECOG)",
            "rationale": "Mitochondrial stress - reduced dose for frail patient",
            "evidence": "Phase 2"
        })
    else:
        metabolic_agents.append({
            "drug": "Metformin",
            "dose": "500mg BID, titrate to 1000mg BID",
            "rationale": "Mitochondrial Complex I inhibition",
            "evidence": "Phase 2"
        })

    # Dieta Ketogenica
    albumin = float(blood.get('albumin', 4.0) or 4.0)
    if albumin >= 3.0 and ecog <= 2:
        metabolic_agents.append({
            "drug": "Ketogenic Diet",
            "dose": "<50g carbs/day, dietitian supervised",
            "rationale": "Glucose deprivation - exploit Warburg effect",
            "evidence": "Phase 2"
        })
    else:
        warnings.append(f"Ketogenic diet NOT recommended (Albumin {albumin:.1f} g/dL, ECOG {ecog})")

    # Agenti sperimentali per LOW sensitivity
    metabolic_agents.append({
        "drug": "DCA (Dichloroacetate)",
        "dose": "10-15 mg/kg/day (if available in trial)",
        "rationale": "PDK inhibitor - forces oxidative phosphorylation",
        "evidence": "Phase 1 - Experimental"
    })

    # === 3. IMMUNOTHERAPY ===

    # Check controindicazioni
    stk11_mut = stk11 in ['mutated', 'mut', 'loss']
    keap1_mut = keap1 in ['mutated', 'mut', 'loss']

    if stk11_mut and keap1_mut:
        immunotherapy_excluded_reason = "STK11+KEAP1 double loss - checkpoint inhibitors likely ineffective"
    elif stk11_mut:
        immunotherapy_excluded_reason = "STK11 loss - reduced immunotherapy efficacy expected"
    elif pdl1 < 1:
        immunotherapy_excluded_reason = f"PD-L1 <1% ({pdl1:.0f}%) - limited immunotherapy benefit"
    else:
        # Immunoterapia possibile
        if pdl1 >= 50:
            immunotherapy_option = {
                "drug": "Pembrolizumab",
                "dose": "200mg IV q3w",
                "rationale": f"PD-L1 high ({pdl1:.0f}%) - excellent candidate",
                "evidence": "FDA Approved",
                "line": "Consider as combo or sequential"
            }
        elif pdl1 >= 1:
            immunotherapy_option = {
                "drug": "Pembrolizumab + Chemotherapy",
                "dose": "Pembro 200mg + Carboplatin/Pemetrexed",
                "rationale": f"PD-L1 intermediate ({pdl1:.0f}%) - combo preferred",
                "evidence": "FDA Approved",
                "line": "If targeted therapy fails"
            }

    # === 4. FALLBACK OPTIONS ===

    if 'squamous' in histology or 'squamoso' in histology:
        fallback_options.append({
            "drug": "Carboplatin + Gemcitabine",
            "dose": "Standard dosing",
            "rationale": "Platinum doublet for squamous histology",
            "evidence": "NCCN Guideline"
        })
    else:
        fallback_options.append({
            "drug": "Carboplatin + Pemetrexed",
            "dose": "Standard dosing, Pemetrexed maintenance",
            "rationale": "Platinum doublet for non-squamous histology",
            "evidence": "NCCN Guideline"
        })

    fallback_options.append({
        "drug": "Metronomic Chemotherapy",
        "dose": "Low-dose continuous (e.g., oral Vinorelbine)",
        "rationale": "Anti-angiogenic + immunomodulatory, good tolerability",
        "evidence": "Phase 2"
    })

    # Docetaxel come ultima linea
    fallback_options.append({
        "drug": "Docetaxel ± Ramucirumab",
        "dose": "Docetaxel 75mg/m2 q3w",
        "rationale": "Standard 2nd/3rd line option",
        "evidence": "FDA Approved"
    })

    # === 5. WARNINGS ===

    if ecog >= 3:
        warnings.append("ECOG ≥3: Dose reductions required, avoid aggressive combinations")
    if age > 75:
        warnings.append(f"Age {age}: Consider geriatric assessment, watch for toxicity")
    if albumin < 3.0:
        warnings.append(f"Low albumin ({albumin:.1f} g/dL): Malnutrition, poor drug tolerance expected")

    # Sort priority therapies
    priority_therapies.sort(key=lambda x: x.get('priority', 99))

    return {
        "priority_therapies": priority_therapies,
        "metabolic_agents": metabolic_agents,
        "immunotherapy": immunotherapy_option,
        "immunotherapy_excluded": immunotherapy_excluded_reason,
        "fallback": fallback_options,
        "warnings": warnings
    }


def _draw_elephant_protocol_legacy(pdf, base: Dict, ldh: float, tp53_mut: bool, ai_result: Dict):
    """Fallback al protocollo Elephant legacy (statico) con tabella quantitativa"""

    pdf.set_fill_color(255, 69, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, safe_text("[!] PROTOCOL ELEPHANT ACTIVATED"), 1, 1, 'C', 1)
    pdf.set_text_color(0)
    pdf.ln(5)

    # RATIONALE
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "RATIONALE:", 0, 1)
    pdf.set_font('Arial', '', 11)

    if ldh > 350:
        pdf.cell(0, 6, safe_text(f"- High LDH {ldh:.0f} U/L (Warburg Effect)"), 0, 1)
    if tp53_mut:
        pdf.cell(0, 6, safe_text("- TP53 Mutation (Genomic Instability)"), 0, 1)
    pdf.ln(5)

    # === QUANTITATIVE PROJECTIONS ===
    pdf.set_fill_color(240, 248, 255)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, safe_text("  QUANTITATIVE PROJECTIONS"), 0, 1, 'L', 1)
    pdf.ln(2)

    # Calcola Metabolic Sensitivity
    # Calcola Metabolic Sensitivity (Formula Realistica v18.1)
    # Basata su evidenze cliniche per approcci metabolici in NSCLC
    sensitivity = 0
    genetics = base.get('genetics', {})

    # LDH contribuisce max 40% (marker principale Warburg effect)
    if ldh > 350:
        sensitivity += min(40, (ldh - 350) / 15)

    # TP53 mutato: instabilità genomica, può beneficiare da stress metabolico
    if tp53_mut:
        sensitivity += 15

    # KRAS mutato: tumori spesso Warburg-dipendenti
    kras = str(genetics.get('kras_mutation', '')).lower()
    if kras not in ['wt', 'none', '', 'wild-type', 'wildtype']:
        sensitivity += 10

    # STK11 loss: regolatore metabolico, sensibilità aumentata
    stk11 = str(genetics.get('stk11_status', '')).lower()
    if stk11 in ['mutated', 'mut', 'loss']:
        sensitivity += 10

    # KEAP1 loss: stress ossidativo alterato
    keap1 = str(genetics.get('keap1_status', '')).lower()
    if keap1 in ['mutated', 'mut', 'loss']:
        sensitivity += 10

    # Penalità ECOG (paziente fragile = minore tolleranza ai trattamenti)
    ecog = int(base.get('ecog_ps', 1))
    if ecog >= 3:
        sensitivity *= 0.6  # -40% per ECOG 3-4 (paziente molto compromesso)
    elif ecog == 2:
        sensitivity *= 0.8  # -20% per ECOG 2 (paziente moderatamente compromesso)

    sensitivity = min(100, max(0, sensitivity))

    # Label sensibilità
    # Label sensibilità (Filosofia Elephant: MAI arrendersi, adatta la strategia)
    if sensitivity >= 50:
        sens_label = "HIGH (Excellent candidate)"
        sens_strategy = "standard"
        pdf.set_text_color(0, 128, 0)
    elif sensitivity >= 25:
        sens_label = "MODERATE (Good candidate)"
        sens_strategy = "enhanced"
        pdf.set_text_color(255, 140, 0)
    else:
        sens_label = "LOW (Intensify strategy)"
        sens_strategy = "intensive"
        pdf.set_text_color(200, 100, 0)  # Arancione scuro, non rosso

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, safe_text(f"Metabolic Sensitivity: {sensitivity:.0f}% - {sens_label}"), 0, 1)
    pdf.set_text_color(0)

    # === NOTA STRATEGICA PER LOW SENSITIVITY ===
    if sens_strategy == "intensive":
        pdf.ln(2)
        pdf.set_fill_color(255, 248, 220)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 6, safe_text("  [!] INTENSIFIED CONTAINMENT REQUIRED"), 0, 1, 'L', 1)
        pdf.set_font('Arial', '', 8)
        pdf.set_text_color(100, 60, 0)
        pdf.cell(0, 4, safe_text("  Tumor shows metabolic flexibility - standard protocol may be insufficient."), 0, 1)
        pdf.cell(0, 4, safe_text("  Strategy: Multi-target approach + aggressive monitoring + early adaptation."), 0, 1)
        pdf.set_text_color(0)
    elif sens_strategy == "enhanced":
        pdf.ln(2)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, safe_text("  Moderate sensitivity: Consider combination metabolic agents for optimal response."),
                 0, 1)
        pdf.set_text_color(0)

    pdf.ln(3)

    # Calcola proiezioni
    base_regression = sensitivity * 0.42

    # Tabella header
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(50, 6, "Phase", 1, 0, 'C', 1)
    pdf.cell(25, 6, "Duration", 1, 0, 'C', 1)
    pdf.cell(50, 6, "Expected Regression", 1, 0, 'C', 1)
    pdf.cell(50, 6, "Cumulative", 1, 1, 'C', 1)

    # Righe tabella
    pdf.set_font('Arial', '', 8)
    pdf.set_fill_color(255, 255, 255)

    # Phase 1
    p1_min, p1_max = base_regression * 0.5, base_regression
    pdf.cell(50, 5, "PHASE 1: INDUCTION", 1, 0, 'L')
    pdf.cell(25, 5, "4-6 weeks", 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"-{p1_min:.1f}% to -{p1_max:.1f}%"), 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"-{p1_min:.1f}% to -{p1_max:.1f}%"), 1, 1, 'C')

    # Phase 2
    p2_min, p2_max = base_regression * 0.2, base_regression * 0.5
    c2_min, c2_max = p1_min + p2_min, p1_max + p2_max
    pdf.cell(50, 5, "PHASE 2: CONSOLIDATION", 1, 0, 'L')
    pdf.cell(25, 5, "6-12 weeks", 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"-{p2_min:.1f}% to -{p2_max:.1f}%"), 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"-{c2_min:.1f}% to -{c2_max:.1f}%"), 1, 1, 'C')

    # Phase 3
    p3_min, p3_max = 0, base_regression * 0.1
    c3_min, c3_max = c2_min + p3_min, c2_max + p3_max
    pdf.cell(50, 5, "PHASE 3: MAINTENANCE", 1, 0, 'L')
    pdf.cell(25, 5, "Indefinite", 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"{p3_min:.1f}% to -{p3_max:.1f}%"), 1, 0, 'C')
    pdf.cell(50, 5, safe_text(f"-{c3_min:.1f}% to -{c3_max:.1f}%"), 1, 1, 'C')

    # Total row
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(230, 230, 250)
    pdf.cell(50, 6, "TOTAL:", 1, 0, 'L', 1)
    pdf.cell(25, 6, "", 1, 0, 'C', 1)
    pdf.cell(50, 6, "", 1, 0, 'C', 1)
    pdf.cell(50, 6, safe_text(f"-{c3_min:.1f}% to -{c3_max:.1f}%"), 1, 1, 'C', 1)
    pdf.ln(5)

    # === ELEPHANT TRIAD STRATEGY ===
    triad = _get_elephant_triad(base, ai_result)

    pdf.set_fill_color(200, 230, 200)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, safe_text(f"  ELEPHANT TRIAD STRATEGY ({triad['summary']['strength']})"), 0, 1, 'L', 1)
    pdf.ln(2)

    # Summary box
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 5, safe_text(
        f"  Active Arms: {triad['summary']['active_arms']}/3 | {triad['summary']['prognosis_modifier']}"), 0, 1)
    pdf.set_text_color(0)
    pdf.ln(3)

    # --- ARM 1: METABOLIC ATTACK ---
    arm1 = triad["metabolic_attack"]
    if arm1["indicated"]:
        pdf.set_fill_color(255, 250, 205)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(200, 150, 0)
        pdf.cell(0, 7, safe_text(f"  [ACTIVE] {arm1['name']}"), 0, 1, 'L', 1)
    else:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 7, safe_text(f"  [--] {arm1['name']}"), 0, 1, 'L', 1)

    pdf.set_text_color(0)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 4, safe_text(f"    Rationale: {arm1['rationale']}"), 0, 1)

    pdf.set_font('Arial', '', 8)
    for therapy in arm1["therapies"][:2]:
        pdf.cell(0, 4, safe_text(f"    - {therapy['drug']}: {therapy['dose']}"), 0, 1)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 4, safe_text(f"      Mechanism: {therapy['mechanism']}"), 0, 1)
        pdf.cell(0, 4, safe_text(f"      Evidence: {therapy['evidence']}"), 0, 1)
        pdf.set_text_color(0)

    if arm1["warnings"]:
        pdf.set_text_color(200, 100, 0)
        for warn in arm1["warnings"]:
            pdf.cell(0, 4, safe_text(f"    [!] {warn}"), 0, 1)
        pdf.set_text_color(0)
    pdf.ln(2)

    # --- ARM 2: DNA REPAIR COLLAPSE ---
    arm2 = triad["dna_repair_collapse"]
    if arm2["indicated"]:
        pdf.set_fill_color(230, 240, 255)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 100, 200)
        pdf.cell(0, 7, safe_text(f"  [ACTIVE] {arm2['name']}"), 0, 1, 'L', 1)

        pdf.set_text_color(0)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 4, safe_text(f"    Rationale: {arm2['rationale'][:80]}..."), 0, 1)

        pdf.set_font('Arial', '', 8)
        for therapy in arm2["therapies"][:2]:
            pdf.cell(0, 4, safe_text(f"    - {therapy['drug']}: {therapy['dose']}"), 0, 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"      {therapy['mechanism'][:70]}..."), 0, 1)
            ev_note = therapy.get('note', therapy['evidence'])
            pdf.cell(0, 4, safe_text(f"      [{therapy['evidence']}] {therapy.get('note', '')}"), 0, 1)
            pdf.set_text_color(0)
    else:
        pdf.set_fill_color(245, 245, 245)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 7, safe_text(f"  [NOT INDICATED] {arm2['name']}"), 0, 1, 'L', 1)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(150, 100, 100)
        pdf.cell(0, 4, safe_text(f"    {arm2['contraindication'][:90]}"), 0, 1)

    pdf.set_text_color(0)
    if arm2["warnings"]:
        pdf.set_text_color(200, 100, 0)
        for warn in arm2["warnings"][:2]:
            pdf.cell(0, 4, safe_text(f"    [!] {warn}"), 0, 1)
        pdf.set_text_color(0)
    pdf.ln(2)

    # --- ARM 3: APOPTOSIS BOOST ---
    arm3 = triad["apoptosis_boost"]
    if arm3["indicated"]:
        pdf.set_fill_color(255, 230, 230)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, safe_text(f"  [ACTIVE] {arm3['name']}"), 0, 1, 'L', 1)

        pdf.set_text_color(0)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 4, safe_text(f"    Rationale: {arm3['rationale'][:80]}..."), 0, 1)

        pdf.set_font('Arial', '', 8)
        for therapy in arm3["therapies"][:2]:
            pdf.cell(0, 4, safe_text(f"    - {therapy['drug']}: {therapy['dose']}"), 0, 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"      {therapy['mechanism'][:70]}..."), 0, 1)
            pdf.cell(0, 4, safe_text(f"      [{therapy['evidence']}]"), 0, 1)
            pdf.set_text_color(0)

        # Warnings MDM2
        pdf.set_text_color(200, 0, 0)
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(0, 4, safe_text("    EXPERIMENTAL - Requires clinical trial enrollment"), 0, 1)
        pdf.set_text_color(0)
    else:
        pdf.set_fill_color(245, 245, 245)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 7, safe_text(f"  [NOT INDICATED] {arm3['name']}"), 0, 1, 'L', 1)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(150, 100, 100)
        pdf.cell(0, 4, safe_text(f"    {arm3['contraindication'][:90]}"), 0, 1)

    pdf.set_text_color(0)
    if arm3["warnings"]:
        pdf.set_text_color(200, 100, 0)
        for warn in arm3["warnings"][:2]:
            pdf.cell(0, 4, safe_text(f"    [!] {warn}"), 0, 1)
        pdf.set_text_color(0)
    pdf.ln(5)

    # === INTERVENTION STRATEGY ===
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "  INTERVENTION STRATEGY", 0, 1, 'L', 1)

    # VETO note se attivo
    if ai_result.get('veto_active'):
        pdf.set_text_color(200, 0, 0)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, safe_text(f"  [!] VETO: {ai_result.get('veto_reason', '')}"), 0, 1)
        pdf.set_text_color(0)

    # Protocollo
    neural_drug = ai_result.get('vittoria_drug', 'Metformin 500mg BID')
    if neural_drug == 'None' or neural_drug is None:
        neural_drug = 'Metformin 500mg BID'

    current_therapy = base.get('current_therapy', '').lower()
    if "pembrolizumab" in current_therapy or "nivolumab" in current_therapy:
        phase_2_action = "- Switch to CTLA-4 Inhibitors (Immuno-Resistance detected)."
    else:
        phase_2_action = "- Evaluate Checkpoint Inhibitors (PD-1/PD-L1)."

    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "PHASE 1: INDUCTION (Shock)", 0, 1)
    pdf.set_text_color(0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, safe_text("  - Virotherapy (T-VEC): Selective viral lysis"), 0, 1)
    pdf.cell(0, 5, safe_text(f"  - Metabolic Exhaustion: {neural_drug} (Mitochondrial stress)"), 0, 1)
    pdf.cell(0, 5, safe_text("  - Dietary Restriction: Ketogenic Diet"), 0, 1)
    pdf.ln(2)

    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "PHASE 2: CONSOLIDATION (Cage)", 0, 1)
    pdf.set_text_color(0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, safe_text("  - Physical Encapsulation strategies"), 0, 1)
    pdf.cell(0, 5, safe_text(f"  {phase_2_action}"), 0, 1)
    pdf.ln(2)

    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "PHASE 3: MAINTENANCE (Chronic)", 0, 1)
    pdf.set_text_color(0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, safe_text("  - Adaptive Therapy: Clone competition"), 0, 1)
    pdf.cell(0, 5, safe_text("  - Low-dose Metformin maintenance"), 0, 1)
    pdf.ln(5)

    # === INTENSIFIED STRATEGY DETAILS (per LOW sensitivity) ===
    # === INTENSIFIED STRATEGY DETAILS (per LOW sensitivity) ===
    if sens_strategy == "intensive":
        pdf.ln(3)
        pdf.set_fill_color(255, 243, 205)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, safe_text("  INTENSIFIED CONTAINMENT PROTOCOL"), 0, 1, 'L', 1)

        # Ottieni terapie personalizzate
        intensified = _get_intensified_therapies(base, ai_result)

        pdf.set_font('Arial', '', 9)
        pdf.set_text_color(0)

        # --- PRIORITY THERAPIES ---
        if intensified['priority_therapies']:
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 6, safe_text("Priority Targeted Therapies (Correct First):"), 0, 1)
            pdf.set_font('Arial', '', 8)
            pdf.set_text_color(0)

            for i, therapy in enumerate(intensified['priority_therapies'][:4], 1):
                ev_tag = "[FDA]" if "FDA" in therapy.get('evidence', '') else "[Trial]"
                pdf.cell(0, 4, safe_text(f"  {i}. {therapy['drug']} {ev_tag}"), 0, 1)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 4, safe_text(f"     Dose: {therapy['dose']}"), 0, 1)
                pdf.cell(0, 4, safe_text(f"     Rationale: {therapy['rationale']}"), 0, 1)
                pdf.set_text_color(0)

        # --- METABOLIC AGENTS ---
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(255, 140, 0)
        pdf.cell(0, 6, safe_text("Metabolic Agents (Containment Strategy):"), 0, 1)
        pdf.set_font('Arial', '', 8)
        pdf.set_text_color(0)

        for agent in intensified['metabolic_agents'][:3]:
            ev_tag = "[EXP]" if "Experimental" in agent.get('evidence', '') or "Phase 1" in agent.get('evidence',
                                                                                                      '') else "[P2]"
            pdf.cell(0, 4, safe_text(f"  - {agent['drug']} {ev_tag}: {agent['rationale']}"), 0, 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"    Dose: {agent['dose']}"), 0, 1)
            pdf.set_text_color(0)

        # --- IMMUNOTHERAPY ---
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(0, 100, 150)
        pdf.cell(0, 6, safe_text("Immunotherapy Evaluation:"), 0, 1)
        pdf.set_font('Arial', '', 8)
        pdf.set_text_color(0)

        if intensified['immunotherapy']:
            imm = intensified['immunotherapy']
            pdf.cell(0, 4, safe_text(f"  [CANDIDATE] {imm['drug']} - {imm['rationale']}"), 0, 1)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, safe_text(f"    Dose: {imm['dose']}"), 0, 1)
            pdf.cell(0, 4, safe_text(f"    Timing: {imm.get('line', 'Per clinical judgment')}"), 0, 1)
            pdf.set_text_color(0)
        elif intensified['immunotherapy_excluded']:
            pdf.set_text_color(150, 0, 0)
            pdf.cell(0, 4, safe_text(f"  [EXCLUDED] {intensified['immunotherapy_excluded']}"), 0, 1)
            pdf.set_text_color(0)

        # --- FALLBACK OPTIONS ---
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, safe_text("Fallback Options (If Above Strategies Fail):"), 0, 1)
        pdf.set_font('Arial', '', 8)
        pdf.set_text_color(0)

        for fb in intensified['fallback'][:3]:
            pdf.cell(0, 4, safe_text(f"  - {fb['drug']}: {fb['rationale']}"), 0, 1)

        # --- ENHANCED MONITORING ---
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 6, safe_text("Enhanced Monitoring (Mandatory for LOW Sensitivity):"), 0, 1)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 4, safe_text("  - ctDNA: Every 4 weeks (early resistance detection)"), 0, 1)
        pdf.cell(0, 4, safe_text("  - Imaging: Every 6 weeks (rapid response assessment)"), 0, 1)
        pdf.cell(0, 4, safe_text("  - LDH + tumor markers: Weekly during induction"), 0, 1)

        # --- EARLY SWITCH TRIGGERS ---
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 6, safe_text("Early Switch Triggers (Don't Wait for RECIST PD):"), 0, 1)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 4, safe_text("  - LDH increase >25% from nadir"), 0, 1)
        pdf.cell(0, 4, safe_text("  - ctDNA: New mutation or VAF increase >50%"), 0, 1)
        pdf.cell(0, 4, safe_text("  - Clinical deterioration (ECOG +1 or weight loss >5%)"), 0, 1)

        # --- WARNINGS ---
        if intensified['warnings']:
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 9)
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 6, safe_text("Patient-Specific Warnings:"), 0, 1)
            pdf.set_font('Arial', '', 8)
            for warn in intensified['warnings']:
                pdf.cell(0, 4, safe_text(f"  [!] {warn}"), 0, 1)
            pdf.set_text_color(0)

        # --- GOAL ---
        pdf.ln(3)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 6, safe_text("GOAL: Chronic containment through aggressive adaptation. Never surrender."), 0, 1)
        pdf.set_text_color(0)
        pdf.ln(3)

    # === DISCLAIMER ===
    pdf.set_draw_color(200, 0, 0)
    pdf.set_line_width(0.5)
    pdf.rect(10, pdf.get_y(), 190, 25)
    pdf.set_line_width(0.2)

    pdf.set_text_color(150, 0, 0)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 6, safe_text("  WARNING"), 0, 1)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 4, safe_text(
        "  This protocol targets metabolic vulnerabilities. Requires Multidisciplinary Board Approval. "
        "The final therapeutic decision must be made by the treating oncologist in consultation with "
        "the patient, considering individual circumstances and preferences."
    ))
    pdf.set_text_color(0)

def generate_pdf(patient_id):
    """
    Genera report PDF longitudinale completo.

    Struttura:
    1. Report COMPLETO per Baseline
    2. Report COMPLETO per ogni visita (in ordine cronologico)
    3. Pagina Longitudinal Analysis alla fine
    """
    print(f"\n{'='*60}")
    print(f"SENTINEL REPORT GENERATOR v18.0 LONGITUDINAL FULL")
    print(f"Patient: {patient_id}")
    print(f"{'='*60}")

    # 1. Load data
    json_path = DATA_DIR / f"{patient_id}.json"
    if not json_path.exists():
        print(f"❌ File {patient_id}.json not found")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    baseline_original = data.get('baseline', {})
    visits = data.get('visits', [])

    print(f"📊 Found {len(visits)} visits")

    # 2. Create PDF
    pdf = SentinelReport()

    # =========================================================================
    # BASELINE REPORT (Prima sezione)
    # =========================================================================
    print("📄 Generating BASELINE report...")

    # Analizza baseline
    try:
        ai_result_baseline = analyze_patient_risk({'baseline': baseline_original})
    except Exception as e:
        print(f"⚠️ Baseline analysis error: {e}")
        ai_result_baseline = {}

    # Separatore Baseline
    pdf.visit_separator(
        "BASELINE ASSESSMENT",
        baseline_original.get('therapy_start_date', datetime.date.today().strftime('%Y-%m-%d')),
        is_current=(len(visits) == 0)
    )

    # Report completo baseline
    generate_complete_visit_report(
        pdf=pdf,
        visit_data=baseline_original,
        ai_result=ai_result_baseline,
        visit_label="BASELINE",
        visit_date=baseline_original.get('therapy_start_date', ''),
        is_baseline=True,
        previous_data=None

    )

    # =========================================================================
    # AI INITIAL SYNTHESIS (se presente nel JSON, mostrata per tutti)
    # =========================================================================
    if data.get('ai_initial_synthesis'):
        try:
            print("📄 Adding AI Initial Synthesis to PDF...")
            synthesis_info = data['ai_initial_synthesis']
            synthesis_text = synthesis_info.get('text', '')
            generated_at = synthesis_info.get('generated_at', '')
            model_used = synthesis_info.get('model', 'Gemini')

            if synthesis_text:
                pdf.add_page()

                # Header
                pdf.set_fill_color(70, 130, 180)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, safe_text("AI CLINICAL SYNTHESIS - INITIAL ASSESSMENT"), 0, 1, 'C', 1)
                pdf.set_text_color(0)

                pdf.ln(3)
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, safe_text(
                    f"Generated by: {model_used} | Date: {generated_at[:19] if generated_at else 'N/A'}"), 0, 1, 'C')
                pdf.set_text_color(0)
                pdf.ln(5)

                # Contenuto della sintesi
                pdf.set_font('Arial', '', 10)

                sections = synthesis_text.split('\n')
                for line in sections:
                    line = line.strip()
                    if not line:
                        pdf.ln(2)
                        continue

                    is_header = (line.startswith('1.') or line.startswith('2.') or
                                 line.startswith('3.') or line.startswith('4.') or
                                 line.startswith('5.') or line.startswith('**'))

                    is_keyword = any(kw in line.upper() for kw in
                                     ['QUADRO', 'ANDAMENTO', 'MUTAZIONI', 'CONSIDERAZIONI', 'PUNTI',
                                      'ATTENZIONE', 'PROFILO', 'RISCHIO'])

                    try:
                        pdf.set_x(pdf.l_margin)  # Reset posizione x
                        if is_header or is_keyword:
                            pdf.ln(3)
                            pdf.set_font('Arial', 'B', 10)
                            pdf.set_text_color(70, 130, 180)
                            clean_line = line.replace('**', '')
                            pdf.set_x(pdf.l_margin)
                            pdf.multi_cell(0, 6, safe_text(clean_line))
                            pdf.set_text_color(0)
                            pdf.set_font('Arial', '', 10)
                        else:
                            pdf.set_x(pdf.l_margin)
                            pdf.multi_cell(0, 5, safe_text(line))
                    except Exception:
                        pdf.ln(5)
                        continue

                # Disclaimer
                pdf.ln(10)
                pdf.set_fill_color(255, 250, 240)
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(150, 100, 50)
                disclaimer = "DISCLAIMER: Questa sintesi e' generata da intelligenza artificiale a scopo di supporto decisionale. Non sostituisce il giudizio clinico. Tutte le decisioni terapeutiche devono essere prese dal medico curante in accordo con il paziente."
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4, safe_text(disclaimer), 0, 'C', 1)
                pdf.set_text_color(0)

                print("   ✅ AI Initial Synthesis added to PDF")
        except Exception as e:
            print(f"   ⚠️ Error adding AI synthesis: {e}")

    # =========================================================================
    # VISITE SUCCESSIVE (in ordine cronologico)
    # =========================================================================
    previous_data = baseline_original

    for i, visit in enumerate(visits):
        visit_num = i + 1
        is_current = (i == len(visits) - 1)  # Ultima visita = corrente

        print(f"📄 Generating VISIT {visit_num} report...")

        # Costruisci stato per questa visita
        visit_state = copy.deepcopy(baseline_original)

        # Override con dati della visita
        if visit.get('therapy_changed') and visit.get('new_therapy'):
            visit_state['current_therapy'] = visit['new_therapy']
        elif visit.get('therapy_at_visit'):
            visit_state['current_therapy'] = visit['therapy_at_visit']

        if visit.get('blood_markers'):
            visit_state['blood_markers'] = visit['blood_markers']

        if visit.get('genetics'):
            visit_state['genetics'].update(visit['genetics'])

        if visit.get('ecog_ps') is not None:
            visit_state['ecog_ps'] = visit['ecog_ps']

        # Analizza questa visita
        try:
            ai_result_visit = analyze_patient_risk({'baseline': visit_state})
        except Exception as e:
            print(f"⚠️ Visit {visit_num} analysis error: {e}")
            ai_result_visit = {}

        # Determina se la visita è compatta (stessa genetica e terapia)
        _visit_compact = False
        if previous_data:
            _prev_gen = previous_data.get('genetics', {})
            _curr_gen = visit_state.get('genetics', {})
            _prev_ther = str(previous_data.get('current_therapy', '')).lower()
            _curr_ther = str(visit_state.get('current_therapy', '')).lower()
            _visit_compact = (_prev_gen == _curr_gen) and (_prev_ther == _curr_ther)

        if _visit_compact:
            # Separatore leggero per visite compatte (senza nuova pagina dedicata)
            pdf.add_page()
            visit_label_text = f"VISIT {visit_num} - Week {visit.get('week_on_therapy') or visit.get('week_offset', '?')}"
            if is_current:
                pdf.set_fill_color(0, 102, 0)
            else:
                pdf.set_fill_color(0, 51, 102)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 12, safe_text(f'  {visit_label_text}'), 0, 1, 'L', 1)
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 6, safe_text(f'  Date: {format_date(visit.get("date", ""))}'), 0, 1, 'L', 1)
            pdf.set_text_color(0)
            pdf.ln(3)
        else:
            # Separatore visita completo (pagina intera)
            pdf.visit_separator(
                f"VISIT {visit_num} - Week {visit.get('week_on_therapy') or visit.get('week_offset', '?')}",
                visit.get('date', ''),
                is_current=is_current
            )

        # Report completo visita
        generate_complete_visit_report(
            pdf=pdf,
            visit_data=visit_state,
            ai_result=ai_result_visit,
            visit_label=f"VISIT {visit_num}",
            visit_date=visit.get('date', ''),
            is_baseline=False,
            previous_data=previous_data
        )

        # [NEW] === LLM CLINICAL NOTES (per questa visita) ===
        try:
            has_notes_field = 'notes' in visit if visit else False
            original_notes = visit.get('notes', '') if visit else ''
            llm_summary = visit.get('llm_notes_summary', '') if visit else ''
            print(f"   DEBUG V{visit_num}: notes={bool(original_notes)}, llm_summary_exists={bool(llm_summary)}")
            llm_flags = visit.get('llm_urgency_flags', []) if visit else []

            # Se ci sono note ma manca il sommario LLM, generalo e salvalo nel JSON
            if original_notes and original_notes.strip() and not llm_summary:
                try:
                    from clinical_notes_llm import enrich_visit_data, is_available as llm_available
                    if llm_available():
                        print(f"🤖 Generating AI summary for Visit {visit_num}...")
                        enriched = enrich_visit_data({'notes': original_notes})
                        llm_summary = enriched.get('llm_notes_summary', '')
                        llm_flags = enriched.get('llm_urgency_flags', [])

                        # Salva nel JSON per non rigenerare
                        if llm_summary and "Impossibile" not in llm_summary and "non disponibile" not in llm_summary:
                            visit['llm_notes_summary'] = llm_summary
                            visit['llm_urgency_flags'] = llm_flags

                            # Salva il JSON aggiornato
                            patient_id = baseline_original.get('patient_id', 'unknown')
                            json_path = DATA_DIR / f"{patient_id}.json"
                            print(f"   DEBUG: patient_id={patient_id}, json_path={json_path}")
                            print(f"   DEBUG: json_path.exists()={json_path.exists()}")
                            if json_path.exists():
                                with open(json_path, 'r') as f:
                                    patient_json = json.load(f)
                                print(f"   DEBUG: i={i}, len(visits)={len(patient_json.get('visits', []))}")
                                # i è l'indice del loop delle visite
                                if 'visits' in patient_json and i < len(patient_json['visits']):
                                    patient_json['visits'][i]['llm_notes_summary'] = llm_summary
                                    patient_json['visits'][i]['llm_urgency_flags'] = llm_flags
                                    with open(json_path, 'w') as f:
                                        json.dump(patient_json, f, indent=2, ensure_ascii=False)
                                    print(f"   ✅ Summary saved to JSON for visit {i}")
                                else:
                                    print(
                                        f"   ❌ Index mismatch: i={i}, visits in json={len(patient_json.get('visits', []))}")
                            else:
                                print(f"   ❌ JSON file not found: {json_path}")
                except Exception as llm_err:
                    print(f"⚠️ LLM enrichment error: {llm_err}")

            # Render nel PDF
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 11)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(0, 8, safe_text(f"Clinical Notes - Visit {visit_num}"), 0, 1, 'L', 1)

            if not has_notes_field:
                pdf.set_font('Arial', 'I', 9)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 5, safe_text("None"), 0, 1)
                pdf.set_text_color(0)
            elif not original_notes or original_notes.strip() == '':
                pdf.set_font('Arial', 'I', 9)
                pdf.set_text_color(150, 150, 150)
                pdf.cell(0, 5, safe_text(""), 0, 1)
                pdf.set_text_color(0)
            else:
                if llm_summary and "Impossibile" not in llm_summary and "non disponibile" not in llm_summary:
                    pdf.set_font('Arial', 'I', 9)
                    pdf.set_text_color(0, 80, 0)
                    pdf.multi_cell(0, 5, safe_text(f"AI Summary: {llm_summary}"))
                    pdf.set_text_color(0)

                if llm_flags:
                    pdf.set_font('Arial', 'B', 9)
                    pdf.set_text_color(200, 0, 0)
                    pdf.cell(0, 6, safe_text(f"Alerts: {', '.join(llm_flags)}"), 0, 1)
                    pdf.set_text_color(0)

                pdf.set_font('Arial', '', 8)
                pdf.set_text_color(100, 100, 100)
                notes_short = original_notes[:150] + "..." if len(original_notes) > 150 else original_notes
                pdf.multi_cell(0, 4, safe_text(f"[Original: {notes_short}]"))
                pdf.set_text_color(0)

        except Exception as e:
            print(f"⚠️ LLM Notes section error: {e}")
        # === END LLM ===

        # Aggiorna previous per la prossima iterazione
        previous_data = visit_state

        # Salva ultimo tank_score e ai_result per la sintesi finale
        try:
            final_tank_score = tank_score
            final_ai_result = ai_result
        except:
            final_tank_score = ai_result_baseline.get('tank_score', 0) if 'ai_result_baseline' in dir() else 0
            final_ai_result = ai_result_baseline if 'ai_result_baseline' in dir() else {}

    # =========================================================================
    # LONGITUDINAL ANALYSIS (Ultima pagina - FUORI dal loop visite)
    # =========================================================================
    if visits:
        print("📄 Generating LONGITUDINAL ANALYSIS...")
        draw_longitudinal_analysis(pdf, baseline_original, visits)

    # =========================================================================
    # SEZIONE FINALE: AI CLINICAL SYNTHESIS (Gemini) - PER TUTTI I PAZIENTI
    # =========================================================================
    try:
        from clinical_notes_llm import is_available as llm_available
        import google.generativeai as genai

        if llm_available():
            print("🧠 Generating AI Clinical Synthesis...")

            base_data = baseline_original

            # Raccogli tutti i dati del paziente per la sintesi
            synthesis_data = {
                'patient_id': base_data.get('patient_id', 'Unknown'),
                'age': base_data.get('age', 'N/A'),
                'sex': base_data.get('sex', 'N/A'),
                'histology': base_data.get('histology', 'N/A'),
                'stage': base_data.get('stage', 'N/A'),
                'smoking': base_data.get('smoking_status', 'N/A'),
                'ecog_baseline': base_data.get('ecog_ps', 'N/A'),
                'genetics': base_data.get('genetics', {}),
                'tank_score': final_tank_score if 'final_tank_score' in dir() else ai_result_baseline.get('tank_score', 0),
                'ferrari_score': (final_ai_result if 'final_ai_result' in dir() else ai_result_baseline).get('ferrari_score', 'N/A'),
                'ldh_baseline': (base_data.get('blood_markers') or {}).get('ldh', 'N/A'),
                'current_therapy': base_data.get('current_therapy', 'N/A'),
                'num_visits': len(visits) if visits else 0,
            }

            # Raccogli info dalle visite
            visit_summaries = []
            ldh_values = []
            ecog_values = []
            therapies = set()

            if (base_data.get('blood_markers') or {}).get('ldh'):
                ldh_values.append(('Baseline', base_data['blood_markers']['ldh']))
            if base_data.get('ecog_ps') is not None:
                ecog_values.append(('Baseline', base_data['ecog_ps']))
            therapies.add(base_data.get('current_therapy', 'Unknown'))

            for idx, visit in enumerate(visits or []):
                v_num = idx + 1
                v_ldh = (visit.get('blood_markers') or {}).get('ldh')
                v_ecog = visit.get('ecog_ps')
                v_therapy = visit.get('current_therapy', '')
                v_recist = visit.get('recist_response', '')
                v_notes_summary = visit.get('llm_notes_summary', '')

                if v_ldh:
                    ldh_values.append((f'V{v_num}', v_ldh))
                if v_ecog is not None:
                    ecog_values.append((f'V{v_num}', v_ecog))
                if v_therapy:
                    therapies.add(v_therapy)

                visit_summaries.append({
                    'visit': v_num,
                    'ldh': v_ldh,
                    'ecog': v_ecog,
                    'therapy': v_therapy,
                    'recist': v_recist,
                    'notes': v_notes_summary[:200] if v_notes_summary else ''
                })

            # Costruisci RECIST history
            recist_history = []
            for idx2, v in enumerate(visits or []):
                v_recist_val = v.get('recist_response', (v.get('imaging') or {}).get('response', 'NE'))
                if v_recist_val:
                    recist_history.append(f"V{idx2 + 1}: {v_recist_val}")
                else:
                    recist_history.append(f"V{idx2 + 1}: NE")

            synthesis_data['recist_history'] = recist_history
            synthesis_data['recist_current'] = recist_history[-1].split(': ')[-1] if recist_history else 'NE'

            synthesis_data['ldh_trend'] = ldh_values
            synthesis_data['ecog_trend'] = ecog_values
            synthesis_data['therapies_used'] = list(therapies)
            synthesis_data['visits'] = visit_summaries
            synthesis_data['synergies'] = (final_ai_result if 'final_ai_result' in dir() else ai_result_baseline).get('explainability', {}).get('synergies', [])

            # Estrazione sicura dei biomarcatori immunitari chiave
            biomarkers = base_data.get("biomarkers", {})
            tmb_score = biomarkers.get("tmb_score", "Non misurato")
            pdl1_percent = biomarkers.get("pdl1_percent", "Non misurato")

            # Prompt per Gemini
            synthesis_prompt = f"""Sei un oncologo esperto. Analizza i dati di questo paziente oncologico e fornisci una sintesi clinica strutturata.

DATI PAZIENTE:
- ID: {synthesis_data['patient_id']}
- Età/Sesso: {synthesis_data['age']}/{synthesis_data['sex']}
- Istologia: {synthesis_data['histology']}, Stadio: {synthesis_data['stage']}
- Fumo: {synthesis_data['smoking']}
- ECOG baseline: {synthesis_data['ecog_baseline']}
- TMB (Tumor Mutational Burden): {tmb_score}
- Espressione PD-L1: {pdl1_percent}%

ATTENZIONE: Valuta sempre il TMB e il PD-L1 per giustificare il successo o il fallimento dell'Immunoterapia. Un TMB alto (>10) o un PD-L1 alto spiegano una forte risposta immunitaria.

PROFILO GENOMICO:
{json.dumps(synthesis_data['genetics'], indent=2, default=str)}

SCORE DI RISCHIO:
- Tank (clinico): {synthesis_data['tank_score']}/100
- Ferrari (biologico): {synthesis_data['ferrari_score']}%

SINERGIE MUTAZIONALI RILEVATE:
{synthesis_data['synergies']}

ANDAMENTO LDH: {synthesis_data['ldh_trend']}
ANDAMENTO ECOG: {synthesis_data['ecog_trend']}
TERAPIE UTILIZZATE: {synthesis_data['therapies_used']}

VISITE ({synthesis_data['num_visits']}):
{json.dumps(synthesis_data['visits'], indent=2, default=str)}

RECIST HISTORY (usa SOLO questi valori, NON inferire):
{synthesis_data.get('recist_history', ['Non disponibile'])}
Risposta corrente: {synthesis_data.get('recist_current', 'NE')}

REGOLE OBBLIGATORIE (DEVI rispettarle):
1. RECIST: Usa SOLO i valori forniti. Se RECIST = PR, scrivi "Partial Response" o "risposta parziale". 
   NON usare MAI "Risposta Completa" o "CR" se RECIST != CR.
2. CEA: NON menzionare CEA (antigene carcinoembrionario) - non è presente in questo dataset.
3. Distingui SEMPRE tra:
   - Risposta CLINICA (sintomi, PS, qualità vita)
   - Risposta RADIOLOGICA (imaging, RECIST) 
   - Risposta MOLECOLARE (ctDNA, VAF)
4. Se ctDNA è negativo ma RECIST è PR, scrivi "risposta molecolare completa con risposta radiologica parziale mantenuta".
5. NON inferire miglioramenti oltre i dati forniti.
6. Basa TUTTO sui dati espliciti, non su assunzioni.

FORNISCI UNA SINTESI STRUTTURATA IN ITALIANO CON:
1. QUADRO CLINICO COMPLESSIVO (2-3 frasi)
2. ANDAMENTO MALATTIA (trend positivo/negativo/stabile, con evidenze)
3. MUTAZIONI CHIAVE E IMPATTO CLINICO (quali sono actionable, quali conferiscono resistenza)
4. CONSIDERAZIONI TERAPEUTICHE (basate su evidenze, NON prescrizioni)
5. PUNTI DI ATTENZIONE (cosa monitorare, red flags)

Sii conciso ma completo. Usa linguaggio medico appropriato. NON fare diagnosi o prescrizioni, solo considerazioni basate sui dati."""

            # Chiama Gemini
            from clinical_notes_llm import GOOGLE_API_KEY, MODEL_ID, _call_gemini_safe
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(MODEL_ID)

            response = _call_gemini_safe(model, synthesis_prompt)

            if response and response.text:
                synthesis_text = response.text.strip()

                # Aggiungi nuova pagina per la sintesi
                pdf.add_page()

                # Header
                pdf.set_fill_color(70, 130, 180)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, safe_text("AI CLINICAL SYNTHESIS"), 0, 1, 'C', 1)
                pdf.set_text_color(0)

                pdf.ln(3)
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(100, 100, 100)
                now_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')
                pdf.cell(0, 5, safe_text(f"Generated by: Google Gemini ({MODEL_ID}) | Date: {now_str}"), 0, 1, 'C')
                pdf.set_text_color(0)
                pdf.ln(5)

                # Contenuto della sintesi
                pdf.set_font('Arial', '', 10)

                # Dividi per sezioni
                sections = synthesis_text.split('\n')
                for line in sections:
                    line = line.strip()
                    if not line:
                        pdf.ln(2)
                        continue

                    # Formatta i titoli delle sezioni
                    is_header = (line.startswith('1.') or line.startswith('2.') or
                                 line.startswith('3.') or line.startswith('4.') or
                                 line.startswith('5.') or line.startswith('**'))

                    is_keyword = any(kw in line.upper() for kw in
                                     ['QUADRO', 'ANDAMENTO', 'MUTAZIONI', 'CONSIDERAZIONI', 'PUNTI',
                                      'ATTENZIONE'])

                    try:
                        pdf.set_x(pdf.l_margin)  # Reset posizione x
                        if is_header or is_keyword:
                            pdf.ln(3)
                            pdf.set_font('Arial', 'B', 10)
                            pdf.set_text_color(70, 130, 180)
                            clean_line = line.replace('**', '')
                            pdf.set_x(pdf.l_margin)
                            pdf.multi_cell(0, 6, safe_text(clean_line))
                            pdf.set_text_color(0)
                            pdf.set_font('Arial', '', 10)
                        else:
                            pdf.set_x(pdf.l_margin)
                            pdf.multi_cell(0, 5, safe_text(line))
                    except Exception:
                        # Fallback: se la riga causa problemi, salta
                        pdf.ln(5)
                        continue

                # Disclaimer
                pdf.ln(10)
                pdf.set_fill_color(255, 250, 240)
                pdf.set_font('Arial', 'I', 8)
                pdf.set_text_color(150, 100, 50)
                disclaimer = "DISCLAIMER: Questa sintesi e' generata da intelligenza artificiale a scopo di supporto decisionale. Non sostituisce il giudizio clinico. Tutte le decisioni terapeutiche devono essere prese dal medico curante in accordo con il paziente."
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4, safe_text(disclaimer), 0, 'C', 1)
                pdf.set_text_color(0)

                print("   ✅ AI Clinical Synthesis generated")
            else:
                print("   ⚠️ AI Synthesis: No response from Gemini")

    except Exception as synth_err:
        print(f"⚠️ AI Clinical Synthesis error: {synth_err}")

    # 3. Save
    filename = f"SENTINEL_REPORT_{patient_id}.pdf"
    full_path = OUTPUT_DIR / filename
    pdf.output(str(full_path))

    print(f"\n✅ Report generated: {filename}")
    print(f"   Total pages: {pdf.page_no()}")
    print(f"   Sections: Baseline + {len(visits)} visits + Longitudinal")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_pdf(sys.argv[1])
    else:
        files = list(DATA_DIR.glob("*.json"))
        if files:
            latest = max(files, key=os.path.getctime)
            generate_pdf(latest.stem)
        else:
            p_id = input("Patient ID: ")
            generate_pdf(p_id)