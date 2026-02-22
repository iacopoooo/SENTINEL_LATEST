"""
SENTINEL FOLLOW-UP & TIME TRAVEL MODULE (v1.0)
===============================================
Gestione longitudinale del paziente oncologico.

Features:
- Tracking visite successive
- Calcolo delta tra visite (LDH, mutations, imaging)
- Rilevamento pattern di resistenza
- Integrazione con Ferrari temporal modifiers
- Time Travel simulation (what-if scenarios)

Schema JSON Extended:
{
    "baseline": {...},
    "visits": [
        {
            "visit_id": "V1",
            "date": "2026-02-15",
            "week_on_therapy": 6,
            "genetics": {...},
            "blood_markers": {...},
            "imaging": {"response": "PR", "tumor_change_percent": -30},
            "clinical_notes": "..."
        }
    ]
}
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class RECISTResponse(Enum):
    """RECIST 1.1 Response Categories"""
    CR = "Complete Response"  # Scomparsa tutte le lesioni
    PR = "Partial Response"  # ≥30% decrease
    SD = "Stable Disease"  # Neither PR nor PD
    PD = "Progressive Disease"  # ≥20% increase or new lesions
    NE = "Not Evaluable"


class ResistancePattern(Enum):
    """Pattern di resistenza rilevabili"""
    NONE = "No resistance detected"
    T790M_ACQUIRED = "T790M Acquired Resistance"
    MET_AMPLIFICATION = "MET Amplification (Bypass)"
    SCLC_TRANSFORMATION = "SCLC Histological Transformation"
    EMERGENT_CLONE = "Emergent Resistant Clone (VAF rising)"
    EMT_PHENOTYPE = "EMT Phenotype Shift"
    UNKNOWN = "Unknown Resistance Mechanism"


class TrendDirection(Enum):
    """Direzione trend marker"""
    RISING = "↑ RISING"
    STABLE = "→ STABLE"
    FALLING = "↓ FALLING"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImagingResult:
    """Risultato imaging (CT/PET)"""
    date: str
    response: RECISTResponse = RECISTResponse.NE
    tumor_change_percent: float = 0.0  # Negativo = shrinkage
    new_lesions: bool = False
    sites_evaluated: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BloodMarkers:
    """Marker ematici per visita"""
    ldh: float = 0.0
    neutrophils: float = 0.0
    lymphocytes: float = 0.0
    nlr: float = 0.0
    albumin: float = 0.0
    cea: float = 0.0  # Carcinoembryonic antigen (optional)


@dataclass
class GeneticSnapshot:
    """Snapshot genetico per visita (da ctDNA o biopsy)"""
    source: str = "ctDNA"  # ctDNA, biopsy, liquid_biopsy
    tp53_status: str = "unknown"
    kras_mutation: str = "unknown"
    egfr_status: str = "unknown"
    met_status: str = "unknown"
    met_cn: float = 0.0
    stk11_status: str = "unknown"
    keap1_status: str = "unknown"
    rb1_status: str = "unknown"
    # VAF tracking (Variant Allele Frequency)
    vaf_values: Dict[str, float] = field(default_factory=dict)  # {"EGFR": 0.15, "TP53": 0.22}


@dataclass
class Visit:
    """Singola visita di follow-up"""
    visit_id: str
    date: str
    week_on_therapy: int
    therapy_at_visit: str
    ecog_ps: int = 1

    # Clinical data
    genetics: Optional[GeneticSnapshot] = None
    blood: Optional[BloodMarkers] = None
    imaging: Optional[ImagingResult] = None

    # Computed deltas (filled by PatientTimeline)
    deltas: Dict = field(default_factory=dict)

    ai_snapshot: Dict = field(default_factory=dict)
    # Clinical notes
    notes: str = ""
    adverse_events: List[str] = field(default_factory=list)

    # Flags
    therapy_changed: bool = False
    new_therapy: str = ""


@dataclass
class ResistanceAlert:
    """Alert per resistenza rilevata"""
    pattern: ResistancePattern
    confidence: str  # "High", "Medium", "Low"
    evidence: List[str]
    recommendation: str
    urgency: str  # "Immediate", "Soon", "Monitor"


@dataclass
class TimelineAnalysis:
    """Analisi completa della timeline paziente"""
    patient_id: str
    total_visits: int
    weeks_on_therapy: int
    current_therapy: str

    # Trends
    ldh_trend: TrendDirection
    response_trajectory: str  # "Improving", "Stable", "Declining"

    # Resistance
    resistance_alerts: List[ResistanceAlert]

    # Prognosis update
    updated_risk_score: int
    risk_change: int  # Rispetto a baseline

    # Recommendations
    recommendations: List[str]


# ============================================================================
# PATIENT TIMELINE
# ============================================================================

class PatientTimeline:
    """
    Gestisce la timeline longitudinale di un paziente.
    Calcola delta, trend, e rileva pattern di resistenza.
    """

    def __init__(self, patient_id: str, baseline_data: Dict):
        self.patient_id = patient_id
        self.baseline = baseline_data.get('baseline', baseline_data)
        self.therapy_start_date = self._parse_date(
            self.baseline.get('therapy_start_date', datetime.now().strftime('%Y-%m-%d'))
        )
        self.visits: List[Visit] = []

        # Load existing visits if present
        existing_visits = baseline_data.get('visits', [])
        for v_data in existing_visits:
            visit = self._dict_to_visit(v_data)
            self.visits.append(visit)

        logger.info(f"📅 Timeline initialized for {patient_id} with {len(self.visits)} existing visits")

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return datetime.now()

    def _dict_to_visit(self, data: Dict) -> Visit:
        """Convert dictionary to Visit object"""
        genetics = None
        if data.get('genetics'):
            genetics = GeneticSnapshot(**data['genetics'])

        blood = None
        if data.get('blood_markers'):
            bm = data['blood_markers']
            blood = BloodMarkers(
                ldh=float(bm.get('ldh', 0)),
                neutrophils=float(bm.get('neutrophils', 0)),
                lymphocytes=float(bm.get('lymphocytes', 1)),
                nlr=float(bm.get('nlr', 0)),
                albumin=float(bm.get('albumin', 0)),
                cea=float(bm.get('cea', 0))
            )

        imaging = None
        if data.get('imaging'):
            img = data['imaging']
            imaging = ImagingResult(
                date=img.get('date', data.get('date', '')),
                response=RECISTResponse[img.get('response', 'NE')],
                tumor_change_percent=float(img.get('tumor_change_percent', 0)),
                new_lesions=img.get('new_lesions', False),
                notes=img.get('notes', '')
            )

        return Visit(
            visit_id=data.get('visit_id', f"V{len(self.visits) + 1}"),
            date=data.get('date', ''),
            week_on_therapy=int(data.get('week_on_therapy', 0)),
            therapy_at_visit=data.get('therapy_at_visit', data.get('current_therapy', '')),
            ecog_ps=int(data.get('ecog_ps', 1)),
            genetics=genetics,
            blood=blood,
            imaging=imaging,
            notes=data.get('notes', ''),
            adverse_events=data.get('adverse_events', []),
            therapy_changed=data.get('therapy_changed', False),
            new_therapy=data.get('new_therapy', '')
        )

    def add_visit(self, visit_data: Dict) -> Visit:
        """
        Aggiunge una nuova visita e calcola i delta rispetto alla precedente.

        Args:
            visit_data: Dizionario con dati della visita

        Returns:
            Visit object con delta calcolati
        """
        visit = self._dict_to_visit(visit_data)

        # Calculate week on therapy if not provided
        if visit.week_on_therapy == 0:
            visit_date = self._parse_date(visit.date)
            delta_days = (visit_date - self.therapy_start_date).days
            visit.week_on_therapy = max(0, delta_days // 7)

        # Calculate deltas if we have previous data
        if self.visits:
            prev_visit = self.visits[-1]
            visit.deltas = self._calculate_deltas(prev_visit, visit)
        else:
            # Compare to baseline
            visit.deltas = self._calculate_deltas_from_baseline(visit)

        self.visits.append(visit)

        logger.info(f"📝 Added visit {visit.visit_id} at week {visit.week_on_therapy}")

        return visit

    def _calculate_deltas(self, prev: Visit, curr: Visit) -> Dict:
        """Calcola delta tra due visite consecutive"""
        deltas = {
            "weeks_elapsed": curr.week_on_therapy - prev.week_on_therapy,
            "ldh_change": 0,
            "ldh_trend": TrendDirection.STABLE.value,
            "nlr_change": 0,
            "ecog_change": curr.ecog_ps - prev.ecog_ps,
            "new_mutations": [],
            "vaf_changes": {},
            "imaging_response": None
        }

        # Blood markers delta
        if prev.blood and curr.blood:
            ldh_prev = prev.blood.ldh or 0
            ldh_curr = curr.blood.ldh or 0

            if ldh_prev > 0:
                deltas["ldh_change"] = ldh_curr - ldh_prev
                ldh_ratio = ldh_curr / ldh_prev

                if ldh_ratio > 1.2:
                    deltas["ldh_trend"] = TrendDirection.RISING.value
                elif ldh_ratio < 0.8:
                    deltas["ldh_trend"] = TrendDirection.FALLING.value
                else:
                    deltas["ldh_trend"] = TrendDirection.STABLE.value

            if prev.blood.nlr > 0:
                deltas["nlr_change"] = (curr.blood.nlr or 0) - prev.blood.nlr

        # Genetic changes
        if prev.genetics and curr.genetics:
            deltas["new_mutations"] = self._detect_new_mutations(prev.genetics, curr.genetics)
            deltas["vaf_changes"] = self._track_vaf_changes(prev.genetics, curr.genetics)

        # Imaging
        if curr.imaging:
            deltas["imaging_response"] = curr.imaging.response.name

        return deltas

    def _calculate_deltas_from_baseline(self, visit: Visit) -> Dict:
        """Calcola delta rispetto al baseline"""
        deltas = {
            "weeks_elapsed": visit.week_on_therapy,
            "ldh_change": 0,
            "ldh_trend": TrendDirection.STABLE.value,
            "from_baseline": True
        }

        baseline_blood = self.baseline.get('blood_markers', {})
        baseline_ldh = float(baseline_blood.get('ldh', 0))

        if visit.blood and baseline_ldh > 0:
            deltas["ldh_change"] = (visit.blood.ldh or 0) - baseline_ldh
            ldh_ratio = (visit.blood.ldh or baseline_ldh) / baseline_ldh

            if ldh_ratio > 1.2:
                deltas["ldh_trend"] = TrendDirection.RISING.value
            elif ldh_ratio < 0.8:
                deltas["ldh_trend"] = TrendDirection.FALLING.value

        return deltas

    def _detect_new_mutations(self, prev: GeneticSnapshot, curr: GeneticSnapshot) -> List[str]:
        """Rileva nuove mutazioni acquisite"""
        new_muts = []

        # Check each gene
        gene_pairs = [
            ('tp53_status', 'TP53'),
            ('egfr_status', 'EGFR'),
            ('met_status', 'MET'),
            ('stk11_status', 'STK11'),
            ('keap1_status', 'KEAP1'),
            ('rb1_status', 'RB1')
        ]

        for attr, gene_name in gene_pairs:
            prev_val = getattr(prev, attr, 'wt').lower()
            curr_val = getattr(curr, attr, 'wt').lower()

            # Was WT, now mutated
            if prev_val in ['wt', 'unknown', 'none', ''] and curr_val not in ['wt', 'unknown', 'none', '']:
                new_muts.append(f"{gene_name}: {curr_val}")

        # Special: T790M detection
        prev_egfr = getattr(prev, 'egfr_status', '').lower()
        curr_egfr = getattr(curr, 'egfr_status', '').lower()
        if 't790m' not in prev_egfr and 't790m' in curr_egfr:
            new_muts.append("T790M_ACQUIRED")

        # MET amplification
        if prev.met_cn < 5.0 and curr.met_cn >= 5.0:
            new_muts.append(f"MET_AMPLIFICATION (CN: {curr.met_cn})")

        return new_muts

    def _track_vaf_changes(self, prev: GeneticSnapshot, curr: GeneticSnapshot) -> Dict[str, Dict]:
        """Traccia cambiamenti VAF"""
        changes = {}

        for gene in set(list(prev.vaf_values.keys()) + list(curr.vaf_values.keys())):
            prev_vaf = prev.vaf_values.get(gene, 0)
            curr_vaf = curr.vaf_values.get(gene, 0)

            if prev_vaf > 0 or curr_vaf > 0:
                change = curr_vaf - prev_vaf
                changes[gene] = {
                    "previous": prev_vaf,
                    "current": curr_vaf,
                    "change": change,
                    "trend": "rising" if change > 0.05 else ("falling" if change < -0.05 else "stable")
                }

        return changes

    def detect_resistance_patterns(self) -> List[ResistanceAlert]:
        """
        Analizza la timeline per rilevare pattern di resistenza.

        Returns:
            Lista di ResistanceAlert
        """
        alerts = []

        if len(self.visits) < 1:
            return alerts

        latest = self.visits[-1]

        # Pattern 1: T790M acquired
        if latest.deltas.get('new_mutations'):
            for mut in latest.deltas['new_mutations']:
                if 'T790M' in mut:
                    alerts.append(ResistanceAlert(
                        pattern=ResistancePattern.T790M_ACQUIRED,
                        confidence="High",
                        evidence=[f"T790M detected at week {latest.week_on_therapy}"],
                        recommendation="Switch to Osimertinib (3rd-gen EGFR-TKI)",
                        urgency="Immediate"
                    ))

        # Pattern 2: MET amplification
        if latest.genetics and latest.genetics.met_cn >= 5.0:
            # Check if new
            is_new = True
            if len(self.visits) > 1 and self.visits[-2].genetics:
                if self.visits[-2].genetics.met_cn >= 5.0:
                    is_new = False

            if is_new:
                alerts.append(ResistanceAlert(
                    pattern=ResistancePattern.MET_AMPLIFICATION,
                    confidence="High",
                    evidence=[f"MET CN={latest.genetics.met_cn} at week {latest.week_on_therapy}"],
                    recommendation="Add Capmatinib or Tepotinib to regimen",
                    urgency="Soon"
                ))

        # Pattern 3: Emergent clone (VAF rising + stable/PD imaging)
        if latest.deltas.get('vaf_changes'):
            rising_vafs = [g for g, v in latest.deltas['vaf_changes'].items() if v.get('trend') == 'rising']

            if rising_vafs and latest.imaging:
                if latest.imaging.response in [RECISTResponse.SD, RECISTResponse.PD]:
                    alerts.append(ResistanceAlert(
                        pattern=ResistancePattern.EMERGENT_CLONE,
                        confidence="Medium",
                        evidence=[
                            f"Rising VAF in: {', '.join(rising_vafs)}",
                            f"Imaging: {latest.imaging.response.value}"
                        ],
                        recommendation="Consider rebiopsy to characterize resistant clone",
                        urgency="Soon"
                    ))

        # Pattern 4: LDH rising + imaging stable (occult progression)
        if latest.deltas.get('ldh_trend') == TrendDirection.RISING.value:
            if latest.imaging and latest.imaging.response == RECISTResponse.SD:
                alerts.append(ResistanceAlert(
                    pattern=ResistancePattern.UNKNOWN,
                    confidence="Low",
                    evidence=[
                        f"LDH rising ({latest.deltas.get('ldh_change', 0):+.0f} U/L)",
                        "Imaging stable - possible occult progression"
                    ],
                    recommendation="Increase monitoring frequency, consider PET-CT",
                    urgency="Monitor"
                ))

        # Pattern 5: RB1 loss + TP53 (SCLC transformation risk)
        if latest.genetics:
            rb1 = latest.genetics.rb1_status.lower()
            tp53 = latest.genetics.tp53_status.lower()

            if rb1 in ['loss', 'mutated'] and tp53 in ['loss', 'mutated']:
                # Check histology change
                alerts.append(ResistanceAlert(
                    pattern=ResistancePattern.SCLC_TRANSFORMATION,
                    confidence="Medium",
                    evidence=["TP53 + RB1 double loss detected"],
                    recommendation="Rebiopsy to rule out SCLC transformation. If confirmed, switch to Platinum-Etoposide",
                    urgency="Soon"
                ))

        return alerts

    def get_ldh_trend(self) -> TrendDirection:
        """Calcola trend LDH globale"""
        if len(self.visits) < 2:
            return TrendDirection.STABLE

        # Usa ultimi 3 valori
        recent = self.visits[-3:] if len(self.visits) >= 3 else self.visits
        ldh_values = [v.blood.ldh for v in recent if v.blood and v.blood.ldh > 0]

        if len(ldh_values) < 2:
            return TrendDirection.STABLE

        # Linear trend
        avg_change = (ldh_values[-1] - ldh_values[0]) / len(ldh_values)

        if avg_change > 20:
            return TrendDirection.RISING
        elif avg_change < -20:
            return TrendDirection.FALLING
        return TrendDirection.STABLE

    def get_response_trajectory(self) -> str:
        """Determina traiettoria risposta"""
        if len(self.visits) < 1:
            return "Unknown"

        responses = [v.imaging.response for v in self.visits if v.imaging]

        if not responses:
            return "No imaging data"

        latest = responses[-1]

        if latest == RECISTResponse.CR:
            return "Complete Response Achieved"
        elif latest == RECISTResponse.PR:
            if len(responses) > 1 and responses[-2] == RECISTResponse.SD:
                return "Improving (SD → PR)"
            return "Partial Response Maintained"
        elif latest == RECISTResponse.SD:
            if len(responses) > 1 and responses[-2] == RECISTResponse.PR:
                return "Declining (PR → SD)"
            return "Stable Disease"
        elif latest == RECISTResponse.PD:
            return "Progressive Disease - Treatment Failure"

        return "Indeterminate"

    def analyze(self) -> TimelineAnalysis:
        """
        Esegue analisi completa della timeline.

        Returns:
            TimelineAnalysis con tutti i risultati
        """
        current_therapy = self.baseline.get('current_therapy', 'Unknown')
        if self.visits and self.visits[-1].therapy_changed:
            current_therapy = self.visits[-1].new_therapy

        weeks = 0
        if self.visits:
            weeks = self.visits[-1].week_on_therapy

        # Resistance detection
        resistance_alerts = self.detect_resistance_patterns()

        # Calculate updated risk (simplified)
        base_risk = 50  # Would come from SENTINEL engine
        risk_change = 0

        # Modifiers based on timeline
        if self.get_ldh_trend() == TrendDirection.RISING:
            risk_change += 15
        elif self.get_ldh_trend() == TrendDirection.FALLING:
            risk_change -= 10

        for alert in resistance_alerts:
            if alert.confidence == "High":
                risk_change += 20
            elif alert.confidence == "Medium":
                risk_change += 10

        # Recommendations
        recommendations = []

        if resistance_alerts:
            for alert in resistance_alerts:
                recommendations.append(f"[{alert.urgency}] {alert.recommendation}")

        if self.get_ldh_trend() == TrendDirection.RISING:
            recommendations.append("Consider Elephant Protocol activation (metabolic intervention)")

        if not recommendations:
            recommendations.append("Continue current therapy with standard monitoring")

        return TimelineAnalysis(
            patient_id=self.patient_id,
            total_visits=len(self.visits),
            weeks_on_therapy=weeks,
            current_therapy=current_therapy,
            ldh_trend=self.get_ldh_trend(),
            response_trajectory=self.get_response_trajectory(),
            resistance_alerts=resistance_alerts,
            updated_risk_score=min(100, max(0, base_risk + risk_change)),
            risk_change=risk_change,
            recommendations=recommendations
        )

    def to_dict(self) -> Dict:
        """Converte timeline in dizionario per salvataggio JSON"""
        return {
            "patient_id": self.patient_id,
            "therapy_start_date": self.therapy_start_date.strftime('%Y-%m-%d'),
            "visits": [
                {
                    "visit_id": v.visit_id,
                    "date": v.date,
                    "week_on_therapy": v.week_on_therapy,
                    "therapy_at_visit": v.therapy_at_visit,
                    "ecog_ps": v.ecog_ps,
                    "genetics": asdict(v.genetics) if v.genetics else None,
                    "blood_markers": asdict(v.blood) if v.blood else None,
                    "imaging": {
                        "date": v.imaging.date,
                        "response": v.imaging.response.name,
                        "tumor_change_percent": v.imaging.tumor_change_percent,
                        "new_lesions": v.imaging.new_lesions,
                        "notes": v.imaging.notes
                    } if v.imaging else None,
                    "deltas": v.deltas,
                    "notes": v.notes,
                    "adverse_events": v.adverse_events,
                    "therapy_changed": v.therapy_changed,
                    "new_therapy": v.new_therapy
                }
                for v in self.visits
            ]
        }


# ============================================================================
# TIME TRAVEL SIMULATOR
# ============================================================================

class TimeTravelSimulator:
    """
    Simula scenari what-if per evoluzione paziente.
    """

    @staticmethod
    def simulate_scenario(timeline: PatientTimeline,
                          intervention: str,
                          months_forward: int = 6) -> Dict:
        """
        Simula uno scenario futuro basato su intervento.

        Args:
            timeline: Timeline attuale del paziente
            intervention: Tipo di intervento ("standard", "switch_therapy", "elephant")
            months_forward: Mesi da simulare

        Returns:
            Dict con proiezione
        """
        analysis = timeline.analyze()

        # Base survival estimate
        if analysis.resistance_alerts:
            base_pfs = 3.0  # Mesi se resistenza
        else:
            base_pfs = 12.0  # Mesi standard

        # Modify based on intervention
        if intervention == "standard":
            projected_pfs = base_pfs
            outcome = "Continue current trajectory"

        elif intervention == "switch_therapy":
            # Switching to appropriate therapy
            projected_pfs = base_pfs * 1.8  # 80% improvement
            outcome = "Expected response to new targeted therapy"

        elif intervention == "elephant":
            # Metabolic protocol
            if analysis.ldh_trend == TrendDirection.RISING:
                projected_pfs = base_pfs * 1.5  # 50% improvement
                outcome = "Metabolic control may stabilize disease"
            else:
                projected_pfs = base_pfs * 1.2
                outcome = "Modest benefit expected (LDH not elevated)"

        else:
            projected_pfs = base_pfs
            outcome = "Unknown intervention"

        return {
            "scenario": intervention,
            "months_simulated": months_forward,
            "baseline_pfs_months": base_pfs,
            "projected_pfs_months": round(projected_pfs, 1),
            "improvement_percent": round((projected_pfs - base_pfs) / base_pfs * 100, 0),
            "outcome_description": outcome,
            "confidence": "Medium" if len(timeline.visits) >= 2 else "Low",
            "caveats": [
                "Simulation based on population-level data",
                "Individual response may vary",
                "Requires clinical validation"
            ]
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_patient_state(data: Dict) -> Dict:
    """
    Ritorna lo stato ATTUALE del paziente.
    Fonde baseline con ultima visita se disponibile.

    Logica "Parabrezza": il clinico deve vedere la situazione OGGI,
    non quella di 3 mesi fa.

    Args:
        data: Dizionario completo del paziente (con baseline e visits)

    Returns:
        Dict con stato attuale (baseline + override da ultima visita)
    """
    baseline = data.get('baseline', {})
    visits = data.get('visits', [])

    # Se non ci sono visite, ritorna baseline
    if not visits:
        baseline['_data_source'] = 'baseline_only'
        return baseline

    # Altrimenti, fonde con ultima visita
    latest = visits[-1]
    current = {}

    # Copia profonda del baseline
    for key, value in baseline.items():
        if isinstance(value, dict):
            current[key] = value.copy()
        elif isinstance(value, list):
            current[key] = value.copy()
        else:
            current[key] = value

    # === TERAPIA ATTUALE ===
    if latest.get('therapy_changed') and latest.get('new_therapy'):
        current['current_therapy'] = latest['new_therapy']
    elif latest.get('therapy_at_visit'):
        current['current_therapy'] = latest['therapy_at_visit']

    # === BLOOD MARKERS ATTUALI ===
    if latest.get('blood_markers'):
        latest_blood = latest['blood_markers']
        current_blood = current.get('blood_markers', {}).copy()

        # Sovrascrivi solo i valori presenti nella visita
        for key in ['ldh', 'neutrophils', 'lymphocytes', 'nlr', 'albumin', 'cea']:
            if key in latest_blood and latest_blood[key]:
                current_blood[key] = latest_blood[key]

        current['blood_markers'] = current_blood

    # === GENETICA ATTUALE (merge) ===
    if latest.get('genetics'):
        current_gen = current.get('genetics', {}).copy()
        latest_gen = latest['genetics']

        # Sovrascrivi solo se il nuovo valore non è 'unknown'
        for key, value in latest_gen.items():
            if value and str(value).lower() not in ['unknown', 'none', '']:
                current_gen[key] = value

        current['genetics'] = current_gen

    # === ECOG ATTUALE ===
    if latest.get('ecog_ps') is not None:
        current['ecog_ps'] = latest['ecog_ps']

    # === METADATA ===
    current['_data_source'] = 'merged_with_followup'
    current['_last_visit_id'] = latest.get('visit_id')
    current['_last_visit_date'] = latest.get('date')
    current['_weeks_on_therapy'] = latest.get('week_on_therapy', 0)
    current['_total_visits'] = len(visits)

    # === BASELINE ORIGINALE (per riferimento) ===
    current['_original_baseline'] = {
        'therapy': baseline.get('current_therapy'),
        'ldh': baseline.get('blood_markers', {}).get('ldh'),
        'ecog_ps': baseline.get('ecog_ps')
    }

    return current


def load_patient_timeline(patient_id: str, data_dir: Path) -> Optional[PatientTimeline]:
    """Carica timeline paziente da file JSON"""
    json_path = data_dir / f"{patient_id}.json"

    if not json_path.exists():
        logger.error(f"Patient file not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return PatientTimeline(patient_id, data)


def save_patient_timeline(timeline: PatientTimeline, data_dir: Path):
    """Salva timeline paziente su file JSON"""
    json_path = data_dir / f"{timeline.patient_id}.json"

    # Load existing data
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Update with timeline data
    timeline_dict = timeline.to_dict()
    data['visits'] = timeline_dict['visits']

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"💾 Saved timeline for {timeline.patient_id}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_timeline_summary(timeline: PatientTimeline):
    """Stampa riassunto timeline a console"""
    analysis = timeline.analyze()

    print("\n" + "=" * 70)
    print(f"📅 PATIENT TIMELINE: {analysis.patient_id}")
    print("=" * 70)
    print(f"Total Visits: {analysis.total_visits}")
    print(f"Weeks on Therapy: {analysis.weeks_on_therapy}")
    print(f"Current Therapy: {analysis.current_therapy}")
    print(f"LDH Trend: {analysis.ldh_trend.value}")
    print(f"Response: {analysis.response_trajectory}")
    print(f"Updated Risk: {analysis.updated_risk_score}/100 ({analysis.risk_change:+d} from baseline)")

    if analysis.resistance_alerts:
        print("\n⚠️  RESISTANCE ALERTS:")
        for alert in analysis.resistance_alerts:
            print(f"   [{alert.urgency}] {alert.pattern.value}")
            print(f"      Evidence: {', '.join(alert.evidence)}")
            print(f"      Action: {alert.recommendation}")

    print("\n📋 RECOMMENDATIONS:")
    for rec in analysis.recommendations:
        print(f"   • {rec}")

    print("=" * 70 + "\n")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FOLLOW-UP MODULE - Test")
    print("=" * 70)

    # Create mock baseline
    baseline_data = {
        "baseline": {
            "patient_id": "TEST-FU-001",
            "current_therapy": "Osimertinib",
            "therapy_start_date": "2025-10-01",
            "genetics": {
                "tp53_status": "mutated",
                "egfr_status": "L858R",
                "kras_mutation": "wt"
            },
            "blood_markers": {
                "ldh": 280,
                "neutrophils": 5000,
                "lymphocytes": 1500
            }
        }
    }

    # Initialize timeline
    timeline = PatientTimeline("TEST-FU-001", baseline_data)

    # Add visit 1 (Week 6 - Good response)
    timeline.add_visit({
        "visit_id": "V1",
        "date": "2025-11-12",
        "week_on_therapy": 6,
        "therapy_at_visit": "Osimertinib",
        "ecog_ps": 0,
        "genetics": {
            "source": "ctDNA",
            "tp53_status": "mutated",
            "egfr_status": "L858R",
            "vaf_values": {"EGFR": 0.05, "TP53": 0.08}
        },
        "blood_markers": {
            "ldh": 220,
            "neutrophils": 4500,
            "lymphocytes": 1600
        },
        "imaging": {
            "response": "PR",
            "tumor_change_percent": -35,
            "new_lesions": False
        }
    })

    # Add visit 2 (Week 12 - Signs of resistance)
    timeline.add_visit({
        "visit_id": "V2",
        "date": "2025-12-24",
        "week_on_therapy": 12,
        "therapy_at_visit": "Osimertinib",
        "ecog_ps": 1,
        "genetics": {
            "source": "ctDNA",
            "tp53_status": "mutated",
            "egfr_status": "L858R + T790M",  # Acquired resistance!
            "met_cn": 6.5,  # MET amplification!
            "vaf_values": {"EGFR": 0.15, "TP53": 0.18}
        },
        "blood_markers": {
            "ldh": 380,
            "neutrophils": 6000,
            "lymphocytes": 1200
        },
        "imaging": {
            "response": "SD",
            "tumor_change_percent": -5,
            "new_lesions": False
        }
    })

    # Print summary
    print_timeline_summary(timeline)

    # Time Travel simulation
    print("\n🕐 TIME TRAVEL SIMULATION:")

    scenarios = ["standard", "switch_therapy", "elephant"]
    for scenario in scenarios:
        result = TimeTravelSimulator.simulate_scenario(timeline, scenario)
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"    Projected PFS: {result['projected_pfs_months']} months ({result['improvement_percent']:+.0f}%)")
        print(f"    Outcome: {result['outcome_description']}")

    print("\n✅ Test completed")