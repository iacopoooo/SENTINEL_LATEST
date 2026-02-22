"""
SENTINEL TRIAL - FOLLOW-UP MODULE v2.0
======================================
Modulo per gestione visite di follow-up clinicamente complete.

Features:
- Tracking longitudinale completo
- Peso, Albumina, CEA
- Eventi avversi con grading CTCAE
- Compliance terapeutica
- Motivo cambio terapia
- Nuove lesioni con sedi
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
import datetime


# =============================================================================
# ENUMS
# =============================================================================

class RECISTResponse(Enum):
    """RECIST 1.1 Response Categories"""
    CR = "Complete Response"
    PR = "Partial Response"
    SD = "Stable Disease"
    PD = "Progressive Disease"
    NE = "Not Evaluable"


class TrendDirection(Enum):
    """Trend direction for markers"""
    RISING = "↑ RISING"
    STABLE = "→ STABLE"
    FALLING = "↓ FALLING"


class AdverseEventGrade(Enum):
    """CTCAE v5.0 Grading"""
    NONE = "None"
    G1 = "Grade 1 - Mild"
    G2 = "Grade 2 - Moderate"
    G3 = "Grade 3 - Severe"
    G4 = "Grade 4 - Life-threatening"
    G5 = "Grade 5 - Death"


class TherapyChangeReason(Enum):
    """Motivi per cambio terapia"""
    NONE = "No change"
    PROGRESSION = "Disease Progression"
    TOXICITY = "Unacceptable Toxicity"
    PATIENT_CHOICE = "Patient Choice"
    PHYSICIAN_CHOICE = "Physician Decision"
    PROTOCOL = "Per Protocol"
    COMPLETED = "Treatment Completed"
    OTHER = "Other"


class ComplianceLevel(Enum):
    """Livelli di compliance"""
    FULL = "100% - Full compliance"
    HIGH = "80-99% - High compliance"
    MODERATE = "50-79% - Moderate compliance"
    LOW = "<50% - Low compliance"
    UNKNOWN = "Unknown"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class AdverseEvent:
    """Singolo evento avverso"""
    term: str  # Es: "Fatigue", "Rash", "Diarrhea"
    grade: AdverseEventGrade = AdverseEventGrade.NONE
    onset_date: str = ""
    resolved: bool = False
    action_taken: str = ""  # "None", "Dose reduced", "Drug held", "Drug discontinued"
    
    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "grade": self.grade.name,
            "onset_date": self.onset_date,
            "resolved": self.resolved,
            "action_taken": self.action_taken
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AdverseEvent':
        # Gestisce valori legacy per grade
        grade_str = data.get('grade', 'NONE')
        try:
            grade = AdverseEventGrade[grade_str]
        except KeyError:
            grade = AdverseEventGrade.NONE

        return cls(
            term=data.get('term', ''),
            grade=grade,
            onset_date=data.get('onset_date', ''),
            resolved=data.get('resolved', False),
            action_taken=data.get('action_taken', '')
        )


@dataclass
class BloodMarkers:
    """Marker ematici - versione estesa"""
    ldh: float = 0.0
    neutrophils: float = 0.0
    lymphocytes: float = 0.0
    nlr: float = 0.0
    albumin: float = 0.0  # g/dL - NUOVO
    cea: float = 0.0  # ng/mL - NUOVO (opzionale, per adenocarcinoma)
    hemoglobin: float = 0.0  # g/dL - NUOVO (opzionale)
    platelets: float = 0.0  # x10^9/L - NUOVO (opzionale)

    def calculate_nlr(self):
        """Calcola NLR automaticamente"""
        if self.lymphocytes > 0:
            self.nlr = round(self.neutrophils / self.lymphocytes, 2)

    def to_dict(self) -> Dict:
        return {
            "ldh": self.ldh,
            "neutrophils": self.neutrophils,
            "lymphocytes": self.lymphocytes,
            "nlr": self.nlr,
            "albumin": self.albumin,
            "cea": self.cea,
            "hemoglobin": self.hemoglobin,
            "platelets": self.platelets
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BloodMarkers':
        markers = cls(
            ldh=float(data.get('ldh', 0)),
            neutrophils=float(data.get('neutrophils', 0)),
            lymphocytes=float(data.get('lymphocytes', 0)),
            nlr=float(data.get('nlr', 0)),
            albumin=float(data.get('albumin', 0)),
            cea=float(data.get('cea', 0)),
            hemoglobin=float(data.get('hemoglobin', 0)),
            platelets=float(data.get('platelets', 0))
        )
        if markers.nlr == 0:
            markers.calculate_nlr()
        return markers


@dataclass
class ImagingResult:
    """Risultato imaging - versione estesa"""
    date: str = ""
    response: RECISTResponse = RECISTResponse.NE
    tumor_change_percent: float = 0.0
    new_lesions: bool = False
    new_lesion_sites: List[str] = field(default_factory=list)  # NUOVO: sedi nuove lesioni
    target_lesion_sum: float = 0.0  # mm - somma diametri lesioni target
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "date": self.date,
            "response": self.response.name,
            "tumor_change_percent": self.tumor_change_percent,
            "new_lesions": self.new_lesions,
            "new_lesion_sites": self.new_lesion_sites,
            "target_lesion_sum": self.target_lesion_sum,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ImagingResult':
        return cls(
            date=data.get('date', ''),
            response=RECISTResponse[data.get('response', 'NE')],
            tumor_change_percent=float(data.get('tumor_change_percent', 0)),
            new_lesions=data.get('new_lesions', False),
            new_lesion_sites=data.get('new_lesion_sites', []),
            target_lesion_sum=float(data.get('target_lesion_sum', 0)),
            notes=data.get('notes', '')
        )


@dataclass
class GeneticSnapshot:
    """Snapshot genetico da ctDNA o rebiopsy"""
    source: str = "ctDNA"  # "ctDNA", "tissue_rebiopsy", "liquid_biopsy"
    date: str = ""

    # Mutazioni rilevate
    tp53_status: str = "unknown"
    kras_mutation: str = "unknown"
    egfr_status: str = "unknown"
    met_status: str = "unknown"
    met_cn: float = 0.0  # Copy number
    stk11_status: str = "unknown"
    keap1_status: str = "unknown"
    rb1_status: str = "unknown"

    # Nuove mutazioni di resistenza
    t790m_detected: bool = False
    c797s_detected: bool = False
    met_amplification_acquired: bool = False

    # VAF values (Variant Allele Frequency) - per tracking clonale
    vaf_values: Dict[str, float] = field(default_factory=dict)

    # Nuove mutazioni rispetto a baseline
    new_mutations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "date": self.date,
            "tp53_status": self.tp53_status,
            "kras_mutation": self.kras_mutation,
            "egfr_status": self.egfr_status,
            "met_status": self.met_status,
            "met_cn": self.met_cn,
            "stk11_status": self.stk11_status,
            "keap1_status": self.keap1_status,
            "rb1_status": self.rb1_status,
            "t790m_detected": self.t790m_detected,
            "c797s_detected": self.c797s_detected,
            "met_amplification_acquired": self.met_amplification_acquired,
            "vaf_values": self.vaf_values,
            "new_mutations": self.new_mutations
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneticSnapshot':
        return cls(
            source=data.get('source', 'ctDNA'),
            date=data.get('date', ''),
            tp53_status=data.get('tp53_status', 'unknown'),
            kras_mutation=data.get('kras_mutation', 'unknown'),
            egfr_status=data.get('egfr_status', 'unknown'),
            met_status=data.get('met_status', 'unknown'),
            met_cn=float(data.get('met_cn', 0)),
            stk11_status=data.get('stk11_status', 'unknown'),
            keap1_status=data.get('keap1_status', 'unknown'),
            rb1_status=data.get('rb1_status', 'unknown'),
            t790m_detected=data.get('t790m_detected', False),
            c797s_detected=data.get('c797s_detected', False),
            met_amplification_acquired=data.get('met_amplification_acquired', False),
            vaf_values=data.get('vaf_values', {}),
            new_mutations=data.get('new_mutations', [])
        )


@dataclass
class TherapyInfo:
    """Informazioni sulla terapia - NUOVO"""
    current_therapy: str = ""
    therapy_changed: bool = False
    new_therapy: str = ""
    change_reason: TherapyChangeReason = TherapyChangeReason.NONE
    change_reason_detail: str = ""  # Dettagli aggiuntivi
    dose_reduced: bool = False
    dose_reduction_detail: str = ""  # Es: "80mg -> 40mg"
    compliance: ComplianceLevel = ComplianceLevel.UNKNOWN

    def to_dict(self) -> Dict:
        return {
            "current_therapy": self.current_therapy,
            "therapy_changed": self.therapy_changed,
            "new_therapy": self.new_therapy,
            "change_reason": self.change_reason.name,
            "change_reason_detail": self.change_reason_detail,
            "dose_reduced": self.dose_reduced,
            "dose_reduction_detail": self.dose_reduction_detail,
            "compliance": self.compliance.name
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TherapyInfo':
        # Gestisce valori legacy per change_reason
        change_reason_str = data.get('change_reason', 'NONE')
        try:
            change_reason = TherapyChangeReason[change_reason_str]
        except KeyError:
            # Valore legacy non riconosciuto, usa OTHER
            change_reason = TherapyChangeReason.OTHER

        # Gestisce valori legacy per compliance
        compliance_str = data.get('compliance', 'UNKNOWN')
        try:
            compliance = ComplianceLevel[compliance_str]
        except KeyError:
            compliance = ComplianceLevel.UNKNOWN

        return cls(
            current_therapy=data.get('current_therapy', ''),
            therapy_changed=data.get('therapy_changed', False),
            new_therapy=data.get('new_therapy', ''),
            change_reason=change_reason,
            change_reason_detail=data.get('change_reason_detail', str(change_reason_str) if change_reason == TherapyChangeReason.OTHER else ''),
            dose_reduced=data.get('dose_reduced', False),
            dose_reduction_detail=data.get('dose_reduction_detail', ''),
            compliance=compliance
        )


@dataclass
class ClinicalStatus:
    """Stato clinico del paziente - NUOVO"""
    ecog_ps: int = 1
    weight_kg: float = 0.0  # NUOVO
    weight_change_kg: float = 0.0  # Delta rispetto a visita precedente
    height_cm: float = 0.0  # Per calcolo BSA se necessario
    bsa: float = 0.0  # Body Surface Area

    def calculate_bsa(self):
        """Calcola BSA con formula Mosteller"""
        if self.weight_kg > 0 and self.height_cm > 0:
            self.bsa = round(((self.weight_kg * self.height_cm) / 3600) ** 0.5, 2)

    def to_dict(self) -> Dict:
        return {
            "ecog_ps": self.ecog_ps,
            "weight_kg": self.weight_kg,
            "weight_change_kg": self.weight_change_kg,
            "height_cm": self.height_cm,
            "bsa": self.bsa
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalStatus':
        status = cls(
            ecog_ps=int(data.get('ecog_ps', 1)),
            weight_kg=float(data.get('weight_kg', 0)),
            weight_change_kg=float(data.get('weight_change_kg', 0)),
            height_cm=float(data.get('height_cm', 0)),
            bsa=float(data.get('bsa', 0))
        )
        if status.bsa == 0:
            status.calculate_bsa()
        return status


@dataclass
class Visit:
    """Singola visita di follow-up - VERSIONE COMPLETA"""
    visit_id: str
    date: str
    week_on_therapy: int

    # Stato clinico
    clinical_status: ClinicalStatus = field(default_factory=ClinicalStatus)

    # Eventi avversi
    adverse_events: List[AdverseEvent] = field(default_factory=list)

    # Esami ematici
    blood_markers: BloodMarkers = field(default_factory=BloodMarkers)

    # Imaging
    imaging: Optional[ImagingResult] = None

    # Genetica (opzionale)
    genetics: Optional[GeneticSnapshot] = None

    # Terapia
    therapy_info: TherapyInfo = field(default_factory=TherapyInfo)

    # Deltas calcolati (rispetto a visita precedente)
    deltas: Dict = field(default_factory=dict)

    # Note cliniche
    notes: str = ""

    # === CAMPI LEGACY per compatibilità ===
    @property
    def ecog_ps(self) -> int:
        return self.clinical_status.ecog_ps

    @property
    def therapy_changed(self) -> bool:
        return self.therapy_info.therapy_changed

    @property
    def new_therapy(self) -> str:
        return self.therapy_info.new_therapy

    @property
    def therapy_at_visit(self) -> str:
        return self.therapy_info.current_therapy

    @property
    def blood(self) -> BloodMarkers:
        return self.blood_markers

    def to_dict(self) -> Dict:
        return {
            "visit_id": self.visit_id,
            "date": self.date,
            "week_on_therapy": self.week_on_therapy,
            "clinical_status": self.clinical_status.to_dict(),
            "adverse_events": [ae.to_dict() for ae in self.adverse_events],
            "blood_markers": self.blood_markers.to_dict(),
            "imaging": self.imaging.to_dict() if self.imaging else None,
            "genetics": self.genetics.to_dict() if self.genetics else None,
            "therapy_info": self.therapy_info.to_dict(),
            "deltas": self.deltas,
            "notes": self.notes,
            # Legacy fields per compatibilità
            "ecog_ps": self.clinical_status.ecog_ps,
            "therapy_changed": self.therapy_info.therapy_changed,
            "new_therapy": self.therapy_info.new_therapy,
            "therapy_at_visit": self.therapy_info.current_therapy
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Visit':
        # Gestisci sia nuovo formato che legacy
        if 'clinical_status' in data:
            clinical_status = ClinicalStatus.from_dict(data['clinical_status'])
        else:
            # Legacy format
            clinical_status = ClinicalStatus(ecog_ps=int(data.get('ecog_ps', 1)))

        if 'therapy_info' in data:
            therapy_info = TherapyInfo.from_dict(data['therapy_info'])
        else:
            # Legacy format
            therapy_info = TherapyInfo(
                current_therapy=data.get('therapy_at_visit', ''),
                therapy_changed=data.get('therapy_changed', False),
                new_therapy=data.get('new_therapy', '')
            )

        adverse_events = []
        if 'adverse_events' in data:
            for ae_data in data['adverse_events']:
                if isinstance(ae_data, dict):
                    adverse_events.append(AdverseEvent.from_dict(ae_data))
                elif isinstance(ae_data, str):
                    # Legacy: lista di stringhe
                    adverse_events.append(AdverseEvent(term=ae_data))

        blood_data = data.get('blood_markers', {})
        if not blood_data:
            # Prova legacy format
            blood_data = {
                'ldh': data.get('ldh', 0),
                'neutrophils': data.get('neutrophils', 0),
                'lymphocytes': data.get('lymphocytes', 0)
            }

        return cls(
            visit_id=data.get('visit_id', ''),
            date=data.get('date', ''),
            week_on_therapy=int(data.get('week_on_therapy', 0)),
            clinical_status=clinical_status,
            adverse_events=adverse_events,
            blood_markers=BloodMarkers.from_dict(blood_data),
            imaging=ImagingResult.from_dict(data['imaging']) if data.get('imaging') else None,
            genetics=GeneticSnapshot.from_dict(data['genetics']) if data.get('genetics') else None,
            therapy_info=therapy_info,
            deltas=data.get('deltas', {}),
            notes=data.get('notes', '')
        )


# =============================================================================
# RESISTANCE DETECTION
# =============================================================================

class ResistancePattern(Enum):
    """Pattern di resistenza riconosciuti"""
    T790M_ACQUIRED = "T790M Acquired Resistance"
    C797S_ACQUIRED = "C797S Acquired (Osimertinib Resistance)"
    MET_AMPLIFICATION = "MET Amplification"
    SCLC_TRANSFORMATION = "Small Cell Transformation Risk"
    EMERGENT_CLONE = "Emergent Resistant Clone"
    OCCULT_PROGRESSION = "Occult Progression (LDH rising, imaging stable)"


@dataclass
class ResistanceAlert:
    """Alert per pattern di resistenza"""
    pattern: ResistancePattern
    confidence: float  # 0-100
    evidence: List[str]
    recommendation: str
    urgency: str  # "immediate", "next_visit", "monitor"

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "urgency": self.urgency
        }


# =============================================================================
# PATIENT TIMELINE
# =============================================================================

@dataclass
class TimelineAnalysis:
    """Risultato analisi timeline"""
    total_visits: int
    weeks_on_therapy: int
    ldh_trend: TrendDirection
    response_trajectory: str
    resistance_alerts: List[ResistanceAlert]
    recommendations: List[str]
    weight_trend: str = ""  # NUOVO
    compliance_average: str = ""  # NUOVO


class PatientTimeline:
    """Gestisce la timeline longitudinale del paziente"""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.visits: List[Visit] = []
        self.baseline_data: Dict = {}

    def add_visit(self, visit: Visit, previous_visit_data: Optional[Dict] = None):
        """Aggiunge una visita e calcola i delta"""
        # Calcola delta rispetto a visita precedente o baseline
        if self.visits:
            prev = self.visits[-1]
            prev_blood = prev.blood_markers
            prev_clinical = prev.clinical_status
        elif previous_visit_data:
            prev_blood = BloodMarkers.from_dict(previous_visit_data.get('blood_markers', {}))
            prev_clinical = ClinicalStatus(ecog_ps=previous_visit_data.get('ecog_ps', 1))
        else:
            prev_blood = BloodMarkers()
            prev_clinical = ClinicalStatus()

        # Calcola deltas
        visit.deltas = self._calculate_deltas(visit, prev_blood, prev_clinical)

        # Calcola weight change
        if self.visits and visit.clinical_status.weight_kg > 0:
            prev_weight = self.visits[-1].clinical_status.weight_kg
            if prev_weight > 0:
                visit.clinical_status.weight_change_kg = round(
                    visit.clinical_status.weight_kg - prev_weight, 1
                )

        self.visits.append(visit)

    def _calculate_deltas(self, visit: Visit, prev_blood: BloodMarkers,
                          prev_clinical: ClinicalStatus) -> Dict:
        """Calcola i delta tra visite"""
        deltas = {}

        # LDH
        if prev_blood.ldh > 0:
            deltas['ldh_change'] = round(visit.blood_markers.ldh - prev_blood.ldh, 1)
            deltas['ldh_trend'] = self._get_trend(prev_blood.ldh, visit.blood_markers.ldh)

        # NLR
        if prev_blood.nlr > 0:
            deltas['nlr_change'] = round(visit.blood_markers.nlr - prev_blood.nlr, 2)

        # ECOG
        deltas['ecog_change'] = visit.clinical_status.ecog_ps - prev_clinical.ecog_ps

        # Weight
        if visit.clinical_status.weight_kg > 0 and prev_clinical.weight_kg > 0:
            deltas['weight_change'] = round(
                visit.clinical_status.weight_kg - prev_clinical.weight_kg, 1
            )

        # Albumin
        if visit.blood_markers.albumin > 0 and prev_blood.albumin > 0:
            deltas['albumin_change'] = round(
                visit.blood_markers.albumin - prev_blood.albumin, 2
            )

        # Imaging
        if visit.imaging:
            deltas['imaging_response'] = visit.imaging.response.name
            if visit.imaging.new_lesions:
                deltas['new_lesions'] = True
                deltas['new_lesion_sites'] = visit.imaging.new_lesion_sites

        # Nuove mutazioni
        if visit.genetics and visit.genetics.new_mutations:
            deltas['new_mutations'] = visit.genetics.new_mutations

        return deltas

    def _get_trend(self, old_val: float, new_val: float, threshold: float = 0.1) -> str:
        """Determina trend basato su variazione percentuale"""
        if old_val == 0:
            return TrendDirection.STABLE.value

        pct_change = (new_val - old_val) / old_val

        if pct_change > threshold:
            return TrendDirection.RISING.value
        elif pct_change < -threshold:
            return TrendDirection.FALLING.value
        else:
            return TrendDirection.STABLE.value

    def get_ldh_trend(self) -> TrendDirection:
        """Calcola trend LDH dagli ultimi 3 valori"""
        if len(self.visits) < 2:
            return TrendDirection.STABLE

        recent_ldh = [v.blood_markers.ldh for v in self.visits[-3:]]

        if all(recent_ldh[i] < recent_ldh[i-1] for i in range(1, len(recent_ldh))):
            return TrendDirection.FALLING
        elif all(recent_ldh[i] > recent_ldh[i-1] for i in range(1, len(recent_ldh))):
            return TrendDirection.RISING
        else:
            return TrendDirection.STABLE

    def get_weight_trend(self) -> str:
        """Calcola trend peso"""
        weights = [v.clinical_status.weight_kg for v in self.visits if v.clinical_status.weight_kg > 0]

        if len(weights) < 2:
            return "Insufficient data"

        total_change = weights[-1] - weights[0]

        if total_change < -5:
            return f"↓ Significant loss ({total_change:.1f} kg)"
        elif total_change < -2:
            return f"↓ Mild loss ({total_change:.1f} kg)"
        elif total_change > 2:
            return f"↑ Weight gain (+{total_change:.1f} kg)"
        else:
            return "→ Stable"

    def detect_resistance_patterns(self) -> List[ResistanceAlert]:
        """Rileva pattern di resistenza"""
        alerts = []

        if not self.visits:
            return alerts

        latest = self.visits[-1]

        # 1. T790M acquired
        if latest.genetics and latest.genetics.t790m_detected:
            alerts.append(ResistanceAlert(
                pattern=ResistancePattern.T790M_ACQUIRED,
                confidence=95,
                evidence=["T790M detected in ctDNA/rebiopsy"],
                recommendation="Switch to Osimertinib (3rd gen EGFR-TKI)",
                urgency="immediate"
            ))

        # 2. C797S acquired
        if latest.genetics and latest.genetics.c797s_detected:
            alerts.append(ResistanceAlert(
                pattern=ResistancePattern.C797S_ACQUIRED,
                confidence=90,
                evidence=["C797S mutation detected - Osimertinib resistance"],
                recommendation="Consider combination therapy or clinical trial",
                urgency="immediate"
            ))

        # 3. MET amplification
        if latest.genetics:
            met_cn = latest.genetics.met_cn
            if met_cn >= 5.0 or latest.genetics.met_amplification_acquired:
                alerts.append(ResistanceAlert(
                    pattern=ResistancePattern.MET_AMPLIFICATION,
                    confidence=85,
                    evidence=[f"MET CN={met_cn}" if met_cn else "MET amplification acquired"],
                    recommendation="Add MET inhibitor (Capmatinib/Tepotinib)",
                    urgency="immediate"
                ))

        # 4. SCLC transformation risk
        if latest.genetics:
            if (latest.genetics.tp53_status == 'mutated' and
                latest.genetics.rb1_status == 'mutated'):
                alerts.append(ResistanceAlert(
                    pattern=ResistancePattern.SCLC_TRANSFORMATION,
                    confidence=70,
                    evidence=["TP53 + RB1 co-mutation detected"],
                    recommendation="Consider tissue rebiopsy to rule out SCLC transformation",
                    urgency="next_visit"
                ))

        # 5. Occult progression
        ldh_trend = self.get_ldh_trend()
        if (ldh_trend == TrendDirection.RISING and
            latest.imaging and
            latest.imaging.response in [RECISTResponse.SD, RECISTResponse.PR]):
            alerts.append(ResistanceAlert(
                pattern=ResistancePattern.OCCULT_PROGRESSION,
                confidence=60,
                evidence=["LDH rising with stable/responding imaging"],
                recommendation="Increase monitoring frequency, consider PET-CT",
                urgency="next_visit"
            ))

        # 6. Emergent clone (VAF rising)
        if latest.genetics and latest.genetics.vaf_values:
            for gene, vaf in latest.genetics.vaf_values.items():
                # Check if VAF increased significantly from previous
                if len(self.visits) >= 2:
                    prev_genetics = self.visits[-2].genetics
                    if prev_genetics and prev_genetics.vaf_values:
                        prev_vaf = prev_genetics.vaf_values.get(gene, 0)
                        if vaf > prev_vaf * 1.5 and vaf > 5:  # >50% increase and >5% VAF
                            alerts.append(ResistanceAlert(
                                pattern=ResistancePattern.EMERGENT_CLONE,
                                confidence=65,
                                evidence=[f"{gene} VAF increased: {prev_vaf}% -> {vaf}%"],
                                recommendation="Consider rebiopsy and resistance profiling",
                                urgency="next_visit"
                            ))
                            break

        return alerts

    def analyze(self) -> TimelineAnalysis:
        """Analizza la timeline completa"""
        if not self.visits:
            return TimelineAnalysis(
                total_visits=0,
                weeks_on_therapy=0,
                ldh_trend=TrendDirection.STABLE,
                response_trajectory="No data",
                resistance_alerts=[],
                recommendations=["Schedule first follow-up visit"]
            )

        latest = self.visits[-1]

        # Response trajectory
        responses = [v.imaging.response for v in self.visits if v.imaging]
        if responses:
            if responses[-1] == RECISTResponse.CR:
                trajectory = "Complete Response Achieved"
            elif responses[-1] == RECISTResponse.PR:
                trajectory = "Partial Response Maintained"
            elif responses[-1] == RECISTResponse.PD:
                trajectory = "Progressive Disease"
            elif responses[-1] == RECISTResponse.SD:
                trajectory = "Stable Disease"
            else:
                trajectory = "Not Evaluable"
        else:
            trajectory = "No imaging data"

        # Recommendations
        recommendations = []
        resistance_alerts = self.detect_resistance_patterns()

        if resistance_alerts:
            for alert in resistance_alerts:
                if alert.urgency == "immediate":
                    recommendations.append(f"[URGENT] {alert.recommendation}")
                else:
                    recommendations.append(alert.recommendation)
        else:
            recommendations.append("Continue current therapy with standard monitoring")

        # Weight trend
        weight_trend = self.get_weight_trend()
        if "loss" in weight_trend.lower():
            recommendations.append("Consider nutritional assessment")

        # Compliance
        compliances = [v.therapy_info.compliance for v in self.visits
                      if v.therapy_info.compliance != ComplianceLevel.UNKNOWN]
        if compliances:
            if ComplianceLevel.LOW in compliances or ComplianceLevel.MODERATE in compliances:
                recommendations.append("Address compliance issues")

        return TimelineAnalysis(
            total_visits=len(self.visits),
            weeks_on_therapy=latest.week_on_therapy,
            ldh_trend=self.get_ldh_trend(),
            response_trajectory=trajectory,
            resistance_alerts=resistance_alerts,
            recommendations=recommendations,
            weight_trend=weight_trend,
            compliance_average=compliances[-1].value if compliances else "Unknown"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_patient_state(data: Dict) -> Dict:
    """
    Ritorna lo stato ATTUALE del paziente.
    Fonde baseline con ultima visita se disponibile.
    """
    baseline = data.get('baseline', {})
    visits = data.get('visits', [])

    if not visits:
        baseline['_data_source'] = 'baseline_only'
        return baseline

    latest = visits[-1]
    current = {}

    # Deep copy baseline
    for key, value in baseline.items():
        if isinstance(value, dict):
            current[key] = value.copy()
        elif isinstance(value, list):
            current[key] = value.copy()
        else:
            current[key] = value

    # Override con dati ultima visita

    # Terapia
    if latest.get('therapy_info'):
        therapy_info = latest['therapy_info']
        if therapy_info.get('therapy_changed') and therapy_info.get('new_therapy'):
            current['current_therapy'] = therapy_info['new_therapy']
        elif therapy_info.get('current_therapy'):
            current['current_therapy'] = therapy_info['current_therapy']
    elif latest.get('therapy_changed') and latest.get('new_therapy'):
        # Legacy format
        current['current_therapy'] = latest['new_therapy']
    elif latest.get('therapy_at_visit'):
        current['current_therapy'] = latest['therapy_at_visit']

    # Blood markers
    blood_data = latest.get('blood_markers', {})
    if blood_data:
        current_blood = current.get('blood_markers', {}).copy()
        for key in ['ldh', 'neutrophils', 'lymphocytes', 'nlr', 'albumin', 'cea']:
            if key in blood_data and blood_data[key]:
                current_blood[key] = blood_data[key]
        current['blood_markers'] = current_blood

    # Clinical status
    if latest.get('clinical_status'):
        current['ecog_ps'] = latest['clinical_status'].get('ecog_ps', current.get('ecog_ps', 1))
        if latest['clinical_status'].get('weight_kg'):
            current['weight_kg'] = latest['clinical_status']['weight_kg']
    elif latest.get('ecog_ps') is not None:
        current['ecog_ps'] = latest['ecog_ps']

    # Genetics
    if latest.get('genetics'):
        current_gen = current.get('genetics', {}).copy()
        for key, value in latest['genetics'].items():
            if value and str(value).lower() not in ['unknown', 'none', '']:
                current_gen[key] = value
        current['genetics'] = current_gen

    # Metadata
    current['_data_source'] = 'merged_with_followup'
    current['_last_visit_id'] = latest.get('visit_id')
    current['_last_visit_date'] = latest.get('date')
    current['_weeks_on_therapy'] = latest.get('week_on_therapy', 0)
    current['_total_visits'] = len(visits)

    return current


def load_patient_timeline(patient_id: str, data_dir: Path) -> Optional[PatientTimeline]:
    """Carica timeline paziente da file JSON"""
    json_path = data_dir / f"{patient_id}.json"

    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    timeline = PatientTimeline(patient_id)
    timeline.baseline_data = data.get('baseline', {})

    for visit_data in data.get('visits', []):
        visit = Visit.from_dict(visit_data)
        timeline.visits.append(visit)

    return timeline


def save_patient_timeline(timeline: PatientTimeline, data_dir: Path):
    """Salva timeline paziente su file JSON"""
    json_path = data_dir / f"{timeline.patient_id}.json"

    # Carica dati esistenti
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {'baseline': timeline.baseline_data}

    # Aggiorna visite
    data['visits'] = [v.to_dict() for v in timeline.visits]

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def print_timeline_summary(timeline: PatientTimeline):
    """Stampa summary della timeline"""
    analysis = timeline.analyze()

    print(f"\n{'='*60}")
    print(f"📊 TIMELINE SUMMARY - Patient {timeline.patient_id}")
    print(f"{'='*60}")
    print(f"Total Visits: {analysis.total_visits}")
    print(f"Weeks on Therapy: {analysis.weeks_on_therapy}")
    print(f"LDH Trend: {analysis.ldh_trend.value}")
    print(f"Weight Trend: {analysis.weight_trend}")
    print(f"Response: {analysis.response_trajectory}")
    print(f"Compliance: {analysis.compliance_average}")

    if analysis.resistance_alerts:
        print(f"\n⚠️ RESISTANCE ALERTS:")
        for alert in analysis.resistance_alerts:
            print(f"  [{alert.urgency.upper()}] {alert.pattern.value}")
            print(f"    Confidence: {alert.confidence}%")
            print(f"    Action: {alert.recommendation}")

    print(f"\n📋 Recommendations:")
    for rec in analysis.recommendations:
        print(f"  • {rec}")

    print(f"{'='*60}\n")