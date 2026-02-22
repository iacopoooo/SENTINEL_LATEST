"""
SENTINEL v18.1 - Adaptive Therapy Optimizer
============================================
Implementa strategie di terapia adattiva basate su:
- Teoria dei giochi evolutiva (Gatenby et al.)
- Competizione clonale
- Drug holidays strategici
- Metronomic scheduling

Obiettivo: Prolungare PFS sfruttando la fitness cost della resistenza.

Principio base:
- Cloni resistenti hanno un "costo" metabolico quando il farmaco è assente
- Drug holidays permettono ai cloni sensibili di ri-espandersi
- Questo mantiene la competizione e ritarda la dominanza dei resistenti

References:
- Gatenby RA et al. "Adaptive Therapy" Nat Rev Cancer 2019
- Zhang J et al. "Integrating evolutionary dynamics into treatment" Nat Commun 2017
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class TherapyPhase(Enum):
    """Fasi del ciclo adattivo"""
    INDUCTION = "INDUCTION"  # Dose piena, ridurre tumor burden
    MAINTENANCE = "MAINTENANCE"  # Dose ridotta, mantenere controllo
    DRUG_HOLIDAY = "DRUG_HOLIDAY"  # Pausa, permettere competizione clonale
    INTENSIFICATION = "INTENSIFICATION"  # Dose aumentata per progression
    METRONOMIC = "METRONOMIC"  # Low-dose continuo


class ResponseStatus(Enum):
    """Status risposta tumorale"""
    DEEP_RESPONSE = "DEEP_RESPONSE"  # >50% reduction
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"  # 30-50% reduction
    STABLE = "STABLE"  # <30% change
    MINOR_PROGRESSION = "MINOR_PROGRESSION"  # 10-20% increase
    PROGRESSION = "PROGRESSION"  # >20% increase


@dataclass
class AdaptiveCycle:
    """Singolo ciclo di terapia adattiva"""
    phase: TherapyPhase
    duration_weeks: int
    drug: str
    dose: str
    rationale: str
    monitoring: List[str]
    transition_triggers: List[str]  # Quando passare alla fase successiva


@dataclass
class AdaptiveProtocol:
    """Protocollo completo di terapia adattiva"""
    patient_id: str
    current_therapy: str
    current_phase: TherapyPhase

    # Cicli pianificati
    cycles: List[AdaptiveCycle]

    # Parametri paziente-specifici
    baseline_tumor_burden: float  # Sum of target lesions (mm)
    current_tumor_burden: float
    response_status: ResponseStatus

    # Soglie personalizzate
    holiday_threshold: float  # Tumor burden sotto cui è safe holiday
    resume_threshold: float  # Tumor burden sopra cui riprendere

    # Metriche
    estimated_pfs_standard: float  # PFS con terapia continua
    estimated_pfs_adaptive: float  # PFS con terapia adattiva
    benefit_months: float

    # Eligibilità
    eligible_for_adaptive: bool
    ineligibility_reason: Optional[str]

    # Raccomandazioni
    immediate_action: str
    next_assessment_weeks: int

    # Metadata
    generated_at: str


class AdaptiveTherapyOptimizer:
    """
    Motore per ottimizzazione terapia adattiva.

    Strategie implementate:
    1. Standard Adaptive: Cycles of full dose -> holiday -> full dose
    2. Metronomic: Continuous low-dose
    3. Dose-Dense: Intensified cycles for aggressive disease
    """

    # Criteri eligibilità per adaptive therapy
    ELIGIBILITY_CRITERIA = {
        'min_response': 30,  # Almeno 30% reduction per holiday
        'max_ecog': 2,  # ECOG ≤2
        'min_pfs_expected': 6,  # Almeno 6 mesi PFS atteso
        'stable_duration_weeks': 12,  # Almeno 12 settimane di stabilità
    }

    # Durate standard cicli (settimane)
    CYCLE_DURATIONS = {
        TherapyPhase.INDUCTION: 12,
        TherapyPhase.MAINTENANCE: 8,
        TherapyPhase.DRUG_HOLIDAY: 4,
        TherapyPhase.INTENSIFICATION: 8,
        TherapyPhase.METRONOMIC: 12,
    }

    # Beneficio atteso (moltiplicatore PFS)
    ADAPTIVE_BENEFIT = {
        'deep_response': 1.4,  # 40% longer PFS
        'partial_response': 1.25,  # 25% longer PFS
        'stable': 1.15,  # 15% longer PFS
    }

    def __init__(self):
        self.generated_at = datetime.now().isoformat()

    def assess_eligibility(self, patient_data: Dict,
                           tumor_response: float,
                           weeks_stable: int) -> Tuple[bool, Optional[str]]:
        """
        Valuta se il paziente è eleggibile per terapia adattiva.

        Args:
            patient_data: Dati paziente
            tumor_response: % change dal baseline (negativo = riduzione)
            weeks_stable: Settimane di stabilità

        Returns:
            Tuple[eligible, reason_if_not]
        """
        base = patient_data.get('baseline', patient_data)

        # Check ECOG
        ecog = int(base.get('ecog_ps', 2))
        if ecog > self.ELIGIBILITY_CRITERIA['max_ecog']:
            return False, f"ECOG {ecog} too high (max {self.ELIGIBILITY_CRITERIA['max_ecog']})"

        # Check response
        if tumor_response > -self.ELIGIBILITY_CRITERIA['min_response']:
            return False, f"Insufficient response ({tumor_response:.0f}%, need ≤-{self.ELIGIBILITY_CRITERIA['min_response']}%)"

        # Check stability
        if weeks_stable < self.ELIGIBILITY_CRITERIA['stable_duration_weeks']:
            return False, f"Insufficient stability ({weeks_stable}w, need ≥{self.ELIGIBILITY_CRITERIA['stable_duration_weeks']}w)"

        # Check specific contraindications
        genetics = base.get('genetics', {})

        # TP53+RB1 = rapid progression, not good for holidays
        tp53 = str(genetics.get('tp53_status', '')).lower()
        rb1 = str(genetics.get('rb1_status', '')).lower()
        if tp53 in ['mutated', 'mut', 'loss'] and rb1 in ['mutated', 'mut', 'loss']:
            return False, "TP53+RB1 double loss - too aggressive for drug holidays"

        # Very high LDH = Warburg, aggressive
        ldh = float(base.get('blood_markers', {}).get('ldh', 200) or 200)
        if ldh > 500:
            return False, f"LDH {ldh:.0f} too high - aggressive disease"

        # Brain mets = risky for holidays
        # Would need imaging data to check this

        return True, None

    def calculate_response_status(self, tumor_change: float) -> ResponseStatus:
        """Calcola status risposta da % change"""
        if tumor_change <= -50:
            return ResponseStatus.DEEP_RESPONSE
        elif tumor_change <= -30:
            return ResponseStatus.PARTIAL_RESPONSE
        elif tumor_change <= 20:
            return ResponseStatus.STABLE
        elif tumor_change <= 40:
            return ResponseStatus.MINOR_PROGRESSION
        else:
            return ResponseStatus.PROGRESSION

    def calculate_thresholds(self, baseline_burden: float,
                             best_response_burden: float) -> Tuple[float, float]:
        """
        Calcola soglie per drug holiday.

        Holiday threshold: Possiamo fare holiday se burden < X
        Resume threshold: Dobbiamo riprendere se burden > Y

        Strategy:
        - Holiday quando siamo al 50% del baseline O al best response + 20%
        - Resume quando raggiungiamo 75% del baseline
        """
        holiday = min(
            baseline_burden * 0.50,  # 50% del baseline
            best_response_burden * 1.20  # 20% sopra best response
        )

        resume = baseline_burden * 0.75  # 75% del baseline

        return holiday, resume

    def design_adaptive_cycles(self, patient_data: Dict,
                               current_therapy: str,
                               response_status: ResponseStatus) -> List[AdaptiveCycle]:
        """
        Disegna i cicli di terapia adattiva personalizzati.
        """
        cycles = []
        base = patient_data.get('baseline', patient_data)
        ecog = int(base.get('ecog_ps', 1))

        # Determina dose based on ECOG
        if 'osimertinib' in current_therapy.lower():
            full_dose = "80mg QD"
            reduced_dose = "40mg QD"
            metronomic_dose = "40mg QOD"
        elif 'gefitinib' in current_therapy.lower():
            full_dose = "250mg QD"
            reduced_dose = "250mg QOD"
            metronomic_dose = "250mg every 3 days"
        else:
            full_dose = "Standard dose"
            reduced_dose = "50% dose"
            metronomic_dose = "25% dose continuous"

        # === CYCLE 1: INDUCTION ===
        cycles.append(AdaptiveCycle(
            phase=TherapyPhase.INDUCTION,
            duration_weeks=12,
            drug=current_therapy,
            dose=full_dose,
            rationale="Maximize initial tumor reduction. Eliminate sensitive clones.",
            monitoring=[
                "ctDNA at week 4, 8, 12",
                "Imaging at week 12",
                "LDH weekly x4, then biweekly"
            ],
            transition_triggers=[
                "Tumor reduction ≥50% → Consider DRUG_HOLIDAY",
                "Tumor reduction 30-50% → Continue MAINTENANCE",
                "Progression → INTENSIFICATION"
            ]
        ))

        # === CYCLE 2: Depends on response ===
        if response_status == ResponseStatus.DEEP_RESPONSE:
            # Deep response → Drug holiday safe
            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.DRUG_HOLIDAY,
                duration_weeks=4,
                drug=current_therapy,
                dose="HOLD",
                rationale="Allow sensitive clones to re-expand. Exploit fitness cost of resistance.",
                monitoring=[
                    "ctDNA at week 2, 4 of holiday",
                    "Symptoms weekly",
                    "LDH weekly"
                ],
                transition_triggers=[
                    "ctDNA VAF increase >50% → Resume therapy",
                    "New symptoms → Resume therapy",
                    "Week 4 reached → Resume MAINTENANCE",
                    "LDH increase >25% → Resume therapy"
                ]
            ))

            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.MAINTENANCE,
                duration_weeks=8,
                drug=current_therapy,
                dose=reduced_dose if ecog <= 1 else full_dose,
                rationale="Maintain control with reduced dose. Continue clonal competition.",
                monitoring=[
                    "ctDNA at week 4, 8",
                    "Imaging at week 8",
                    "LDH biweekly"
                ],
                transition_triggers=[
                    "Stable → Repeat DRUG_HOLIDAY",
                    "Minor progression → Return to INDUCTION",
                    "Progression → INTENSIFICATION"
                ]
            ))

        elif response_status == ResponseStatus.PARTIAL_RESPONSE:
            # Partial response → Shorter holiday or maintenance only
            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.MAINTENANCE,
                duration_weeks=8,
                drug=current_therapy,
                dose=full_dose,
                rationale="Consolidate partial response before attempting holiday.",
                monitoring=[
                    "ctDNA at week 4, 8",
                    "Imaging at week 8"
                ],
                transition_triggers=[
                    "Deepening response → DRUG_HOLIDAY (short)",
                    "Stable → Continue MAINTENANCE",
                    "Progression → INTENSIFICATION"
                ]
            ))

            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.DRUG_HOLIDAY,
                duration_weeks=2,  # Shorter holiday for PR
                drug=current_therapy,
                dose="HOLD",
                rationale="Brief holiday to test clonal dynamics. Shorter due to partial response.",
                monitoring=[
                    "ctDNA at week 2",
                    "Symptoms daily",
                    "LDH at day 7, 14"
                ],
                transition_triggers=[
                    "Any concerning signal → Resume immediately",
                    "Week 2 reached → Resume MAINTENANCE"
                ]
            ))

        else:
            # Stable/Minor progression → Metronomic
            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.METRONOMIC,
                duration_weeks=12,
                drug=current_therapy,
                dose=metronomic_dose,
                rationale="Continuous low-dose to maintain pressure without selecting for resistance.",
                monitoring=[
                    "ctDNA monthly",
                    "Imaging at week 12",
                    "LDH monthly"
                ],
                transition_triggers=[
                    "Deepening response → DRUG_HOLIDAY",
                    "Progression → Full dose INDUCTION"
                ]
            ))

        # Frailty Check (Per evitare accanimento terapeutico in over 75 o ECOG >= 2)
        age = int(base.get("age", 50))
        is_frail = age >= 75 or ecog >= 2

        # === CONTINGENCY: INTENSIFICATION / RE-INDUCTION ===
        if is_frail:
            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.INTENSIFICATION,
                duration_weeks=8,
                drug=current_therapy,  # Evita la double-chemo
                dose="Standard Dose (Monotherapy)",
                rationale="[!] De-escalated protocol due to patient age/frailty. Avoid highly toxic multi-agent chemotherapy. Attempt re-challenge with initial therapy if progression occurs.",
                monitoring=[
                    "ctDNA at week 4",
                    "Symptoms weekly",
                    "Close QoL monitoring"
                ],
                transition_triggers=[
                    "Response → Return to MAINTENANCE",
                    "Rapid progression → Transition to Best Supportive Care (BSC)"
                ]
            ))
        else:
            cycles.append(AdaptiveCycle(
                phase=TherapyPhase.INTENSIFICATION,
                duration_weeks=8,
                drug=current_therapy + " + Chemotherapy" if 'chemo' not in current_therapy.lower() else current_therapy,
                dose="Full dose + Platinum doublet",
                rationale="Reserved for progression. Aggressive multi-agent attack to eradicate resistant clones.",
                monitoring=[
                    "ctDNA at week 2, 4, 8",
                    "Imaging at week 8",
                    "CBC weekly (myelosuppression)"
                ],
                transition_triggers=[
                    "Response → Return to INDUCTION",
                    "Continued progression → Re-biopsy, consider trial"
                ]
            ))

        return cycles

    def estimate_pfs_benefit(self, response_status: ResponseStatus,
                             baseline_pfs: float) -> Tuple[float, float]:
        """
        Stima il beneficio in PFS dalla terapia adattiva.

        Args:
            response_status: Status risposta attuale
            baseline_pfs: PFS atteso con terapia standard (mesi)

        Returns:
            Tuple[standard_pfs, adaptive_pfs]
        """
        if response_status == ResponseStatus.DEEP_RESPONSE:
            multiplier = self.ADAPTIVE_BENEFIT['deep_response']
        elif response_status == ResponseStatus.PARTIAL_RESPONSE:
            multiplier = self.ADAPTIVE_BENEFIT['partial_response']
        else:
            multiplier = self.ADAPTIVE_BENEFIT['stable']

        adaptive_pfs = baseline_pfs * multiplier

        return baseline_pfs, adaptive_pfs

    def generate_protocol(self, patient_data: Dict,
                          tumor_change: float = -40,
                          weeks_stable: int = 16,
                          baseline_burden: float = 100,
                          current_burden: float = 60,
                          baseline_pfs: float = 18) -> AdaptiveProtocol:
        """
        Genera protocollo adattivo completo.

        Args:
            patient_data: Dati paziente
            tumor_change: % change dal baseline (negativo = riduzione)
            weeks_stable: Settimane di stabilità
            baseline_burden: Sum of target lesions at baseline (mm)
            current_burden: Sum of target lesions current (mm)
            baseline_pfs: PFS atteso standard (mesi)

        Returns:
            AdaptiveProtocol completo
        """
        base = patient_data.get('baseline', patient_data)
        patient_id = base.get('patient_id', 'Unknown')
        current_therapy = base.get('current_therapy', 'Unknown')

        # Check eligibility
        eligible, reason = self.assess_eligibility(patient_data, tumor_change, weeks_stable)

        # Response status
        response_status = self.calculate_response_status(tumor_change)

        # Thresholds
        best_burden = min(baseline_burden, current_burden)
        holiday_threshold, resume_threshold = self.calculate_thresholds(baseline_burden, best_burden)

        # Design cycles
        if eligible:
            cycles = self.design_adaptive_cycles(patient_data, current_therapy, response_status)
            current_phase = cycles[0].phase
        else:
            cycles = []
            current_phase = TherapyPhase.INDUCTION

        # PFS estimates
        standard_pfs, adaptive_pfs = self.estimate_pfs_benefit(response_status, baseline_pfs)

        # Immediate action
        if tumor_change == -100:
            immediate_action = "Scenario: Deep Complete Response (CR) achieved. STRICT MAINTENANCE OR DRUG HOLIDAY ONLY. Intensification is absolutely contraindicated."
            next_assessment = 12
        elif not eligible:
            immediate_action = f"Continue standard therapy. Not eligible for adaptive: {reason}"
            next_assessment = 8
        elif response_status == ResponseStatus.DEEP_RESPONSE:
            immediate_action = "Excellent response! Consider initiating drug holiday protocol."
            next_assessment = 4
        elif response_status == ResponseStatus.PARTIAL_RESPONSE:
            immediate_action = "Good response. Continue induction, reassess for holiday at week 12."
            next_assessment = 4
        else:
            immediate_action = "Stable disease. Consider metronomic dosing."
            next_assessment = 8

        return AdaptiveProtocol(
            patient_id=patient_id,
            current_therapy=current_therapy,
            current_phase=current_phase,
            cycles=cycles,
            baseline_tumor_burden=baseline_burden,
            current_tumor_burden=current_burden,
            response_status=response_status,
            holiday_threshold=holiday_threshold,
            resume_threshold=resume_threshold,
            estimated_pfs_standard=standard_pfs,
            estimated_pfs_adaptive=adaptive_pfs,
            benefit_months=adaptive_pfs - standard_pfs,
            eligible_for_adaptive=eligible,
            ineligibility_reason=reason,
            immediate_action=immediate_action,
            next_assessment_weeks=next_assessment,
            generated_at=self.generated_at
        )


def generate_adaptive_protocol(patient_data: Dict,
                               tumor_change: float = -40,
                               weeks_stable: int = 16,
                               baseline_burden: float = 100,
                               current_burden: float = 60,
                               baseline_pfs: float = 18) -> AdaptiveProtocol:
    """Factory function per generare protocollo adattivo."""
    optimizer = AdaptiveTherapyOptimizer()
    return optimizer.generate_protocol(
        patient_data, tumor_change, weeks_stable,
        baseline_burden, current_burden, baseline_pfs
    )