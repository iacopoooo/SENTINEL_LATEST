"""
SENTINEL v18.1 - Predictive Resistance Timeline
================================================
Predice QUANDO emergerà la resistenza basandosi su:
- Rischio attuale (Overall Risk)
- Velocità di evoluzione (VAF dynamics)
- Fattori biologici (LDH trend, mutazioni)
- Dati storici (letteratura EGFR+ NSCLC)

Modello: Weibull hazard function con modificatori paziente-specifici
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


@dataclass
class TimelineProjection:
    """Proiezione temporale del rischio"""
    month: int
    risk_percent: float
    confidence_low: float
    confidence_high: float
    cumulative_hazard: float


@dataclass
class ResistanceAlert:
    """Alert per soglia di rischio raggiunta"""
    threshold_percent: int
    expected_month: float
    confidence_range: Tuple[float, float]
    recommended_action: str


@dataclass
class PredictiveTimelineResult:
    """Risultato completo della predizione temporale"""
    current_risk: float
    baseline_risk: float

    # Proiezioni mensili (12 mesi)
    monthly_projections: List[TimelineProjection]

    # Alert per soglie chiave
    alerts: List[ResistanceAlert]

    # Raccomandazioni monitoring
    next_ctdna_week: int
    next_imaging_week: int
    monitoring_intensity: str  # "ROUTINE", "ENHANCED", "INTENSIVE"

    # Fattori che accelerano/rallentano
    accelerating_factors: List[str]
    protective_factors: List[str]

    # Metadata
    model_confidence: str
    generated_at: str


class PredictiveTimeline:
    """
    Motore predittivo per timeline resistenza.

    Usa un modello Weibull modificato:
    - Shape parameter (k): determina se il rischio accelera o decelera nel tempo
    - Scale parameter (λ): tempo mediano alla resistenza
    - Modificatori paziente-specifici: aggiustano λ basandosi su biologia
    """

    # Parametri base da letteratura EGFR+ NSCLC con Osimertinib (default)
    # Mediana PFS ~18-22 mesi, 50% resistenza a ~18 mesi
    BASE_MEDIAN_MONTHS = 18.0
    BASE_SHAPE = 1.8  # k > 1 = rischio aumenta nel tempo (tipico cancro)

    # Parametri Weibull per tipo di tumore (mediana PFS mesi, shape k)
    CANCER_TYPE_PARAMS = {
        # NSCLC
        'adenocarcinoma': (18.0, 1.8, 'NSCLC'),
        'squamous cell': (12.0, 1.7, 'Squamous Cell Lung Cancer'),
        'large cell': (10.0, 1.9, 'Large Cell Lung Cancer'),
        'nsclc': (18.0, 1.8, 'NSCLC'),
        # Uroteliale
        'urothelial': (14.0, 1.6, 'Urothelial Carcinoma'),
        'bladder': (14.0, 1.6, 'Urothelial Carcinoma'),
        # Gastrico / GEJ
        'gastroesophageal': (8.0, 2.0, 'Gastroesophageal Adenocarcinoma'),
        'gastric': (8.0, 2.0, 'Gastric Cancer'),
        'esophageal': (9.0, 1.9, 'Esophageal Cancer'),
        # Mammella
        'breast': (16.0, 1.5, 'Breast Cancer'),
        # Colon-retto
        'colorectal': (12.0, 1.7, 'Colorectal Cancer'),
        'colon': (12.0, 1.7, 'Colorectal Cancer'),
        # Melanoma
        'melanoma': (11.0, 1.6, 'Melanoma'),
        # Pancreas
        'pancreatic': (6.0, 2.2, 'Pancreatic Cancer'),
        'pancreas': (6.0, 2.2, 'Pancreatic Cancer'),
        # Renale
        'renal': (15.0, 1.5, 'Renal Cell Carcinoma'),
        # Default
        'default': (18.0, 1.8, 'Solid Tumor (generic)'),
    }

    # Modificatori per fattori di rischio
    RISK_MODIFIERS = {
        # Fattori che ACCELERANO resistenza (riducono tempo mediano)
        'high_ldh': 0.70,  # LDH > 350: 30% faster
        'very_high_ldh': 0.50,  # LDH > 600: 50% faster
        'tp53_mutated': 0.75,  # TP53: 25% faster
        'rb1_loss': 0.65,  # RB1: 35% faster
        'tp53_rb1_double': 0.45,  # Double loss: 55% faster
        'met_amplification': 0.60,  # MET amp: 40% faster
        'high_tmb': 0.80,  # TMB > 10: 20% faster
        'ecog_2': 0.85,  # ECOG 2: 15% faster
        'ecog_3_4': 0.60,  # ECOG 3-4: 40% faster
        'stk11_loss': 0.75,  # STK11: 25% faster
        'keap1_loss': 0.80,  # KEAP1: 20% faster
        'rising_vaf': 0.70,  # VAF increasing: 30% faster
        'new_mutations': 0.50,  # New mutations detected: 50% faster

        # Fattori che RALLENTANO resistenza (aumentano tempo mediano)
        'low_ldh': 1.20,  # LDH < 200: 20% slower
        'ecog_0': 1.25,  # ECOG 0: 25% slower
        'complete_response': 1.40,  # CR: 40% slower
        'partial_response': 1.15,  # PR: 15% slower
        'ctdna_clearance': 1.50,  # ctDNA cleared: 50% slower
        'low_tmb': 1.10,  # TMB < 5: 10% slower
        'young_age': 1.10,  # Age < 50: 10% slower
    }

    # Soglie di alert
    ALERT_THRESHOLDS = [30, 50, 70, 90]

    def __init__(self):
        self.generated_at = datetime.now().isoformat()

    def calculate_patient_modifiers(self, patient_data: Dict) -> Tuple[float, List[str], List[str]]:
        """
        Calcola il modificatore combinato basato sul profilo paziente.

        Returns:
            Tuple[modifier, accelerating_factors, protective_factors]
        """
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})
        blood = base.get('blood_markers', {})
        biomarkers = base.get('biomarkers', {})

        modifier = 1.0
        accelerating = []
        protective = []

        # === LDH ===
        ldh = float(blood.get('ldh', 200) or 200)
        if ldh > 600:
            modifier *= self.RISK_MODIFIERS['very_high_ldh']
            accelerating.append(f"Very high LDH ({ldh:.0f} U/L)")
        elif ldh > 350:
            modifier *= self.RISK_MODIFIERS['high_ldh']
            accelerating.append(f"High LDH ({ldh:.0f} U/L)")
        elif ldh < 200:
            modifier *= self.RISK_MODIFIERS['low_ldh']
            protective.append(f"Normal LDH ({ldh:.0f} U/L)")

        # === ECOG ===
        ecog = int(base.get('ecog_ps', 1))
        if ecog >= 3:
            modifier *= self.RISK_MODIFIERS['ecog_3_4']
            accelerating.append(f"Poor performance status (ECOG {ecog})")
        elif ecog == 2:
            modifier *= self.RISK_MODIFIERS['ecog_2']
            accelerating.append(f"Reduced performance status (ECOG {ecog})")
        elif ecog == 0:
            modifier *= self.RISK_MODIFIERS['ecog_0']
            protective.append("Excellent performance status (ECOG 0)")

        # === Genetic Markers ===
        tp53 = str(genetics.get('tp53_status', '')).lower()
        rb1 = str(genetics.get('rb1_status', '')).lower()
        stk11 = str(genetics.get('stk11_status', '')).lower()
        keap1 = str(genetics.get('keap1_status', '')).lower()
        met = str(genetics.get('met_status', '')).lower()
        met_cn = float(genetics.get('met_cn', 0) or 0)

        tp53_mut = tp53 in ['mutated', 'mut', 'loss']
        rb1_mut = rb1 in ['mutated', 'mut', 'loss']

        if tp53_mut and rb1_mut:
            modifier *= self.RISK_MODIFIERS['tp53_rb1_double']
            accelerating.append("TP53 + RB1 double loss (transformation risk)")
        elif tp53_mut:
            modifier *= self.RISK_MODIFIERS['tp53_mutated']
            accelerating.append("TP53 mutation (genomic instability)")

        if rb1_mut and not tp53_mut:
            modifier *= self.RISK_MODIFIERS['rb1_loss']
            accelerating.append("RB1 loss")

        if stk11 in ['mutated', 'mut', 'loss']:
            modifier *= self.RISK_MODIFIERS['stk11_loss']
            accelerating.append("STK11 loss (metabolic resistance)")

        if keap1 in ['mutated', 'mut', 'loss']:
            modifier *= self.RISK_MODIFIERS['keap1_loss']
            accelerating.append("KEAP1 loss (oxidative stress)")

        if 'amplification' in met or met_cn >= 5:
            modifier *= self.RISK_MODIFIERS['met_amplification']
            accelerating.append(f"MET amplification (CN={met_cn:.1f})")

        # === TMB ===
        tmb = float(biomarkers.get('tmb_score', 0) or base.get('tmb', 0) or 0)
        if tmb > 10:
            modifier *= self.RISK_MODIFIERS['high_tmb']
            accelerating.append(f"High TMB ({tmb:.1f} mut/Mb)")
        elif tmb < 5 and tmb > 0:
            modifier *= self.RISK_MODIFIERS['low_tmb']
            protective.append(f"Low TMB ({tmb:.1f} mut/Mb)")

        # === Age ===
        age = int(base.get('age', 65))
        if age < 50:
            modifier *= self.RISK_MODIFIERS['young_age']
            protective.append(f"Young age ({age})")

        # === Response (if available) ===
        # Questo verrebbe da visits/imaging

        return modifier, accelerating, protective

    def weibull_survival(self, t: float, k: float, lam: float) -> float:
        """
        Calcola probabilità di sopravvivenza (no resistenza) al tempo t.
        S(t) = exp(-(t/λ)^k)
        """
        if t <= 0:
            return 1.0
        return math.exp(-((t / lam) ** k))

    def weibull_risk(self, t: float, k: float, lam: float) -> float:
        """
        Calcola probabilità cumulativa di resistenza al tempo t.
        F(t) = 1 - S(t) = 1 - exp(-(t/λ)^k)
        """
        return 1.0 - self.weibull_survival(t, k, lam)

    def find_time_to_threshold(self, threshold: float, k: float, lam: float) -> float:
        """
        Trova il tempo in cui il rischio raggiunge una certa soglia.
        t = λ * (-ln(1-F))^(1/k)
        """
        if threshold >= 1.0:
            return float('inf')
        if threshold <= 0:
            return 0.0
        return lam * ((-math.log(1 - threshold)) ** (1 / k))

    def _get_cancer_type_params(self, histology: str) -> tuple:
        """Determina parametri Weibull dal tipo di tumore."""
        hist_lower = histology.lower() if histology else ''
        for key, params in self.CANCER_TYPE_PARAMS.items():
            if key in hist_lower:
                return params
        return self.CANCER_TYPE_PARAMS['default']

    def generate_timeline(self, patient_data: Dict,
                          current_risk: float,
                          months_on_therapy: int = 0) -> PredictiveTimelineResult:
        """
        Genera la timeline predittiva completa.

        Args:
            patient_data: Dati paziente
            current_risk: Overall Resistance Risk attuale (0-100)
            months_on_therapy: Mesi già in terapia

        Returns:
            PredictiveTimelineResult con proiezioni e raccomandazioni
        """
        # Calcola modificatori paziente-specifici
        modifier, accelerating, protective = self.calculate_patient_modifiers(patient_data)

        # Determina parametri base dal tipo di tumore
        base = patient_data.get('baseline', patient_data)
        histology = base.get('histology', '')
        base_median, base_shape, self._cancer_type_label = self._get_cancer_type_params(histology)

        # Calcola parametri Weibull personalizzati
        # λ (scale) = tempo mediano modificato
        patient_lambda = base_median * modifier
        k = base_shape

        # Se già in terapia, aggiusta per conditional probability
        # P(resist by t+Δt | survived to t)

        # Genera proiezioni mensili (dynamic scale based on patient lambda)
        projections = []
        
        # If lambda (median time to resistance) is very small, we shouldn't project 12 months out.
        max_months = min(12, int(max(3, math.ceil(patient_lambda * 1.5))))
        
        for month in range(1, max_months + 1):
            future_month = months_on_therapy + month

            # Rischio cumulativo al mese futuro
            risk = self.weibull_risk(future_month, k, patient_lambda) * 100

            # Confidence interval (±15% del rischio, clamped)
            ci_width = risk * 0.15
            ci_low = max(0, risk - ci_width)
            ci_high = min(100, risk + ci_width)

            # Hazard cumulativo
            cum_hazard = -math.log(max(0.001, 1 - risk / 100))

            projections.append(TimelineProjection(
                month=month,
                risk_percent=round(risk, 1),
                confidence_low=round(ci_low, 1),
                confidence_high=round(ci_high, 1),
                cumulative_hazard=round(cum_hazard, 3)
            ))

        # Genera alerts per soglie chiave
        alerts = []
        for threshold in self.ALERT_THRESHOLDS:
            time_to_threshold = self.find_time_to_threshold(
                threshold / 100, k, patient_lambda
            )

            # Sottrai mesi già trascorsi
            months_remaining = time_to_threshold - months_on_therapy

            if months_remaining > 0:
                # Confidence range
                ci_low_time = months_remaining * 0.75
                ci_high_time = months_remaining * 1.35

                # Raccomandazione basata su urgenza
                if threshold <= 30:
                    action = "Baseline ctDNA monitoring"
                elif threshold <= 50:
                    action = "Increase ctDNA frequency to q4 weeks"
                elif threshold <= 70:
                    action = "Consider re-biopsy if ctDNA negative"
                else:
                    action = "Prepare alternative therapy options"

                alerts.append(ResistanceAlert(
                    threshold_percent=threshold,
                    expected_month=round(months_remaining, 1),
                    confidence_range=(round(ci_low_time, 1), round(ci_high_time, 1)),
                    recommended_action=action
                ))

        # Determina intensità monitoring
        if current_risk >= 70 or (projections and projections[2].risk_percent >= 60):
            monitoring_intensity = "INTENSIVE"
            next_ctdna = 2
            next_imaging = 4
        elif current_risk >= 40 or (projections and projections[2].risk_percent >= 40):
            monitoring_intensity = "ENHANCED"
            next_ctdna = 4
            next_imaging = 6
        else:
            monitoring_intensity = "ROUTINE"
            next_ctdna = 8
            next_imaging = 12

        # Model confidence
        if len(accelerating) + len(protective) >= 3:
            model_confidence = "HIGH"
        elif len(accelerating) + len(protective) >= 1:
            model_confidence = "MEDIUM"
        else:
            model_confidence = "LOW"

        return PredictiveTimelineResult(
            current_risk=current_risk,
            baseline_risk=self.weibull_risk(months_on_therapy, k, patient_lambda) * 100,
            monthly_projections=projections,
            alerts=alerts,
            next_ctdna_week=next_ctdna,
            next_imaging_week=next_imaging,
            monitoring_intensity=monitoring_intensity,
            accelerating_factors=accelerating,
            protective_factors=protective,
            model_confidence=model_confidence,
            generated_at=self.generated_at
        )


def generate_predictive_timeline(patient_data: Dict,
                                 current_risk: float,
                                 months_on_therapy: int = 0) -> PredictiveTimelineResult:
    """
    Factory function per generare timeline predittiva.
    """
    engine = PredictiveTimeline()
    return engine.generate_timeline(patient_data, current_risk, months_on_therapy)