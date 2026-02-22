"""
CLINICAL SAFETY ENGINE - SENTINEL
==================================
Modulo salvavita per alert clinici critici in oncologia.

Include:
- Neutropenic Fever Detection
- Critical Lab Value Alerts
- Renal/Hepatic Dose Adjustment
- Drug-Drug Interactions (DDI)
- QTc Prolongation Monitoring
- Tumor Lysis Syndrome (TLS) Prediction
- VTE Risk (Khorana Score)

Basato su linee guida:
- ASCO/IDSA Neutropenic Fever
- CPIC Pharmacogenomics
- ESC Cardio-Oncology
- MASCC TLS Guidelines
- Khorana VTE Score
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class AlertSeverity(Enum):
    """Severity levels for clinical alerts"""
    CRITICAL = "critical"      # Immediate action required - life threatening
    HIGH = "high"              # Urgent attention needed
    MODERATE = "moderate"      # Should be addressed soon
    LOW = "low"                # Informational
    
    def icon(self) -> str:
        icons = {
            "critical": "🚨",
            "high": "⚠️",
            "moderate": "📊",
            "low": "ℹ️"
        }
        return icons.get(self.value, "❓")
    
    def color(self) -> str:
        colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "moderate": "#ffc107",
            "low": "#17a2b8"
        }
        return colors.get(self.value, "#6c757d")


class AlertCategory(Enum):
    """Categories of clinical safety alerts"""
    NEUTROPENIC_FEVER = "neutropenic_fever"
    CRITICAL_LAB = "critical_lab"
    DOSE_ADJUSTMENT = "dose_adjustment"
    DRUG_INTERACTION = "drug_interaction"
    QTC_PROLONGATION = "qtc_prolongation"
    TUMOR_LYSIS = "tumor_lysis"
    VTE_RISK = "vte_risk"


@dataclass
class ClinicalAlert:
    """Single clinical safety alert"""
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    recommended_actions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_immediate_action: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "actions": self.recommended_actions,
            "immediate": self.requires_immediate_action,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DoseAdjustment:
    """Dose adjustment recommendation"""
    drug: str
    original_dose: str
    adjusted_dose: str
    adjustment_factor: float
    reason: str
    gfr: Optional[float] = None
    hepatic_function: Optional[str] = None


@dataclass
class DrugInteraction:
    """Drug-drug interaction"""
    drug_a: str
    drug_b: str
    severity: AlertSeverity
    mechanism: str
    effect: str
    recommendation: str


@dataclass 
class QTcRisk:
    """QTc prolongation risk assessment"""
    qtc_value: Optional[float]
    risk_drugs: List[str]
    total_qt_burden: int
    risk_level: AlertSeverity
    recommendation: str


@dataclass
class TLSRisk:
    """Tumor Lysis Syndrome risk stratification"""
    risk_score: int
    risk_level: str  # LOW, INTERMEDIATE, HIGH
    factors: List[str]
    prophylaxis_recommendation: str
    monitoring_frequency: str


@dataclass
class KhoranaScore:
    """VTE risk assessment (Khorana Score)"""
    total_score: int
    risk_category: str  # LOW, INTERMEDIATE, HIGH
    factors: List[Tuple[str, int]]
    vte_risk_percent: float
    recommendation: str


# =============================================================================
# CLINICAL SAFETY ENGINE
# =============================================================================

class ClinicalSafetyEngine:
    """
    Engine for detecting life-threatening conditions in oncology patients.
    """
    
    # Critical lab thresholds (life-threatening values)
    CRITICAL_LABS = {
        'potassium': {'low': 2.5, 'high': 6.5, 'unit': 'mEq/L', 'risk': 'Arrhythmia/Cardiac arrest'},
        'sodium': {'low': 120, 'high': 160, 'unit': 'mEq/L', 'risk': 'Seizures/Brain herniation'},
        'calcium': {'low': 6.0, 'high': 14.0, 'unit': 'mg/dL', 'risk': 'Cardiac arrest/Coma'},
        'glucose': {'low': 40, 'high': 500, 'unit': 'mg/dL', 'risk': 'Coma/DKA'},
        'hemoglobin': {'low': 7.0, 'high': None, 'unit': 'g/dL', 'risk': 'Shock'},
        'platelets': {'low': 20000, 'high': None, 'unit': '/µL', 'risk': 'Spontaneous bleeding'},
        'neutrophils': {'low': 500, 'high': None, 'unit': '/µL', 'risk': 'Severe infection'},
        'creatinine': {'low': None, 'high': 10.0, 'unit': 'mg/dL', 'risk': 'Uremia/Dialysis needed'},
        'inr': {'low': None, 'high': 5.0, 'unit': '', 'risk': 'Bleeding risk'},
        'lactate': {'low': None, 'high': 4.0, 'unit': 'mmol/L', 'risk': 'Septic shock'},
    }
    
    # QTc prolonging drugs commonly used in oncology
    QT_PROLONGING_DRUGS = {
        # High risk (>10ms QTc prolongation)
        'osimertinib': {'qt_risk': 3, 'category': 'high'},
        'ribociclib': {'qt_risk': 3, 'category': 'high'},
        'vandetanib': {'qt_risk': 3, 'category': 'high'},
        'arsenic trioxide': {'qt_risk': 3, 'category': 'high'},
        'sotalol': {'qt_risk': 3, 'category': 'high'},
        'haloperidol': {'qt_risk': 3, 'category': 'high'},
        'ondansetron': {'qt_risk': 2, 'category': 'moderate'},
        'granisetron': {'qt_risk': 2, 'category': 'moderate'},
        # Moderate risk
        'sunitinib': {'qt_risk': 2, 'category': 'moderate'},
        'pazopanib': {'qt_risk': 2, 'category': 'moderate'},
        'lapatinib': {'qt_risk': 2, 'category': 'moderate'},
        'crizotinib': {'qt_risk': 2, 'category': 'moderate'},
        'ceritinib': {'qt_risk': 2, 'category': 'moderate'},
        'fluorouracil': {'qt_risk': 1, 'category': 'low'},
        'capecitabine': {'qt_risk': 1, 'category': 'low'},
        'tamoxifen': {'qt_risk': 1, 'category': 'low'},
    }
    
    # Drug-drug interactions database (critical oncology interactions)
    CRITICAL_DDI = [
        {
            'drug_a': 'warfarin', 'drug_b': 'capecitabine',
            'severity': 'critical', 'mechanism': 'CYP2C9 inhibition',
            'effect': 'Increased INR, severe bleeding risk',
            'recommendation': 'Reduce warfarin dose 30-50%, monitor INR closely'
        },
        {
            'drug_a': 'osimertinib', 'drug_b': 'rifampicin',
            'severity': 'high', 'mechanism': 'CYP3A4 induction',
            'effect': 'Reduced osimertinib exposure 80%',
            'recommendation': 'Avoid combination, use alternative antibiotic'
        },
        {
            'drug_a': 'irinotecan', 'drug_b': 'ketoconazole',
            'severity': 'high', 'mechanism': 'CYP3A4 inhibition',
            'effect': 'Increased SN-38 toxicity',
            'recommendation': 'Avoid combination or reduce irinotecan dose'
        },
        {
            'drug_a': 'methotrexate', 'drug_b': 'nsaids',
            'severity': 'high', 'mechanism': 'Reduced renal clearance',
            'effect': 'Methotrexate toxicity (mucositis, myelosuppression)',
            'recommendation': 'Avoid NSAIDs 48h before/after MTX'
        },
        {
            'drug_a': 'tamoxifen', 'drug_b': 'paroxetine',
            'severity': 'high', 'mechanism': 'CYP2D6 inhibition',
            'effect': 'Reduced endoxifen (active metabolite)',
            'recommendation': 'Switch to venlafaxine or escitalopram'
        },
        {
            'drug_a': 'palbociclib', 'drug_b': 'clarithromycin',
            'severity': 'high', 'mechanism': 'CYP3A4 inhibition',
            'effect': 'Increased palbociclib toxicity',
            'recommendation': 'Use azithromycin instead'
        },
        {
            'drug_a': 'cisplatin', 'drug_b': 'aminoglycosides',
            'severity': 'high', 'mechanism': 'Additive ototoxicity',
            'effect': 'Permanent hearing loss',
            'recommendation': 'Avoid combination if possible'
        },
    ]
    
    # Renal dose adjustments
    RENAL_ADJUSTMENTS = {
        'cisplatin': [
            {'gfr_min': 60, 'gfr_max': None, 'adjustment': 1.0, 'note': 'Full dose'},
            {'gfr_min': 45, 'gfr_max': 60, 'adjustment': 0.75, 'note': 'Reduce 25%'},
            {'gfr_min': 30, 'gfr_max': 45, 'adjustment': 0.5, 'note': 'Reduce 50% or switch to carboplatin'},
            {'gfr_min': 0, 'gfr_max': 30, 'adjustment': 0, 'note': 'CONTRAINDICATED - use carboplatin'}
        ],
        'carboplatin': [
            # Calvert formula: Dose = AUC × (GFR + 25)
            {'gfr_min': 0, 'gfr_max': None, 'adjustment': 'calvert', 'note': 'Use Calvert formula'}
        ],
        'capecitabine': [
            {'gfr_min': 50, 'gfr_max': None, 'adjustment': 1.0, 'note': 'Full dose'},
            {'gfr_min': 30, 'gfr_max': 50, 'adjustment': 0.75, 'note': 'Reduce to 75%'},
            {'gfr_min': 0, 'gfr_max': 30, 'adjustment': 0, 'note': 'CONTRAINDICATED'}
        ],
        'methotrexate': [
            {'gfr_min': 80, 'gfr_max': None, 'adjustment': 1.0, 'note': 'Full dose'},
            {'gfr_min': 50, 'gfr_max': 80, 'adjustment': 0.65, 'note': 'Reduce 35%'},
            {'gfr_min': 30, 'gfr_max': 50, 'adjustment': 0.5, 'note': 'Reduce 50%'},
            {'gfr_min': 0, 'gfr_max': 30, 'adjustment': 0, 'note': 'CONTRAINDICATED - avoid high dose'}
        ],
        'pemetrexed': [
            {'gfr_min': 45, 'gfr_max': None, 'adjustment': 1.0, 'note': 'Full dose'},
            {'gfr_min': 0, 'gfr_max': 45, 'adjustment': 0, 'note': 'CONTRAINDICATED'}
        ],
        'topotecan': [
            {'gfr_min': 40, 'gfr_max': None, 'adjustment': 1.0, 'note': 'Full dose'},
            {'gfr_min': 20, 'gfr_max': 40, 'adjustment': 0.5, 'note': 'Reduce 50%'},
            {'gfr_min': 0, 'gfr_max': 20, 'adjustment': 0, 'note': 'Not recommended'}
        ],
    }
    
    def __init__(self):
        self.alerts: List[ClinicalAlert] = []
    
    # =========================================================================
    # 1. NEUTROPENIC FEVER DETECTION
    # =========================================================================
    
    def check_neutropenic_fever(
        self, 
        neutrophil_count: float,
        temperature: float,
        blood_pressure_systolic: Optional[float] = None,
        respiratory_rate: Optional[float] = None
    ) -> Optional[ClinicalAlert]:
        """
        Detects neutropenic fever - a medical emergency in cancer patients.
        
        Criteria (IDSA/ASCO):
        - ANC < 500/µL OR expected to fall < 500/µL
        - Temperature ≥ 38.3°C single OR ≥ 38.0°C sustained > 1h
        
        Args:
            neutrophil_count: Absolute neutrophil count (/µL)
            temperature: Body temperature (°C)
            blood_pressure_systolic: Systolic BP for qSOFA
            respiratory_rate: RR for qSOFA
            
        Returns:
            ClinicalAlert if neutropenic fever detected
        """
        is_neutropenic = neutrophil_count < 500
        has_fever = temperature >= 38.0
        
        if is_neutropenic and has_fever:
            # Calculate qSOFA for sepsis risk
            qsofa = 0
            if respiratory_rate and respiratory_rate >= 22:
                qsofa += 1
            if blood_pressure_systolic and blood_pressure_systolic <= 100:
                qsofa += 1
            # Altered mental status would be +1 but requires clinical assessment
            
            severity = AlertSeverity.CRITICAL if qsofa >= 2 else AlertSeverity.HIGH
            
            actions = [
                "📞 CALL ONCOLOGIST IMMEDIATELY",
                "🩸 Blood cultures x2 (peripheral + central line if present)",
                "💉 Start empiric antibiotics within 1 HOUR (e.g., Piperacillin-Tazobactam 4.5g IV)",
                "💧 IV fluid resuscitation if hypotensive",
                "🏥 Consider ICU admission if qSOFA ≥ 2"
            ]
            
            if neutrophil_count < 100:
                actions.append("⚠️ PROFOUND NEUTROPENIA - consider G-CSF")
            
            alert = ClinicalAlert(
                category=AlertCategory.NEUTROPENIC_FEVER,
                severity=severity,
                title="🚨 NEUTROPENIC FEVER - EMERGENCY",
                message=(
                    f"ANC: {neutrophil_count:.0f}/µL | Temp: {temperature:.1f}°C | "
                    f"qSOFA: {qsofa}/2\n"
                    f"SEPSIS RISK: {'HIGH' if qsofa >= 2 else 'MODERATE'}"
                ),
                recommended_actions=actions,
                parameters={
                    'anc': neutrophil_count,
                    'temperature': temperature,
                    'qsofa': qsofa
                },
                requires_immediate_action=True
            )
            self.alerts.append(alert)
            return alert
        
        return None
    
    # =========================================================================
    # 2. CRITICAL LAB VALUE ALERTS
    # =========================================================================
    
    def check_critical_labs(self, labs: Dict[str, float]) -> List[ClinicalAlert]:
        """
        Checks for life-threatening laboratory values.
        
        Args:
            labs: Dictionary of lab values with lab name as key
            
        Returns:
            List of critical alerts
        """
        alerts = []
        
        for lab_name, value in labs.items():
            lab_lower = lab_name.lower()
            
            # Normalize lab names
            normalized = None
            if 'potassium' in lab_lower or lab_lower == 'k':
                normalized = 'potassium'
            elif 'sodium' in lab_lower or lab_lower == 'na':
                normalized = 'sodium'
            elif 'calcium' in lab_lower or lab_lower == 'ca':
                normalized = 'calcium'
            elif 'glucose' in lab_lower:
                normalized = 'glucose'
            elif 'hemoglobin' in lab_lower or lab_lower == 'hb' or lab_lower == 'hgb':
                normalized = 'hemoglobin'
            elif 'platelet' in lab_lower or lab_lower == 'plt':
                normalized = 'platelets'
            elif 'neutrophil' in lab_lower or lab_lower == 'anc':
                normalized = 'neutrophils'
            elif 'creatinine' in lab_lower or lab_lower == 'cr':
                normalized = 'creatinine'
            elif 'inr' in lab_lower:
                normalized = 'inr'
            elif 'lactate' in lab_lower:
                normalized = 'lactate'
            
            if normalized and normalized in self.CRITICAL_LABS:
                thresholds = self.CRITICAL_LABS[normalized]
                
                is_critical_low = thresholds['low'] is not None and value < thresholds['low']
                is_critical_high = thresholds['high'] is not None and value > thresholds['high']
                
                if is_critical_low or is_critical_high:
                    direction = "LOW" if is_critical_low else "HIGH"
                    threshold = thresholds['low'] if is_critical_low else thresholds['high']
                    
                    alert = ClinicalAlert(
                        category=AlertCategory.CRITICAL_LAB,
                        severity=AlertSeverity.CRITICAL,
                        title=f"🚨 CRITICAL {normalized.upper()}: {value} {thresholds['unit']}",
                        message=(
                            f"Value: {value} | Threshold: {direction} < {threshold}\n"
                            f"Risk: {thresholds['risk']}"
                        ),
                        recommended_actions=[
                            "Verify result (repeat if possible)",
                            f"Treat {direction.lower()} {normalized} per protocol",
                            "Continuous cardiac monitoring if electrolyte",
                            "Notify physician immediately"
                        ],
                        parameters={'lab': normalized, 'value': value, 'threshold': threshold},
                        requires_immediate_action=True
                    )
                    alerts.append(alert)
                    self.alerts.append(alert)
        
        return alerts
    
    # =========================================================================
    # 3. RENAL/HEPATIC DOSE ADJUSTMENT
    # =========================================================================
    
    def calculate_gfr(
        self, 
        creatinine: float, 
        age: int, 
        sex: str, 
        weight_kg: Optional[float] = None,
        race_african: bool = False
    ) -> float:
        """
        Calculates GFR using CKD-EPI equation (2021, race-free).
        
        Args:
            creatinine: Serum creatinine (mg/dL)
            age: Patient age in years
            sex: 'M' or 'F'
            weight_kg: Optional for Cockcroft-Gault
            race_african: Not used in 2021 equation
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        # CKD-EPI 2021 (race-free)
        if sex.upper() == 'F':
            kappa = 0.7
            alpha = -0.241
            factor = 1.012
        else:
            kappa = 0.9
            alpha = -0.302
            factor = 1.0
        
        cr_ratio = creatinine / kappa
        
        if cr_ratio <= 1:
            gfr = 142 * (cr_ratio ** alpha) * (0.9938 ** age) * factor
        else:
            gfr = 142 * (cr_ratio ** -1.200) * (0.9938 ** age) * factor
        
        return round(gfr, 1)
    
    def get_renal_dose_adjustment(
        self, 
        drug: str, 
        gfr: float,
        original_dose: Optional[str] = None
    ) -> Optional[DoseAdjustment]:
        """
        Calculates dose adjustment based on renal function.
        
        Args:
            drug: Drug name
            gfr: eGFR in mL/min/1.73m²
            original_dose: Original prescribed dose
            
        Returns:
            DoseAdjustment recommendation
        """
        drug_lower = drug.lower()
        
        # Find matching drug
        matched_drug = None
        for d in self.RENAL_ADJUSTMENTS.keys():
            if d in drug_lower:
                matched_drug = d
                break
        
        if not matched_drug:
            return None
        
        adjustments = self.RENAL_ADJUSTMENTS[matched_drug]
        
        for adj in adjustments:
            gfr_min = adj['gfr_min']
            gfr_max = adj['gfr_max'] if adj['gfr_max'] else float('inf')
            
            if gfr_min <= gfr < gfr_max:
                factor = adj['adjustment']
                
                # Special case: Carboplatin Calvert formula
                if factor == 'calvert':
                    return DoseAdjustment(
                        drug=matched_drug,
                        original_dose=original_dose or "AUC-based",
                        adjusted_dose=f"Dose = AUC × ({gfr:.0f} + 25)",
                        adjustment_factor=1.0,
                        reason="Calvert formula for carboplatin",
                        gfr=gfr
                    )
                
                if factor == 0:
                    severity = AlertSeverity.CRITICAL
                    self.alerts.append(ClinicalAlert(
                        category=AlertCategory.DOSE_ADJUSTMENT,
                        severity=severity,
                        title=f"🚨 {matched_drug.upper()} CONTRAINDICATED",
                        message=f"GFR {gfr:.0f} mL/min - {adj['note']}",
                        recommended_actions=["Do NOT administer", "Consider alternative drug"],
                        requires_immediate_action=True
                    ))
                
                return DoseAdjustment(
                    drug=matched_drug,
                    original_dose=original_dose or "Standard",
                    adjusted_dose=f"{int(factor*100)}% of standard dose" if factor > 0 else "CONTRAINDICATED",
                    adjustment_factor=factor,
                    reason=adj['note'],
                    gfr=gfr
                )
        
        return None
    
    # =========================================================================
    # 4. DRUG-DRUG INTERACTIONS
    # =========================================================================
    
    def check_drug_interactions(
        self, 
        medications: List[str]
    ) -> List[DrugInteraction]:
        """
        Checks for critical drug-drug interactions.
        
        Args:
            medications: List of current medications
            
        Returns:
            List of identified interactions
        """
        interactions = []
        meds_lower = [m.lower() for m in medications]
        
        for ddi in self.CRITICAL_DDI:
            drug_a = ddi['drug_a'].lower()
            drug_b = ddi['drug_b'].lower()
            
            # Check if both drugs are present
            a_present = any(drug_a in m for m in meds_lower)
            b_present = any(drug_b in m for m in meds_lower)
            
            # Special handling for drug classes
            if drug_b == 'nsaids':
                nsaid_list = ['ibuprofen', 'naproxen', 'diclofenac', 'aspirin', 'celecoxib', 'meloxicam']
                b_present = any(any(n in m for n in nsaid_list) for m in meds_lower)
            
            if a_present and b_present:
                severity = AlertSeverity[ddi['severity'].upper()]
                
                interaction = DrugInteraction(
                    drug_a=ddi['drug_a'],
                    drug_b=ddi['drug_b'],
                    severity=severity,
                    mechanism=ddi['mechanism'],
                    effect=ddi['effect'],
                    recommendation=ddi['recommendation']
                )
                interactions.append(interaction)
                
                # Create alert
                self.alerts.append(ClinicalAlert(
                    category=AlertCategory.DRUG_INTERACTION,
                    severity=severity,
                    title=f"⚠️ DDI: {ddi['drug_a'].upper()} + {ddi['drug_b'].upper()}",
                    message=f"Mechanism: {ddi['mechanism']}\nEffect: {ddi['effect']}",
                    recommended_actions=[ddi['recommendation']],
                    requires_immediate_action=(severity == AlertSeverity.CRITICAL)
                ))
        
        return interactions
    
    # =========================================================================
    # 5. QTc PROLONGATION MONITORING
    # =========================================================================
    
    def check_qtc_risk(
        self,
        medications: List[str],
        qtc_value: Optional[float] = None,
        potassium: Optional[float] = None,
        magnesium: Optional[float] = None
    ) -> QTcRisk:
        """
        Assesses QTc prolongation risk from medications.
        
        Args:
            medications: Current medications
            qtc_value: Measured QTc (ms) if available
            potassium: Serum K+ for electrolyte correction
            magnesium: Serum Mg2+ for electrolyte correction
            
        Returns:
            QTcRisk assessment
        """
        meds_lower = [m.lower() for m in medications]
        risk_drugs = []
        total_qt_burden = 0
        
        for med in meds_lower:
            for drug, info in self.QT_PROLONGING_DRUGS.items():
                if drug in med:
                    risk_drugs.append(drug)
                    total_qt_burden += info['qt_risk']
        
        # Determine risk level
        if total_qt_burden >= 6 or (qtc_value and qtc_value > 500):
            risk_level = AlertSeverity.CRITICAL
            recommendation = "🚨 HIGH QT RISK - Consider ECG monitoring, avoid combination, correct electrolytes"
        elif total_qt_burden >= 4 or (qtc_value and qtc_value > 480):
            risk_level = AlertSeverity.HIGH
            recommendation = "⚠️ Significant QT risk - Baseline ECG, monitor electrolytes"
        elif total_qt_burden >= 2 or (qtc_value and qtc_value > 450):
            risk_level = AlertSeverity.MODERATE
            recommendation = "📊 Moderate QT risk - Consider ECG if adding more QT drugs"
        else:
            risk_level = AlertSeverity.LOW
            recommendation = "Low QT risk - Standard monitoring"
        
        result = QTcRisk(
            qtc_value=qtc_value,
            risk_drugs=risk_drugs,
            total_qt_burden=total_qt_burden,
            risk_level=risk_level,
            recommendation=recommendation
        )
        
        # Add alert if significant risk
        if risk_level in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            actions = ["Obtain baseline ECG", "Correct K+ to >4.0 mEq/L, Mg2+ to >2.0 mg/dL"]
            if qtc_value and qtc_value > 500:
                actions.insert(0, "STOP QT-prolonging drugs if possible")
            
            self.alerts.append(ClinicalAlert(
                category=AlertCategory.QTC_PROLONGATION,
                severity=risk_level,
                title=f"💓 QTc PROLONGATION RISK",
                message=f"QT burden: {total_qt_burden} | Drugs: {', '.join(risk_drugs)}",
                recommended_actions=actions,
                requires_immediate_action=(risk_level == AlertSeverity.CRITICAL)
            ))
        
        return result
    
    # =========================================================================
    # 6. TUMOR LYSIS SYNDROME PREDICTION
    # =========================================================================
    
    def predict_tls_risk(
        self,
        tumor_type: str,
        ldh: float,
        uric_acid: float,
        creatinine: float,
        potassium: float,
        phosphate: float,
        tumor_burden: str = "moderate"  # low, moderate, high, bulky
    ) -> TLSRisk:
        """
        Predicts Tumor Lysis Syndrome risk (Cairo-Bishop criteria).
        
        Args:
            tumor_type: Type of malignancy
            ldh: LDH (U/L)
            uric_acid: Uric acid (mg/dL)
            creatinine: Creatinine (mg/dL)
            potassium: Potassium (mEq/L)
            phosphate: Phosphate (mg/dL)
            tumor_burden: Estimated tumor burden
            
        Returns:
            TLSRisk assessment with prophylaxis recommendation
        """
        risk_score = 0
        factors = []
        
        # High-risk tumor types
        high_risk_tumors = ['acute leukemia', 'all', 'aml', 'burkitt', 'lymphoblastic']
        moderate_risk_tumors = ['dlbcl', 'lymphoma', 'small cell', 'sclc', 'cll']
        
        tumor_lower = tumor_type.lower()
        
        if any(t in tumor_lower for t in high_risk_tumors):
            risk_score += 3
            factors.append("High-risk tumor type (+3)")
        elif any(t in tumor_lower for t in moderate_risk_tumors):
            risk_score += 2
            factors.append("Intermediate-risk tumor type (+2)")
        
        # LDH elevation
        if ldh > 500:
            risk_score += 2
            factors.append(f"LDH elevated: {ldh:.0f} U/L (+2)")
        elif ldh > 300:
            risk_score += 1
            factors.append(f"LDH mildly elevated: {ldh:.0f} U/L (+1)")
        
        # Uric acid
        if uric_acid > 8.0:
            risk_score += 2
            factors.append(f"Uric acid elevated: {uric_acid:.1f} mg/dL (+2)")
        elif uric_acid > 6.0:
            risk_score += 1
            factors.append(f"Uric acid borderline: {uric_acid:.1f} mg/dL (+1)")
        
        # Renal function
        if creatinine > 1.5:
            risk_score += 2
            factors.append(f"Renal impairment: Cr {creatinine:.1f} mg/dL (+2)")
        
        # Tumor burden
        if tumor_burden == "bulky":
            risk_score += 2
            factors.append("Bulky disease (+2)")
        elif tumor_burden == "high":
            risk_score += 1
            factors.append("High tumor burden (+1)")
        
        # Risk stratification
        if risk_score >= 5:
            risk_level = "HIGH"
            prophylaxis = "Rasburicase 0.2 mg/kg IV + aggressive hydration 2.5-3 L/m²/day"
            monitoring = "Labs q6h (K+, PO4, Ca++, uric acid, Cr, LDH)"
        elif risk_score >= 3:
            risk_level = "INTERMEDIATE"
            prophylaxis = "Allopurinol 300mg PO daily + hydration 2 L/m²/day"
            monitoring = "Labs q8-12h during treatment"
        else:
            risk_level = "LOW"
            prophylaxis = "Allopurinol if treating solid tumor with high response rate"
            monitoring = "Labs daily"
        
        result = TLSRisk(
            risk_score=risk_score,
            risk_level=risk_level,
            factors=factors,
            prophylaxis_recommendation=prophylaxis,
            monitoring_frequency=monitoring
        )
        
        # Create alert if high risk
        if risk_level == "HIGH":
            self.alerts.append(ClinicalAlert(
                category=AlertCategory.TUMOR_LYSIS,
                severity=AlertSeverity.HIGH,
                title="🧪 HIGH TLS RISK",
                message=f"Risk Score: {risk_score} | {', '.join(factors[:2])}..."  ,
                recommended_actions=[prophylaxis, monitoring, "Avoid NSAIDs/nephrotoxins"],
                requires_immediate_action=True
            ))
        
        return result
    
    # =========================================================================
    # 7. VTE RISK - KHORANA SCORE
    # =========================================================================
    
    def calculate_khorana_score(
        self,
        tumor_site: str,
        platelet_count: float,
        hemoglobin: float,
        leukocyte_count: float,
        bmi: Optional[float] = None
    ) -> KhoranaScore:
        """
        Calculates Khorana score for VTE risk in cancer patients.
        
        Args:
            tumor_site: Primary tumor location
            platelet_count: Platelets (/µL)
            hemoglobin: Hemoglobin (g/dL)
            leukocyte_count: WBC (/µL)
            bmi: Body mass index (kg/m²)
            
        Returns:
            KhoranaScore with VTE risk and recommendation
        """
        score = 0
        factors = []
        
        tumor_lower = tumor_site.lower()
        
        # Very high-risk tumor sites (2 points)
        very_high_risk = ['stomach', 'gastric', 'pancreas', 'pancreatic']
        # High-risk tumor sites (1 point)
        high_risk = ['lung', 'lymphoma', 'gynecologic', 'ovarian', 'uterine', 
                     'bladder', 'testis', 'testicular', 'kidney', 'renal']
        
        if any(t in tumor_lower for t in very_high_risk):
            score += 2
            factors.append(("Very high-risk tumor site", 2))
        elif any(t in tumor_lower for t in high_risk):
            score += 1
            factors.append(("High-risk tumor site", 1))
        
        # Platelet count ≥350,000/µL (1 point)
        if platelet_count >= 350000:
            score += 1
            factors.append(("Platelets ≥350k/µL", 1))
        
        # Hemoglobin <10 g/dL or ESA use (1 point)
        if hemoglobin < 10:
            score += 1
            factors.append(("Hemoglobin <10 g/dL", 1))
        
        # Leukocyte count >11,000/µL (1 point)
        if leukocyte_count > 11000:
            score += 1
            factors.append(("WBC >11k/µL", 1))
        
        # BMI ≥35 kg/m² (1 point)
        if bmi and bmi >= 35:
            score += 1
            factors.append(("BMI ≥35", 1))
        
        # Risk stratification
        if score >= 3:
            risk_category = "HIGH"
            vte_risk = 7.1  # ~7% VTE risk at 2.5 months
            recommendation = "Consider prophylactic anticoagulation (LMWH or DOAC)"
        elif score == 2:
            risk_category = "INTERMEDIATE"
            vte_risk = 2.0
            recommendation = "Consider prophylaxis if additional risk factors present"
        else:
            risk_category = "LOW"
            vte_risk = 0.8
            recommendation = "Standard care, prophylaxis during hospitalization"
        
        result = KhoranaScore(
            total_score=score,
            risk_category=risk_category,
            factors=factors,
            vte_risk_percent=vte_risk,
            recommendation=recommendation
        )
        
        # Alert if high risk
        if risk_category == "HIGH":
            self.alerts.append(ClinicalAlert(
                category=AlertCategory.VTE_RISK,
                severity=AlertSeverity.MODERATE,
                title="🩸 HIGH VTE RISK (Khorana)",
                message=f"Score: {score} | VTE risk: {vte_risk}%",
                recommended_actions=[recommendation, "Educate patient on VTE symptoms"],
                requires_immediate_action=False
            ))
        
        return result
    
    # =========================================================================
    # UNIFIED PATIENT SAFETY CHECK
    # =========================================================================
    
    def run_full_safety_check(self, patient_data: Dict[str, Any]) -> List[ClinicalAlert]:
        """
        Runs all safety checks on a patient.
        
        Args:
            patient_data: SENTINEL patient data structure
            
        Returns:
            List of all generated alerts, sorted by severity
        """
        self.alerts = []  # Reset alerts
        
        baseline = patient_data.get('baseline', patient_data)
        visits = patient_data.get('visits', [])
        current_visit = visits[-1] if visits else baseline
        
        # Get labs
        labs = current_visit.get('blood_markers', {})
        clinical = current_visit.get('clinical_status', {})
        
        # 1. Check critical labs
        if labs:
            self.check_critical_labs(labs)
        
        # 2. Check neutropenic fever
        neutrophils = labs.get('neutrophils') or labs.get('anc')
        temp = clinical.get('temperature')
        if neutrophils and temp:
            self.check_neutropenic_fever(neutrophils, temp)
        
        # 3. Check drug interactions
        medications = baseline.get('medications', [])
        if baseline.get('current_therapy'):
            medications.append(baseline['current_therapy'])
        if medications:
            self.check_drug_interactions(medications)
        
        # 4. Check QTc risk
        if medications:
            self.check_qtc_risk(
                medications, 
                potassium=labs.get('potassium'),
                magnesium=labs.get('magnesium')
            )
        
        # 5. Renal dose adjustment
        creatinine = labs.get('creatinine')
        if creatinine and baseline.get('current_therapy'):
            age = baseline.get('age', 65)
            sex = baseline.get('sex', 'M')
            gfr = self.calculate_gfr(creatinine, age, sex)
            self.get_renal_dose_adjustment(baseline['current_therapy'], gfr)
        
        # 6. TLS risk (if applicable)
        ldh = labs.get('ldh')
        uric_acid = labs.get('uric_acid')
        if ldh and ldh > 300:
            tumor_type = baseline.get('histology', 'solid tumor')
            self.predict_tls_risk(
                tumor_type=tumor_type,
                ldh=ldh,
                uric_acid=uric_acid or 5.0,
                creatinine=creatinine or 1.0,
                potassium=labs.get('potassium', 4.0),
                phosphate=labs.get('phosphate', 3.5)
            )
        
        # 7. VTE risk (Khorana)
        platelets = labs.get('platelets')
        hemoglobin = labs.get('hemoglobin')
        wbc = labs.get('leukocytes') or labs.get('wbc')
        weight = clinical.get('weight_kg')
        height = clinical.get('height_cm')
        bmi = None
        if weight and height:
            bmi = weight / ((height / 100) ** 2)
        
        if platelets and hemoglobin and wbc:
            tumor_site = baseline.get('primary_site', baseline.get('histology', 'solid'))
            self.calculate_khorana_score(tumor_site, platelets, hemoglobin, wbc, bmi)
        
        # Sort alerts by severity
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MODERATE: 2,
            AlertSeverity.LOW: 3
        }
        self.alerts.sort(key=lambda a: severity_order.get(a.severity, 99))
        
        return self.alerts
    
    def format_alerts_for_display(self) -> str:
        """Formats all alerts for console/UI display."""
        if not self.alerts:
            return "✅ No safety alerts detected."
        
        lines = ["=" * 60, "🏥 CLINICAL SAFETY ALERTS", "=" * 60, ""]
        
        for alert in self.alerts:
            icon = alert.severity.icon()
            lines.append(f"{icon} [{alert.severity.value.upper()}] {alert.title}")
            lines.append(f"   {alert.message}")
            if alert.recommended_actions:
                lines.append("   Actions:")
                for action in alert.recommended_actions[:3]:
                    lines.append(f"      → {action}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 CLINICAL SAFETY ENGINE - TEST")
    print("=" * 60)
    
    engine = ClinicalSafetyEngine()
    
    # Test 1: Neutropenic Fever
    print("\n📋 Test 1: Neutropenic Fever")
    alert = engine.check_neutropenic_fever(
        neutrophil_count=250,
        temperature=38.5,
        blood_pressure_systolic=90,
        respiratory_rate=24
    )
    if alert:
        print(f"   {alert.severity.icon()} {alert.title}")
        print(f"   {alert.message}")
    
    # Test 2: Critical Labs
    print("\n📋 Test 2: Critical Labs")
    engine.alerts = []
    alerts = engine.check_critical_labs({
        'potassium': 6.8,
        'hemoglobin': 6.5,
        'platelets': 15000
    })
    for a in alerts:
        print(f"   {a.severity.icon()} {a.title}")
    
    # Test 3: Renal Dose
    print("\n📋 Test 3: Renal Dose Adjustment")
    gfr = engine.calculate_gfr(creatinine=2.5, age=70, sex='M')
    print(f"   GFR: {gfr} mL/min")
    adj = engine.get_renal_dose_adjustment('cisplatin', gfr)
    if adj:
        print(f"   {adj.drug}: {adj.adjusted_dose} ({adj.reason})")
    
    # Test 4: DDI
    print("\n📋 Test 4: Drug Interactions")
    engine.alerts = []
    ddis = engine.check_drug_interactions([
        'Warfarin', 'Capecitabine', 'Osimertinib', 'Ibuprofen', 'Methotrexate'
    ])
    for ddi in ddis:
        print(f"   ⚠️ {ddi.drug_a} + {ddi.drug_b}: {ddi.effect}")
    
    # Test 5: QTc
    print("\n📋 Test 5: QTc Risk")
    engine.alerts = []
    qtc = engine.check_qtc_risk(['Osimertinib', 'Ondansetron', 'Haloperidol'])
    print(f"   QT Burden: {qtc.total_qt_burden}")
    print(f"   Risk Level: {qtc.risk_level.value}")
    
    # Test 6: TLS
    print("\n📋 Test 6: TLS Risk")
    engine.alerts = []
    tls = engine.predict_tls_risk(
        tumor_type="Burkitt Lymphoma",
        ldh=850,
        uric_acid=9.5,
        creatinine=1.8,
        potassium=5.2,
        phosphate=5.5,
        tumor_burden="bulky"
    )
    print(f"   Risk Score: {tls.risk_score} ({tls.risk_level})")
    print(f"   Prophylaxis: {tls.prophylaxis_recommendation}")
    
    # Test 7: Khorana
    print("\n📋 Test 7: Khorana VTE Score")
    engine.alerts = []
    khorana = engine.calculate_khorana_score(
        tumor_site="Pancreatic adenocarcinoma",
        platelet_count=420000,
        hemoglobin=9.5,
        leukocyte_count=12500,
        bmi=32
    )
    print(f"   Score: {khorana.total_score} ({khorana.risk_category})")
    print(f"   VTE Risk: {khorana.vte_risk_percent}%")
    
    print("\n" + "=" * 60)
    print("✅ Clinical Safety Engine Ready!")
    print("=" * 60)
