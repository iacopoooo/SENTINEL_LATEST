"""
SENTINEL v3.0 - Bayesian Network Early Warning System for Cancer Resistance
=============================================================================
Sistema di inferenza Bayesiana AVANZATO per predizione resistenza tumorale.

Features v3.0:
- Correlation Priors: meccanismi biologicamente correlati
- Temporal Dependencies: l'ordine temporale delle evidenze modifica i LR
- Pathway Clustering: raggruppamento per pathway biologici
- Conditional Probability Updates: P(A|B) quando B è già osservato
- Time-weighted evidence decay: evidenze recenti pesano di più

Features v3.1 (Cancer-Specific):
- Cancer-specific priors per ogni tipo di tumore
- Meccanismi di resistenza appropriati per istologia
- Selezione automatica basata su diagnosi/genetica
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


# ============================================================================
# ENUMS
# ============================================================================

class RiskLevel(Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Urgency(Enum):
    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TemporalPhase(Enum):
    """Fase temporale del trattamento"""
    EARLY = "EARLY"  # 0-6 mesi
    INTERMEDIATE = "INTERMEDIATE"  # 6-12 mesi
    LATE = "LATE"  # 12-24 mesi
    VERY_LATE = "VERY_LATE"  # >24 mesi


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CredibilityInterval:
    point_estimate: float
    ci_low: float
    ci_high: float
    uncertainty: float
    evidence_weight: float

    def __str__(self):
        return f"{self.point_estimate:.1%} (95% CI: {self.ci_low:.1%}-{self.ci_high:.1%})"


@dataclass
class TreatmentRecommendation:
    mechanism: str
    probability: float
    risk_level: RiskLevel
    action: str
    primary_drug: str
    alternative_drug: str
    evidence_level: str
    urgency: Urgency
    monitoring_interval: str
    correlated_risks: List[str] = field(default_factory=list)
    additional_notes: str = ""


@dataclass
class TemporalEvidence:
    """Evidenza con timestamp per tracking temporale"""
    evidence: str
    timestamp: datetime
    visit_week: int
    source: str  # 'baseline', 'ctdna', 'biopsy', 'imaging'


# ============================================================================
# BIOLOGICAL CORRELATION NETWORKS
# ============================================================================

CORRELATION_NETWORK = {
    'EMT_phenotype': {
        'MET_amplification': 0.6,
        'STK11_loss': 0.7,
        'KEAP1_loss': 0.6,
        'C797S_mutation': 0.3,
        'HER2_amplification': 0.2,
    },
    'MET_amplification': {
        'EMT_phenotype': 0.6,
        'HER2_amplification': 0.4,
        'PIK3CA_mutation': 0.3,
        'EGFR_amplification': 0.3,
    },
    'SCLC_transformation': {
        'Squamous_transformation': -0.8,
        'EMT_phenotype': 0.4,
        'PIK3CA_mutation': 0.2,
    },
    'Squamous_transformation': {
        'SCLC_transformation': -0.8,
        'EMT_phenotype': 0.3,
    },
    'C797S_mutation': {
        'T790M_acquired': -0.6,
        'EGFR_amplification': 0.4,
    },
    'T790M_acquired': {
        'C797S_mutation': -0.6,
        'EGFR_amplification': 0.3,
    },
    'HER2_amplification': {
        'MET_amplification': 0.4,
        'PIK3CA_mutation': 0.5,
        'BRAF_fusion_mutation': 0.2,
    },
    'PIK3CA_mutation': {
        'HER2_amplification': 0.5,
        'MET_amplification': 0.3,
        'KRAS_activation': -0.7,
    },
    'BRAF_fusion_mutation': {
        'KRAS_activation': -0.9,
        'HER2_amplification': 0.2,
    },
    'KRAS_activation': {
        'PIK3CA_mutation': -0.7,
        'BRAF_fusion_mutation': -0.9,
    },
    'EGFR_amplification': {
        'C797S_mutation': 0.4,
        'T790M_acquired': 0.3,
        'MET_amplification': 0.2,
    },
}


# ============================================================================
# TEMPORAL MODIFIERS
# ============================================================================

TEMPORAL_LR_MODIFIERS = {
    'C797S_mutation': {
        TemporalPhase.EARLY: 1.5,
        TemporalPhase.INTERMEDIATE: 1.2,
        TemporalPhase.LATE: 0.8,
        TemporalPhase.VERY_LATE: 0.6,
    },
    'T790M_acquired': {
        TemporalPhase.EARLY: 1.3,
        TemporalPhase.INTERMEDIATE: 1.0,
        TemporalPhase.LATE: 0.7,
        TemporalPhase.VERY_LATE: 0.5,
    },
    'SCLC_transformation': {
        TemporalPhase.EARLY: 0.5,
        TemporalPhase.INTERMEDIATE: 0.8,
        TemporalPhase.LATE: 1.3,
        TemporalPhase.VERY_LATE: 1.5,
    },
    'Squamous_transformation': {
        TemporalPhase.EARLY: 0.4,
        TemporalPhase.INTERMEDIATE: 0.7,
        TemporalPhase.LATE: 1.2,
        TemporalPhase.VERY_LATE: 1.4,
    },
    'EMT_phenotype': {
        TemporalPhase.EARLY: 0.6,
        TemporalPhase.INTERMEDIATE: 1.0,
        TemporalPhase.LATE: 1.3,
        TemporalPhase.VERY_LATE: 1.4,
    },
    'MET_amplification': {
        TemporalPhase.EARLY: 1.0,
        TemporalPhase.INTERMEDIATE: 1.1,
        TemporalPhase.LATE: 1.2,
        TemporalPhase.VERY_LATE: 1.0,
    },
    'HER2_amplification': {
        TemporalPhase.EARLY: 0.8,
        TemporalPhase.INTERMEDIATE: 1.0,
        TemporalPhase.LATE: 1.2,
        TemporalPhase.VERY_LATE: 1.1,
    },
}

TEMPORAL_PATTERNS = {
    'rapid_vaf_stable_imaging': {
        'evidences': ['VAF_rising_rapid', 'Stable_disease'],
        'boost': {'C797S_mutation': 1.5, 'MET_amplification': 1.3},
        'description': 'Clone emergente pre-clinico'
    },
    'pd_stable_vaf': {
        'evidences': ['Progressive_disease', 'VAF_stable'],
        'boost': {'SCLC_transformation': 1.8, 'EMT_phenotype': 1.6, 'Squamous_transformation': 1.4},
        'description': 'Possibile transformation (EGFR-independent)'
    },
    'c797s_with_emt': {
        'evidences': ['ctDNA_C797S_confirmed', 'Vimentin_high'],
        'boost': {'EMT_phenotype': 2.0},
        'description': 'Resistenza multifattoriale'
    },
}


# ============================================================================
# MUTUAL EXCLUSIVITY GROUPS
# ============================================================================

MUTEX_GROUPS = {
    'MET_cn': ['MET_cn_gain_low', 'MET_cn_gain_medium', 'MET_cn_gain_high'],
    'HER2_cn': ['HER2_cn_gain_low', 'HER2_cn_gain_medium', 'HER2_cn_gain_high'],
    'VAF': ['VAF_decreasing', 'VAF_stable', 'VAF_rising_mild', 'VAF_rising_moderate', 'VAF_rising_rapid'],
    'response': ['Complete_response', 'Partial_response', 'Stable_disease', 'Progressive_disease'],
    'C797S': ['ctDNA_C797S_trace', 'ctDNA_C797S_confirmed'],
}

MUTEX_PRIORITY = {
    'MET_cn': ['MET_cn_gain_high', 'MET_cn_gain_medium', 'MET_cn_gain_low'],
    'HER2_cn': ['HER2_cn_gain_high', 'HER2_cn_gain_medium', 'HER2_cn_gain_low'],
    'VAF': ['VAF_rising_rapid', 'VAF_rising_moderate', 'VAF_rising_mild', 'VAF_stable', 'VAF_decreasing'],
    'response': ['Progressive_disease', 'Stable_disease', 'Partial_response', 'Complete_response'],
    'C797S': ['ctDNA_C797S_confirmed', 'ctDNA_C797S_trace'],
}


# ============================================================================
# CANCER-SPECIFIC PRIORS (Fase 3)
# ============================================================================

CANCER_SPECIFIC_PRIORS = {
    # =========================================================================
    # NSCLC (Non-Small Cell Lung Cancer)
    # =========================================================================
    'NSCLC_EGFR': {
        'C797S_mutation': 0.15,
        'T790M_acquired': 0.08,
        'MET_amplification': 0.15,
        'HER2_amplification': 0.05,
        'EGFR_amplification': 0.08,
        'PIK3CA_mutation': 0.05,
        'BRAF_fusion_mutation': 0.03,
        'KRAS_activation': 0.02,
        'SCLC_transformation': 0.05,
        'Squamous_transformation': 0.02,
        'EMT_phenotype': 0.05,
    },
    'NSCLC_KRAS': {
        'STK11_loss': 0.25,
        'KEAP1_loss': 0.20,
        'MET_amplification': 0.10,
        'PIK3CA_mutation': 0.08,
        'EMT_phenotype': 0.10,
        'CDKN2A_loss': 0.15,
        'Immune_evasion': 0.20,
        'SCLC_transformation': 0.02,
    },
    'NSCLC_ALK': {
        'ALK_resistance_mutations': 0.25,
        'ALK_amplification': 0.10,
        'MET_amplification': 0.12,
        'EGFR_activation': 0.05,
        'EMT_phenotype': 0.08,
        'PIK3CA_mutation': 0.05,
    },
    'NSCLC_OTHER': {
        'MET_amplification': 0.10,
        'HER2_amplification': 0.05,
        'PIK3CA_mutation': 0.05,
        'EMT_phenotype': 0.08,
        'Immune_evasion': 0.15,
        'SCLC_transformation': 0.03,
    },

    # =========================================================================
    # BREAST CANCER
    # =========================================================================
    'BREAST_HER2': {
        'HER2_mutations': 0.15,
        'PIK3CA_mutation': 0.20,
        'PTEN_loss': 0.15,
        'ER_loss': 0.10,
        'MET_amplification': 0.05,
        'EMT_phenotype': 0.10,
        'CDK4_6_resistance': 0.08,
    },
    'BREAST_ER': {
        'ESR1_mutations': 0.25,
        'PIK3CA_mutation': 0.20,
        'CDK4_6_resistance': 0.15,
        'FGFR_amplification': 0.08,
        'HER2_activation': 0.05,
        'EMT_phenotype': 0.08,
        'PTEN_loss': 0.10,
    },
    'BREAST_TNBC': {
        'EMT_phenotype': 0.20,
        'PIK3CA_mutation': 0.15,
        'Immune_evasion': 0.20,
        'BRCA_reversion': 0.10,
        'MET_amplification': 0.08,
        'DNA_repair_restoration': 0.12,
    },
    'BREAST_OTHER': {
        'PIK3CA_mutation': 0.15,
        'EMT_phenotype': 0.12,
        'Immune_evasion': 0.15,
        'PTEN_loss': 0.10,
        'MET_amplification': 0.05,
    },

    # =========================================================================
    # BLADDER / UROTHELIAL CANCER
    # =========================================================================
    'BLADDER': {
        'FGFR_alterations': 0.20,
        'RB1_loss': 0.12,
        'Immune_evasion': 0.25,
        'EMT_phenotype': 0.15,
        'DNA_repair_defects': 0.10,
        'PIK3CA_mutation': 0.08,
        'CDKN2A_loss': 0.15,
        'Neuroendocrine_transformation': 0.03,
    },

    # =========================================================================
    # MELANOMA
    # =========================================================================
    'MELANOMA_BRAF': {
        'BRAF_amplification': 0.15,
        'MEK_mutations': 0.12,
        'NRAS_mutation': 0.10,
        'PTEN_loss': 0.15,
        'Immune_evasion': 0.20,
        'Beta_catenin_activation': 0.08,
        'COX2_upregulation': 0.05,
    },
    'MELANOMA_OTHER': {
        'Immune_evasion': 0.30,
        'PTEN_loss': 0.12,
        'Beta_catenin_activation': 0.10,
        'JAK_mutations': 0.08,
        'B2M_loss': 0.10,
        'EMT_phenotype': 0.08,
    },

    # =========================================================================
    # COLORECTAL CANCER
    # =========================================================================
    'COLORECTAL_KRAS': {
        'KRAS_amplification': 0.10,
        'MET_amplification': 0.12,
        'HER2_amplification': 0.08,
        'PIK3CA_mutation': 0.15,
        'BRAF_mutation': 0.05,
        'EMT_phenotype': 0.10,
    },
    'COLORECTAL_BRAF': {
        'BRAF_amplification': 0.12,
        'KRAS_mutation': 0.05,
        'PIK3CA_mutation': 0.12,
        'Immune_evasion': 0.15,
        'EMT_phenotype': 0.10,
    },
    'COLORECTAL_OTHER': {
        'KRAS_mutation': 0.15,
        'PIK3CA_mutation': 0.10,
        'MET_amplification': 0.08,
        'HER2_amplification': 0.05,
        'Immune_evasion': 0.12,
    },

    # =========================================================================
    # GASTRIC / GEJ CANCER
    # =========================================================================
    'GASTRIC': {
        'HER2_loss': 0.15,
        'MET_amplification': 0.12,
        'FGFR2_amplification': 0.10,
        'PIK3CA_mutation': 0.08,
        'Claudin_loss': 0.10,
        'EMT_phenotype': 0.12,
        'Immune_evasion': 0.15,
    },

    # =========================================================================
    # OVARIAN CANCER
    # =========================================================================
    'OVARIAN': {
        'BRCA_reversion': 0.20,
        'HR_restoration': 0.15,
        'ABCB1_upregulation': 0.12,
        'EMT_phenotype': 0.10,
        'PIK3CA_mutation': 0.08,
        'TP53_GOF': 0.10,
        'Immune_evasion': 0.12,
    },

    # =========================================================================
    # PROSTATE CANCER
    # =========================================================================
    'PROSTATE': {
        'AR_amplification': 0.25,
        'AR_mutations': 0.20,
        'AR_splice_variants': 0.15,
        'Glucocorticoid_receptor': 0.08,
        'Neuroendocrine_transformation': 0.10,
        'PTEN_loss': 0.12,
        'DNA_repair_defects': 0.08,
    },

    # =========================================================================
    # RENAL CELL CARCINOMA
    # =========================================================================
    'RENAL': {
        'MET_amplification': 0.12,
        'Immune_evasion': 0.25,
        'EMT_phenotype': 0.15,
        'mTOR_pathway': 0.10,
        'PBRM1_loss': 0.08,
        'Sarcomatoid_transformation': 0.05,
    },

    # =========================================================================
    # PANCREATIC CANCER
    # =========================================================================
    'PANCREATIC': {
        'KRAS_amplification': 0.15,
        'Stromal_barrier': 0.25,
        'EMT_phenotype': 0.20,
        'DNA_repair_defects': 0.10,
        'Immune_evasion': 0.20,
        'Metabolic_reprogramming': 0.12,
    },

    # =========================================================================
    # HEAD AND NECK CANCER
    # =========================================================================
    'HEAD_NECK': {
        'PIK3CA_mutation': 0.15,
        'Immune_evasion': 0.25,
        'EGFR_amplification': 0.12,
        'EMT_phenotype': 0.12,
        'HPV_status_change': 0.05,
        'NOTCH_mutations': 0.08,
    },

    # =========================================================================
    # HEPATOCELLULAR CARCINOMA
    # =========================================================================
    'LIVER': {
        'Immune_evasion': 0.25,
        'Beta_catenin_activation': 0.15,
        'MET_amplification': 0.10,
        'VEGF_independence': 0.12,
        'AFP_independence': 0.08,
        'EMT_phenotype': 0.10,
    },

    # =========================================================================
    # DEFAULT (fallback per tipi non specificati)
    # =========================================================================
    'DEFAULT': {
        'Immune_evasion': 0.15,
        'EMT_phenotype': 0.10,
        'PIK3CA_mutation': 0.08,
        'MET_amplification': 0.08,
        'Drug_efflux': 0.10,
        'Target_downregulation': 0.10,
        'Bypass_signaling': 0.10,
        'Metabolic_reprogramming': 0.08,
    },
}


def get_cancer_type_key(diagnosis: str, histology: str, genetics: dict) -> str:
    """
    Determina la chiave del tipo di tumore basandosi su diagnosi, istologia e genetica.

    Args:
        diagnosis: Diagnosi del paziente (es: "NSCLC", "Breast Cancer")
        histology: Istologia (es: "Adenocarcinoma", "Urothelial")
        genetics: Dict con status genetici

    Returns:
        Chiave per CANCER_SPECIFIC_PRIORS
    """
    diag_lower = str(diagnosis).lower() if diagnosis else ""
    hist_lower = str(histology).lower() if histology else ""

    # Estrai mutazioni chiave
    egfr = str(genetics.get('egfr_status', '')).lower() if genetics else ""
    kras = str(genetics.get('kras_mutation', '')).lower() if genetics else ""
    alk = str(genetics.get('alk_status', '')).lower() if genetics else ""
    braf = str(genetics.get('braf_status', genetics.get('braf_mutation', ''))).lower() if genetics else ""
    her2 = str(genetics.get('her2_status', '')).lower() if genetics else ""

    # =========================================================================
    # LUNG CANCER
    # =========================================================================
    if any(x in diag_lower for x in ['nsclc', 'lung', 'polmone', 'polmonare']):
        if 'small cell' in hist_lower or 'sclc' in diag_lower:
            return 'DEFAULT'
        if egfr and egfr not in ['wt', 'wild-type', 'negative', '']:
            return 'NSCLC_EGFR'
        if kras and kras not in ['wt', 'wild-type', 'negative', '']:
            return 'NSCLC_KRAS'
        if alk and 'rearrangement' in alk:
            return 'NSCLC_ALK'
        return 'NSCLC_OTHER'

    # =========================================================================
    # BREAST CANCER
    # =========================================================================
    if any(x in diag_lower for x in ['breast', 'mammella', 'mammario', 'ductal', 'lobular']):
        if her2 and her2 not in ['wt', 'wild-type', 'negative', '']:
            return 'BREAST_HER2'
        if 'er+' in diag_lower or 'er positive' in diag_lower or 'hormone' in diag_lower:
            return 'BREAST_ER'
        if 'triple negative' in diag_lower or 'tnbc' in diag_lower:
            return 'BREAST_TNBC'
        return 'BREAST_OTHER'
    if any(x in hist_lower for x in ['ductal', 'lobular']):
        return 'BREAST_OTHER'

    # =========================================================================
    # BLADDER / UROTHELIAL
    # =========================================================================
    if any(x in diag_lower for x in ['bladder', 'urothelial', 'vescica', 'uroteliale']):
        return 'BLADDER'
    if any(x in hist_lower for x in ['urothelial', 'uroteliale', 'transitional']):
        return 'BLADDER'

    # =========================================================================
    # MELANOMA
    # =========================================================================
    if any(x in diag_lower for x in ['melanoma', 'melanocytic']):
        if braf and 'v600' in braf:
            return 'MELANOMA_BRAF'
        return 'MELANOMA_OTHER'

    # =========================================================================
    # COLORECTAL
    # =========================================================================
    if any(x in diag_lower for x in ['colorectal', 'colon', 'rectal', 'retto', 'colon-retto']):
        if kras and kras not in ['wt', 'wild-type', 'negative', '']:
            return 'COLORECTAL_KRAS'
        if braf and braf not in ['wt', 'wild-type', 'negative', '']:
            return 'COLORECTAL_BRAF'
        return 'COLORECTAL_OTHER'

    # =========================================================================
    # GASTRIC / GEJ
    # =========================================================================
    if any(x in diag_lower for x in ['gastric', 'stomach', 'gastroesophageal', 'gej', 'stomaco']):
        return 'GASTRIC'

    # =========================================================================
    # OVARIAN
    # =========================================================================
    if any(x in diag_lower for x in ['ovarian', 'ovary', 'ovaio', 'ovarico']):
        return 'OVARIAN'

    # =========================================================================
    # PROSTATE
    # =========================================================================
    if any(x in diag_lower for x in ['prostate', 'prostata', 'prostatico']):
        return 'PROSTATE'

    # =========================================================================
    # RENAL
    # =========================================================================
    if any(x in diag_lower for x in ['renal', 'kidney', 'rene', 'renale', 'rcc']):
        return 'RENAL'

    # =========================================================================
    # PANCREATIC
    # =========================================================================
    if any(x in diag_lower for x in ['pancrea', 'pdac']):
        return 'PANCREATIC'

    # =========================================================================
    # HEAD AND NECK
    # =========================================================================
    if any(x in diag_lower for x in ['head and neck', 'hnscc', 'oral', 'oropharynx', 'larynx']):
        return 'HEAD_NECK'

    # =========================================================================
    # LIVER / HCC
    # =========================================================================
    if any(x in diag_lower for x in ['hepatocellular', 'liver', 'hcc', 'fegato', 'epatico']):
        return 'LIVER'

    # =========================================================================
    # FALLBACK
    # =========================================================================
    return 'DEFAULT'


# ============================================================================
# SENTINEL v3.0 ENGINE
# ============================================================================

class SentinelV30:
    """
    Bayesian Network Early Warning System v3.0

    Miglioramenti rispetto a v2.5:
    - Correlation priors tra meccanismi
    - Temporal dependencies
    - Evidence decay (evidenze recenti pesano di più)
    - Pattern recognition temporale
    - Cancer-specific priors (v3.1)
    """

    VERSION = "3.1"

    def __init__(self, patient_id: str = "UNKNOWN", therapy_start_date: str = None,
                 diagnosis: str = None, histology: str = None, genetics: dict = None):
        """
        Inizializza il motore Bayesiano con priors cancer-specific.

        Args:
            patient_id: ID del paziente
            therapy_start_date: Data inizio terapia (YYYY-MM-DD)
            diagnosis: Diagnosi (es: "NSCLC", "Breast Cancer", "Bladder Urothelial Carcinoma")
            histology: Istologia (es: "Adenocarcinoma", "Mixed Ductal and Lobular")
            genetics: Dict con status genetici (egfr_status, kras_mutation, etc.)
        """
        self.patient_id = patient_id
        self.creation_time = datetime.now()
        self.last_update = datetime.now()
        self.n_updates = 0

        # Therapy timeline
        if therapy_start_date:
            try:
                self.therapy_start = datetime.strptime(therapy_start_date, '%Y-%m-%d')
            except:
                self.therapy_start = datetime.now()
        else:
            self.therapy_start = datetime.now()

        # =====================================================================
        # CANCER-SPECIFIC PRIORS (Fase 3)
        # =====================================================================
        if diagnosis or histology:
            self.cancer_type_key = get_cancer_type_key(
                diagnosis or "",
                histology or "",
                genetics or {}
            )
        else:
            self.cancer_type_key = 'NSCLC_EGFR'  # Default legacy

        # Usa priors specifici per questo tipo di tumore
        self.base_priors = CANCER_SPECIFIC_PRIORS.get(
            self.cancer_type_key,
            CANCER_SPECIFIC_PRIORS['DEFAULT']
        ).copy()

        self.hypotheses = self.base_priors.copy()

        # Log del tipo selezionato
        print(f"🏎️  [FERRARI] Cancer type: {self.cancer_type_key} ({len(self.base_priors)} mechanisms)")

        # Severity multiplier (will be set by calibrate_priors)
        self.severity_multiplier = 1.0

        # Likelihood Ratios base
        self.evidence_strength = {
            # Baseline genomic
            'TP53_loss': {'SCLC_transformation': 3.0, 'EMT_phenotype': 1.5, 'Squamous_transformation': 1.3},
            'RB1_loss': {'SCLC_transformation': 8.0},
            'TP53_RB1_double_loss': {'SCLC_transformation': 25.0, 'EMT_phenotype': 2.0},
            'PIK3CA_baseline': {'PIK3CA_mutation': 5.0, 'HER2_amplification': 1.3},

            'STK11_loss': {
                'EMT_phenotype': 8.0,
                'MET_amplification': 3.0,
                'SCLC_transformation': 2.0
            },
            'KEAP1_loss': {
                'EMT_phenotype': 6.0,
                'MET_amplification': 2.5,
                'PIK3CA_mutation': 2.0
            },
            'STK11_KEAP1_double_loss': {
                'EMT_phenotype': 15.0,
                'MET_amplification': 4.0,
                'SCLC_transformation': 3.5
            },

            # Copy number
            'MET_cn_gain_low': {'MET_amplification': 2.5, 'EMT_phenotype': 1.2},
            'MET_cn_gain_medium': {'MET_amplification': 8.0, 'EMT_phenotype': 1.5},
            'MET_cn_gain_high': {'MET_amplification': 50.0, 'EMT_phenotype': 2.0, 'HER2_amplification': 1.3},
            'HER2_cn_gain_low': {'HER2_amplification': 2.5},
            'HER2_cn_gain_medium': {'HER2_amplification': 8.0, 'PIK3CA_mutation': 1.2},
            'HER2_cn_gain_high': {'HER2_amplification': 30.0, 'PIK3CA_mutation': 1.5},
            'EGFR_cn_gain': {'EGFR_amplification': 10.0, 'C797S_mutation': 1.3},

            # ctDNA mutations
            'ctDNA_C797S_trace': {'C797S_mutation': 20.0},
            'ctDNA_C797S_confirmed': {'C797S_mutation': 100.0, 'EGFR_amplification': 1.5},
            'ctDNA_T790M_trace': {'T790M_acquired': 15.0},
            'ctDNA_PIK3CA_detected': {'PIK3CA_mutation': 25.0, 'HER2_amplification': 1.5},
            'ctDNA_BRAF_detected': {'BRAF_fusion_mutation': 30.0},
            'ctDNA_KRAS_detected': {'KRAS_activation': 50.0},

            # VAF dynamics
            'VAF_decreasing': {
                'C797S_mutation': 0.7, 'MET_amplification': 0.7,
                'SCLC_transformation': 0.8, 'EMT_phenotype': 0.6
            },
            'VAF_stable': {},
            'VAF_rising_mild': {
                'C797S_mutation': 1.5, 'MET_amplification': 1.5,
                'SCLC_transformation': 1.3, 'EMT_phenotype': 1.4
            },
            'VAF_rising_moderate': {
                'C797S_mutation': 2.5, 'MET_amplification': 2.5,
                'SCLC_transformation': 2.0, 'EMT_phenotype': 2.0
            },
            'VAF_rising_rapid': {
                'C797S_mutation': 5.0, 'MET_amplification': 5.0,
                'SCLC_transformation': 4.0, 'EMT_phenotype': 3.5
            },

            # Clinical response
            'Complete_response': {
                'C797S_mutation': 0.3, 'MET_amplification': 0.4,
                'SCLC_transformation': 0.5, 'EMT_phenotype': 0.3
            },
            'Partial_response': {
                'C797S_mutation': 0.6, 'MET_amplification': 0.7,
                'SCLC_transformation': 0.7, 'EMT_phenotype': 0.6
            },
            'Stable_disease': {},
            'Progressive_disease': {
                'C797S_mutation': 2.0, 'MET_amplification': 2.0,
                'SCLC_transformation': 2.5, 'EMT_phenotype': 2.5
            },

            # New mutations
            'New_mutation_detected': {
                'C797S_mutation': 1.3, 'MET_amplification': 1.2, 'EMT_phenotype': 1.5
            },

            # Biopsy markers
            'Vimentin_high': {'EMT_phenotype': 4.0, 'MET_amplification': 1.5},
            'E_cadherin_loss': {'EMT_phenotype': 5.0, 'SCLC_transformation': 1.3},
            'Synaptophysin_positive': {'SCLC_transformation': 15.0},
            'Chromogranin_positive': {'SCLC_transformation': 15.0},
            'p40_positive': {'Squamous_transformation': 20.0},
            'AXL_overexpression': {'EMT_phenotype': 3.0, 'C797S_mutation': 1.5, 'MET_amplification': 1.8},

            # AlphaFold integration
            'structural_docking_high': {
                'C797S_mutation': 0.05,
                'T790M_acquired': 500.0,
                'KRAS_activation': 500.0,
                'MET_amplification': 450.0,
                'BRAF_fusion_mutation': 450.0,
                'HER2_amplification': 400.0
            },
            'structural_docking_low': {
                'C797S_mutation': 1000.0,
                'T790M_acquired': 0.01,
                'KRAS_activation': 0.01,
                'SCLC_transformation': 50.0
            },
        }

        # Treatment recommendations
        self.treatments = {
            'C797S_mutation': ('Brigatinib + Cetuximab', 'Amivantamab', 'Phase I/II', Urgency.HIGH),
            'T790M_acquired': ('Osimertinib', 'Lazertinib', 'FDA approved', Urgency.HIGH),
            'MET_amplification': ('Capmatinib o Savolitinib', 'Tepotinib', 'FDA approved', Urgency.HIGH),
            'HER2_amplification': ('Trastuzumab Deruxtecan', 'Poziotinib', 'Phase II', Urgency.ELEVATED),
            'EGFR_amplification': ('Aumenta EGFR-TKI', 'Amivantamab', 'Limited', Urgency.ELEVATED),
            'PIK3CA_mutation': ('Alpelisib', 'Copanlisib', 'Phase I/II', Urgency.ELEVATED),
            'BRAF_fusion_mutation': ('Dabrafenib + Trametinib', 'Encorafenib', 'FDA approved', Urgency.HIGH),
            'KRAS_activation': ('Sotorasib (G12C)', 'Adagrasib', 'FDA approved', Urgency.HIGH),
            'SCLC_transformation': ('Platinum-Etoposide', 'Tarlatamab', 'Standard', Urgency.CRITICAL),
            'Squamous_transformation': ('Platinum + Immunoterapia', 'Afatinib', 'Limited', Urgency.CRITICAL),
            'EMT_phenotype': ('AXL inhibitor + EGFR-TKI', 'Chemioterapia', 'Experimental', Urgency.ELEVATED),
            # Cancer-specific treatments
            'Immune_evasion': ('Immunotherapy combo', 'Clinical Trial', 'Experimental', Urgency.ELEVATED),
            'FGFR_alterations': ('Erdafitinib', 'Pemigatinib', 'FDA approved', Urgency.HIGH),
            'ESR1_mutations': ('Elacestrant', 'Fulvestrant', 'FDA approved', Urgency.HIGH),
            'AR_amplification': ('Abiraterone + Prednisone', 'Enzalutamide', 'FDA approved', Urgency.HIGH),
            'BRCA_reversion': ('Platinum rechallenge', 'Clinical Trial', 'Limited', Urgency.HIGH),
        }

        # Config
        self.config = {
            'min_prob': 0.001,
            'max_prob': 0.999,
            'high_threshold': 0.50,
            'critical_threshold': 0.75,
            'max_cumulative_lr': 500.0,
            'base_evidence_weight': 5.0,
            'correlation_strength': 0.5,
            'temporal_weight': 0.7,
            'evidence_decay_rate': 0.95,
            'evidence_decay_weeks': 12,
        }

        # Tracking
        self.cumulative_weight = defaultdict(float)
        self.cumulative_lr = defaultdict(lambda: 1.0)
        self.temporal_evidence_log: List[TemporalEvidence] = []
        self.active_patterns: List[str] = []
        self.correlation_boosts: Dict[str, float] = defaultdict(lambda: 1.0)
        self.audit_log = []
        self.clinical_overrides = []

    def calibrate_priors(self, ldh: float = 200, ecog: int = 1,
                         tp53_mutated: bool = False, rb1_mutated: bool = False,
                         stk11_mutated: bool = False, keap1_mutated: bool = False,
                         met_cn: float = 0, her2_cn: float = 0):
        """
        Calibra i priors basandosi sulla severità del paziente.
        """
        severity = 0
        protective = 0

        if ldh > 600:
            severity += 3
        elif ldh > 450:
            severity += 2
        elif ldh > 350:
            severity += 1
        elif ldh < 200:
            protective += 2
        elif ldh < 250:
            protective += 1

        if ecog >= 3:
            severity += 2
        elif ecog >= 2:
            severity += 1
        elif ecog == 0:
            protective += 2
        elif ecog == 1:
            protective += 1

        if tp53_mutated:
            severity += 1
        if rb1_mutated:
            severity += 1
        if tp53_mutated and rb1_mutated:
            severity += 1

        if stk11_mutated:
            severity += 1
        if keap1_mutated:
            severity += 1
        if stk11_mutated and keap1_mutated:
            severity += 1

        if met_cn >= 10:
            severity += 2
        elif met_cn >= 5:
            severity += 1

        if her2_cn >= 10:
            severity += 1

        net_score = severity - protective

        if net_score >= 0:
            self.severity_multiplier = 1.0 + min(net_score, 10) * 0.15
        else:
            reduction = max(0.3, 1.0 + net_score * 0.175)
            self.severity_multiplier = reduction

        calibrated_priors = {}
        for mech, base_prior in self.base_priors.items():
            if self.severity_multiplier >= 1.0:
                new_prior = min(0.40, base_prior * self.severity_multiplier)
            else:
                new_prior = max(0.02, base_prior * self.severity_multiplier)
            calibrated_priors[mech] = new_prior

        if severity > 0:
            if tp53_mutated and rb1_mutated and 'SCLC_transformation' in calibrated_priors:
                calibrated_priors['SCLC_transformation'] = min(0.50, calibrated_priors['SCLC_transformation'] * 2.0)

            if stk11_mutated and keap1_mutated and 'EMT_phenotype' in calibrated_priors:
                calibrated_priors['EMT_phenotype'] = min(0.45, calibrated_priors['EMT_phenotype'] * 2.0)

            if met_cn >= 5 and 'MET_amplification' in calibrated_priors:
                calibrated_priors['MET_amplification'] = min(0.50, calibrated_priors['MET_amplification'] * 1.5)

        self.hypotheses = calibrated_priors.copy()

        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'PRIOR_CALIBRATION',
            'cancer_type': self.cancer_type_key,
            'severity_score': severity,
            'protective_score': protective,
            'net_score': net_score,
            'multiplier': self.severity_multiplier,
        })

        return self.severity_multiplier

    # =========================================================================
    # TEMPORAL METHODS
    # =========================================================================

    def get_temporal_phase(self, reference_date: datetime = None) -> TemporalPhase:
        if reference_date is None:
            reference_date = datetime.now()
        months_on_therapy = (reference_date - self.therapy_start).days / 30.0
        if months_on_therapy <= 6:
            return TemporalPhase.EARLY
        elif months_on_therapy <= 12:
            return TemporalPhase.INTERMEDIATE
        elif months_on_therapy <= 24:
            return TemporalPhase.LATE
        else:
            return TemporalPhase.VERY_LATE

    def get_weeks_on_therapy(self, reference_date: datetime = None) -> int:
        if reference_date is None:
            reference_date = datetime.now()
        return int((reference_date - self.therapy_start).days / 7)

    def _apply_temporal_modifier(self, mechanism: str, base_lr: float,
                                 phase: TemporalPhase = None) -> float:
        if phase is None:
            phase = self.get_temporal_phase()
        if mechanism in TEMPORAL_LR_MODIFIERS:
            modifier = TEMPORAL_LR_MODIFIERS[mechanism].get(phase, 1.0)
            effective_modifier = 1.0 + (modifier - 1.0) * self.config['temporal_weight']
            return base_lr * effective_modifier
        return base_lr

    def _apply_evidence_decay(self, evidence: str, weeks_ago: int) -> float:
        if weeks_ago <= self.config['evidence_decay_weeks']:
            return 1.0
        decay_periods = (weeks_ago - self.config['evidence_decay_weeks']) / 4
        return self.config['evidence_decay_rate'] ** decay_periods

    # =========================================================================
    # CORRELATION METHODS
    # =========================================================================

    def _update_correlation_boosts(self):
        for mech, correlated in CORRELATION_NETWORK.items():
            current_prob = self.hypotheses.get(mech, 0)
            for corr_mech, strength in correlated.items():
                if corr_mech not in self.hypotheses:
                    continue
                if current_prob > 0.30 and strength > 0:
                    boost = 1.0 + (current_prob - 0.30) * strength * self.config['correlation_strength']
                    self.correlation_boosts[corr_mech] = max(self.correlation_boosts[corr_mech], boost)
                elif current_prob > 0.50 and strength < 0:
                    reduction = 1.0 + (current_prob - 0.50) * strength * self.config['correlation_strength']
                    self.correlation_boosts[corr_mech] = min(self.correlation_boosts[corr_mech], reduction)

    def _apply_correlation_prior(self, mechanism: str, prior: float) -> float:
        boost = self.correlation_boosts.get(mechanism, 1.0)
        if boost > 1.0:
            new_prior = prior + (1 - prior) * (boost - 1.0) * 0.5
        elif boost < 1.0:
            new_prior = prior * boost
        else:
            new_prior = prior
        return max(self.config['min_prob'], min(self.config['max_prob'], new_prior))

    # =========================================================================
    # PATTERN RECOGNITION
    # =========================================================================

    def _check_temporal_patterns(self, current_evidences: List[str]) -> Dict[str, float]:
        boosts = defaultdict(lambda: 1.0)
        recent_evidences = set(current_evidences)
        for ev_log in self.temporal_evidence_log[-10:]:
            recent_evidences.add(ev_log.evidence)
        for pattern_name, pattern_def in TEMPORAL_PATTERNS.items():
            required = set(pattern_def['evidences'])
            if required.issubset(recent_evidences):
                self.active_patterns.append(pattern_name)
                for mech, boost in pattern_def['boost'].items():
                    boosts[mech] = max(boosts[mech], boost)
                self.audit_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'PATTERN_MATCH',
                    'pattern': pattern_name,
                    'description': pattern_def['description']
                })
        return dict(boosts)

    # =========================================================================
    # CORE UPDATE
    # =========================================================================

    def _resolve_mutex(self, evidences: List[str]) -> List[str]:
        resolved = set(evidences)
        for group_name, members in MUTEX_GROUPS.items():
            present = [e for e in evidences if e in members]
            if len(present) > 1:
                priority = MUTEX_PRIORITY.get(group_name, members)
                for p in priority:
                    if p in present:
                        for e in present:
                            if e != p:
                                resolved.discard(e)
                        break
        return list(resolved)

    def update_probabilities(self, evidence_list: List[str],
                             visit_week: int = None,
                             source: str = 'unknown') -> Dict[str, float]:
        resolved = self._resolve_mutex(evidence_list)
        self.n_updates += 1
        phase = self.get_temporal_phase()
        current_week = visit_week or self.get_weeks_on_therapy()

        for ev in resolved:
            self.temporal_evidence_log.append(TemporalEvidence(
                evidence=ev,
                timestamp=datetime.now(),
                visit_week=current_week,
                source=source
            ))

        pattern_boosts = self._check_temporal_patterns(resolved)
        self._update_correlation_boosts()

        for mech, prior in list(self.hypotheses.items()):
            adj_prior = self._apply_correlation_prior(mech, prior)
            adj_prior = max(self.config['min_prob'], min(self.config['max_prob'], adj_prior))
            prior_odds = adj_prior / (1 - adj_prior)
            lr_this_update = 1.0

            for ev in resolved:
                if ev in self.evidence_strength and mech in self.evidence_strength[ev]:
                    base_lr = self.evidence_strength[ev][mech]
                    lr = self._apply_temporal_modifier(mech, base_lr, phase)
                    if mech in pattern_boosts:
                        lr *= pattern_boosts[mech]
                    lr_this_update *= lr
                    self.cumulative_weight[mech] += 1

            proposed = self.cumulative_lr[mech] * lr_this_update
            if proposed > self.config['max_cumulative_lr']:
                dampen = (self.config['max_cumulative_lr'] / proposed) ** 0.5
                lr_this_update *= dampen

            self.cumulative_lr[mech] *= lr_this_update
            posterior_odds = prior_odds * lr_this_update
            posterior = posterior_odds / (1 + posterior_odds)
            posterior = max(self.config['min_prob'], min(self.config['max_prob'], posterior))
            self.hypotheses[mech] = posterior

        self._update_correlation_boosts()
        self.last_update = datetime.now()
        self.audit_log.append({
            'timestamp': self.last_update.isoformat(),
            'evidences': resolved,
            'update': self.n_updates,
            'phase': phase.value,
            'week': current_week,
            'patterns': self.active_patterns[-3:] if self.active_patterns else []
        })
        return self.hypotheses.copy()

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def get_credibility_interval(self, mech: str) -> CredibilityInterval:
        try:
            from scipy import stats
            prob = self.hypotheses[mech]
            weight = self.cumulative_weight[mech]
            n = self.config['base_evidence_weight'] + weight
            alpha = prob * n + 1
            beta = (1 - prob) * n + 1
            ci_low, ci_high = stats.beta.ppf([0.025, 0.975], alpha, beta)
        except:
            prob = self.hypotheses[mech]
            ci_low, ci_high = max(0, prob - 0.2), min(1, prob + 0.2)
            weight = self.cumulative_weight[mech]
        return CredibilityInterval(prob, ci_low, ci_high, ci_high - ci_low, weight)

    def get_risk_level(self, prob: float) -> RiskLevel:
        if prob >= self.config['critical_threshold']:
            return RiskLevel.CRITICAL
        elif prob >= self.config['high_threshold']:
            return RiskLevel.HIGH
        elif prob >= 0.30:
            return RiskLevel.MEDIUM
        elif prob >= 0.15:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def get_sorted_risks(self) -> List[Tuple[str, float, RiskLevel, CredibilityInterval]]:
        risks = []
        for mech, prob in self.hypotheses.items():
            ci = self.get_credibility_interval(mech)
            level = self.get_risk_level(prob)
            risks.append((mech, prob, level, ci))
        return sorted(risks, key=lambda x: x[1], reverse=True)

    def get_correlated_risks(self, mechanism: str) -> List[Tuple[str, float, float]]:
        correlated = []
        if mechanism in CORRELATION_NETWORK:
            for corr_mech, strength in CORRELATION_NETWORK[mechanism].items():
                if corr_mech in self.hypotheses:
                    correlated.append((corr_mech, self.hypotheses[corr_mech], strength))
        return sorted(correlated, key=lambda x: abs(x[2]), reverse=True)

    def get_dominant_risk(self) -> Dict:
        top = self.get_sorted_risks()[0]
        mech, prob, level, ci = top
        tx = self.treatments.get(mech, ('Nessun cambio', 'N/A', 'N/A', Urgency.ROUTINE))

        correlated = self.get_correlated_risks(mech)
        corr_warnings = []
        for corr_mech, corr_prob, strength in correlated:
            if corr_prob > 0.20 and strength > 0:
                corr_warnings.append(f"{corr_mech} ({corr_prob:.0%})")

        if level == RiskLevel.CRITICAL:
            action, monitoring = "SWITCH TERAPIA IMMEDIATO", "ctDNA settimanale"
        elif level == RiskLevel.HIGH:
            action, monitoring = "CONSIDERARE SWITCH O COMBO", "ctDNA ogni 2 settimane"
        elif level == RiskLevel.MEDIUM:
            action, monitoring = "AUMENTARE MONITORAGGIO", "ctDNA ogni 4 settimane"
        else:
            action, monitoring = "CONTINUARE TERAPIA ATTUALE", "ctDNA ogni 6-8 settimane"

        return {
            'mechanism': mech,
            'probability': prob,
            'risk_level': level,
            'credibility_interval': ci,
            'correlated_risks': corr_warnings,
            'active_patterns': self.active_patterns[-3:] if self.active_patterns else [],
            'temporal_phase': self.get_temporal_phase().value,
            'weeks_on_therapy': self.get_weeks_on_therapy(),
            'cancer_type': self.cancer_type_key,
            'recommendation': {
                'action': action,
                'primary_drug': tx[0],
                'alternative_drug': tx[1],
                'evidence_level': tx[2],
                'urgency': tx[3].value if hasattr(tx[3], 'value') else str(tx[3]),
                'monitoring_interval': monitoring
            }
        }

    def get_alerts(self) -> List[Dict]:
        alerts = []
        for mech, prob, level, ci in self.get_sorted_risks():
            if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                tx = self.treatments.get(mech, ('Consult', 'N/A', 'N/A', Urgency.ELEVATED))
                correlated = self.get_correlated_risks(mech)
                alerts.append({
                    'mechanism': mech,
                    'probability': prob,
                    'risk_level': level.value,
                    'recommended_action': tx[0],
                    'alternative': tx[1],
                    'evidence_level': tx[2],
                    'urgency': tx[3].value,
                    'correlated_risks': [(c[0], c[1]) for c in correlated if c[1] > 0.15]
                })
        return alerts

    def get_treatment_recommendation(self) -> TreatmentRecommendation:
        top = self.get_sorted_risks()[0]
        mech, prob, level, ci = top
        tx = self.treatments.get(mech, ('Nessun cambio', 'N/A', 'N/A', Urgency.ROUTINE))

        correlated = self.get_correlated_risks(mech)
        corr_list = [c[0] for c in correlated if c[1] > 0.20]

        if level == RiskLevel.CRITICAL:
            action, monitoring = "SWITCH TERAPIA IMMEDIATO", "ctDNA settimanale"
        elif level == RiskLevel.HIGH:
            action, monitoring = "CONSIDERARE SWITCH O COMBO", "ctDNA ogni 2 settimane"
        elif level == RiskLevel.MEDIUM:
            action, monitoring = "AUMENTARE MONITORAGGIO", "ctDNA ogni 4 settimane"
        else:
            action, monitoring = "CONTINUARE TERAPIA ATTUALE", "ctDNA ogni 6-8 settimane"

        notes = f"Cancer type: {self.cancer_type_key}"
        if self.active_patterns:
            notes += f" | Pattern rilevati: {', '.join(self.active_patterns[-3:])}"

        return TreatmentRecommendation(
            mechanism=mech,
            probability=prob,
            risk_level=level,
            action=action,
            primary_drug=tx[0],
            alternative_drug=tx[1],
            evidence_level=tx[2],
            urgency=tx[3],
            monitoring_interval=monitoring,
            correlated_risks=corr_list,
            additional_notes=notes
        )

    def apply_clinical_override(self, mechanism: str, decision: str, reason: str, clinician: str):
        override = {
            'timestamp': datetime.now().isoformat(),
            'mechanism': mechanism,
            'sentinel_prob': self.hypotheses.get(mechanism, 0),
            'clinical_decision': decision,
            'reason': reason,
            'clinician_id': clinician
        }
        self.clinical_overrides.append(override)
        self.audit_log.append({'type': 'OVERRIDE', **override})
        return override

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_report(self, include_all=False, include_correlations=True) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"SENTINEL v{self.VERSION} - REPORT")
        lines.append("=" * 70)
        lines.append(f"Patient: {self.patient_id}")
        lines.append(f"Cancer Type: {self.cancer_type_key}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Updates: {self.n_updates}")
        lines.append(f"Temporal Phase: {self.get_temporal_phase().value}")
        lines.append(f"Weeks on Therapy: {self.get_weeks_on_therapy()}")
        lines.append("")

        if self.active_patterns:
            lines.append("-" * 70)
            lines.append("ACTIVE PATTERNS")
            lines.append("-" * 70)
            for pattern in set(self.active_patterns[-5:]):
                if pattern in TEMPORAL_PATTERNS:
                    lines.append(f"  • {pattern}: {TEMPORAL_PATTERNS[pattern]['description']}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("RISK ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"{'Mechanism':<28} {'Prob':>7} {'Risk':<10} {'CI'}")
        lines.append("-" * 70)

        risks = self.get_sorted_risks()
        for mech, prob, level, ci in risks:
            if prob >= 0.10 or include_all:
                lines.append(f"{mech:<28} {prob:>6.1%} {level.value:<10} {ci}")

        if include_correlations:
            top_mech = risks[0][0]
            correlated = self.get_correlated_risks(top_mech)
            if correlated:
                lines.append("")
                lines.append("-" * 70)
                lines.append(f"CORRELATED RISKS (with {top_mech})")
                lines.append("-" * 70)
                for corr_mech, corr_prob, strength in correlated[:5]:
                    direction = "↑" if strength > 0 else "↓"
                    lines.append(f"  {corr_mech:<25} {corr_prob:>6.1%} (corr: {direction}{abs(strength):.1f})")

        lines.append("")
        lines.append("-" * 70)
        lines.append("RECOMMENDATION")
        lines.append("-" * 70)
        rec = self.get_treatment_recommendation()
        lines.append(f"Action: {rec.action}")
        lines.append(f"Primary: {rec.primary_drug}")
        lines.append(f"Alternative: {rec.alternative_drug}")
        lines.append(f"Monitoring: {rec.monitoring_interval}")
        if rec.additional_notes:
            lines.append(f"Notes: {rec.additional_notes}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

SentinelV25 = SentinelV30


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_sentinel(patient_id: str, therapy_start_date: str = None,
                    diagnosis: str = None, histology: str = None,
                    genetics: dict = None, version: str = "3.0") -> SentinelV30:
    """Factory function per creare istanza SENTINEL"""
    return SentinelV30(
        patient_id=patient_id,
        therapy_start_date=therapy_start_date,
        diagnosis=diagnosis,
        histology=histology,
        genetics=genetics
    )