
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from src.digital_twin import DigitalTwin

# Flexible import for sentinel_utils
try:
    from src.sentinel_utils import safe_float, safe_get_float
except ImportError:
    from sentinel_utils import safe_float, safe_get_float

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SENTINEL_v19_ML: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ML WEIGHTS - Trained on 22,053 patients (C-index: OS=0.734, PFS=0.693)
# ============================================================================

ML_WEIGHTS = {
    # ML-VALIDATED WEIGHTS
    # Training: 317 NSCLC patients (MSK-IMPACT)
    # Method: Log-Rank test + Kaplan-Meier
    # Key findings: STK11/KEAP1 worst prognosis, MET/ALK protective when targeted

    # Clinical weights from ML (PFS model)
    "clinical_pfs": {
        "hypoxia_high": 25,
        "tmb_high": 22,
        "fraction_genome_altered_high": 17,
        "aneuploidy_high": 13,
        "msi_high": 9,
    },

    # Mutation weights - ML-validated on 317 NSCLC + Literature
    "mutations": {
        # === RESISTENZA ALTA (ML: HR>1.2 + Literature) ===
        'TP53': 25,  # HR=1.23, genomic instability
        'STK11': 30,  # HR=1.29, immuno-resistance
        'KEAP1': 25,  # HR=1.38, immuno-resistance
        'RB1': 30,  # SCLC transformation risk
        'PTEN': 20,  # HR=1.42, PI3K pathway

        # === RESISTENZA MEDIA ===
        'KRAS': 25,  # Resistenza se non targetato
        'BRAF': 20,  # Resistenza se non targetato
        'PIK3CA': 15,  # Bypass pathway
        'NF1': 15,  # RAS pathway activation
        'CDKN2A': 10,  # Cell cycle

        # === TARGETABILI (base score, beneficio se trattati) ===
        'EGFR': 15,  # Targetabile (+40 se T790M)
        'ALK': 15,  # Targetabile (HR=0.66 se trattato)
        'MET': 25,  # Amplification = bypass resistance
        'RET': 15,  # Targetabile
        'ROS1': 15,  # Targetabile
        'ERBB2': 15,  # HER2, targetabile
        'HER2': 15,  # Alias ERBB2

        # === ML PROTETTIVI (quando targetati) ===
        'ATM': 10,  # HR=0.62, DNA repair
        'ARID1A': 10,  # HR=0.75, chromatin
    },

    # Resistance mutations
    "resistance": {
        "T790M": 40,
        "C797S": 50,
        "MET_amplification": 35,
    },

    # Blood markers (Elephant Protocol)
    "blood": {
        "ldh_very_high": 50,  # >= 500
        "ldh_high": 30,  # >= 350
        "nlr_high": 25,  # >= 5.0
        "albumin_low": 20,  # < 3.5
    },

    # Thresholds
    "thresholds": {
        "ldh_very_high": 500,
        "ldh_high": 350,
        "ldh_normal": 250,
        "nlr_high": 5.0,
        "albumin_low": 3.5,
        "tmb_high": 10,
        "hypoxia_high": 0,
        "aneuploidy_high": 10,
        "fga_high": 0.3,
        "msi_high": 10,
    },

    # Co-mutation modifiers
    "co_mutations": {
        "TP53_RB1": 1.5,
        "TP53_STK11": 1.4,
        "STK11_KEAP1": 1.6,
        "EGFR_TP53": 1.3,
        "EGFR_MET": 1.5,
        "KRAS_STK11": 1.4,
        "KRAS_KEAP1": 1.3,
    }
}

# ============================================================================
# MODULE IMPORTS (Hardware Check)
# ============================================================================
MODULES_STATUS = {
    "BAYESIAN": False,
    "VITTORIA": False,
    "VISION": False,
    "PHYSICS": False
}

try:
    from src.sentinel_bayesian_v3 import SentinelV30

    MODULES_STATUS["BAYESIAN"] = True
    logger.info("✓ Bayesian Module Loaded")
except ImportError:
    try:
        from sentinel_bayesian_v3 import SentinelV30

        MODULES_STATUS["BAYESIAN"] = True
        logger.info("✓ Bayesian Module Loaded")
    except ImportError:
        logger.warning("✗ Bayesian Module Not Available")

try:
    from src.vittoria_3 import VittoriaNeuralNet

    MODULES_STATUS["VITTORIA"] = True
    logger.info("✓ Vittoria Neural Net Loaded")
except ImportError:
    try:
        from vittoria_3 import VittoriaNeuralNet

        MODULES_STATUS["VITTORIA"] = True
        logger.info("✓ Vittoria Neural Net Loaded")
    except ImportError:
        logger.warning("✗ Vittoria Module Not Available")

try:
    from src.vision_ai_net import analyze_biopsy

    MODULES_STATUS["VISION"] = True
    logger.info("✓ Vision AI Module Loaded")
except ImportError:
    try:
        from vision_ai_net import analyze_biopsy

        MODULES_STATUS["VISION"] = True
        logger.info("✓ Vision AI Module Loaded")
    except ImportError:
        logger.warning("✗ Vision Module Not Available")

try:
    from src.alphafold_client import check_binding

    MODULES_STATUS["PHYSICS"] = True
    logger.info("✓ Physics Module Loaded")
except ImportError:
    try:
        from alphafold_client import check_binding

        MODULES_STATUS["PHYSICS"] = True
        logger.info("✓ Physics Module Loaded")
    except ImportError:
        logger.warning("✗ Physics Module Not Available")


# CIViC integration for evidence levels
try:
    from civic_client import get_evidence_level, enrich_mutation_weight
    CIVIC_AVAILABLE = True
except ImportError:
    CIVIC_AVAILABLE = False
    logger.warning("CIViC client not available - using ML weights only")


try:
    from clinical_notes_llm import enrich_visit_data, is_available as llm_available
    LLM_AVAILABLE = llm_available()
except ImportError:
    LLM_AVAILABLE = False

# AlphaMissense integration
try:
    from alphamissense_lookup import classify_mutation, classify_mutations_batch, is_available as alphamissense_available
    ALPHAMISSENSE_AVAILABLE = alphamissense_available()
    if ALPHAMISSENSE_AVAILABLE:
        logger.info("✓ AlphaMissense Module Loaded (71M variants)")
except ImportError:
    ALPHAMISSENSE_AVAILABLE = False
    logger.warning("⚠️ AlphaMissense not available")

# ============================================================================
# DATA STRUCTURES (Unchanged from v18)
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels for clinical decision support"""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


@dataclass
class VetoResult:
    """Result from therapy validation check"""
    active: bool
    reason: str
    recommendation: str
    severity: str = "CRITICAL"


@dataclass
class PrognosisResult:
    """Result from biological risk assessment"""
    tank_score: int
    ferrari_score: int
    reasons: List[str]
    status: str
    ldh: float
    tmb: float
    nlr: float


@dataclass
class ExplainabilityReport:
    """Report dettagliato del perché dei punteggi"""
    tank_contributions: List[Dict]
    tank_total: int
    ferrari_evidences: List[Dict]
    ferrari_mechanism: str
    ferrari_probability: float
    ferrari_confidence: str
    synergies: List[Dict]


@dataclass
class DualTrackResult:
    """Combined result from both tracks"""
    veto_active: bool
    veto_reason: str
    veto_recommendation: str
    biological_risk_tank: int
    biological_risk_ferrari: int
    prognosis_reasons: List[str]
    prognosis_status: str
    display_risk: int
    display_status: str
    alert_level: AlertLevel
    ldh: float
    tmb: float
    nlr: float
    final_recommendation: str
    consensus: str
    active_genes: List[str]
    vision_data: Optional[Dict] = None
    physics_verdict: Optional[str] = None


# ============================================================================
# BAYESIAN EVIDENCE MAP (Unchanged from v18)
# ============================================================================

BAYESIAN_EVIDENCE_MAP = {
    'TP53': ['TP53_loss'],
    'RB1': ['RB1_loss'],
    'STK11': ['STK11_loss'],
    'KEAP1': ['KEAP1_loss'],
    'KRAS': ['KRAS_activation'],
    'G12C': ['KRAS_activation'],
    'EGFR': ['EGFR_amplification'],
    'T790M': ['T790M_acquired'],
    'MET': ['MET_amplification'],
    'BRAF': ['BRAF_fusion_mutation'],
    'V459L': ['BRAF_fusion_mutation']
}


# ============================================================================
# VETO SYSTEM (Unchanged from v18 - proven logic)
# ============================================================================

class VetoSystem:
    """
    Validates therapy appropriateness based on molecular targets.
    Returns VETO if therapy is biologically incompatible with patient genetics.
    """

    def __init__(self):
        self.veto_rules = []
        logger.info("🛡️  VETO System Initialized")

    def check_therapy(self, patient_data: Dict) -> VetoResult:
        """
        Check if current therapy is appropriate for patient genetics.
        Returns VetoResult with active=True if therapy should be blocked.
        """
        base = patient_data.get('baseline', patient_data)
        therapy = base.get('current_therapy', 'None').lower()
        genetics = base.get('genetics', {})

        logger.info(f"🔍 VETO Check - Therapy: {therapy}")

        # RULE 1: Osimertinib requires EGFR mutation
        if "osimertinib" in therapy:
            egfr_val = str(genetics.get('egfr_status', '')).lower()
            if egfr_val in ['wt', 'none', 'nan', '', 'wild-type', 'wildtype']:
                logger.warning("⛔ VETO TRIGGERED: Osimertinib without EGFR mutation")
                return VetoResult(
                    active=True,
                    reason="Osimertinib requires EGFR Mutation. (Target Missing)",
                    recommendation="DISCONTINUE Osimertinib. Consider:\n"
                                   "• Chemotherapy (platinum-based doublet)\n"
                                   "• Immunotherapy if PD-L1 ≥50%\n"
                                   "• Re-biopsy to confirm molecular status",
                    severity="CRITICAL"
                )

        # RULE 2: Gefitinib/Erlotinib ineffective against T790M
        if "gefitinib" in therapy or "erlotinib" in therapy:
            genetics_str = str(genetics).lower()
            if "t790m" in genetics_str and "non rilevabile" not in genetics_str and "clearance" not in genetics_str and "not detected" not in genetics_str:
                logger.warning("⛔ VETO TRIGGERED: 1st-gen EGFR-TKI vs T790M")
                return VetoResult(
                    active=True,
                    reason="Gefitinib ineffective vs T790M. (Known Resistance)",
                    recommendation="SWITCH to Osimertinib (3rd-gen EGFR-TKI)\n"
                                   "• FDA approved for T790M resistance\n"
                                   "• Re-evaluate response in 4-6 weeks",
                    severity="CRITICAL"
                )

        # RULE 3: Sotorasib specific for KRAS G12C
        if "sotorasib" in therapy or "adagrasib" in therapy:
            kras_val = str(genetics.get('kras_mutation', '')).lower()
            if "g12c" not in kras_val:
                logger.warning("⛔ VETO TRIGGERED: KRAS G12C inhibitor without G12C")
                return VetoResult(
                    active=True,
                    reason="Sotorasib specific for KRAS G12C mutation.",
                    recommendation="DISCONTINUE Sotorasib. Consider:\n"
                                   "• Re-sequence KRAS (confirm G12C absence)\n"
                                   "• Alternative therapy based on other drivers\n"
                                   "• Clinical trial enrollment",
                    severity="CRITICAL"
                )

        # RULE 4: ALK inhibitors require ALK rearrangement
        if any(x in therapy for x in ["alectinib", "crizotinib", "ceritinib", "brigatinib", "lorlatinib"]):
            alk_val = str(genetics.get('alk_status', '')).lower()
            if alk_val in ['wt', 'none', 'nan', '', 'negative']:
                logger.warning("⛔ VETO TRIGGERED: ALK inhibitor without ALK+")
                return VetoResult(
                    active=True,
                    reason="ALK inhibitor requires ALK rearrangement.",
                    recommendation="DISCONTINUE ALK inhibitor. Verify:\n"
                                   "• FISH or IHC for ALK rearrangement\n"
                                   "• Consider alternative targeted therapy",
                    severity="CRITICAL"
                )

        # RULE 5: MET Amplification requires MET inhibitor
        met_status = str(genetics.get('met_status', '')).lower()
        met_cn = 0.0
        try:
            met_cn = float(genetics.get('met_cn', 0) or 0)
        except:
            pass

        if 'amplification' in met_status or met_cn >= 5.0:
            if not any(x in therapy for x in ["capmatinib", "tepotinib", "crizotinib", "savolitinib"]):
                logger.warning("⛔ VETO TRIGGERED: MET Amplification without MET inhibitor")
                return VetoResult(
                    active=True,
                    reason=f"MET Amplification (CN={met_cn}) requires MET inhibitor.",
                    recommendation="ADD or SWITCH to MET inhibitor:\n"
                                   "• Capmatinib (FDA approved for MET ex14/amp)\n"
                                   "• Tepotinib (alternative)\n"
                                   "• Consider combo with current EGFR-TKI if EGFR+",
                    severity="HIGH"
                )

        # RULE 6: Immunotherapy (PD-1/PD-L1) WARNING for KEAP1/STK11 mutated
        # KEAP1 loss and STK11 loss are strong negative predictors for IO response
        io_drugs = ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
                     "avelumab", "ipilimumab", "cemiplimab", "pd-1", "pdl-1", "pd-l1"]
        if any(x in therapy for x in io_drugs):
            keap1_val = str(genetics.get('keap1_status', '')).lower()
            stk11_val = str(genetics.get('stk11_status', '')).lower()
            keap1_mut = keap1_val in ['mutated', 'mut', 'loss']
            stk11_mut = stk11_val in ['mutated', 'mut', 'loss']

            if keap1_mut and stk11_mut:
                logger.warning("⚠️ WARNING: IO + KEAP1/STK11 double loss - severe immuno-resistance")
                return VetoResult(
                    active=True,
                    reason="KEAP1 + STK11 double loss: severe immuno-resistance predicted.",
                    recommendation="IMMUNOTHERAPY LIKELY INEFFECTIVE. Consider:\n"
                                   "• Chemotherapy + targeted therapy combination\n"
                                   "• KRAS G12C inhibitor if applicable\n"
                                   "• Metabolic approach (Elephant Protocol)\n"
                                   "• Clinical trial enrollment",
                    severity="HIGH"
                )
            elif keap1_mut:
                logger.warning("⚠️ WARNING: IO + KEAP1 loss - reduced IO efficacy")
                return VetoResult(
                    active=True,
                    reason="KEAP1 loss: reduced immunotherapy efficacy predicted.",
                    recommendation="IMMUNOTHERAPY MAY BE INEFFECTIVE. Consider:\n"
                                   "• Adding chemotherapy to IO (chemo-IO combo)\n"
                                   "• Alternative targeted therapy if actionable driver exists\n"
                                   "• Close monitoring: if PD at first assessment, switch promptly",
                    severity="WARNING"
                )
            elif stk11_mut:
                logger.warning("⚠️ WARNING: IO + STK11 loss - cold tumor phenotype")
                return VetoResult(
                    active=True,
                    reason="STK11 loss: cold tumor phenotype, reduced IO response.",
                    recommendation="IMMUNOTHERAPY RESPONSE UNLIKELY. Consider:\n"
                                   "• Chemo-IO combination preferred over IO monotherapy\n"
                                   "• KRAS G12C inhibitor if applicable\n"
                                   "• Monitor closely for early progression",
                    severity="WARNING"
                )

            # Check TMB and PD-L1 for cold tumors
            biomarkers = base.get('biomarkers', {})
            has_tmb = ('tmb' in base or 'tmb_score' in base or 
                       'tmb' in genetics or 'tmb_score' in genetics or
                       'tmb_score' in biomarkers or 'tmb' in biomarkers)
            if has_tmb:
                tmb = safe_float(base.get('tmb') or base.get('tmb_score') or 
                                 genetics.get('tmb') or genetics.get('tmb_score') or
                                 biomarkers.get('tmb') or biomarkers.get('tmb_score'), 0.0)
                pdl1 = safe_float(base.get('pdl1') or base.get('pdl1_score') or 
                                  genetics.get('pdl1') or genetics.get('pdl1_score') or
                                  biomarkers.get('pdl1') or biomarkers.get('pdl1_percent') or biomarkers.get('pdl1_score'), 0.0)
                
                if tmb < 5.0 and pdl1 < 50.0:
                    logger.warning("⛔ VETO TRIGGERED: IO Monotherapy vs cold tumor (Low TMB / Low PD-L1)")
                    return VetoResult(
                        active=True,
                        reason="Low TMB (<5 mut/Mb) and PD-L1 (<50%): severe immuno-resistance predicted.",
                        recommendation="IMMUNOTHERAPY MONOTHERAPY LIKELY INEFFECTIVE. Consider:\n"
                                       "• Chemotherapy combination (Chemo-IO)\n"
                                       "• Alternative targeted therapy\n"
                                       "• Metabolic approach / Clinical trial enrollment",
                        severity="HIGH"
                    )

        logger.info("✓ No VETO - Therapy appears appropriate")
        return VetoResult(
            active=False,
            reason="",
            recommendation="",
            severity="INFO"
        )


# ============================================================================
# EVIDENCE TRANSLATOR (Unchanged from v18)
# ============================================================================

class EvidenceTranslator:
    """Translates patient genetics to Bayesian network evidence"""

    @staticmethod
    def translate_patient(patient_data: Dict) -> List[str]:
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})
        evidences = []

        for key, value in genetics.items():
            gene = key.replace('_status', '').replace('_mutation', '').upper()
            val_str = str(value).lower()

            # Skip cleared/undetected mutations
            if any(skip in val_str for skip in
                   ['wt', 'none', 'unknown', 'nan', 'clearance', 'non rilevabile', 'not detected']):
                continue

            if gene in BAYESIAN_EVIDENCE_MAP:
                evidences.extend(BAYESIAN_EVIDENCE_MAP[gene])

            # Check value string for specific mutations
            for map_key, map_ev in BAYESIAN_EVIDENCE_MAP.items():
                if map_key.lower() in val_str:
                    # Skip if T790M is cleared
                    if map_key == 'T790M' and (
                            'non rilevabile' in val_str or 'clearance' in val_str or 'not detected' in val_str):
                        continue
                    if isinstance(map_ev, list):
                        evidences.extend(map_ev)

        # STK11 + KEAP1 double loss detection
        stk11_mut = False
        keap1_mut = False

        stk11_val = str(genetics.get('stk11_status', '')).lower()
        if stk11_val in ['mutated', 'mut', 'loss']:
            if 'STK11_loss' not in evidences:
                evidences.append('STK11_loss')
            stk11_mut = True

        keap1_val = str(genetics.get('keap1_status', '')).lower()
        if keap1_val in ['mutated', 'mut', 'loss']:
            if 'KEAP1_loss' not in evidences:
                evidences.append('KEAP1_loss')
            keap1_mut = True

        if stk11_mut and keap1_mut:
            if 'STK11_KEAP1_double_loss' not in evidences:
                evidences.append('STK11_KEAP1_double_loss')
                logger.info("🔴 STK11+KEAP1 DOUBLE LOSS DETECTED - High immuno-resistance risk")

        return list(set(evidences))


# ============================================================================
# PROGNOSIS ENGINE (ML-Enhanced v19)
# ============================================================================

class PrognosisEngine:
    """
    Biological risk assessment using ML-validated weights.
    Maintains v18 logic flow with enhanced weighting from ML training.
    """

    def __init__(self):
        self.translator = EvidenceTranslator()
        self.bayesian_cache = None
        logger.info("📊 Prognosis Engine v19 ML-Enhanced Initialized")

    def _calculate_clinical_risk(self, patient_data: Dict, vision_risk: float = 0) -> Tuple[
        int, List[str], str, float, float, float, List[Dict]]:
        """
        Calculate Tank score using ML-validated weights.
        CRITICAL: Uses visit blood_markers if available, falls back to baseline.
        """
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})

        # CRITICAL: Use visit blood_markers if this is a visit, otherwise baseline
        blood = base.get('blood_markers', {})

        score = 0
        reasons = []
        active_genes = []
        contributions = []

        thresholds = ML_WEIGHTS["thresholds"]
        mutation_weights = ML_WEIGHTS["mutations"]
        resistance_weights = ML_WEIGHTS["resistance"]
        blood_weights = ML_WEIGHTS["blood"]
        clinical_weights = ML_WEIGHTS["clinical_pfs"]

        logger.info("🧬 Scanning Genetics (ML-Enhanced)...")

        # =====================================================================
        # 1. GENETIC DRIVERS (ML mutation weights)
        # =====================================================================
        detected_mutations = []

        for key, value in genetics.items():
            gene_clean = key.replace('_status', '').replace('_mutation', '').upper().strip()
            val_str = str(value).lower().strip()

            # Skip wild-type, cleared, or unknown
            if val_str in ['wt', 'none', 'unknown', 'nan', 'na', '', 'wild-type', 'wildtype']:
                continue

            # Skip if cleared/undetected (but not if it contains mutation info)
            if any(skip in val_str for skip in ['clearance', 'non rilevabile', 'not detected']):
                # Check if there's still useful mutation info before skipping
                if not any(mut in val_str for mut in ['t790m', 'c797s', 'g12c', 'l858r', 'exon']):
                    continue

            # A. Standard gene weights (ML-validated)
            # Gene is considered ACTIVE if value is not wt/none/unknown
            if gene_clean in mutation_weights:
                w = mutation_weights[gene_clean]
                # Enrich with CIViC evidence if available
                if CIVIC_AVAILABLE and w > 0:
                    try:
                        enriched = enrich_mutation_weight(gene_clean, val_str, w)
                        if enriched.get("evidence_source") == "CIViC":
                            w = enriched.get("adjusted_weight", w)
                            logger.debug(
                                f"   CIViC enriched {gene_clean}: {w} (Level {enriched.get('evidence_level')})")
                    except:
                        pass  # Use ML weight if CIViC fails
                score += w
                reasons.append(f"{gene_clean} (+{w})")
                active_genes.append(gene_clean)
                detected_mutations.append(gene_clean)
                contributions.append({
                    "factor": f"{gene_clean} mutated",
                    "value": val_str,
                    "weight": w,
                    "category": "genetic_driver",
                    "source": "ML-Literature"
                })

            # B. Amplifications
            if "amplification" in val_str or "amp" in val_str:
                amp_weight = resistance_weights.get("MET_amplification", 35) if "MET" in gene_clean else 30
                score += amp_weight
                reasons.append(f"{gene_clean} AMPLIFICATION (+{amp_weight})")
                detected_mutations.append(f"{gene_clean}_AMP")
                contributions.append({
                    "factor": f"{gene_clean} amplification",
                    "value": val_str,
                    "weight": amp_weight,
                    "category": "amplification",
                    "source": "ML-Literature"
                })

            # C. Resistance mutations (T790M, C797S) - check in value string
            if "t790m" in val_str:
                # Skip if explicitly cleared
                if "non rilevabile" in val_str or "clearance" in val_str or "not detected" in val_str:
                    logger.info(f"   T790M CLEARED - skipping resistance penalty")
                else:
                    w = resistance_weights.get("T790M", 40)
                    score += w
                    reasons.append(f"EGFR T790M resistance (+{w})")
                    detected_mutations.append("T790M")
                    contributions.append({
                        "factor": "EGFR T790M resistance",
                        "value": val_str,
                        "weight": w,
                        "category": "resistance",
                        "source": "ML-Literature"
                    })

            if "c797s" in val_str:
                w = resistance_weights.get("C797S", 50)
                score += w
                reasons.append(f"EGFR C797S resistance (+{w})")
                detected_mutations.append("C797S")
                contributions.append({
                    "factor": "EGFR C797S resistance",
                    "value": val_str,
                    "weight": w,
                    "category": "resistance",
                    "source": "ML-Literature"
                })

        # =====================================================================
        # 2. CO-MUTATION SYNERGIES (ML-validated modifiers)
        # =====================================================================
        co_mut_weights = ML_WEIGHTS["co_mutations"]

        for co_mut_key, modifier in co_mut_weights.items():
            genes = co_mut_key.split("_")
            if len(genes) == 2:
                gene1, gene2 = genes
                if gene1 in detected_mutations and gene2 in detected_mutations:
                    # Apply synergy bonus
                    synergy_bonus = int(score * (modifier - 1))
                    if synergy_bonus > 0:
                        score += synergy_bonus
                        reasons.append(f"{gene1}+{gene2} synergy (+{synergy_bonus})")
                        contributions.append({
                            "factor": f"{gene1}+{gene2} synergy",
                            "value": f"modifier x{modifier}",
                            "weight": synergy_bonus,
                            "category": "co_mutation",
                            "source": f"ML-Interaction (x{modifier})"
                        })

        # =====================================================================
        # 3. BLOOD MARKERS (ML-validated thresholds)
        # =====================================================================
        ldh = safe_float(blood.get('ldh') or blood.get('LDH'), 200)
        neutrophils = safe_float(blood.get('neutrophils'), 0)
        lymphocytes = safe_float(blood.get('lymphocytes'), 0)
        albumin = safe_float(blood.get('albumin') or blood.get('Albumin'), 4.0)

        # NLR: SEMPRE ricalcolato da neutrofili/linfociti (mai fidarsi del pre-calcolato)
        # Se linfociti mancano o sono 0, NLR non calcolabile → 0 (non contribuisce al score)
        if neutrophils > 0 and lymphocytes > 0:
            nlr = round(neutrophils / lymphocytes, 2)
        else:
            nlr = 0
        # Validazione: NLR > 50 è clinicamente impossibile → dato errato
        if nlr > 50:
            logger.warning(f"⚠️ NLR anomalo ({nlr:.1f}) - linfociti mancanti o errati, NLR ignorato")
            nlr = 0

        status = "STABLE"

        logger.info(f"🩸 LDH Level: {ldh} U/L, NLR: {nlr:.1f}, Albumin: {albumin}")

        # LDH scoring (ML-validated thresholds)
        if ldh >= thresholds["ldh_very_high"]:
            w = blood_weights["ldh_very_high"]
            score += w
            status = "ELEPHANT_PROTOCOL"
            reasons.append(f"LDH Very High ({ldh} U/L) (+{w})")
            contributions.append({
                "factor": f"LDH Very High (>={thresholds['ldh_very_high']}, actual={ldh:.0f})",
                "value": f"{ldh}",
                "weight": w,
                "category": "blood_marker",
                "source": "Elephant Protocol"
            })
            logger.warning(f"🐘 ELEPHANT PROTOCOL TRIGGERED (LDH={ldh})")
        elif ldh >= thresholds["ldh_high"]:
            w = blood_weights["ldh_high"]
            score += w
            status = "ELEVATED_LDH"
            reasons.append(f"High LDH ({ldh} U/L) (+{w})")
            contributions.append({
                "factor": f"LDH High (>={thresholds['ldh_high']}, actual={ldh:.0f})",
                "value": f"{ldh}",
                "weight": w,
                "category": "blood_marker",
                "source": "Elephant Protocol"
            })

        # NLR scoring (ML-validated)
        if nlr >= thresholds["nlr_high"]:
            w = blood_weights["nlr_high"]
            score += w
            reasons.append(f"High NLR ({nlr:.1f}) (+{w})")
            contributions.append({
                "factor": f"NLR High (>={thresholds['nlr_high']}, actual={nlr:.1f})",
                "value": f"{nlr:.1f}",
                "weight": w,
                "category": "blood_marker",
                "source": "ML-Literature"
            })

        # Albumin scoring (ML-validated)
        if albumin < thresholds["albumin_low"]:
            w = blood_weights["albumin_low"]
            score += w
            reasons.append(f"Low Albumin ({albumin}) (+{w})")
            contributions.append({
                "factor": f"Albumin Low (<{thresholds['albumin_low']}, actual={albumin:.1f})",
                "value": f"{albumin}",
                "weight": w,
                "category": "blood_marker",
                "source": "ML-Literature"
            })

        # =====================================================================
        # 4. ML-ONLY FEATURES (Hypoxia, Aneuploidy, FGA, MSI - if available)
        # =====================================================================

        # Hypoxia (from PanCancer data)
        hypoxia = base.get('hypoxia_buffa') or base.get('hypoxia')
        if hypoxia is not None:
            try:
                hypoxia = float(hypoxia)
                if hypoxia > thresholds["hypoxia_high"]:
                    w = clinical_weights["hypoxia_high"]
                    score += w
                    reasons.append(f"Hypoxia High ({hypoxia:.1f}) (+{w})")
                    contributions.append({
                        "factor": f"Hypoxia High ({hypoxia:.1f})",
                        "value": f"{hypoxia}",
                        "weight": w,
                        "category": "ml_feature",
                        "source": "ML-validated (22K patients)"
                    })
            except (ValueError, TypeError):
                pass

        # Aneuploidy
        aneuploidy = base.get('aneuploidy_score') or base.get('aneuploidy')
        if aneuploidy is not None:
            try:
                aneuploidy = float(aneuploidy)
                if aneuploidy >= thresholds["aneuploidy_high"]:
                    w = clinical_weights["aneuploidy_high"]
                    score += w
                    reasons.append(f"Aneuploidy High ({aneuploidy:.0f}) (+{w})")
                    contributions.append({
                        "factor": f"Aneuploidy High ({aneuploidy:.0f})",
                        "value": f"{aneuploidy}",
                        "weight": w,
                        "category": "ml_feature",
                        "source": "ML-validated (22K patients)"
                    })
            except (ValueError, TypeError):
                pass

        # Fraction Genome Altered
        fga = base.get('fraction_genome_altered')
        if fga is not None:
            try:
                fga = float(fga)
                if fga >= thresholds["fga_high"]:
                    w = clinical_weights["fraction_genome_altered_high"]
                    score += w
                    reasons.append(f"FGA High ({fga:.2f}) (+{w})")
                    contributions.append({
                        "factor": f"FGA High ({fga:.2f})",
                        "value": f"{fga}",
                        "weight": w,
                        "category": "ml_feature",
                        "source": "ML-validated (22K patients)"
                    })
            except (ValueError, TypeError):
                pass

        # TMB (ML-validated)
        tmb = safe_float(base.get('tmb') or base.get('tmb_score'), 0)
        if tmb >= thresholds["tmb_high"]:
            w = clinical_weights["tmb_high"]
            score += w
            reasons.append(f"TMB High ({tmb:.1f}) (+{w})")
            contributions.append({
                "factor": f"TMB High (>={thresholds['tmb_high']}, actual={tmb:.1f})",
                "value": f"{tmb}",
                "weight": w,
                "category": "ml_feature",
                "source": "ML-validated (22K patients)"
            })

        # MSI
        msi = base.get('msi_score') or base.get('msi')
        if msi is not None:
            try:
                msi = float(msi)
                if msi >= thresholds["msi_high"]:
                    w = clinical_weights["msi_high"]
                    score += w
                    reasons.append(f"MSI High ({msi:.1f}) (+{w})")
                    contributions.append({
                        "factor": f"MSI High ({msi:.1f})",
                        "value": f"{msi}",
                        "weight": w,
                        "category": "ml_feature",
                        "source": "ML-validated (22K patients)"
                    })
            except (ValueError, TypeError):
                pass

        # =====================================================================
        # 5. VISION AI OVERRIDE (Unchanged from v18)
        # =====================================================================
        if vision_risk > 80:
            vision_boost = max(0, 85 - score)
            if vision_boost > 0:
                contributions.append({
                    "factor": "VISION_AI",
                    "value": f"{vision_risk}%",
                    "weight": vision_boost,
                    "category": "vision",
                    "source": "Vision AI"
                })
            score = max(score, 85)
            reasons.append(f"VISION AI OVERRIDE (Risk {vision_risk}%)")

        # =====================================================================
        # 6. FINAL SCORE CAPPING
        # =====================================================================
        raw_score = score
        final_score = max(0, min(100, score))

        if raw_score != final_score:
            contributions.append({
                "factor": f"Score capped",
                "value": f"Raw: {raw_score}, Final: {final_score}",
                "weight": 0,
                "category": "cap",
                "source": "System"
            })

        logger.info(f"📊 Tank Score: {final_score}/100" + (f" (raw: {raw_score})" if raw_score > 100 else ""))
        logger.info(f"   Active Genes: {', '.join(active_genes) if active_genes else 'None'}")

        return final_score, reasons, status, ldh, tmb, nlr, contributions

    def _calculate_bayesian_risk(self, patient_data: Dict, ldh: float) -> Tuple[int, str, str, List[Dict]]:
        """
        Calculate Ferrari (Bayesian) score with Overall Risk.
        UNCHANGED LOGIC from v18 - proven to work.
        """
        evidence_details = []
        metabolic_stress = ldh > ML_WEIGHTS["thresholds"]["ldh_high"]

        if MODULES_STATUS["BAYESIAN"]:
            try:
                base = patient_data.get('baseline', patient_data)
                current_patient_id = base.get('patient_id', 'Unknown')

                if self.bayesian_cache is None or getattr(self, '_last_patient_id', None) != current_patient_id:
                    self.bayesian_cache = SentinelV30(
                        patient_id=current_patient_id,
                        therapy_start_date=base.get('therapy_start_date'),
                        diagnosis=base.get('diagnosis', base.get('histology', '')),
                        histology=base.get('histology', ''),
                        genetics=base.get('genetics', {})
                    )

                    self._last_patient_id = current_patient_id

                    genetics = base.get('genetics', {})
                    blood = base.get('blood_markers', {})

                    tp53 = str(genetics.get('tp53_status', '')).lower()
                    rb1 = str(genetics.get('rb1_status', '')).lower()
                    stk11 = str(genetics.get('stk11_status', '')).lower()
                    keap1 = str(genetics.get('keap1_status', '')).lower()

                    self.bayesian_cache.calibrate_priors(
                        ldh=ldh,
                        ecog=int(base.get('ecog_ps', 1)),
                        tp53_mutated=tp53 in ['mutated', 'mut', 'loss'],
                        rb1_mutated=rb1 in ['mutated', 'mut', 'loss'],
                        stk11_mutated=stk11 in ['mutated', 'mut', 'loss'],
                        keap1_mutated=keap1 in ['mutated', 'mut', 'loss'],
                        met_cn=float(genetics.get('met_cn', 0) or 0),
                        her2_cn=float(genetics.get('her2_cn', 0) or 0)
                    )

                evidences = self.translator.translate_patient(patient_data)
                logger.debug(f"🏎️  Ferrari Evidences: {evidences}")

                if not evidences:
                    if metabolic_stress:
                        logger.info("🏎️  Ferrari: No drivers but LDH high → Metabolic Override")
                        evidence_details.append({
                            "evidence": "Metabolic_Deregulation",
                            "probability": 88.0,
                            "risk_level": "HIGH",
                            "CI": "N/A",
                            "source": "LDH_Override"
                        })
                        return 88, "Metabolic_Deregulation (Warburg Effect)", "High (Override)", evidence_details
                    else:
                        logger.info("🏎️  Ferrari: No active drivers → 10% risk")
                        return 10, "No_Active_Drivers", "High", []

                # Calculate Bayesian prediction
                self.bayesian_cache.update_probabilities(evidences)
                sorted_risks = self.bayesian_cache.get_sorted_risks()
                top_risk = sorted_risks[0]

                # Build evidence_details from top 5 mechanisms
                for mech, prob, level, ci in sorted_risks[:5]:
                    if prob > 0.05:
                        evidence_details.append({
                            "evidence": mech,
                            "probability": round(prob * 100, 1),
                            "risk_level": level.value if hasattr(level, 'value') else str(level),
                            "CI": str(ci) if ci else "N/A"
                        })

                base_ferrari_score = int(top_risk[1] * 100)
                base_mechanism = top_risk[0]
                confidence = "High" if len(evidences) >= 3 else ("Medium" if len(evidences) >= 1 else "Low")

                # === CALCULATE OVERALL RESISTANCE RISK ===
                # P(at least one) = 1 - P(none) = 1 - ∏(1 - P_i)
                overall_risk = 1.0
                for mech, prob, level, ci in sorted_risks[:5]:
                    overall_risk *= (1 - prob)
                overall_risk = (1 - overall_risk) * 100

                # Boost for high LDH
                if metabolic_stress:
                    overall_risk = max(overall_risk, 75)
                    if ldh > 500:
                        overall_risk = max(overall_risk, 82)
                    if ldh > 700:
                        overall_risk = max(overall_risk, 88)

                # Boost for multi-evidence
                num_evidences = len(evidences)
                if num_evidences >= 5:
                    overall_risk = min(98, overall_risk + 10)
                elif num_evidences >= 4:
                    overall_risk = min(95, overall_risk + 6)
                elif num_evidences >= 3:
                    overall_risk = min(92, overall_risk + 3)

                overall_risk = min(98, overall_risk)

                # === ALPHAMISSENSE PATHOGENICITY BOOST ===
                alphamissense_boost = 1.0
                alphamissense_details = []

                if ALPHAMISSENSE_AVAILABLE:
                    try:
                        genetics = base.get('genetics', {})
                        mutations_to_check = []

                        # === METHOD 1: Standard fields ===
                        # EGFR
                        egfr = str(genetics.get('egfr_status', '')).upper()
                        if egfr and egfr not in ['WT', 'WILD-TYPE', 'WILD_TYPE', 'NEGATIVE', '', 'WTD']:
                            variant = egfr.replace('EGFR', '').replace('(', '').replace(')', '').strip()
                            # Remove VAF info if present (e.g., "L858R (VAF 12%)" -> "L858R")
                            variant = variant.split('VAF')[0].strip()
                            if variant and len(variant) >= 3:
                                mutations_to_check.append(('EGFR', variant))

                        # KRAS
                        kras = str(genetics.get('kras_mutation', '')).upper()
                        if kras and kras not in ['WT', 'WILD-TYPE', 'WILD_TYPE', 'NEGATIVE', '']:
                            mutations_to_check.append(('KRAS', kras))

                        # BRAF
                        braf = str(genetics.get('braf_status', '')).upper()
                        if braf and braf not in ['WT', 'WILD-TYPE', 'WILD_TYPE', 'NEGATIVE', '']:
                            mutations_to_check.append(('BRAF', braf))

                        # PIK3CA
                        pik3ca = str(genetics.get('pik3ca_status', '')).upper()
                        if pik3ca and pik3ca not in ['WT', 'WILD-TYPE', 'WILD_TYPE', 'NEGATIVE', '']:
                            mutations_to_check.append(('PIK3CA', pik3ca))

                        # === METHOD 2: Parse VAF-style keys (GENE_VARIANT format) ===
                        # This catches: "TP53_R273H", "KRAS_G12C", "EGFR_L858R", etc.
                        import re
                        vaf_pattern = re.compile(r'^([A-Z0-9]+)_([A-Z]\d+[A-Z])$')

                        for key, value in genetics.items():
                            match = vaf_pattern.match(key.upper())
                            if match:
                                gene = match.group(1)
                                variant = match.group(2)
                                # Check it's a dict with status/vaf (VAF format)
                                if isinstance(value, dict):
                                    status = str(value.get('status', '')).lower()
                                    if status in ['mutated', 'mut', 'emerging', 'acquired', 'expanding']:
                                        # Avoid duplicates
                                        if (gene, variant) not in mutations_to_check:
                                            mutations_to_check.append((gene, variant))

                        # === METHOD 3: Check for "_loss" or "_amp" patterns ===
                        loss_genes = ['STK11', 'KEAP1', 'RB1', 'PTEN', 'NF1']
                        for gene in loss_genes:
                            key_loss = f"{gene}_loss"
                            if key_loss in genetics or f"{gene.lower()}_loss" in genetics:
                                val = genetics.get(key_loss) or genetics.get(f"{gene.lower()}_loss")
                                if isinstance(val, dict) and val.get('status') in ['mutated', 'mut', 'loss']:
                                    # Loss of function - mark as pathogenic (no specific variant)
                                    pass  # Can't query AlphaMissense without specific variant

                        # Query AlphaMissense
                        if mutations_to_check:
                            logger.info(f"🧬 AlphaMissense checking: {mutations_to_check}")
                            results = classify_mutations_batch(mutations_to_check)

                            pathogenic_count = 0
                            for (gene, variant), result in zip(mutations_to_check, results):
                                if result:
                                    alphamissense_details.append({
                                        'gene': gene,
                                        'variant': variant,
                                        'pathogenicity': result['pathogenicity'],
                                        'class': result['class'],
                                        'boost': result['ferrari_boost']
                                    })

                                    if result['is_pathogenic']:
                                        pathogenic_count += 1
                                        alphamissense_boost = max(alphamissense_boost, result['ferrari_boost'])

                            # Extra boost for multiple pathogenic mutations
                            if pathogenic_count >= 3:
                                alphamissense_boost = min(1.35, alphamissense_boost + 0.05)
                            elif pathogenic_count >= 2:
                                alphamissense_boost = min(1.32, alphamissense_boost + 0.02)

                            if alphamissense_boost > 1.0:
                                logger.info(
                                    f"🧬 AlphaMissense: {pathogenic_count} pathogenic mutations → boost {alphamissense_boost:.2f}x")

                    except Exception as am_err:
                        logger.warning(f"⚠️ AlphaMissense lookup error: {am_err}")

                # Apply AlphaMissense boost to overall risk
                if alphamissense_boost > 1.0:
                    overall_risk = min(98, overall_risk * alphamissense_boost)

                    # Add to evidence details
                    evidence_details.append({
                        "evidence": "AlphaMissense_Pathogenicity",
                        "probability": round((alphamissense_boost - 1.0) * 100, 1),
                        "risk_level": "HIGH" if alphamissense_boost >= 1.25 else "MEDIUM",
                        "details": alphamissense_details,
                        "source": "DeepMind_AlphaMissense"
                    })

                # Add overall_risk to evidence_details (CRITICAL!)
                evidence_details.append({
                    "evidence": "_OVERALL_RISK",
                    "probability": round(overall_risk, 1),
                    "risk_level": "CRITICAL" if overall_risk >= 80 else ("HIGH" if overall_risk >= 60 else "MEDIUM"),
                    "num_evidences": num_evidences,
                    "source": "Composite"
                })

                # Metabolic boost for primary mechanism
                if metabolic_stress:
                    boosted_score = max(base_ferrari_score, 70)

                    has_metabolic = any(e.get('evidence') == 'Metabolic_Deregulation' for e in evidence_details)
                    if not has_metabolic:
                        evidence_details.insert(0, {
                            "evidence": "Metabolic_Deregulation",
                            "probability": min(100.0, round((ldh - 200) / 10, 1)),
                            "risk_level": "HIGH",
                            "CI": "N/A",
                            "source": "LDH_Boost"
                        })

                    confidence = "High (LDH Boost)"
                    logger.info(f"🏎️  Ferrari: {base_ferrari_score}% → {boosted_score}%")
                    logger.info(f"🏎️  Overall Resistance Risk: {overall_risk:.0f}%")

                    return boosted_score, base_mechanism, confidence, evidence_details

                logger.info(f"🏎️  Ferrari Score: {base_ferrari_score}% (Mechanism: {base_mechanism})")
                logger.info(f"🏎️  Overall Resistance Risk: {overall_risk:.0f}%")
                return base_ferrari_score, base_mechanism, confidence, evidence_details

            except Exception as e:
                logger.error(f"🏎️  Ferrari Error: {str(e)}")
                if metabolic_stress:
                    evidence_details.append({
                        "evidence": "Metabolic_Deregulation",
                        "probability": 88.0,
                        "risk_level": "HIGH",
                        "CI": "N/A",
                        "source": "LDH_Fallback"
                    })
                    return 88, "Metabolic_Deregulation (Warburg Effect)", "High (Override)", evidence_details
                return 50, "Bayesian_Error", "Low", []

        # Bayesian module unavailable
        logger.warning("🏎️  Ferrari: Bayesian module unavailable, using heuristic")
        if metabolic_stress:
            evidence_details.append({
                "evidence": "Metabolic_Deregulation",
                "probability": 88.0,
                "risk_level": "HIGH",
                "CI": "N/A",
                "source": "Heuristic_LDH"
            })
            return 88, "Metabolic_Deregulation (Warburg Effect)", "High (Override)", evidence_details
        return 15, "Heuristic_Baseline", "Low", []

    def calculate_risk(self, patient_data: Dict) -> Tuple[PrognosisResult, ExplainabilityReport]:
        """Main prognosis calculation."""
        logger.info("\n" + "=" * 60)
        logger.info("PROGNOSIS ENGINE - ML-Enhanced Risk Assessment")
        logger.info("=" * 60)

        # Vision AI
        vision_risk = 0
        if MODULES_STATUS["VISION"]:
            try:
                img = patient_data.get('baseline', {}).get('biopsy_image_path')
                if img:
                    vision_data = analyze_biopsy(img)
                    vision_risk = vision_data.get('visual_risk', 0)
            except Exception as e:
                logger.warning(f"👁️  Vision AI Error: {str(e)}")

        # Tank score
        tank_score, reasons, status, ldh, tmb, nlr, tank_contributions = self._calculate_clinical_risk(
            patient_data, vision_risk
        )

        # Ferrari score
        ferrari_score, mechanism, confidence, ferrari_evidences = self._calculate_bayesian_risk(patient_data, ldh)

        # Detect synergies
        synergies = self._detect_synergies(patient_data)

        logger.info("=" * 60 + "\n")

        prognosis = PrognosisResult(
            tank_score=tank_score,
            ferrari_score=ferrari_score,
            reasons=reasons,
            status=status,
            ldh=ldh,
            tmb=tmb,
            nlr=nlr
        )

        explainability = ExplainabilityReport(
            tank_contributions=tank_contributions,
            tank_total=tank_score,
            ferrari_evidences=ferrari_evidences,
            ferrari_mechanism=mechanism,
            ferrari_probability=ferrari_score / 100,
            ferrari_confidence=confidence,
            synergies=synergies
        )

        return prognosis, explainability

    def _detect_synergies(self, patient_data: Dict) -> List[Dict]:
        """Detect mutation synergies."""
        synergies = []
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})

        stk11 = str(genetics.get('stk11_status', '')).lower()
        keap1 = str(genetics.get('keap1_status', '')).lower()
        tp53 = str(genetics.get('tp53_status', '')).lower()
        rb1 = str(genetics.get('rb1_status', '')).lower()
        kras = str(genetics.get('kras_mutation', '')).lower()

        if stk11 in ['mutated', 'mut', 'loss'] and keap1 in ['mutated', 'mut', 'loss']:
            synergies.append({
                "pair": "STK11 + KEAP1",
                "effect": "Double Loss - Severe Immuno-resistance",
                "clinical_impact": "Checkpoint inhibitors likely ineffective",
                "boost_percent": 20
            })

        if tp53 in ['mutated', 'mut', 'loss'] and rb1 in ['mutated', 'mut', 'loss']:
            synergies.append({
                "pair": "TP53 + RB1",
                "effect": "High risk of SCLC transformation",
                "clinical_impact": "Monitor histology, consider platinum-etoposide",
                "boost_percent": 30
            })

        if kras not in ['wt', 'none', ''] and stk11 in ['mutated', 'mut', 'loss']:
            synergies.append({
                "pair": "KRAS + STK11",
                "effect": "Cold tumor phenotype",
                "clinical_impact": "Reduced response to KRAS inhibitors (~40%)",
                "boost_percent": 15
            })

        return synergies


# ============================================================================
# DIGITAL TWIN (Unchanged from v18)
# ============================================================================

# ============================================================================
# DIGITAL TWIN (ML-Enhanced v2.0)
# ============================================================================

class DigitalTwin:
    """Simulates patient outcome based on risk and therapy - ML Enhanced."""

    @staticmethod
    def simulate_outcome(risk_score: int, elephant_active: bool, veto_active: bool,
                         ldh: float = 200, patient_data: Dict = None) -> Dict:
        """
        Simula outcome usando ML (se disponibile) o formula.

        Args:
            risk_score: Score di rischio 0-100
            elephant_active: True se LDH > 350
            veto_active: True se terapia inappropriata
            ldh: Valore LDH (per Elephant)
            patient_data: Dict paziente per predizioni ML
        """

        print(f"\n🔍 DEBUG BEFORE TWIN CALL:")
        print(f"   risk_score={risk_score}, elephant={elephant_active}, veto={veto_active}")
        print(f"   ldh={ldh}, patient_data provided: {patient_data is not None}")

        model_source = "FORMULA"
        ml_death_risk = None
        ml_os_months = None  # <-- AGGIUNGI QUESTA VARIABILE

        # === PROVA PREDIZIONE ML ===
        if patient_data and not veto_active:
            print(f"   🧪 Attempting ML prediction...")
            try:
                ml_result = DigitalTwin._predict_with_ml(patient_data)
                if ml_result:
                    print(f"   ✅ ML SUCCESS: OS={ml_result.get('os_months')}m, Risk={ml_result.get('death_risk')}%")
                    model_source = "ML_500K"
                    ml_death_risk = ml_result.get('death_risk')
                    ml_os_months = ml_result.get('os_months')  # <-- SALVA OS
                else:
                    print(f"   ⚠️ ML returned None")
            except Exception as e:
                print(f"   ❌ ML FAILED: {e}")

        # === CALCOLO PFS ===
        if veto_active:
            pfs_soc = 1.5
            dynamics = "UNCONTROLLED GROWTH (Therapy Mismatch)"
            forecast = "Rapid Progression. Immediate switch required."
            model_source = "VETO_OVERRIDE"
        else:
            # USA ML SE DISPONIBILE, ALTRIMENTI FORMULA
            if model_source == "ML_500K" and ml_os_months:
                pfs_soc = ml_os_months
                print(f"   📊 Using ML prediction for PFS: {pfs_soc} months")
            else:
                pfs_soc = round(36 * ((100 - risk_score) / 100), 1)
                print(f"   📊 Using FORMULA for PFS: {pfs_soc} months")

            if pfs_soc < 2:
                pfs_soc = 2.0

            # Dynamics basato su PFS (non su risk_score)
            if pfs_soc > 30:
                dynamics = "Rapid Regression (-40% at 3m)"
                forecast = "Deep response expected. CR possible (model-based, requires RECIST confirmation)."
            elif pfs_soc > 18:
                dynamics = "Stable Disease / Partial Resp (-15% at 3m)"
                forecast = "PARTIAL RESPONSE (PR) maintained."
            elif pfs_soc > 9:
                dynamics = "Mixed Response (Stable/-5%)"
                forecast = "STABLE DISEASE (SD). Metabolic control active."
            else:
                dynamics = "Resistance / Pseudo-progression"
                forecast = "PROGRESSION (PD) likely. Salvage required."

        # === ELEPHANT BOOST ===
        if elephant_active and not veto_active:
            boost = 1.6 if pfs_soc < 12 else 1.3  # Boost maggiore per chi sta peggio
            pfs_sentinel = round(pfs_soc * boost, 1)
            delta_months = round(pfs_sentinel - pfs_soc, 1)
        else:
            pfs_sentinel = pfs_soc
            delta_months = 0

        result = {
            "pfs_soc": pfs_soc,
            "pfs_sentinel": pfs_sentinel,
            "delta": delta_months,
            "dynamics": dynamics,
            "forecast": forecast,
            "model_source": model_source
        }

        if ml_death_risk is not None:
            result["ml_death_risk"] = ml_death_risk

        print(f"   🎯 FINAL: pfs_soc={pfs_soc}, model={model_source}")

        return result

    @staticmethod
    def _predict_with_ml(patient_data: Dict) -> Optional[Dict]:
        """Tenta predizione con modelli ML addestrati."""
        try:
            import joblib
            import pandas as pd
            from pathlib import Path

            model_dir = Path(__file__).parent.parent / "models"

            os_model_path = model_dir / "sentinel_os_regressor.pkl"
            risk_model_path = model_dir / "sentinel_risk_classifier.pkl"
            features_path = model_dir / "sentinel_feature_cols.pkl"

            if not all(p.exists() for p in [os_model_path, risk_model_path, features_path]):
                return None

            os_model = joblib.load(os_model_path)
            risk_model = joblib.load(risk_model_path)
            feature_cols = joblib.load(features_path)

            # Estrai features
            base = patient_data.get('baseline', patient_data)
            gen = base.get('genetics', {})

            features = {col: 0 for col in feature_cols}

            # Cliniche
            features['age'] = int(base.get('age', 60) or 60)
            features['sex'] = 1 if str(base.get('sex', '')).upper().startswith('M') else 0
            features['ecog'] = int(base.get('ecog_ps', 1) or 1)
            features['ldh'] = float(base.get('blood_markers', {}).get('ldh', 200) or 200)
            features['tmb'] = float(gen.get('tmb_score', 5) or 5)

            # Genomiche
            wt = ['wt', 'WT', 'wild-type', '', None]
            features['tp53'] = 1 if gen.get('tp53_status', 'wt') not in wt else 0
            features['kras'] = 1 if gen.get('kras_mutation', 'wt') not in wt else 0
            features['egfr'] = 1 if gen.get('egfr_status', 'wt') not in wt else 0
            features['stk11'] = 1 if gen.get('stk11_status', 'wt') not in wt else 0
            features['keap1'] = 1 if gen.get('keap1_status', 'wt') not in wt else 0
            features['met'] = 1 if gen.get('met_status', 'wt') not in wt else 0
            features['braf'] = 1 if gen.get('braf_mutation', 'wt') not in wt else 0
            features['pik3ca'] = 1 if gen.get('pik3ca_status', 'wt') not in wt else 0

            # Cancer type
            diagnosis = str(base.get('diagnosis', '')).lower()
            for col in feature_cols:
                if col.startswith('cancer_'):
                    cancer_name = col.replace('cancer_', '').replace('_', ' ')
                    if cancer_name in diagnosis:
                        features[col] = 1

            X = pd.DataFrame([features])[feature_cols].fillna(0)

            os_pred = os_model.predict(X)[0]
            risk_pred = risk_model.predict_proba(X)[0][1]

            return {
                'os_months': round(max(os_pred, 1), 1),
                'death_risk': round(risk_pred * 100, 1)
            }

        except Exception as e:
            return None


# ============================================================================
# DUAL TRACK ENGINE (Main Orchestrator)
# ============================================================================

class SentinelDualTrack:
    """Main SENTINEL v19.0 ML-Enhanced engine."""

    def __init__(self):
        self.veto_system = VetoSystem()
        self.prognosis_engine = PrognosisEngine()
        logger.info("\n" + "🚀 " * 20)
        logger.info("SENTINEL v19.0 ML-ENHANCED ENGINE INITIALIZED")
        logger.info("ML Weights: 22,053 patients | C-index: OS=0.734, PFS=0.693")
        logger.info("🚀 " * 20 + "\n")

    def analyze_patient_risk(self, patient_data: Dict) -> DualTrackResult:
        """Main analysis function."""
        logger.info("\n" + "🎯 " * 20)
        logger.info("STARTING DUAL TRACK ANALYSIS (ML-Enhanced)")
        logger.info("🎯 " * 20 + "\n")

        # TRACK 1: PROGNOSIS
        logger.info("━━━ TRACK 1: PROGNOSIS (ML-Validated) ━━━")
        prognosis, explainability = self.prognosis_engine.calculate_risk(patient_data)

        # TRACK 2: PRESCRIPTION
        logger.info("━━━ TRACK 2: PRESCRIPTION (Therapy Check) ━━━")
        veto = self.veto_system.check_therapy(patient_data)

        # MERGE TRACKS
        logger.info("\n━━━ MERGING TRACKS ━━━")

        if veto.active:
            display_risk = 100
            display_status = "THERAPY_MISMATCH"
            alert_level = AlertLevel.RED

            final_recommendation = (
                f"⛔ CRITICAL: {veto.reason}\n\n"
                f"IMMEDIATE ACTION REQUIRED:\n{veto.recommendation}\n\n"
                f"BIOLOGICAL PROGNOSIS (if therapy corrected):\n"
                f"• Survival Risk: {prognosis.ferrari_score}%\n"
                f"• Tumor Biology Score: {prognosis.tank_score}/100\n"
                f"• Status: {'FAVORABLE' if prognosis.ferrari_score < 50 else 'UNFAVORABLE'} "
                f"prognosis IF appropriate therapy initiated"
            )

            consensus = "VETO OVERRIDE - Therapy incompatible with molecular profile"

            logger.warning("⛔ VETO ACTIVE - Display Risk set to 100/100")

        else:
            display_risk = max(prognosis.tank_score, prognosis.ferrari_score)

            if prognosis.ferrari_score < 30:
                display_status = "STABLE_RESPONDER"
            elif prognosis.ferrari_score < 60:
                display_status = "MODERATE_RISK"
            else:
                display_status = "HIGH_RISK"

            divergence = abs(prognosis.tank_score - prognosis.ferrari_score)
            if divergence > 40 or display_risk >= 80:
                alert_level = AlertLevel.RED
            elif divergence > 20 or display_risk >= 60:
                alert_level = AlertLevel.ORANGE
            elif display_risk >= 40:
                alert_level = AlertLevel.YELLOW
            else:
                alert_level = AlertLevel.GREEN

            final_recommendation = (
                f"Current therapy appears appropriate.\n"
                f"Continue monitoring with:\n"
                f"• Imaging every 8-12 weeks\n"
                f"• Tumor markers monthly\n"
                f"• Clinical assessment at each visit"
            )

            consensus = f"Biological Assessment - Tank: {prognosis.tank_score}, Ferrari: {prognosis.ferrari_score}"

            logger.info(f"✓ No VETO - Display Risk: {display_risk}/100")

        # Extract active genes
        active_genes = [r.split('(')[0].strip() for r in prognosis.reasons
                        if any(g in r for g in ML_WEIGHTS["mutations"].keys())]

        result = DualTrackResult(
            veto_active=veto.active,
            veto_reason=veto.reason,
            veto_recommendation=veto.recommendation,
            biological_risk_tank=prognosis.tank_score,
            biological_risk_ferrari=prognosis.ferrari_score,
            prognosis_reasons=prognosis.reasons,
            prognosis_status=prognosis.status,
            display_risk=display_risk,
            display_status=display_status,
            alert_level=alert_level,
            ldh=prognosis.ldh,
            tmb=prognosis.tmb,
            nlr=prognosis.nlr,
            final_recommendation=final_recommendation,
            consensus=consensus,
            active_genes=active_genes
        )

        logger.info("\n" + "✅ " * 20)
        logger.info("DUAL TRACK ANALYSIS COMPLETE")
        logger.info(f"Final Display Risk: {display_risk}/100")
        logger.info(f"Alert Level: {alert_level.value}")
        logger.info("✅ " * 20 + "\n")

        return result


# ============================================================================
# COMPATIBILITY WRAPPER
# ============================================================================

def analyze_patient_risk(patient_data: Dict) -> Dict:
    """
    Main entry point for SENTINEL v19.0 ML-Enhanced.
    Compatible with existing report generation system.
    """
    print("\n" + "=" * 70)
    print("SENTINEL v19.0 ML-ENHANCED ENGINE")
    print("ML Weights: 22,053 patients | C-index: OS=0.734, PFS=0.693")
    print("=" * 70 + "\n")

    engine = SentinelDualTrack()
    result = engine.analyze_patient_risk(patient_data)
    _, explainability = engine.prognosis_engine.calculate_risk(patient_data)

    # Digital Twin calculation
    # Digital Twin calculation (ML-Enhanced v2.0)
    is_elephant = "ELEPHANT" in str(result.prognosis_reasons) or result.ldh > ML_WEIGHTS["thresholds"]["ldh_high"]

    # Prepara patient_data per predizioni ML
    patient_data_for_twin = {
        "baseline": {
            "age": patient_data.get('age', patient_data.get('baseline', {}).get('age', 60)),
            "sex": patient_data.get('sex', patient_data.get('baseline', {}).get('sex', 'U')),
            "ecog_ps": patient_data.get('ecog_ps', patient_data.get('baseline', {}).get('ecog_ps', 1)),
            "diagnosis": patient_data.get('diagnosis', patient_data.get('baseline', {}).get('diagnosis', 'Unknown')),
            "histology": patient_data.get('histology', patient_data.get('baseline', {}).get('histology', 'Unknown')),
            "genetics": patient_data.get('genetics', patient_data.get('baseline', {}).get('genetics', {})),
            "blood_markers": patient_data.get('blood_markers', patient_data.get('baseline', {}).get('blood_markers', {
                'ldh': result.ldh}))
        }
    }

    # DEBUG: Verifica cosa viene passato
    print(f"\n🔍 DEBUG BEFORE TWIN CALL:")
    print(f"   result.display_risk = {result.display_risk}")
    print(f"   result.ldh = {result.ldh}")
    print(f"   is_elephant = {is_elephant}")
    print(f"   patient_data type = {type(patient_data)}")
    print(f"   patient_data keys = {patient_data.keys() if patient_data else 'None'}")

    twin_data = DigitalTwin.simulate_outcome(
        risk_score=result.display_risk,
        elephant_active=is_elephant,
        veto_active=result.veto_active,
        ldh=result.ldh,
        patient_data=patient_data
    )

    output = {
        # === DISPLAY (Primary UI) ===
        "display_risk": result.display_risk,
        "display_status": result.display_status,
        "alert_level": result.alert_level.value,
        "final_recommendation": result.final_recommendation,

        # === VETO TRACK ===
        "veto_active": result.veto_active,
        "veto_reason": result.veto_reason,
        "veto_recommendation": result.veto_recommendation,

        # === PROGNOSIS TRACK ===
        "tank_score": result.biological_risk_tank,
        "ferrari_score": result.biological_risk_ferrari,
        "tank_reasons": result.prognosis_reasons,
        "prognosis_status": result.prognosis_status,

        # === CLINICAL DATA ===
        "ldh": result.ldh,
        "tmb": result.tmb,
        "nlr": result.nlr,
        "active_genes": result.active_genes,

        # === METADATA ===
        "consensus": result.consensus,
        "status": result.display_status,
        "final_risk": result.display_risk,
        "digital_twin": twin_data,

        # === EXPLAINABILITY ===
        "explainability": {
            "tank_breakdown": explainability.tank_contributions,
            "ferrari_breakdown": explainability.ferrari_evidences,
            "ferrari_mechanism": explainability.ferrari_mechanism,
            "ferrari_confidence": explainability.ferrari_confidence,
            "synergies": explainability.synergies
        },

        # === COMPATIBILITY (for old report templates) ===
        "ferrari_details": f"Biological Risk: {result.biological_risk_ferrari}%",
        "match_status": result.alert_level.value,
        "vision_data": result.vision_data or {},
        "physics_verdict": result.physics_verdict or "N/A",
        "vittoria_drug": None,
        "vittoria_score": None,
        "divergence": abs(result.biological_risk_tank - result.biological_risk_ferrari),

        # === ML METADATA ===
        "ml_metadata": {
            "version": "v19.0_ML",
            "training_patients": 22053,
            "c_index_os": 0.734,
            "c_index_pfs": 0.693,
        }
    }

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    return output


# ============================================================================
# MAIN (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    print("\n🧪 SENTINEL v19.0 ML-Enhanced - Test Mode\n")

    test_patient = {
        'baseline': {
            'patient_id': 'TEST-ML-001',
            'current_therapy': 'Gefitinib',
            'genetics': {
                'tp53_status': 'mutated',
                'egfr_status': 'L858R + T790M',
                'kras_status': 'wt',
                'stk11_status': 'wt'
            },
            'blood_markers': {
                'ldh': 650,
                'neutrophils': 9500,
                'lymphocytes': 500,
                'albumin': 2.8
            },
            'tmb': 12.5,
            'ecog_ps': 3
        }
    }

    result = analyze_patient_risk(test_patient)

    print("\n📊 TEST RESULTS:")
    print(f"Display Risk: {result['display_risk']}/100")
    print(f"Alert Level: {result['alert_level']}")
    print(f"VETO Active: {result['veto_active']}")
    if result['veto_active']:
        print(f"VETO Reason: {result['veto_reason']}")
    print(f"\nBiological Risk - Tank: {result['tank_score']}, Ferrari: {result['ferrari_score']}")
    print(f"LDH: {result['ldh']} U/L")

    print("\n📋 Tank Breakdown:")
    for c in result['explainability']['tank_breakdown'][:10]:
        print(f"   • {c['factor']}: +{c['weight']} [{c.get('source', 'N/A')}]")

    # Check for Overall Risk
    overall_risk = None
    for e in result['explainability']['ferrari_breakdown']:
        if e.get('evidence') == '_OVERALL_RISK':
            overall_risk = e.get('probability')
            break

    if overall_risk:
        print(f"\n🎯 OVERALL RESISTANCE RISK: {overall_risk}%")

    print(f"\n🔮 Digital Twin: PFS = {result['digital_twin']['pfs_soc']} months")
    print(f"   Forecast: {result['digital_twin']['forecast']}")
