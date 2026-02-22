"""
SENTINEL - ELEPHANT PROTOCOL v2.0 (Dynamic & Personalized)
==========================================================
Protocollo di contenimento tumorale ispirato alla resistenza naturale
degli elefanti al cancro (gene TP53 multiplo).

Questo modulo genera un protocollo PERSONALIZZATO basato su:
- Profilo genetico del paziente
- Controindicazioni cliniche
- ECOG, età, metastasi
- Integrazione con sistema VETO

Author: SENTINEL Development Team
Version: 2.0
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TherapyEvidence(Enum):
    """Livello di evidenza per le terapie suggerite"""
    FDA_APPROVED = "FDA Approved"
    EMA_APPROVED = "EMA Approved"
    NCCN_GUIDELINE = "NCCN Guideline"
    PHASE_3 = "Phase 3 Trial"
    PHASE_2 = "Phase 2 Trial"
    PHASE_1 = "Phase 1/Experimental"
    OFF_LABEL = "Off-Label"
    PRECLINICAL = "Preclinical/Theoretical"


class UrgencyLevel(Enum):
    """Urgenza dell'intervento"""
    IMMEDIATE = "Immediate (within 48h)"
    URGENT = "Urgent (within 1 week)"
    STANDARD = "Standard (next visit)"
    ELECTIVE = "Elective (can wait)"


@dataclass
class TherapySuggestion:
    """Singola terapia suggerita"""
    drug_name: str
    dose: str
    rationale: str
    evidence: TherapyEvidence
    contraindications: List[str] = field(default_factory=list)
    monitoring: str = ""
    experimental: bool = False


@dataclass
class ProtocolPhase:
    """Fase del protocollo Elephant"""
    name: str
    duration: str
    objectives: List[str]
    therapies: List[TherapySuggestion]
    excluded_therapies: List[Tuple[str, str]] = field(default_factory=list)  # (drug, reason)
    warnings: List[str] = field(default_factory=list)


@dataclass 
class ElephantProtocolResult:
    """Risultato completo del Protocollo Elephant"""
    activated: bool
    activation_reasons: List[str]
    metabolic_sensitivity: float  # 0-100%
    phases: List[ProtocolPhase]
    experimental_options: List[TherapySuggestion]
    contraindications_detected: List[str]
    personalization_summary: str
    clinical_rationale: str
    disclaimer: str


# ============================================================================
# KNOWLEDGE BASES
# ============================================================================

# Terapie target per mutazione
TARGETED_THERAPIES = {
    "EGFR": {
        "first_line": [
            ("Osimertinib", "80mg QD", TherapyEvidence.FDA_APPROVED),
            ("Erlotinib", "150mg QD", TherapyEvidence.FDA_APPROVED),
            ("Gefitinib", "250mg QD", TherapyEvidence.FDA_APPROVED),
        ],
        "t790m_resistance": [
            ("Osimertinib", "80mg QD", TherapyEvidence.FDA_APPROVED),
        ],
        "c797s_resistance": [
            ("Amivantamab", "1050/1400mg IV weekly", TherapyEvidence.FDA_APPROVED),
            ("Brigatinib + Cetuximab", "combo", TherapyEvidence.PHASE_2),
        ],
        "exon20_insertion": [
            ("Amivantamab", "1050/1400mg IV", TherapyEvidence.FDA_APPROVED),
            ("Mobocertinib", "160mg QD", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "MET": {
        "amplification": [
            ("Capmatinib", "400mg BID", TherapyEvidence.FDA_APPROVED),
            ("Tepotinib", "450mg QD", TherapyEvidence.FDA_APPROVED),
            ("Savolitinib", "600mg QD", TherapyEvidence.PHASE_3),
        ],
        "exon14_skip": [
            ("Capmatinib", "400mg BID", TherapyEvidence.FDA_APPROVED),
            ("Tepotinib", "450mg QD", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "KRAS": {
        "g12c": [
            ("Sotorasib", "960mg QD", TherapyEvidence.FDA_APPROVED),
            ("Adagrasib", "600mg BID", TherapyEvidence.FDA_APPROVED),
        ],
        "other": [
            ("Chemotherapy", "platinum-based", TherapyEvidence.NCCN_GUIDELINE),
        ]
    },
    "ALK": {
        "fusion": [
            ("Alectinib", "600mg BID", TherapyEvidence.FDA_APPROVED),
            ("Brigatinib", "180mg QD", TherapyEvidence.FDA_APPROVED),
            ("Lorlatinib", "100mg QD", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "BRAF": {
        "v600e": [
            ("Dabrafenib + Trametinib", "combo", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "RET": {
        "fusion": [
            ("Selpercatinib", "160mg BID", TherapyEvidence.FDA_APPROVED),
            ("Pralsetinib", "400mg QD", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "ROS1": {
        "fusion": [
            ("Crizotinib", "250mg BID", TherapyEvidence.FDA_APPROVED),
            ("Entrectinib", "600mg QD", TherapyEvidence.FDA_APPROVED),
        ]
    },
    "HER2": {
        "mutation": [
            ("Trastuzumab Deruxtecan", "5.4mg/kg IV q3w", TherapyEvidence.FDA_APPROVED),
        ],
        "amplification": [
            ("Trastuzumab Deruxtecan", "5.4mg/kg IV q3w", TherapyEvidence.FDA_APPROVED),
        ]
    }
}

# Terapie metaboliche (core del Protocollo Elephant)
METABOLIC_THERAPIES = {
    "metformin": TherapySuggestion(
        drug_name="Metformin",
        dose="500mg BID, titrate to 1000mg BID",
        rationale="Mitochondrial Complex I inhibition, reduces Warburg effect",
        evidence=TherapyEvidence.PHASE_2,
        contraindications=["eGFR < 30", "Metabolic acidosis", "Severe hepatic impairment"],
        monitoring="Creatinine, B12 levels q3 months"
    ),
    "ketogenic_diet": TherapySuggestion(
        drug_name="Ketogenic Diet",
        dose="<50g carbs/day, supervised",
        rationale="Glucose deprivation exploits tumor metabolic inflexibility",
        evidence=TherapyEvidence.PHASE_2,
        contraindications=["Cachexia", "BMI < 18", "Diabetes Type 1", "Pyruvate carboxylase deficiency"],
        monitoring="Ketone levels, weight, albumin weekly"
    ),
    "dichloroacetate": TherapySuggestion(
        drug_name="Dichloroacetate (DCA)",
        dose="10-15 mg/kg/day",
        rationale="PDK inhibitor, forces oxidative phosphorylation",
        evidence=TherapyEvidence.PHASE_1,
        contraindications=["Peripheral neuropathy", "Hepatic impairment"],
        monitoring="Neuropathy assessment, LFTs",
        experimental=True
    ),
    "2dg": TherapySuggestion(
        drug_name="2-Deoxyglucose (2-DG)",
        dose="45 mg/kg IV with radiation",
        rationale="Glycolysis inhibitor, synergizes with radiation",
        evidence=TherapyEvidence.PHASE_1,
        contraindications=["Hypoglycemia risk", "Cardiac arrhythmias"],
        monitoring="Glucose levels, ECG",
        experimental=True
    )
}

# Immunoterapia
IMMUNOTHERAPY = {
    "pd1_inhibitors": [
        TherapySuggestion("Pembrolizumab", "200mg IV q3w", "PD-1 blockade", TherapyEvidence.FDA_APPROVED),
        TherapySuggestion("Nivolumab", "240mg IV q2w", "PD-1 blockade", TherapyEvidence.FDA_APPROVED),
    ],
    "pdl1_inhibitors": [
        TherapySuggestion("Atezolizumab", "1200mg IV q3w", "PD-L1 blockade", TherapyEvidence.FDA_APPROVED),
        TherapySuggestion("Durvalumab", "10mg/kg IV q2w", "PD-L1 blockade", TherapyEvidence.FDA_APPROVED),
    ],
    "ctla4_inhibitors": [
        TherapySuggestion("Ipilimumab", "1mg/kg IV q6w", "CTLA-4 blockade", TherapyEvidence.FDA_APPROVED),
        TherapySuggestion("Tremelimumab", "75mg IV q4w", "CTLA-4 blockade", TherapyEvidence.FDA_APPROVED),
    ]
}


# ============================================================================
# ELEPHANT PROTOCOL ENGINE
# ============================================================================

class ElephantProtocolEngine:
    """
    Genera un Protocollo Elephant personalizzato basato sul profilo del paziente.
    """
    
    def __init__(self):
        logger.info("🐘 Elephant Protocol Engine v2.0 Initialized")
    
    def generate_protocol(self, patient_data: Dict, ai_result: Dict) -> ElephantProtocolResult:
        """
        Genera il protocollo personalizzato.
        
        Args:
            patient_data: Dati del paziente (baseline + visits)
            ai_result: Risultato dell'analisi SENTINEL (include VETO)
        
        Returns:
            ElephantProtocolResult con protocollo completo
        """
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})
        blood = base.get('blood_markers', {})
        
        # Estrai dati clinici
        ldh = float(blood.get('ldh', 200))
        age = int(base.get('age', 65))
        ecog = int(base.get('ecog_ps', 1))
        histology = base.get('histology', 'Adenocarcinoma').lower()
        stage = base.get('stage', 'IV')
        current_therapy = base.get('current_therapy', '')
        
        # Estrai VETO info
        veto_active = ai_result.get('veto_active', False)
        veto_reason = ai_result.get('veto_reason', '')
        
        # 1. Verifica attivazione
        activated, activation_reasons = self._check_activation(ldh, genetics, ai_result)
        
        if not activated:
            return ElephantProtocolResult(
                activated=False,
                activation_reasons=[],
                metabolic_sensitivity=0,
                phases=[],
                experimental_options=[],
                contraindications_detected=[],
                personalization_summary="Protocol not indicated",
                clinical_rationale="",
                disclaimer=""
            )
            
        # ECOG 3/4 Short Circuit
        if ecog >= 3:
            return ElephantProtocolResult(
                activated=True,
                activation_reasons=["ECOG >= 3 Safety Override"],
                metabolic_sensitivity=self._calculate_metabolic_sensitivity(ldh, genetics),
                phases=[
                    ProtocolPhase(
                        name="BEST SUPPORTIVE CARE (BSC)",
                        duration="Immediate",
                        objectives=["Symptom control", "Palliative care", "Improve quality of life"],
                        therapies=[
                            TherapySuggestion("Palliative Care", "As needed", "Symptom management due to poor PS", TherapyEvidence.NCCN_GUIDELINE)
                        ],
                        excluded_therapies=[("All systemic therapies", "Contraindicated due to ECOG >= 3")],
                        warnings=["Patient is ECOG >= 3. Aggressive systemic or metabolic therapy is contraindicated. Focus on best supportive care."]
                    )
                ],
                experimental_options=[],
                contraindications_detected=["ECOG >= 3: aggressive therapy contraindicated"],
                personalization_summary="Protocol overridden due to poor Performance Status (ECOG >= 3). Recommended Best Supportive Care.",
                clinical_rationale="Patient status (ECOG 3+) precludes the use of intense metabolic stress phases. Rapid physiological deterioration risk.",
                disclaimer="This is a safety override protocol."
            )
        
        # 2. Calcola sensibilità metabolica
        metabolic_sensitivity = self._calculate_metabolic_sensitivity(ldh, genetics)
        
        # 3. Rileva controindicazioni
        contraindications = self._detect_contraindications(base, genetics, blood)
        
        # 4. Analizza profilo genetico per target
        genetic_targets = self._analyze_genetic_targets(genetics)
        
        # 5. Genera le fasi personalizzate
        phase1 = self._generate_phase1_induction(
            genetics, ldh, current_therapy, veto_active, veto_reason, contraindications, age, ecog
        )
        
        phase2 = self._generate_phase2_consolidation(
            genetics, genetic_targets, current_therapy, veto_active, veto_reason, contraindications, ecog
        )
        
        phase3 = self._generate_phase3_maintenance(
            genetics, genetic_targets, contraindications, ecog
        )
        
        # 6. Opzioni sperimentali
        experimental = self._get_experimental_options(genetics, histology, contraindications)
        
        # 7. Genera summary personalizzato
        personalization_summary = self._generate_personalization_summary(
            genetics, genetic_targets, ldh, age, ecog, veto_active, veto_reason
        )
        
        # 8. Rationale clinico
        clinical_rationale = self._generate_clinical_rationale(
            activation_reasons, metabolic_sensitivity, genetic_targets, ldh
        )
        
        # 9. Disclaimer
        disclaimer = self._generate_disclaimer(age, ecog, experimental)
        
        return ElephantProtocolResult(
            activated=True,
            activation_reasons=activation_reasons,
            metabolic_sensitivity=metabolic_sensitivity,
            phases=[phase1, phase2, phase3],
            experimental_options=experimental,
            contraindications_detected=contraindications,
            personalization_summary=personalization_summary,
            clinical_rationale=clinical_rationale,
            disclaimer=disclaimer
        )
    
    def _check_activation(self, ldh: float, genetics: Dict, ai_result: Dict) -> Tuple[bool, List[str]]:
        """Verifica se il protocollo deve essere attivato"""
        reasons = []
        
        # Criterio 1: LDH elevato (Warburg effect)
        if ldh > 350:
            reasons.append(f"High LDH {ldh:.0f} U/L (Warburg Effect - metabolic vulnerability)")
        
        # Criterio 2: TP53 mutato (genomic instability)
        tp53 = str(genetics.get('tp53_status', '')).lower()
        if tp53 in ['mutated', 'mut', 'loss']:
            reasons.append("TP53 mutation (genomic instability - may benefit from metabolic stress)")
        
        # Criterio 3: High risk score
        if ai_result.get('display_risk', 0) >= 80:
            reasons.append(f"High biological risk ({ai_result.get('display_risk')}%) - aggressive intervention needed")
        
        # Criterio 4: Therapy mismatch / VETO
        if ai_result.get('veto_active'):
            reasons.append("Therapy mismatch detected - alternative strategy required")
        
        activated = len(reasons) > 0
        return activated, reasons
    
    def _calculate_metabolic_sensitivity(self, ldh: float, genetics: Dict) -> float:
        """Calcola la sensibilità metabolica (0-100%)"""
        sensitivity = 0
        
        # LDH contribuisce fino al 50%
        if ldh > 350:
            ldh_contrib = min(50, (ldh - 350) / 10)
            sensitivity += ldh_contrib
        
        # TP53 mutato aggiunge 20%
        tp53 = str(genetics.get('tp53_status', '')).lower()
        if tp53 in ['mutated', 'mut', 'loss']:
            sensitivity += 20
        
        # KRAS mutato aggiunge 15% (tumori KRAS spesso Warburg-dipendenti)
        kras = str(genetics.get('kras_mutation', '')).lower()
        if kras not in ['wt', 'none', '', 'wild-type']:
            sensitivity += 15
        
        # STK11 loss aggiunge 15% (regola metabolismo)
        stk11 = str(genetics.get('stk11_status', '')).lower()
        if stk11 in ['mutated', 'mut', 'loss']:
            sensitivity += 15
        
        return min(100, sensitivity)
    
    def _detect_contraindications(self, base: Dict, genetics: Dict, blood: Dict) -> List[str]:
        """Rileva controindicazioni per le terapie"""
        contraindications = []
        
        age = int(base.get('age', 65))
        ecog = int(base.get('ecog_ps', 1))
        
        # Età avanzata
        if age > 80:
            contraindications.append("Age >80: reduce dose intensity, avoid experimental agents")
        elif age > 75:
            contraindications.append("Age >75: consider dose reductions")
        
        # ECOG scarso
        if ecog >= 3:
            contraindications.append("ECOG ≥3: avoid aggressive multimodal therapy")
        elif ecog == 2:
            contraindications.append("ECOG 2: careful patient selection for intensive regimens")
        
        # Funzione renale (se disponibile)
        albumin = float(blood.get('albumin', 4.0))
        if albumin < 3.0:
            contraindications.append("Low albumin (<3 g/dL): malnutrition, avoid ketogenic diet")
        
        # NLR elevato (immunosoppressione)
        neutrophils = float(blood.get('neutrophils', 5000))
        lymphocytes = float(blood.get('lymphocytes', 1500))
        nlr = neutrophils / lymphocytes if lymphocytes > 0 else 0
        if nlr > 10:
            contraindications.append("NLR >10: severe inflammation, immunotherapy may be less effective")
        
        # STK11 + KEAP1 double loss
        stk11 = str(genetics.get('stk11_status', '')).lower()
        keap1 = str(genetics.get('keap1_status', '')).lower()
        if stk11 in ['mutated', 'mut', 'loss'] and keap1 in ['mutated', 'mut', 'loss']:
            contraindications.append("STK11+KEAP1 double loss: checkpoint inhibitors likely ineffective")
        
        return contraindications
    
    def _analyze_genetic_targets(self, genetics: Dict) -> Dict[str, List[str]]:
        """Analizza il profilo genetico e identifica target terapeutici"""
        targets = {
            "actionable": [],
            "resistance": [],
            "monitoring": []
        }
        
        # EGFR
        egfr = str(genetics.get('egfr_status', '')).lower()
        if egfr not in ['wt', 'none', '', 'wild-type', 'wildtype']:
            if 't790m' in egfr:
                targets["resistance"].append("EGFR_T790M")
            if 'c797s' in egfr:
                targets["resistance"].append("EGFR_C797S")
            if 'exon 19' in egfr or 'exon19' in egfr or 'del19' in egfr:
                targets["actionable"].append("EGFR_Exon19del")
            if 'l858r' in egfr:
                targets["actionable"].append("EGFR_L858R")
            if 'exon 20' in egfr or 'exon20' in egfr:
                targets["actionable"].append("EGFR_Exon20ins")
        
        # MET
        met = str(genetics.get('met_status', '')).lower()
        met_cn = float(genetics.get('met_cn', 0) or 0)
        if 'amplification' in met or 'amp' in met or met_cn >= 5:
            targets["actionable"].append("MET_Amplification")
        if 'exon14' in met or 'exon 14' in met:
            targets["actionable"].append("MET_Exon14skip")
        
        # KRAS
        kras = str(genetics.get('kras_mutation', '')).lower()
        if kras not in ['wt', 'none', '', 'wild-type']:
            if 'g12c' in kras:
                targets["actionable"].append("KRAS_G12C")
            else:
                targets["monitoring"].append("KRAS_other")
        
        # ALK
        alk = str(genetics.get('alk_status', '')).lower()
        if alk not in ['wt', 'none', '', 'negative']:
            targets["actionable"].append("ALK_Fusion")
        
        # BRAF
        braf = str(genetics.get('braf_status', '')).lower()
        if 'v600' in braf:
            targets["actionable"].append("BRAF_V600E")
        
        # ROS1
        ros1 = str(genetics.get('ros1_status', '')).lower()
        if ros1 not in ['wt', 'none', '', 'negative']:
            targets["actionable"].append("ROS1_Fusion")
        
        # HER2
        her2 = str(genetics.get('her2_status', '')).lower()
        if her2 not in ['wt', 'none', '', 'negative']:
            if 'amplification' in her2:
                targets["actionable"].append("HER2_Amplification")
            else:
                targets["actionable"].append("HER2_Mutation")
        
        return targets
    
    def _generate_phase1_induction(self, genetics: Dict, ldh: float, current_therapy: str,
                                    veto_active: bool, veto_reason: str,
                                    contraindications: List[str], age: int, ecog: int) -> ProtocolPhase:
        """Genera Fase 1: INDUCTION (Shock metabolico)"""
        
        therapies = []
        excluded = []
        warnings = []
        
        # === VETO THERAPY (se applicabile) ===
        if veto_active and veto_reason:
            # Estrai la terapia suggerita dal VETO
            if "MET" in veto_reason.upper():
                therapies.append(TherapySuggestion(
                    drug_name="Capmatinib",
                    dose="400mg BID",
                    rationale=f"VETO: {veto_reason}",
                    evidence=TherapyEvidence.FDA_APPROVED,
                    monitoring="LFTs, creatinine q2 weeks initially"
                ))
            elif "T790M" in veto_reason.upper():
                therapies.append(TherapySuggestion(
                    drug_name="Osimertinib",
                    dose="80mg QD",
                    rationale=f"VETO: {veto_reason}",
                    evidence=TherapyEvidence.FDA_APPROVED,
                    monitoring="ECG, ophthalmologic exam"
                ))
            elif "G12C" in veto_reason.upper():
                therapies.append(TherapySuggestion(
                    drug_name="Sotorasib",
                    dose="960mg QD",
                    rationale=f"VETO: {veto_reason}",
                    evidence=TherapyEvidence.FDA_APPROVED,
                    monitoring="LFTs q2 weeks for first 3 months"
                ))
        
        # === METABOLIC THERAPY ===
        # Metformin (se non controindicato)
        metformin_ok = not any("eGFR" in c or "acidosis" in c.lower() for c in contraindications)
        if metformin_ok:
            met_therapy = METABOLIC_THERAPIES["metformin"]
            # Adatta dose per età
            if age > 75:
                met_therapy = TherapySuggestion(
                    drug_name="Metformin",
                    dose="500mg QD, max 500mg BID (age-adjusted)",
                    rationale=met_therapy.rationale,
                    evidence=met_therapy.evidence,
                    monitoring=met_therapy.monitoring
                )
            therapies.append(met_therapy)
        else:
            excluded.append(("Metformin", "Renal/metabolic contraindication"))
        
        # Ketogenic Diet (se non controindicato)
        keto_ok = not any("cachexia" in c.lower() or "bmi" in c.lower() or "albumin" in c.lower() 
                         for c in contraindications)
        if keto_ok and ecog <= 2:
            therapies.append(METABOLIC_THERAPIES["ketogenic_diet"])
        else:
            excluded.append(("Ketogenic Diet", "Nutritional status or ECOG contraindication"))
        
        # === WARNINGS ===
        if ecog >= 2:
            warnings.append("ECOG ≥2: Close monitoring for treatment tolerance required")
        if age > 75:
            warnings.append("Age >75: Consider reduced dose intensity")
        if ldh > 500:
            warnings.append(f"LDH {ldh:.0f} U/L: High tumor burden, watch for tumor lysis syndrome")
        
        return ProtocolPhase(
            name="PHASE 1: INDUCTION (Metabolic Shock)",
            duration="4-6 weeks",
            objectives=[
                "Correct therapy mismatch (if VETO active)",
                "Initiate metabolic stress on tumor cells",
                "Exploit Warburg effect dependency",
                "Stabilize disease progression"
            ],
            therapies=therapies,
            excluded_therapies=excluded,
            warnings=warnings
        )
    
    def _generate_phase2_consolidation(self, genetics: Dict, genetic_targets: Dict,
                                        current_therapy: str, veto_active: bool, veto_reason: str,
                                        contraindications: List[str], ecog: int) -> ProtocolPhase:
        """Genera Fase 2: CONSOLIDATION (Cage)"""
        
        therapies = []
        excluded = []
        warnings = []
        
        # === TARGETED THERAPY basata su target genetici ===
        for target in genetic_targets.get("actionable", []):
            if "MET" in target:
                if not any(t.drug_name == "Capmatinib" for t in therapies):
                    therapies.append(TherapySuggestion(
                        drug_name="Capmatinib",
                        dose="400mg BID",
                        rationale=f"Target: {target}",
                        evidence=TherapyEvidence.FDA_APPROVED
                    ))
            
            elif "EGFR_C797S" in target:
                therapies.append(TherapySuggestion(
                    drug_name="Amivantamab",
                    dose="1050mg (≤80kg) or 1400mg (>80kg) IV weekly x4, then q2w",
                    rationale="EGFR/MET bispecific for C797S resistance",
                    evidence=TherapyEvidence.FDA_APPROVED
                ))
            
            elif "KRAS_G12C" in target:
                therapies.append(TherapySuggestion(
                    drug_name="Adagrasib",
                    dose="600mg BID",
                    rationale="KRAS G12C inhibitor (alternative to Sotorasib)",
                    evidence=TherapyEvidence.FDA_APPROVED
                ))
        
        # === IMMUNOTHERAPY (se appropriata) ===
        # Check controindicazioni immunoterapia
        immuno_contraindicated = any("STK11+KEAP1" in c or "checkpoint" in c.lower() 
                                     for c in contraindications)
        
        # Check se già su immunoterapia
        already_on_immuno = any(x in current_therapy.lower() 
                                for x in ["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab"])
        
        if immuno_contraindicated:
            excluded.append(("PD-1/PD-L1 Inhibitors", "STK11/KEAP1 mutations - likely ineffective"))
            # Suggerisci alternativa
            if not already_on_immuno:
                warnings.append("Checkpoint inhibitors likely ineffective due to tumor biology")
        elif already_on_immuno:
            # Suggerisci switch a CTLA-4 se fallimento
            therapies.append(TherapySuggestion(
                drug_name="Ipilimumab + Nivolumab",
                dose="Ipi 1mg/kg + Nivo 3mg/kg q3w x4, then Nivo maintenance",
                rationale="Consider dual checkpoint blockade if single-agent PD-1 failing",
                evidence=TherapyEvidence.FDA_APPROVED
            ))
        else:
            # Può iniziare immunoterapia
            therapies.append(TherapySuggestion(
                drug_name="Pembrolizumab",
                dose="200mg IV q3w",
                rationale="Checkpoint inhibitor - evaluate PD-L1 status",
                evidence=TherapyEvidence.FDA_APPROVED,
                monitoring="Immune-related AE monitoring"
            ))
        
        return ProtocolPhase(
            name="PHASE 2: CONSOLIDATION (Immunologic Cage)",
            duration="8-12 weeks",
            objectives=[
                "Consolidate metabolic response",
                "Activate immune surveillance",
                "Prevent clonal escape",
                "Target actionable mutations"
            ],
            therapies=therapies,
            excluded_therapies=excluded,
            warnings=warnings
        )
    
    def _generate_phase3_maintenance(self, genetics: Dict, genetic_targets: Dict,
                                      contraindications: List[str], ecog: int) -> ProtocolPhase:
        """Genera Fase 3: MAINTENANCE (Chronic control)"""
        
        therapies = []
        excluded = []
        warnings = []
        
        # === ADAPTIVE THERAPY ===
        therapies.append(TherapySuggestion(
            drug_name="Adaptive Therapy Protocol",
            dose="Dose modulation based on tumor markers",
            rationale="Exploit clone competition: maintain sensitive clones to suppress resistant ones",
            evidence=TherapyEvidence.PHASE_2,
            monitoring="ctDNA q8 weeks, imaging q12 weeks"
        ))
        
        # === LOW-DOSE METABOLIC MAINTENANCE ===
        metformin_ok = not any("eGFR" in c or "acidosis" in c.lower() for c in contraindications)
        if metformin_ok:
            therapies.append(TherapySuggestion(
                drug_name="Metformin (maintenance)",
                dose="500mg BID (chronic)",
                rationale="Chronic metabolic pressure, possible cancer prevention benefit",
                evidence=TherapyEvidence.PHASE_2,
                monitoring="B12 levels annually, renal function q3 months"
            ))
        
        # === CONTINUED TARGETED THERAPY ===
        if genetic_targets.get("actionable"):
            therapies.append(TherapySuggestion(
                drug_name="Continue targeted agent",
                dose="As tolerated",
                rationale="Maintain selective pressure on driver mutation",
                evidence=TherapyEvidence.NCCN_GUIDELINE,
                monitoring="Resistance monitoring via ctDNA"
            ))
        
        # === ctDNA MONITORING ===
        therapies.append(TherapySuggestion(
            drug_name="ctDNA Surveillance",
            dose="q8-12 weeks",
            rationale="Early detection of resistance mutations (T790M, C797S, MET amp)",
            evidence=TherapyEvidence.NCCN_GUIDELINE,
            monitoring="If new mutations detected → rebiopsy and therapy adjustment"
        ))
        
        if ecog == 0:
            warnings.append("Excellent PS: consider treatment holidays if sustained response")
        
        return ProtocolPhase(
            name="PHASE 3: MAINTENANCE (Chronic Containment)",
            duration="Indefinite (until progression)",
            objectives=[
                "Maintain disease control",
                "Monitor for resistance emergence",
                "Preserve quality of life",
                "Enable adaptive therapy adjustments"
            ],
            therapies=therapies,
            excluded_therapies=excluded,
            warnings=warnings
        )
    
    def _get_experimental_options(self, genetics: Dict, histology: str,
                                   contraindications: List[str]) -> List[TherapySuggestion]:
        """Opzioni sperimentali/trials clinici"""
        experimental = []
        
        # DCA se LDH molto alto
        if not any("neuropathy" in c.lower() for c in contraindications):
            experimental.append(METABOLIC_THERAPIES["dichloroacetate"])
        
        # 2-DG per casi selezionati
        if not any("hypoglycemia" in c.lower() or "cardiac" in c.lower() for c in contraindications):
            experimental.append(METABOLIC_THERAPIES["2dg"])
        
        # Bispecific antibodies per EGFR resistance
        egfr = str(genetics.get('egfr_status', '')).lower()
        if 'c797s' in egfr or 't790m' in egfr:
            experimental.append(TherapySuggestion(
                drug_name="Patritumab Deruxtecan",
                dose="5.6 mg/kg IV q3w",
                rationale="HER3-targeted ADC for EGFR TKI-resistant NSCLC",
                evidence=TherapyEvidence.PHASE_2,
                experimental=True
            ))
        
        # TROP2 ADC
        if histology == "adenocarcinoma":
            experimental.append(TherapySuggestion(
                drug_name="Datopotamab Deruxtecan",
                dose="6 mg/kg IV q3w",
                rationale="TROP2-targeted ADC - emerging data in NSCLC",
                evidence=TherapyEvidence.PHASE_3,
                experimental=True
            ))
        
        return experimental
    
    def _generate_personalization_summary(self, genetics: Dict, genetic_targets: Dict,
                                           ldh: float, age: int, ecog: int,
                                           veto_active: bool, veto_reason: str) -> str:
        """Genera il summary di personalizzazione"""
        
        lines = []
        
        # Target principali
        actionable = genetic_targets.get("actionable", [])
        resistance = genetic_targets.get("resistance", [])
        
        if actionable:
            lines.append(f"Actionable targets identified: {', '.join(actionable)}")
        
        if resistance:
            lines.append(f"Resistance mechanisms detected: {', '.join(resistance)}")
        
        if veto_active:
            lines.append(f"Therapy correction required: {veto_reason}")
        
        # Metabolic
        if ldh > 500:
            lines.append(f"High metabolic activity (LDH {ldh:.0f} U/L) - excellent Elephant Protocol candidate")
        elif ldh > 350:
            lines.append(f"Elevated metabolic activity (LDH {ldh:.0f} U/L) - good Elephant Protocol candidate")
        
        # Patient factors
        if age > 75:
            lines.append(f"Age {age}: dose-adjusted protocol recommended")
        
        if ecog >= 2:
            lines.append(f"ECOG {ecog}: reduced intensity protocol")
        
        return "\n".join(lines) if lines else "Standard protocol parameters"
    
    def _generate_clinical_rationale(self, activation_reasons: List[str],
                                      metabolic_sensitivity: float,
                                      genetic_targets: Dict, ldh: float) -> str:
        """Genera il rationale clinico completo"""
        
        rationale = []
        
        rationale.append("CLINICAL RATIONALE FOR ELEPHANT PROTOCOL:")
        rationale.append("")
        
        # Activation reasons
        rationale.append("Activation Criteria Met:")
        for reason in activation_reasons:
            rationale.append(f"  • {reason}")
        rationale.append("")
        
        # Metabolic sensitivity
        if metabolic_sensitivity >= 70:
            sens_label = "HIGH (Excellent candidate)"
        elif metabolic_sensitivity >= 40:
            sens_label = "MODERATE (Good candidate)"
        else:
            sens_label = "LOW (Consider alternatives)"
        
        rationale.append(f"Metabolic Sensitivity Score: {metabolic_sensitivity:.0f}% - {sens_label}")
        rationale.append("")
        
        # Expected benefit
        rationale.append("Expected Benefit from Metabolic Intervention:")
        
        # Calcola proiezione quantitativa
        base_regression = metabolic_sensitivity * 0.42  # Max 42% regression
        
        rationale.append(f"  Phase 1 (Induction): -{base_regression*0.5:.0f}% to -{base_regression:.0f}% tumor reduction")
        rationale.append(f"  Phase 2 (Consolidation): Additional -{base_regression*0.2:.0f}% to -{base_regression*0.5:.0f}%")
        rationale.append(f"  Phase 3 (Maintenance): Stabilization, -{base_regression*0.1:.0f}% to 0%")
        rationale.append(f"  Cumulative Projected: -{base_regression*0.6:.0f}% to -{base_regression*1.5:.0f}%")
        
        return "\n".join(rationale)
    
    def _generate_disclaimer(self, age: int, ecog: int, experimental: List[TherapySuggestion]) -> str:
        """Genera il disclaimer finale"""
        
        disclaimer_parts = []
        
        disclaimer_parts.append("=" * 60)
        disclaimer_parts.append("IMPORTANT DISCLAIMER")
        disclaimer_parts.append("=" * 60)
        disclaimer_parts.append("")
        disclaimer_parts.append("This protocol is generated by SENTINEL AI as a DECISION SUPPORT TOOL.")
        disclaimer_parts.append("It is NOT a substitute for clinical judgment.")
        disclaimer_parts.append("")
        disclaimer_parts.append("RECOMMENDATIONS:")
        disclaimer_parts.append("• All therapeutic decisions must be validated by the treating oncologist")
        disclaimer_parts.append("• Multidisciplinary Tumor Board review is REQUIRED before implementation")
        disclaimer_parts.append("• Patient informed consent must be obtained for experimental approaches")
        disclaimer_parts.append("• Individual patient factors may necessitate protocol modifications")
        disclaimer_parts.append("")
        
        if experimental:
            disclaimer_parts.append("⚠️ EXPERIMENTAL THERAPIES INCLUDED:")
            disclaimer_parts.append("This protocol includes experimental options that require:")
            disclaimer_parts.append("• IRB/Ethics Committee approval")
            disclaimer_parts.append("• Clinical trial enrollment where available")
            disclaimer_parts.append("• Enhanced safety monitoring")
            disclaimer_parts.append("")
        
        if age > 75 or ecog >= 2:
            disclaimer_parts.append("⚠️ VULNERABLE PATIENT POPULATION:")
            disclaimer_parts.append("This patient may have reduced treatment tolerance.")
            disclaimer_parts.append("Consider geriatric oncology consultation.")
            disclaimer_parts.append("")
        
        disclaimer_parts.append("The final treatment decision rests with the responsible physician")
        disclaimer_parts.append("in consultation with the patient and care team.")
        disclaimer_parts.append("=" * 60)
        
        return "\n".join(disclaimer_parts)


# ============================================================================
# HELPER FUNCTION FOR REPORT GENERATION
# ============================================================================

def generate_elephant_protocol(patient_data: Dict, ai_result: Dict) -> ElephantProtocolResult:
    """
    Entry point per generare il Protocollo Elephant.
    
    Usage:
        from elephant_protocol import generate_elephant_protocol
        
        result = generate_elephant_protocol(patient_data, ai_result)
        if result.activated:
            for phase in result.phases:
                print(phase.name)
                for therapy in phase.therapies:
                    print(f"  - {therapy.drug_name}: {therapy.dose}")
    """
    engine = ElephantProtocolEngine()
    return engine.generate_protocol(patient_data, ai_result)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test con paziente CRASH
    test_patient = {
        'baseline': {
            'patient_id': 'CRASH',
            'age': 45,
            'ecog_ps': 3,
            'histology': 'Adenocarcinoma',
            'current_therapy': 'Osimertinib (EGFR 3rd-gen)',
            'genetics': {
                'tp53_status': 'wt',
                'egfr_status': 'Exon 19 deletion',
                'met_status': 'Amplification',
                'met_cn': 12.5,
                'kras_mutation': 'wt'
            },
            'blood_markers': {
                'ldh': 850,
                'neutrophils': 9000,
                'lymphocytes': 600,
                'albumin': 2.9
            }
        }
    }
    
    test_ai_result = {
        'veto_active': True,
        'veto_reason': 'MET Amplification (CN=12.5) requires MET inhibitor',
        'display_risk': 100
    }
    
    result = generate_elephant_protocol(test_patient, test_ai_result)
    
    print("\n" + "🐘" * 30)
    print("ELEPHANT PROTOCOL TEST")
    print("🐘" * 30 + "\n")
    
    print(f"Activated: {result.activated}")
    print(f"Metabolic Sensitivity: {result.metabolic_sensitivity}%")
    print(f"\nActivation Reasons:")
    for r in result.activation_reasons:
        print(f"  • {r}")
    
    print(f"\nContraindications:")
    for c in result.contraindications_detected:
        print(f"  ⚠️ {c}")
    
    for phase in result.phases:
        print(f"\n{'='*60}")
        print(f"{phase.name}")
        print(f"Duration: {phase.duration}")
        print(f"{'='*60}")
        
        print("\nObjectives:")
        for obj in phase.objectives:
            print(f"  • {obj}")
        
        print("\nTherapies:")
        for t in phase.therapies:
            exp_tag = " [EXPERIMENTAL]" if t.experimental else ""
            print(f"  💊 {t.drug_name}: {t.dose}{exp_tag}")
            print(f"     Rationale: {t.rationale}")
            print(f"     Evidence: {t.evidence.value}")
        
        if phase.excluded_therapies:
            print("\nExcluded:")
            for drug, reason in phase.excluded_therapies:
                print(f"  ❌ {drug}: {reason}")
        
        if phase.warnings:
            print("\nWarnings:")
            for w in phase.warnings:
                print(f"  ⚠️ {w}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENTAL OPTIONS")
    print(f"{'='*60}")
    for exp in result.experimental_options:
        print(f"  🔬 {exp.drug_name}: {exp.dose}")
        print(f"     {exp.rationale}")
    
    print(f"\n{result.personalization_summary}")
    print(f"\n{result.clinical_rationale}")
    print(f"\n{result.disclaimer}")
