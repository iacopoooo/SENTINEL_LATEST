"""
SENTINEL v18.1 - Synthetic Lethality Finder
============================================
Identifica vulnerabilità "sinteticamente letali" basate sul profilo genetico.

Principio: Due geni sono sinteticamente letali se la perdita di uno solo
è tollerata, ma la perdita di ENTRAMBI causa morte cellulare.

Applicazione clinica: Se il tumore ha già perso il gene A (es. BRCA1),
possiamo inibire farmacologicamente il gene B (es. PARP) per uccidere
selettivamente le cellule tumorali.

Database basato su:
- Letteratura peer-reviewed
- Clinical trials results
- Preclinical data (dove clinici non disponibili)

References:
- Lord CJ & Ashworth A. "PARP inhibitors: Synthetic lethality" Science 2017
- Setton J et al. "Synthetic Lethality in Cancer Therapeutics" Nat Rev Cancer 2021
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class EvidenceLevel(Enum):
    """Livello di evidenza clinica"""
    FDA_APPROVED = "FDA Approved"
    PHASE_3 = "Phase 3"
    PHASE_2 = "Phase 2"
    PHASE_1 = "Phase 1"
    PRECLINICAL = "Preclinical"
    THEORETICAL = "Theoretical"


class TumorType(Enum):
    """Tipi tumorali"""
    NSCLC = "NSCLC"
    SCLC = "SCLC"
    BREAST = "Breast"
    OVARIAN = "Ovarian"
    PROSTATE = "Prostate"
    PANCREATIC = "Pancreatic"
    COLORECTAL = "Colorectal"
    PAN_CANCER = "Pan-Cancer"


@dataclass
class SyntheticLethalityPair:
    """Coppia di letalità sintetica"""
    # Gene alterato nel tumore
    tumor_alteration: str  # Es: "BRCA1 mutation"
    tumor_gene: str  # Es: "BRCA1"
    alteration_type: str  # Es: "loss-of-function", "amplification"

    # Target farmacologico (partner sintetico)
    synthetic_partner: str  # Es: "PARP1/2"
    inhibitor_drug: str  # Es: "Olaparib"
    drug_dose: str  # Es: "300mg BID"

    # Meccanismo
    mechanism: str  # Spiegazione biologica

    # Evidenza
    evidence_level: EvidenceLevel
    tumor_types: List[TumorType]
    key_trial: Optional[str]  # Trial di riferimento
    response_rate: Optional[float]  # ORR se noto

    # Confidence score (0-100)
    confidence: int

    # Note cliniche
    biomarker_required: Optional[str]  # Es: "HRD score ≥42"
    contraindications: List[str]
    monitoring: List[str]


@dataclass
class SyntheticLethalityResult:
    """Risultato completo dell'analisi di letalità sintetica"""
    patient_id: str

    # Alterazioni rilevate nel paziente
    detected_alterations: List[str]

    # Opportunità trovate (ordinate per confidence)
    opportunities: List[SyntheticLethalityPair]

    # Top recommendation
    top_recommendation: Optional[SyntheticLethalityPair]

    # Summary
    total_opportunities: int
    high_confidence_count: int  # Confidence ≥70
    fda_approved_count: int

    # Clinical action
    immediate_actionable: bool  # Almeno 1 FDA approved
    clinical_trial_recommended: bool

    # Metadata
    generated_at: str


class SyntheticLethalityFinder:
    """
    Motore per identificazione letalità sintetiche.

    Database curato manualmente da letteratura scientifica.
    """

    # =========================================================================
    # DATABASE LETALITÀ SINTETICHE
    # =========================================================================

    SYNTHETIC_LETHALITY_DATABASE = [
        # === BRCA1/2 - PARP (Gold Standard) ===
        {
            "tumor_alteration": "BRCA1 mutation",
            "tumor_gene": "BRCA1",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "PARP1/2",
            "inhibitor_drug": "Olaparib",
            "drug_dose": "300mg BID",
            "mechanism": "BRCA1 loss impairs homologous recombination (HR). PARP inhibition blocks base excision repair (BER). "
                         "Double DNA repair deficiency causes replication fork collapse and cell death.",
            "evidence_level": EvidenceLevel.FDA_APPROVED,
            "tumor_types": [TumorType.OVARIAN, TumorType.BREAST, TumorType.PANCREATIC, TumorType.PROSTATE],
            "key_trial": "OlympiAD, SOLO-1",
            "response_rate": 0.60,
            "confidence": 95,
            "biomarker_required": "BRCA1/2 germline or somatic mutation",
            "contraindications": ["MDS history", "Severe renal impairment"],
            "monitoring": ["CBC every 4 weeks", "Watch for MDS/AML"]
        },
        {
            "tumor_alteration": "BRCA2 mutation",
            "tumor_gene": "BRCA2",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "PARP1/2",
            "inhibitor_drug": "Olaparib",
            "drug_dose": "300mg BID",
            "mechanism": "BRCA2 loss impairs homologous recombination. PARP trapping causes synthetic lethality.",
            "evidence_level": EvidenceLevel.FDA_APPROVED,
            "tumor_types": [TumorType.OVARIAN, TumorType.BREAST, TumorType.PANCREATIC, TumorType.PROSTATE],
            "key_trial": "PROfound, POLO",
            "response_rate": 0.55,
            "confidence": 95,
            "biomarker_required": "BRCA1/2 mutation",
            "contraindications": ["MDS history"],
            "monitoring": ["CBC monthly", "LFTs"]
        },

        # === HRD (BRCAness) - PARP ===
        {
            "tumor_alteration": "HRD positive (non-BRCA)",
            "tumor_gene": "HRD",
            "alteration_type": "genomic_instability",
            "synthetic_partner": "PARP1/2",
            "inhibitor_drug": "Niraparib",
            "drug_dose": "200-300mg QD (weight-based)",
            "mechanism": "HRD tumors have defective HR repair regardless of BRCA status. "
                         "PARP inhibition exploits this 'BRCAness' phenotype.",
            "evidence_level": EvidenceLevel.FDA_APPROVED,
            "tumor_types": [TumorType.OVARIAN],
            "key_trial": "PRIMA/ENGOT-OV26",
            "response_rate": 0.45,
            "confidence": 85,
            "biomarker_required": "HRD score ≥42 (Myriad myChoice)",
            "contraindications": ["Platelets <100k", "Uncontrolled hypertension"],
            "monitoring": ["CBC weekly x4, then monthly", "BP monitoring"]
        },

        # === TP53 - WEE1 ===
        {
            "tumor_alteration": "TP53 mutation",
            "tumor_gene": "TP53",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "WEE1",
            "inhibitor_drug": "Adavosertib (AZD1775)",
            "drug_dose": "300mg QD days 1-5, q3w",
            "mechanism": "TP53 loss eliminates G1 checkpoint. Cells rely on G2/M checkpoint (WEE1). "
                         "WEE1 inhibition forces mitotic catastrophe in TP53-mutant cells.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.NSCLC, TumorType.OVARIAN, TumorType.PAN_CANCER],
            "key_trial": "NCT02482311",
            "response_rate": 0.30,
            "confidence": 65,
            "biomarker_required": "TP53 mutation",
            "contraindications": ["QTc prolongation", "Strong CYP3A4 inhibitors"],
            "monitoring": ["ECG at baseline and cycle 1", "CBC"]
        },

        # === TP53 - ATR ===
        {
            "tumor_alteration": "TP53 mutation",
            "tumor_gene": "TP53",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "ATR",
            "inhibitor_drug": "Ceralasertib (AZD6738)",
            "drug_dose": "240mg BID days 1-14, q4w",
            "mechanism": "ATR mediates replication stress response. TP53-mutant cells have high replication stress. "
                         "ATR inhibition causes replication catastrophe selectively in TP53-mutant cells.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.NSCLC, TumorType.PAN_CANCER],
            "key_trial": "HUDSON (NCT03334617)",
            "response_rate": 0.25,
            "confidence": 60,
            "biomarker_required": "TP53 mutation preferred",
            "contraindications": ["Severe anemia"],
            "monitoring": ["CBC twice weekly cycle 1"]
        },

        # === ATM loss - PARP ===
        {
            "tumor_alteration": "ATM loss/mutation",
            "tumor_gene": "ATM",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "PARP1/2",
            "inhibitor_drug": "Olaparib",
            "drug_dose": "300mg BID",
            "mechanism": "ATM is critical for DNA double-strand break signaling. ATM-deficient cells "
                         "rely heavily on PARP-mediated repair. Dual deficiency is lethal.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.PROSTATE, TumorType.NSCLC, TumorType.PAN_CANCER],
            "key_trial": "PROfound (subset)",
            "response_rate": 0.35,
            "confidence": 70,
            "biomarker_required": "ATM mutation/loss",
            "contraindications": ["MDS history"],
            "monitoring": ["CBC monthly"]
        },

        # === STK11 (LKB1) - mTOR (metabolic) ===
        {
            "tumor_alteration": "STK11 (LKB1) loss",
            "tumor_gene": "STK11",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "mTOR + Glutaminase",
            "inhibitor_drug": "Everolimus + CB-839",
            "drug_dose": "Everolimus 10mg QD + CB-839 (trial)",
            "mechanism": "STK11 loss deregulates AMPK-mTOR axis and increases glutamine dependency. "
                         "Dual mTOR/glutaminase inhibition exploits this metabolic vulnerability.",
            "evidence_level": EvidenceLevel.PHASE_1,
            "tumor_types": [TumorType.NSCLC],
            "key_trial": "NCT03163667",
            "response_rate": 0.20,
            "confidence": 50,
            "biomarker_required": "STK11 mutation",
            "contraindications": ["Interstitial lung disease", "Uncontrolled diabetes"],
            "monitoring": ["Glucose monitoring", "Lipid panel", "Pneumonitis watch"]
        },

        # === KEAP1 - NRF2 pathway ===
        {
            "tumor_alteration": "KEAP1 loss",
            "tumor_gene": "KEAP1",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "Glutathione synthesis",
            "inhibitor_drug": "Buthionine sulfoximine (BSO) + Auranofin",
            "drug_dose": "Experimental dosing",
            "mechanism": "KEAP1 loss activates NRF2, increasing antioxidant capacity. "
                         "These cells become dependent on glutathione. Depleting glutathione is selectively toxic.",
            "evidence_level": EvidenceLevel.PRECLINICAL,
            "tumor_types": [TumorType.NSCLC],
            "key_trial": "Preclinical (Romero et al. 2017)",
            "response_rate": None,
            "confidence": 40,
            "biomarker_required": "KEAP1 mutation",
            "contraindications": ["Unknown - experimental"],
            "monitoring": ["Unknown - experimental"]
        },

        # === RB1 loss - Aurora Kinase ===
        {
            "tumor_alteration": "RB1 loss",
            "tumor_gene": "RB1",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "Aurora Kinase A/B",
            "inhibitor_drug": "Alisertib",
            "drug_dose": "50mg BID days 1-7, q3w",
            "mechanism": "RB1 loss causes E2F overactivity and mitotic stress. "
                         "Aurora kinase inhibition exacerbates mitotic defects selectively in RB1-null cells.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.SCLC, TumorType.NSCLC],
            "key_trial": "NCT02719691",
            "response_rate": 0.18,
            "confidence": 55,
            "biomarker_required": "RB1 loss",
            "contraindications": ["Neutropenia"],
            "monitoring": ["CBC weekly", "Mucositis monitoring"]
        },

        # === RB1 + TP53 (SCLC-like) - CHK1 ===
        {
            "tumor_alteration": "TP53 + RB1 double loss",
            "tumor_gene": "TP53_RB1",
            "alteration_type": "double_loss",
            "synthetic_partner": "CHK1",
            "inhibitor_drug": "Prexasertib (LY2606368)",
            "drug_dose": "105mg/m2 IV q2w",
            "mechanism": "TP53+RB1 loss creates extreme replication stress and checkpoint dependency. "
                         "CHK1 inhibition removes last checkpoint, causing mitotic catastrophe.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.SCLC, TumorType.NSCLC],
            "key_trial": "NCT02735980",
            "response_rate": 0.25,
            "confidence": 65,
            "biomarker_required": "TP53 + RB1 co-mutation",
            "contraindications": ["Severe myelosuppression risk"],
            "monitoring": ["CBC twice weekly", "Hospitalization may be required"]
        },

        # === ARID1A - EZH2 ===
        {
            "tumor_alteration": "ARID1A loss",
            "tumor_gene": "ARID1A",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "EZH2",
            "inhibitor_drug": "Tazemetostat",
            "drug_dose": "800mg BID",
            "mechanism": "ARID1A is part of SWI/SNF chromatin remodeling complex. Loss creates dependency "
                         "on EZH2-mediated gene silencing. EZH2 inhibition reactivates tumor suppressors.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.OVARIAN, TumorType.PAN_CANCER],
            "key_trial": "NCT02601950",
            "response_rate": 0.15,
            "confidence": 55,
            "biomarker_required": "ARID1A mutation",
            "contraindications": ["Lymphopenia"],
            "monitoring": ["CBC", "Watch for secondary malignancies"]
        },

        # === MYC amplification - CDK/BET ===
        {
            "tumor_alteration": "MYC amplification",
            "tumor_gene": "MYC",
            "alteration_type": "amplification",
            "synthetic_partner": "BRD4/BET + CDK9",
            "inhibitor_drug": "BET inhibitor (OTX015, JQ1) + CDK9i",
            "drug_dose": "Clinical trial dosing",
            "mechanism": "MYC-amplified tumors depend on BRD4-mediated MYC transcription and CDK9 "
                         "for transcriptional elongation. Dual inhibition collapses MYC program.",
            "evidence_level": EvidenceLevel.PHASE_1,
            "tumor_types": [TumorType.NSCLC, TumorType.PAN_CANCER],
            "key_trial": "NCT02259114",
            "response_rate": 0.15,
            "confidence": 45,
            "biomarker_required": "MYC amplification",
            "contraindications": ["Thrombocytopenia", "GI toxicity common"],
            "monitoring": ["CBC", "GI symptoms"]
        },

        # === KRAS + SHP2 ===
        {
            "tumor_alteration": "KRAS mutation (non-G12C)",
            "tumor_gene": "KRAS",
            "alteration_type": "activating_mutation",
            "synthetic_partner": "SHP2",
            "inhibitor_drug": "RMC-4630 (SHP2 inhibitor)",
            "drug_dose": "Clinical trial dosing",
            "mechanism": "KRAS mutations require upstream RTK-SHP2 signaling for full activation. "
                         "SHP2 inhibition suppresses adaptive resistance and enhances KRAS inhibitor efficacy.",
            "evidence_level": EvidenceLevel.PHASE_1,
            "tumor_types": [TumorType.NSCLC, TumorType.PANCREATIC],
            "key_trial": "NCT03634982",
            "response_rate": 0.20,
            "confidence": 50,
            "biomarker_required": "KRAS mutation",
            "contraindications": ["Unknown - experimental"],
            "monitoring": ["Standard oncology monitoring"]
        },

        # === EGFR + MET (bypass resistance) ===
        {
            "tumor_alteration": "EGFR mutation + MET amplification",
            "tumor_gene": "EGFR_MET",
            "alteration_type": "bypass_resistance",
            "synthetic_partner": "MET (dual targeting)",
            "inhibitor_drug": "Osimertinib + Capmatinib",
            "drug_dose": "Osimertinib 80mg QD + Capmatinib 400mg BID",
            "mechanism": "MET amplification bypasses EGFR inhibition. Dual EGFR+MET blockade "
                         "eliminates both pathways simultaneously.",
            "evidence_level": EvidenceLevel.PHASE_2,
            "tumor_types": [TumorType.NSCLC],
            "key_trial": "GEOMETRY mono-1, SAVANNAH",
            "response_rate": 0.45,
            "confidence": 80,
            "biomarker_required": "EGFR mutation + MET amplification (CN≥5 or IHC 3+)",
            "contraindications": ["Severe hepatic impairment", "ILD history"],
            "monitoring": ["LFTs biweekly x8", "Pneumonitis watch"]
        },

        # === PTEN loss - AKT ===
        {
            "tumor_alteration": "PTEN loss",
            "tumor_gene": "PTEN",
            "alteration_type": "loss-of-function",
            "synthetic_partner": "AKT",
            "inhibitor_drug": "Capivasertib (AZD5363)",
            "drug_dose": "480mg BID days 1-4 weekly",
            "mechanism": "PTEN loss causes constitutive PI3K/AKT activation. These cells become "
                         "addicted to AKT signaling for survival.",
            "evidence_level": EvidenceLevel.PHASE_3,
            "tumor_types": [TumorType.BREAST, TumorType.PROSTATE],
            "key_trial": "CAPItello-291",
            "response_rate": 0.35,
            "confidence": 75,
            "biomarker_required": "PTEN loss or PIK3CA/AKT1 mutation",
            "contraindications": ["Uncontrolled diabetes", "Severe diarrhea history"],
            "monitoring": ["Glucose monitoring", "GI toxicity"]
        },

        # === PIK3CA - PI3K/AKT ===
        {
            "tumor_alteration": "PIK3CA mutation",
            "tumor_gene": "PIK3CA",
            "alteration_type": "activating_mutation",
            "synthetic_partner": "PI3K alpha",
            "inhibitor_drug": "Alpelisib",
            "drug_dose": "300mg QD",
            "mechanism": "PIK3CA mutation activates PI3K pathway. Alpelisib selectively inhibits "
                         "PI3K-alpha, exploiting pathway addiction.",
            "evidence_level": EvidenceLevel.FDA_APPROVED,
            "tumor_types": [TumorType.BREAST],
            "key_trial": "SOLAR-1",
            "response_rate": 0.35,
            "confidence": 85,
            "biomarker_required": "PIK3CA mutation",
            "contraindications": ["Diabetes", "History of severe rash"],
            "monitoring": ["Glucose monitoring (hyperglycemia common)", "Rash monitoring"]
        },
    ]

    def __init__(self):
        self.generated_at = datetime.now().isoformat()
        self.database = self._build_database()

    def _build_database(self) -> List[SyntheticLethalityPair]:
        """Costruisce database strutturato da dizionari"""
        pairs = []
        for entry in self.SYNTHETIC_LETHALITY_DATABASE:
            pairs.append(SyntheticLethalityPair(
                tumor_alteration=entry["tumor_alteration"],
                tumor_gene=entry["tumor_gene"],
                alteration_type=entry["alteration_type"],
                synthetic_partner=entry["synthetic_partner"],
                inhibitor_drug=entry["inhibitor_drug"],
                drug_dose=entry["drug_dose"],
                mechanism=entry["mechanism"],
                evidence_level=entry["evidence_level"],
                tumor_types=entry["tumor_types"],
                key_trial=entry.get("key_trial"),
                response_rate=entry.get("response_rate"),
                confidence=entry["confidence"],
                biomarker_required=entry.get("biomarker_required"),
                contraindications=entry.get("contraindications", []),
                monitoring=entry.get("monitoring", [])
            ))
        return pairs

    def extract_alterations(self, patient_data: Dict) -> List[Tuple[str, str]]:
        """
        Estrae alterazioni genetiche dal paziente.

        Returns:
            List of (gene, alteration_description)
        """
        base = patient_data.get('baseline', patient_data)
        genetics = base.get('genetics', {})
        alterations = []

        # TP53
        tp53 = str(genetics.get('tp53_status', '')).lower()
        if tp53 in ['mutated', 'mut', 'loss']:
            alterations.append(("TP53", "TP53 mutation"))

        # RB1
        rb1 = str(genetics.get('rb1_status', '')).lower()
        if rb1 in ['mutated', 'mut', 'loss']:
            alterations.append(("RB1", "RB1 loss"))

        # BRCA1
        brca1 = str(genetics.get('brca1_status', '')).lower()
        if brca1 in ['mutated', 'mut', 'pathogenic']:
            alterations.append(("BRCA1", "BRCA1 mutation"))

        # BRCA2
        brca2 = str(genetics.get('brca2_status', '')).lower()
        if brca2 in ['mutated', 'mut', 'pathogenic']:
            alterations.append(("BRCA2", "BRCA2 mutation"))

        # ATM
        atm = str(genetics.get('atm_status', '')).lower()
        if atm in ['mutated', 'mut', 'loss']:
            alterations.append(("ATM", "ATM loss/mutation"))

        # STK11
        stk11 = str(genetics.get('stk11_status', '')).lower()
        if stk11 in ['mutated', 'mut', 'loss']:
            alterations.append(("STK11", "STK11 (LKB1) loss"))

        # KEAP1
        keap1 = str(genetics.get('keap1_status', '')).lower()
        if keap1 in ['mutated', 'mut', 'loss']:
            alterations.append(("KEAP1", "KEAP1 loss"))

        # PTEN
        pten = str(genetics.get('pten_status', '')).lower()
        if pten in ['mutated', 'mut', 'loss']:
            alterations.append(("PTEN", "PTEN loss"))

        # PIK3CA
        pik3ca = str(genetics.get('pik3ca_status', '')).lower()
        if pik3ca in ['mutated', 'mut'] or 'pik3ca' in str(genetics).lower():
            alterations.append(("PIK3CA", "PIK3CA mutation"))

        # ARID1A
        arid1a = str(genetics.get('arid1a_status', '')).lower()
        if arid1a in ['mutated', 'mut', 'loss']:
            alterations.append(("ARID1A", "ARID1A loss"))

        # MYC
        myc = str(genetics.get('myc_status', '')).lower()
        myc_cn = float(genetics.get('myc_cn', 0) or 0)
        if 'amplification' in myc or myc_cn >= 5:
            alterations.append(("MYC", "MYC amplification"))

        # KRAS
        kras = str(genetics.get('kras_mutation', '')).lower()
        if kras and kras not in ['wt', 'none', '']:
            if 'g12c' in kras:
                alterations.append(("KRAS", "KRAS G12C"))
            else:
                alterations.append(("KRAS", "KRAS mutation (non-G12C)"))

        # EGFR + MET combo
        egfr = str(genetics.get('egfr_status', '')).lower()
        met = str(genetics.get('met_status', '')).lower()
        met_cn = float(genetics.get('met_cn', 0) or 0)

        if egfr not in ['wt', 'none', ''] and ('amplification' in met or met_cn >= 5):
            alterations.append(("EGFR_MET", "EGFR mutation + MET amplification"))

        # HRD (se disponibile)
        hrd = str(genetics.get('hrd_status', '')).lower()
        hrd_score = genetics.get('hrd_score')
        if hrd in ['positive', 'high'] or (hrd_score and float(hrd_score) >= 42):
            alterations.append(("HRD", "HRD positive (non-BRCA)"))

        # TP53 + RB1 double loss
        if ("TP53", "TP53 mutation") in alterations and ("RB1", "RB1 loss") in alterations:
            alterations.append(("TP53_RB1", "TP53 + RB1 double loss"))

        return alterations

    def find_opportunities(self, patient_data: Dict,
                           tumor_type: TumorType = TumorType.NSCLC) -> SyntheticLethalityResult:
        """
        Trova opportunità di letalità sintetica per il paziente.

        Args:
            patient_data: Dati paziente
            tumor_type: Tipo tumorale

        Returns:
            SyntheticLethalityResult con tutte le opportunità
        """
        base = patient_data.get('baseline', patient_data)
        patient_id = base.get('patient_id', 'Unknown')

        # Estrai alterazioni
        alterations = self.extract_alterations(patient_data)
        detected_genes = [gene for gene, _ in alterations]
        detected_descriptions = [desc for _, desc in alterations]

        # Cerca match nel database
        opportunities = []
        for pair in self.database:
            # Check if patient has the required alteration
            if pair.tumor_gene in detected_genes:
                # Check tumor type compatibility
                if tumor_type in pair.tumor_types or TumorType.PAN_CANCER in pair.tumor_types:
                    opportunities.append(pair)

        # Ordina per confidence (decrescente)
        opportunities.sort(key=lambda x: (-x.confidence, x.evidence_level.value))

        # Statistics
        high_confidence = [o for o in opportunities if o.confidence >= 70]
        fda_approved = [o for o in opportunities if o.evidence_level == EvidenceLevel.FDA_APPROVED]

        # Top recommendation
        top_rec = opportunities[0] if opportunities else None

        # Clinical action flags
        immediate_actionable = len(fda_approved) > 0
        trial_recommended = len(opportunities) > len(fda_approved) and len(opportunities) > 0

        return SyntheticLethalityResult(
            patient_id=patient_id,
            detected_alterations=detected_descriptions,
            opportunities=opportunities,
            top_recommendation=top_rec,
            total_opportunities=len(opportunities),
            high_confidence_count=len(high_confidence),
            fda_approved_count=len(fda_approved),
            immediate_actionable=immediate_actionable,
            clinical_trial_recommended=trial_recommended,
            generated_at=self.generated_at
        )


def find_synthetic_lethality(patient_data: Dict,
                             tumor_type: TumorType = TumorType.NSCLC) -> SyntheticLethalityResult:
    """Factory function per trovare letalità sintetiche."""
    finder = SyntheticLethalityFinder()
    return finder.find_opportunities(patient_data, tumor_type)