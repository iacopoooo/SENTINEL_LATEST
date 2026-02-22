"""
PHARMGKB DATABASE - SENTINEL FARMACOGENOMICA v2.0
==================================================
Database locale delle interazioni gene-farmaco.
Basato su PharmGKB (https://www.pharmgkb.org/) e linee guida CPIC.

v2.0 - Aggiunta completa interazioni per:
- EGFR Inhibitors (Osimertinib, Gefitinib, Erlotinib, etc.)
- KRAS Inhibitors (Sotorasib, Adagrasib)
- ALK Inhibitors (Alectinib, Crizotinib, Lorlatinib, etc.)
- MET Inhibitors (Capmatinib, Tepotinib)
- BRAF/MEK Inhibitors
- Immunotherapy
- Chemotherapy

Livelli di evidenza PharmGKB:
- 1A: Annotazione approvata da ente regolatorio (FDA, EMA)
- 1B: Evidenza forte da studi clinici
- 2A: Evidenza moderata, raccomandazione CPIC/DPWG
- 2B: Evidenza moderata, studi multipli
- 3: Evidenza limitata
- 4: Case report / studi preliminari

Per SENTINEL usiamo solo evidenze ≥2A per evitare alert fatigue.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceLevel(Enum):
    """Livelli di evidenza PharmGKB"""
    LEVEL_1A = "1A"  # FDA/EMA approved
    LEVEL_1B = "1B"  # Strong clinical evidence
    LEVEL_2A = "2A"  # CPIC/DPWG guideline
    LEVEL_2B = "2B"  # Moderate evidence
    LEVEL_3 = "3"    # Limited evidence
    LEVEL_4 = "4"    # Case reports
    UNKNOWN = "unknown"

    @property
    def is_actionable(self) -> bool:
        """True se il livello richiede azione clinica"""
        return self in [EvidenceLevel.LEVEL_1A, EvidenceLevel.LEVEL_1B, EvidenceLevel.LEVEL_2A]

    @property
    def priority(self) -> int:
        """Priorità numerica (più basso = più importante)"""
        priority_map = {
            EvidenceLevel.LEVEL_1A: 1,
            EvidenceLevel.LEVEL_1B: 2,
            EvidenceLevel.LEVEL_2A: 3,
            EvidenceLevel.LEVEL_2B: 4,
            EvidenceLevel.LEVEL_3: 5,
            EvidenceLevel.LEVEL_4: 6,
            EvidenceLevel.UNKNOWN: 7
        }
        return priority_map.get(self, 7)


class InteractionType(Enum):
    """Tipo di interazione gene-farmaco"""
    TOXICITY = "toxicity"           # Rischio tossicità aumentato
    EFFICACY = "efficacy"           # Rischio inefficacia
    DOSING = "dosing"               # Richiede aggiustamento dose
    CONTRAINDICATION = "contraindication"  # Controindicato
    MONITORING = "monitoring"       # Richiede monitoraggio


@dataclass
class DrugGeneInteraction:
    """Singola interazione gene-farmaco"""
    gene: str
    drug: str
    variant: str  # es. "*2", "rs12345", "poor metabolizer"
    phenotype: str  # es. "Poor Metabolizer", "Intermediate Metabolizer"
    interaction_type: InteractionType
    effect: str  # Descrizione effetto
    recommendation: str  # Raccomandazione clinica
    evidence_level: EvidenceLevel
    source: str  # "CPIC", "DPWG", "FDA", etc.
    pmid: Optional[str] = None  # PubMed ID

    # Rischi quantificati (0-100)
    toxicity_risk: int = 0
    efficacy_risk: int = 0  # Rischio di inefficacia

    # Flag critici
    is_life_threatening: bool = False
    requires_immediate_action: bool = False


@dataclass
class GeneInfo:
    """Informazioni su un gene farmacogenomico"""
    gene: str
    full_name: str
    chromosome: str
    function: str
    key_variants: List[str]
    affected_drugs: List[str]
    testing_recommended: bool = True


class PharmGKBDatabase:
    """
    Database locale interazioni farmacogenomiche.
    Contiene le interazioni più critiche per oncologia.
    """

    def __init__(self, custom_data_path: Optional[str] = None):
        """
        Args:
            custom_data_path: Path a file JSON con dati custom (opzionale)
        """
        self.interactions: Dict[str, List[DrugGeneInteraction]] = {}
        self.genes: Dict[str, GeneInfo] = {}
        self.drugs: Dict[str, List[str]] = {}  # drug -> [genes]

        # Carica database built-in
        self._load_builtin_database()

        # Carica dati custom se forniti
        if custom_data_path and os.path.exists(custom_data_path):
            self._load_custom_data(custom_data_path)

        logger.info(f"✅ PharmGKB Database v2.0: {len(self.interactions)} geni, "
                   f"{sum(len(v) for v in self.interactions.values())} interazioni")

    def _load_builtin_database(self):
        """Carica il database built-in con interazioni oncologiche critiche"""

        # =====================================================================
        # DPYD - 5-FU / Capecitabina (CRITICO: può essere FATALE)
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="DPYD",
            full_name="Dihydropyrimidine Dehydrogenase",
            chromosome="1p21.3",
            function="Metabolismo fluoropirimidine (5-FU, capecitabina)",
            key_variants=["*2A (rs3918290)", "*13 (rs55886062)", "c.2846A>T", "HapB3"],
            affected_drugs=["5-FU", "Capecitabina", "Tegafur", "Fluorouracil"],
            testing_recommended=True
        ))

        # DPYD *2A - Poor Metabolizer (FATALE)
        self._add_interaction(DrugGeneInteraction(
            gene="DPYD",
            drug="5-FU",
            variant="*2A/*2A",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.CONTRAINDICATION,
            effect="Tossicità severa/fatale: neutropenia, mucositis, diarrea, neurotossicità",
            recommendation="CONTROINDICATO. Non somministrare 5-FU. Considerare alternative (es. raltitrexed).",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=95,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        # DPYD *1/*2A - Intermediate Metabolizer
        self._add_interaction(DrugGeneInteraction(
            gene="DPYD",
            drug="5-FU",
            variant="*1/*2A",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.DOSING,
            effect="Rischio aumentato tossicità severa (neutropenia G3-4, mucositis)",
            recommendation="Ridurre dose iniziale al 50%. Monitoraggio emocromo settimanale.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC",
            toxicity_risk=70,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="DPYD",
            drug="Capecitabina",
            variant="*2A/*2A",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.CONTRAINDICATION,
            effect="Tossicità severa/fatale (metabolizzata a 5-FU)",
            recommendation="CONTROINDICATO. Non somministrare capecitabina.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=95,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="DPYD",
            drug="Capecitabina",
            variant="*1/*2A",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.DOSING,
            effect="Rischio aumentato tossicità severa",
            recommendation="Ridurre dose iniziale al 50%. Monitoraggio stretto.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC",
            toxicity_risk=70,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        # =====================================================================
        # UGT1A1 - Irinotecano
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="UGT1A1",
            full_name="UDP Glucuronosyltransferase 1A1",
            chromosome="2q37.1",
            function="Glucuronidazione SN-38 (metabolita attivo irinotecano)",
            key_variants=["*28 (TA repeat)", "*6 (rs4148323)", "*27"],
            affected_drugs=["Irinotecano", "Irinotecan", "Belinostat"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="UGT1A1",
            drug="Irinotecano",
            variant="*28/*28",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Neutropenia severa G3-4, diarrea grado 3-4",
            recommendation="Ridurre dose iniziale del 30%. Considerare regime FOLFIRI modificato.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=80,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="UGT1A1",
            drug="Irinotecano",
            variant="*1/*28",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.MONITORING,
            effect="Rischio moderatamente aumentato di tossicità",
            recommendation="Monitoraggio stretto emocromo. Considerare riduzione dose 20% se tossicità.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="CPIC",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="UGT1A1",
            drug="Irinotecan",
            variant="*28/*28",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Neutropenia severa, diarrea grado 3-4",
            recommendation="Ridurre dose iniziale del 30%.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=80,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        # =====================================================================
        # CYP2D6 - Tamoxifene, Codeina, Ondansetron
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="CYP2D6",
            full_name="Cytochrome P450 2D6",
            chromosome="22q13.2",
            function="Conversione tamoxifene → endoxifene (metabolita attivo)",
            key_variants=["*3", "*4", "*5", "*6", "*10", "*17", "*41"],
            affected_drugs=["Tamoxifene", "Codeina", "Tramadolo", "Ondansetron"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2D6",
            drug="Tamoxifene",
            variant="*4/*4",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.EFFICACY,
            effect="Concentrazione endoxifene subottimale. Rischio aumentato recidiva mammaria.",
            recommendation="Considerare inibitore aromatasi (se post-menopausa) o dose aumentata tamoxifene 40mg.",
            evidence_level=EvidenceLevel.LEVEL_1B,
            source="CPIC",
            toxicity_risk=0,
            efficacy_risk=75,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2D6",
            drug="Tamoxifene",
            variant="*1/*4",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.EFFICACY,
            effect="Concentrazione endoxifene ridotta",
            recommendation="Monitorare risposta. Considerare switch a inibitore aromatasi se post-menopausa.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="CPIC",
            toxicity_risk=0,
            efficacy_risk=50,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2D6",
            drug="Ondansetron",
            variant="*4/*4",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.EFFICACY,
            effect="Ridotta efficacia antiemetica",
            recommendation="Considerare antiemetico alternativo (granisetron, palonosetron).",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=0,
            efficacy_risk=40,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # =====================================================================
        # TPMT - Mercaptopurina / Azatioprina
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="TPMT",
            full_name="Thiopurine S-Methyltransferase",
            chromosome="6p22.3",
            function="Metabolismo tiopurine",
            key_variants=["*2", "*3A", "*3B", "*3C"],
            affected_drugs=["Mercaptopurina", "Azatioprina", "Tioguanina", "6-MP"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="TPMT",
            drug="Mercaptopurina",
            variant="*3A/*3A",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Mielosoppressione severa, potenzialmente fatale",
            recommendation="Ridurre dose al 10% della standard. Considerare alternativa.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=90,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="TPMT",
            drug="Mercaptopurina",
            variant="*1/*3A",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.DOSING,
            effect="Rischio aumentato mielosoppressione",
            recommendation="Ridurre dose al 50%. Monitoraggio emocromo frequente.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC",
            toxicity_risk=60,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="TPMT",
            drug="Azatioprina",
            variant="*3A/*3A",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Mielosoppressione severa",
            recommendation="Ridurre dose al 10%. Considerare alternativa.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=90,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        # =====================================================================
        # NUDT15 - Tiopurine (specialmente popolazioni asiatiche)
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="NUDT15",
            full_name="Nudix Hydrolase 15",
            chromosome="13q14.2",
            function="Metabolismo tiopurine (complementare a TPMT)",
            key_variants=["*2", "*3", "*4", "*5", "*6"],
            affected_drugs=["Mercaptopurina", "Azatioprina", "Tioguanina"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="NUDT15",
            drug="Mercaptopurina",
            variant="*3/*3",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Mielosoppressione severa (specialmente in popolazioni asiatiche)",
            recommendation="Ridurre dose al 10%. Test NUDT15 essenziale in pazienti asiatici.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC",
            toxicity_risk=90,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        # =====================================================================
        # G6PD - Rasburicase (CRITICO)
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="G6PD",
            full_name="Glucose-6-Phosphate Dehydrogenase",
            chromosome="Xq28",
            function="Protezione stress ossidativo eritrociti",
            key_variants=["A-", "Mediterranean", "Canton", "Deficiency"],
            affected_drugs=["Rasburicase", "Dapsone", "Primachina"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="G6PD",
            drug="Rasburicase",
            variant="Deficiency",
            phenotype="G6PD Deficient",
            interaction_type=InteractionType.CONTRAINDICATION,
            effect="Emolisi severa, potenzialmente fatale",
            recommendation="CONTROINDICATO. Non somministrare rasburicase. Usare allopurinolo per TLS.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="FDA",
            toxicity_risk=95,
            efficacy_risk=0,
            is_life_threatening=True,
            requires_immediate_action=True
        ))

        # =====================================================================
        # CYP3A4 - TKI, ALK Inhibitors, etc.
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="CYP3A4",
            full_name="Cytochrome P450 3A4",
            chromosome="7q22.1",
            function="Metabolismo principale TKI, ALK inhibitors, immunosoppressori",
            key_variants=["*1B", "*22"],
            affected_drugs=[
                "Osimertinib", "Gefitinib", "Erlotinib", "Afatinib", "Dacomitinib",
                "Sotorasib", "Adagrasib",
                "Alectinib", "Crizotinib", "Ceritinib", "Brigatinib", "Lorlatinib",
                "Capmatinib", "Tepotinib",
                "Dabrafenib", "Trametinib",
                "Docetaxel", "Paclitaxel"
            ],
            testing_recommended=True
        ))

        # EGFR Inhibitors
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Osimertinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione al farmaco. Rischio aumentato QTc prolongation, ILD.",
            recommendation="Monitorare ECG. Evitare inibitori forti CYP3A4. Considerare riduzione dose a 40mg.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB/EMA",
            toxicity_risk=55,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Gefitinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio aumentato diarrea, rash, epatotossicità.",
            recommendation="Monitoraggio LFT. Evitare inibitori CYP3A4. Considerare riduzione dose.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Erlotinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio aumentato rash, diarrea, ILD.",
            recommendation="Monitoraggio clinico stretto. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Afatinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.MONITORING,
            effect="Afatinib è substrato P-gp, minore impatto CYP3A4",
            recommendation="Monitoraggio per diarrea e rash.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=30,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # KRAS Inhibitors
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Sotorasib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio aumentato epatotossicità, diarrea.",
            recommendation="Monitoraggio LFT frequente. Evitare inibitori forti CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="FDA/PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Adagrasib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio QTc prolongation, epatotossicità.",
            recommendation="ECG baseline e monitoraggio. Monitoraggio LFT. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="FDA/PharmGKB",
            toxicity_risk=55,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        # ALK Inhibitors
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Alectinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio mialgia, CPK elevation, epatotossicità.",
            recommendation="Monitoraggio CPK e LFT. Evitare inibitori forti CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Crizotinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio QTc prolongation, epatotossicità, bradicardia.",
            recommendation="ECG baseline. Monitoraggio LFT. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="FDA/PharmGKB",
            toxicity_risk=60,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Ceritinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio QTc prolongation, diarrea severa, pancreatite.",
            recommendation="ECG baseline. Evitare inibitori CYP3A4. Considerare riduzione dose.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="FDA/PharmGKB",
            toxicity_risk=60,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Brigatinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio ILD, ipertensione, bradicardia.",
            recommendation="Monitoraggio polmonare. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=55,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Lorlatinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio effetti CNS (confusione, allucinazioni), ipercolesterolemia, edema.",
            recommendation="Monitoraggio neurologico e lipidico. Evitare inibitori forti CYP3A4. Considerare riduzione dose a 75mg.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="FDA/PharmGKB",
            toxicity_risk=60,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Lorlatinib",
            variant="*1/*22",
            phenotype="Intermediate Metabolizer",
            interaction_type=InteractionType.MONITORING,
            effect="Moderato aumento esposizione.",
            recommendation="Monitoraggio per effetti CNS e ipercolesterolemia.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=40,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # MET Inhibitors
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Capmatinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio edema, nausea, epatotossicità.",
            recommendation="Monitoraggio LFT. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Tepotinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio edema, ILD.",
            recommendation="Monitoraggio per edema e sintomi polmonari. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        # BRAF/MEK Inhibitors
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Dabrafenib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio febbre, rash, artralgia.",
            recommendation="Monitoraggio per sindrome febbrile. Evitare inibitori CYP3A4.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=45,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Trametinib",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.MONITORING,
            effect="Impatto CYP3A4 minore (metabolismo epatico non-CYP).",
            recommendation="Monitoraggio per rash, diarrea, cardiomiopatia.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=30,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # Chemotherapy
        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Docetaxel",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio aumentato neutropenia, neuropatia, edema.",
            recommendation="Considerare riduzione dose 20-25%. Monitoraggio emocromo stretto.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=60,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP3A4",
            drug="Paclitaxel",
            variant="*22/*22",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio aumentato neuropatia periferica.",
            recommendation="Monitoraggio per neuropatia. Considerare riduzione dose se tossicità.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # =====================================================================
        # CYP2C19 - Alcuni TKI e anticoagulanti
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="CYP2C19",
            full_name="Cytochrome P450 2C19",
            chromosome="10q23.33",
            function="Metabolismo alcuni TKI, IPP, clopidogrel",
            key_variants=["*2", "*3", "*17"],
            affected_drugs=["Clopidogrel", "Voriconazolo", "Lapatinib", "Omeprazolo"],
            testing_recommended=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2C19",
            drug="Clopidogrel",
            variant="*2/*2",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.EFFICACY,
            effect="Ridotta conversione a metabolita attivo. Rischio eventi trombotici.",
            recommendation="Considerare prasugrel o ticagrelor come alternativa.",
            evidence_level=EvidenceLevel.LEVEL_1A,
            source="CPIC/FDA",
            toxicity_risk=0,
            efficacy_risk=80,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2C19",
            drug="Voriconazolo",
            variant="*2/*2",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio epatotossicità, tossicità visiva.",
            recommendation="Ridurre dose o considerare alternativa (posaconazolo).",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=55,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2C19",
            drug="Voriconazolo",
            variant="*17/*17",
            phenotype="Ultra-rapid Metabolizer",
            interaction_type=InteractionType.EFFICACY,
            effect="Ridotta esposizione. Rischio fallimento terapeutico.",
            recommendation="Considerare dose aumentata o TDM.",
            evidence_level=EvidenceLevel.LEVEL_2A,
            source="PharmGKB",
            toxicity_risk=0,
            efficacy_risk=60,
            is_life_threatening=False,
            requires_immediate_action=True
        ))

        # =====================================================================
        # CYP2C8 - Paclitaxel
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="CYP2C8",
            full_name="Cytochrome P450 2C8",
            chromosome="10q23.33",
            function="Metabolismo paclitaxel, repaglinide",
            key_variants=["*2", "*3", "*4"],
            affected_drugs=["Paclitaxel", "Repaglinide"],
            testing_recommended=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="CYP2C8",
            drug="Paclitaxel",
            variant="*3/*3",
            phenotype="Poor Metabolizer",
            interaction_type=InteractionType.TOXICITY,
            effect="Aumento esposizione. Rischio neuropatia periferica severa.",
            recommendation="Monitoraggio neurologico. Considerare riduzione dose.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=50,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # =====================================================================
        # MTHFR - Metotrexato, Pemetrexed
        # =====================================================================
        self._add_gene_info(GeneInfo(
            gene="MTHFR",
            full_name="Methylenetetrahydrofolate Reductase",
            chromosome="1p36.22",
            function="Metabolismo folati",
            key_variants=["C677T", "A1298C"],
            affected_drugs=["Metotrexato", "Pemetrexed"],
            testing_recommended=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="MTHFR",
            drug="Metotrexato",
            variant="677TT",
            phenotype="Reduced Activity",
            interaction_type=InteractionType.TOXICITY,
            effect="Rischio aumentato mucositis, mielosoppressione.",
            recommendation="Supplementazione acido folico. Monitoraggio tossicità.",
            evidence_level=EvidenceLevel.LEVEL_2B,
            source="PharmGKB",
            toxicity_risk=45,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        self._add_interaction(DrugGeneInteraction(
            gene="MTHFR",
            drug="Pemetrexed",
            variant="677TT",
            phenotype="Reduced Activity",
            interaction_type=InteractionType.TOXICITY,
            effect="Possibile aumento tossicità.",
            recommendation="Assicurare supplementazione B12 e acido folico.",
            evidence_level=EvidenceLevel.LEVEL_3,
            source="PharmGKB",
            toxicity_risk=30,
            efficacy_risk=0,
            is_life_threatening=False,
            requires_immediate_action=False
        ))

        # Costruisci indice farmaci
        self._build_drug_index()

    def _add_gene_info(self, gene_info: GeneInfo):
        """Aggiunge info gene al database"""
        self.genes[gene_info.gene] = gene_info

    def _add_interaction(self, interaction: DrugGeneInteraction):
        """Aggiunge interazione al database"""
        if interaction.gene not in self.interactions:
            self.interactions[interaction.gene] = []
        self.interactions[interaction.gene].append(interaction)

    def _build_drug_index(self):
        """Costruisce indice inverso farmaco -> geni"""
        for gene, interactions in self.interactions.items():
            for inter in interactions:
                drug_lower = inter.drug.lower()
                if drug_lower not in self.drugs:
                    self.drugs[drug_lower] = []
                if gene not in self.drugs[drug_lower]:
                    self.drugs[drug_lower].append(gene)

    def _load_custom_data(self, path: str):
        """Carica dati custom da file JSON"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for inter_data in data.get('interactions', []):
                inter = DrugGeneInteraction(
                    gene=inter_data['gene'],
                    drug=inter_data['drug'],
                    variant=inter_data['variant'],
                    phenotype=inter_data['phenotype'],
                    interaction_type=InteractionType(inter_data['interaction_type']),
                    effect=inter_data['effect'],
                    recommendation=inter_data['recommendation'],
                    evidence_level=EvidenceLevel(inter_data['evidence_level']),
                    source=inter_data.get('source', 'Custom'),
                    toxicity_risk=inter_data.get('toxicity_risk', 0),
                    efficacy_risk=inter_data.get('efficacy_risk', 0),
                    is_life_threatening=inter_data.get('is_life_threatening', False),
                    requires_immediate_action=inter_data.get('requires_immediate_action', False)
                )
                self._add_interaction(inter)

            logger.info(f"Caricati {len(data.get('interactions', []))} interazioni custom")

        except Exception as e:
            logger.error(f"Errore caricamento dati custom: {e}")

    def get_interactions_for_gene(self, gene: str) -> List[DrugGeneInteraction]:
        """Ritorna tutte le interazioni per un gene"""
        return self.interactions.get(gene.upper(), [])

    def get_interactions_for_drug(self, drug: str) -> List[DrugGeneInteraction]:
        """Ritorna tutte le interazioni per un farmaco"""
        drug_lower = drug.lower()
        result = []

        # Cerca match esatto e parziale
        for drug_key, genes in self.drugs.items():
            if drug_lower in drug_key or drug_key in drug_lower:
                for gene in genes:
                    result.extend([i for i in self.interactions[gene]
                                  if drug_lower in i.drug.lower() or i.drug.lower() in drug_lower])

        return result

    def get_gene_info(self, gene: str) -> Optional[GeneInfo]:
        """Ritorna info su un gene"""
        return self.genes.get(gene.upper())

    def get_critical_genes(self) -> List[str]:
        """Ritorna lista geni con interazioni life-threatening"""
        critical = set()
        for gene, interactions in self.interactions.items():
            if any(i.is_life_threatening for i in interactions):
                critical.add(gene)
        return list(critical)

    def get_actionable_interactions(self,
                                    min_evidence: EvidenceLevel = EvidenceLevel.LEVEL_2A
                                    ) -> List[DrugGeneInteraction]:
        """Ritorna tutte le interazioni con evidenza >= min_evidence"""
        result = []
        for interactions in self.interactions.values():
            for inter in interactions:
                if inter.evidence_level.priority <= min_evidence.priority:
                    result.append(inter)
        return result

    def search(self, query: str) -> Dict[str, Any]:
        """
        Cerca nel database per gene, farmaco o variante.

        Returns:
            Dict con risultati categorizzati
        """
        query_lower = query.lower()
        results = {
            'genes': [],
            'drugs': [],
            'interactions': []
        }

        # Cerca nei geni
        for gene, info in self.genes.items():
            if query_lower in gene.lower() or query_lower in info.full_name.lower():
                results['genes'].append(info)

        # Cerca nei farmaci
        for drug in self.drugs.keys():
            if query_lower in drug:
                results['drugs'].append(drug)

        # Cerca nelle interazioni
        for gene_interactions in self.interactions.values():
            for inter in gene_interactions:
                if (query_lower in inter.gene.lower() or
                    query_lower in inter.drug.lower() or
                    query_lower in inter.variant.lower()):
                    results['interactions'].append(inter)

        return results


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHARMGKB DATABASE v2.0 - TEST")
    print("=" * 60)

    db = PharmGKBDatabase()

    print(f"\n📊 Database Stats:")
    print(f"   Geni: {len(db.genes)}")
    print(f"   Farmaci: {len(db.drugs)}")
    print(f"   Interazioni totali: {sum(len(v) for v in db.interactions.values())}")

    print(f"\n🚨 Geni critici (life-threatening):")
    for gene in db.get_critical_genes():
        print(f"   - {gene}")

    print(f"\n💊 Test ricerca 'Lorlatinib':")
    for inter in db.get_interactions_for_drug("Lorlatinib"):
        print(f"   {inter.gene} {inter.variant}: {inter.effect[:50]}...")
        print(f"      Tox Risk: {inter.toxicity_risk}%, Evidence: {inter.evidence_level.value}")

    print(f"\n💊 Test ricerca 'CYP3A4':")
    interactions = db.get_interactions_for_gene("CYP3A4")
    print(f"   Trovate {len(interactions)} interazioni")
    for inter in interactions[:5]:
        print(f"   - {inter.drug}: {inter.phenotype}")

    print("\n✅ PharmGKB Database v2.0 pronto!")