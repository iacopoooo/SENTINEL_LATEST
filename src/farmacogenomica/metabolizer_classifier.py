"""
METABOLIZER CLASSIFIER - SENTINEL FARMACOGENOMICA
==================================================
Classifica il fenotipo metabolizzatore del paziente basandosi
sulle varianti genetiche. Segue nomenclatura CPIC standard.

Fenotipi:
- Ultra-Rapid Metabolizer (UM): Metabolismo aumentato
- Normal Metabolizer (NM): Metabolismo normale (wild-type)
- Intermediate Metabolizer (IM): Metabolismo ridotto
- Poor Metabolizer (PM): Metabolismo assente/molto ridotto

Activity Score (AS):
Sistema numerico usato per CYP2D6 e altri geni.
AS = somma dei valori di attività degli alleli.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetabolizerPhenotype(Enum):
    """Fenotipi metabolizzatori standard CPIC"""
    ULTRA_RAPID = "Ultra-Rapid Metabolizer"
    NORMAL = "Normal Metabolizer"
    INTERMEDIATE = "Intermediate Metabolizer"
    POOR = "Poor Metabolizer"
    INDETERMINATE = "Indeterminate"
    UNKNOWN = "Unknown"
    
    @property
    def abbreviation(self) -> str:
        abbrev = {
            MetabolizerPhenotype.ULTRA_RAPID: "UM",
            MetabolizerPhenotype.NORMAL: "NM",
            MetabolizerPhenotype.INTERMEDIATE: "IM",
            MetabolizerPhenotype.POOR: "PM",
            MetabolizerPhenotype.INDETERMINATE: "IND",
            MetabolizerPhenotype.UNKNOWN: "UNK"
        }
        return abbrev.get(self, "UNK")
    
    @property
    def risk_level(self) -> int:
        """Livello di rischio (più alto = più attenzione)"""
        levels = {
            MetabolizerPhenotype.NORMAL: 0,
            MetabolizerPhenotype.ULTRA_RAPID: 2,
            MetabolizerPhenotype.INTERMEDIATE: 3,
            MetabolizerPhenotype.POOR: 5,
            MetabolizerPhenotype.INDETERMINATE: 2,
            MetabolizerPhenotype.UNKNOWN: 1
        }
        return levels.get(self, 1)


@dataclass
class PhenotypeResult:
    """Risultato classificazione fenotipo"""
    gene: str
    phenotype: MetabolizerPhenotype
    activity_score: Optional[float] = None
    genotype: Optional[str] = None
    alleles: List[str] = None
    confidence: float = 1.0
    notes: List[str] = None
    
    def __post_init__(self):
        if self.alleles is None:
            self.alleles = []
        if self.notes is None:
            self.notes = []


class MetabolizerClassifier:
    """
    Classifica fenotipi metabolizzatori basandosi su genotipi.
    """
    
    # Activity Score per alleli CYP2D6 (da CPIC)
    CYP2D6_ACTIVITY = {
        '*1': 1.0,    # Normal function
        '*2': 1.0,    # Normal function
        '*3': 0.0,    # No function
        '*4': 0.0,    # No function
        '*5': 0.0,    # No function (gene deletion)
        '*6': 0.0,    # No function
        '*9': 0.5,    # Decreased function
        '*10': 0.25,  # Decreased function
        '*17': 0.5,   # Decreased function
        '*29': 0.5,   # Decreased function
        '*41': 0.5,   # Decreased function
        '*1xN': 2.0,  # Increased function (gene duplication)
        '*2xN': 2.0,  # Increased function (gene duplication)
    }
    
    # Soglie Activity Score -> Fenotipo per CYP2D6
    CYP2D6_THRESHOLDS = {
        (0, 0): MetabolizerPhenotype.POOR,
        (0.25, 0.5): MetabolizerPhenotype.INTERMEDIATE,
        (0.5, 1.0): MetabolizerPhenotype.INTERMEDIATE,
        (1.0, 1.25): MetabolizerPhenotype.NORMAL,
        (1.25, 2.25): MetabolizerPhenotype.NORMAL,
        (2.25, float('inf')): MetabolizerPhenotype.ULTRA_RAPID,
    }
    
    # DPYD: diplotipi e fenotipi
    DPYD_PHENOTYPES = {
        # Normale
        ('*1', '*1'): MetabolizerPhenotype.NORMAL,
        ('normal', 'normal'): MetabolizerPhenotype.NORMAL,
        
        # Intermediate
        ('*1', '*2A'): MetabolizerPhenotype.INTERMEDIATE,
        ('*1', 'c.2846A>T'): MetabolizerPhenotype.INTERMEDIATE,
        ('*1', 'HapB3'): MetabolizerPhenotype.INTERMEDIATE,
        ('normal', '*2A'): MetabolizerPhenotype.INTERMEDIATE,
        
        # Poor
        ('*2A', '*2A'): MetabolizerPhenotype.POOR,
        ('*2A', '*13'): MetabolizerPhenotype.POOR,
        ('*13', '*13'): MetabolizerPhenotype.POOR,
    }
    
    # UGT1A1: genotipi *28
    UGT1A1_PHENOTYPES = {
        '*1/*1': MetabolizerPhenotype.NORMAL,
        '*1/*28': MetabolizerPhenotype.INTERMEDIATE,
        '*28/*28': MetabolizerPhenotype.POOR,
        '*1/*6': MetabolizerPhenotype.INTERMEDIATE,
        '*6/*6': MetabolizerPhenotype.POOR,
        '*6/*28': MetabolizerPhenotype.POOR,
    }
    
    # TPMT Activity Score
    TPMT_ACTIVITY = {
        '*1': 1.0,
        '*2': 0.0,
        '*3A': 0.0,
        '*3B': 0.0,
        '*3C': 0.0,
    }
    
    # NUDT15 Activity Score
    NUDT15_ACTIVITY = {
        '*1': 1.0,
        '*2': 0.0,
        '*3': 0.0,
        '*4': 0.0,
        '*5': 0.5,
        '*6': 0.0,
    }
    
    def __init__(self):
        pass
    
    def classify(self, gene: str, genotype: str) -> PhenotypeResult:
        """
        Classifica fenotipo per un gene dato il genotipo.
        
        Args:
            gene: Nome gene (es. "CYP2D6", "DPYD")
            genotype: Genotipo (es. "*1/*4", "*2A heterozygous")
            
        Returns:
            PhenotypeResult con fenotipo e dettagli
        """
        gene_upper = gene.upper()
        
        # Dispatch a metodo specifico per gene
        if gene_upper == 'CYP2D6':
            return self._classify_cyp2d6(genotype)
        elif gene_upper == 'DPYD':
            return self._classify_dpyd(genotype)
        elif gene_upper == 'UGT1A1':
            return self._classify_ugt1a1(genotype)
        elif gene_upper == 'TPMT':
            return self._classify_tpmt(genotype)
        elif gene_upper == 'NUDT15':
            return self._classify_nudt15(genotype)
        elif gene_upper == 'G6PD':
            return self._classify_g6pd(genotype)
        else:
            return self._classify_generic(gene_upper, genotype)
    
    def _classify_cyp2d6(self, genotype: str) -> PhenotypeResult:
        """Classifica CYP2D6 usando Activity Score"""
        alleles = self._parse_diplotype(genotype)
        
        if not alleles or len(alleles) != 2:
            return PhenotypeResult(
                gene="CYP2D6",
                phenotype=MetabolizerPhenotype.UNKNOWN,
                genotype=genotype,
                notes=["Impossibile determinare alleli dal genotipo"]
            )
        
        # Calcola Activity Score
        as1 = self.CYP2D6_ACTIVITY.get(alleles[0], 1.0)
        as2 = self.CYP2D6_ACTIVITY.get(alleles[1], 1.0)
        total_as = as1 + as2
        
        # Determina fenotipo
        phenotype = MetabolizerPhenotype.UNKNOWN
        for (low, high), pheno in self.CYP2D6_THRESHOLDS.items():
            if low <= total_as <= high:
                phenotype = pheno
                break
        
        return PhenotypeResult(
            gene="CYP2D6",
            phenotype=phenotype,
            activity_score=total_as,
            genotype=genotype,
            alleles=alleles,
            notes=[f"Activity Score: {as1} + {as2} = {total_as}"]
        )
    
    def _classify_dpyd(self, genotype: str) -> PhenotypeResult:
        """Classifica DPYD"""
        alleles = self._parse_diplotype(genotype)
        notes = []
        
        # Check per pattern noti
        if 'poor' in genotype.lower():
            return PhenotypeResult(
                gene="DPYD",
                phenotype=MetabolizerPhenotype.POOR,
                genotype=genotype,
                notes=["Classificato come Poor Metabolizer dal report"]
            )
        
        if 'intermediate' in genotype.lower() or 'het' in genotype.lower():
            return PhenotypeResult(
                gene="DPYD",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
                genotype=genotype,
                notes=["Classificato come Intermediate Metabolizer"]
            )
        
        # Check varianti specifiche
        critical_variants = ['*2A', '*13', 'rs3918290', 'rs55886062']
        has_critical = any(v.lower() in genotype.lower() for v in critical_variants)
        
        if has_critical:
            # Check se omozigote
            if genotype.count('*2A') >= 2 or 'hom' in genotype.lower():
                return PhenotypeResult(
                    gene="DPYD",
                    phenotype=MetabolizerPhenotype.POOR,
                    genotype=genotype,
                    notes=["⚠️ DPYD deficit omozigote - CONTROINDICAZIONE ASSOLUTA fluoropirimidine"]
                )
            else:
                return PhenotypeResult(
                    gene="DPYD",
                    phenotype=MetabolizerPhenotype.INTERMEDIATE,
                    genotype=genotype,
                    notes=["⚠️ DPYD deficit eterozigote - ridurre dose 50%"]
                )
        
        # Se sembra normale
        if 'wt' in genotype.lower() or 'normal' in genotype.lower() or '*1/*1' in genotype:
            return PhenotypeResult(
                gene="DPYD",
                phenotype=MetabolizerPhenotype.NORMAL,
                genotype=genotype
            )
        
        return PhenotypeResult(
            gene="DPYD",
            phenotype=MetabolizerPhenotype.INDETERMINATE,
            genotype=genotype,
            notes=["Fenotipo non determinabile con certezza"]
        )
    
    def _classify_ugt1a1(self, genotype: str) -> PhenotypeResult:
        """Classifica UGT1A1"""
        # Cerca pattern *28/*28, *1/*28 etc
        genotype_clean = genotype.replace(' ', '').replace('UGT1A1', '').strip()
        
        # Check dizionario
        if genotype_clean in self.UGT1A1_PHENOTYPES:
            return PhenotypeResult(
                gene="UGT1A1",
                phenotype=self.UGT1A1_PHENOTYPES[genotype_clean],
                genotype=genotype
            )
        
        # Inferenza da pattern
        if '*28/*28' in genotype or '*28' in genotype and 'hom' in genotype.lower():
            return PhenotypeResult(
                gene="UGT1A1",
                phenotype=MetabolizerPhenotype.POOR,
                genotype=genotype,
                notes=["*28 omozigote - ridurre dose irinotecano"]
            )
        
        if '*28' in genotype:
            return PhenotypeResult(
                gene="UGT1A1",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
                genotype=genotype,
                notes=["*28 eterozigote - monitorare tossicità irinotecano"]
            )
        
        if '*1/*1' in genotype or 'normal' in genotype.lower():
            return PhenotypeResult(
                gene="UGT1A1",
                phenotype=MetabolizerPhenotype.NORMAL,
                genotype=genotype
            )
        
        return PhenotypeResult(
            gene="UGT1A1",
            phenotype=MetabolizerPhenotype.UNKNOWN,
            genotype=genotype
        )
    
    def _classify_tpmt(self, genotype: str) -> PhenotypeResult:
        """Classifica TPMT usando Activity Score"""
        alleles = self._parse_diplotype(genotype)
        
        if len(alleles) == 2:
            as1 = self.TPMT_ACTIVITY.get(alleles[0], 1.0)
            as2 = self.TPMT_ACTIVITY.get(alleles[1], 1.0)
            total_as = as1 + as2
            
            if total_as == 0:
                phenotype = MetabolizerPhenotype.POOR
            elif total_as < 2:
                phenotype = MetabolizerPhenotype.INTERMEDIATE
            else:
                phenotype = MetabolizerPhenotype.NORMAL
            
            return PhenotypeResult(
                gene="TPMT",
                phenotype=phenotype,
                activity_score=total_as,
                genotype=genotype,
                alleles=alleles
            )
        
        # Pattern matching fallback
        if any(v in genotype for v in ['*3A', '*3B', '*3C', '*2']):
            if 'hom' in genotype.lower() or genotype.count('*3') >= 2:
                return PhenotypeResult(
                    gene="TPMT",
                    phenotype=MetabolizerPhenotype.POOR,
                    genotype=genotype
                )
            return PhenotypeResult(
                gene="TPMT",
                phenotype=MetabolizerPhenotype.INTERMEDIATE,
                genotype=genotype
            )
        
        return PhenotypeResult(
            gene="TPMT",
            phenotype=MetabolizerPhenotype.NORMAL,
            genotype=genotype
        )
    
    def _classify_nudt15(self, genotype: str) -> PhenotypeResult:
        """Classifica NUDT15"""
        alleles = self._parse_diplotype(genotype)
        
        if len(alleles) == 2:
            as1 = self.NUDT15_ACTIVITY.get(alleles[0], 1.0)
            as2 = self.NUDT15_ACTIVITY.get(alleles[1], 1.0)
            total_as = as1 + as2
            
            if total_as == 0:
                phenotype = MetabolizerPhenotype.POOR
            elif total_as < 2:
                phenotype = MetabolizerPhenotype.INTERMEDIATE
            else:
                phenotype = MetabolizerPhenotype.NORMAL
            
            return PhenotypeResult(
                gene="NUDT15",
                phenotype=phenotype,
                activity_score=total_as,
                genotype=genotype,
                alleles=alleles,
                notes=["NUDT15 importante specialmente in pazienti asiatici"]
            )
        
        return PhenotypeResult(
            gene="NUDT15",
            phenotype=MetabolizerPhenotype.UNKNOWN,
            genotype=genotype
        )
    
    def _classify_g6pd(self, genotype: str) -> PhenotypeResult:
        """Classifica G6PD"""
        genotype_lower = genotype.lower()
        
        if any(v in genotype_lower for v in ['deficient', 'deficit', 'deficiency', 'low']):
            return PhenotypeResult(
                gene="G6PD",
                phenotype=MetabolizerPhenotype.POOR,
                genotype=genotype,
                notes=["⚠️ G6PD deficit - CONTROINDICATO rasburicase"]
            )
        
        if any(v in genotype_lower for v in ['a-', 'mediterranean', 'canton', 'mahidol']):
            return PhenotypeResult(
                gene="G6PD",
                phenotype=MetabolizerPhenotype.POOR,
                genotype=genotype,
                notes=["⚠️ Variante G6PD deficit noto - CONTROINDICATO rasburicase"]
            )
        
        if any(v in genotype_lower for v in ['normal', 'sufficient', 'wt', 'wild']):
            return PhenotypeResult(
                gene="G6PD",
                phenotype=MetabolizerPhenotype.NORMAL,
                genotype=genotype
            )
        
        return PhenotypeResult(
            gene="G6PD",
            phenotype=MetabolizerPhenotype.UNKNOWN,
            genotype=genotype
        )
    
    def _classify_generic(self, gene: str, genotype: str) -> PhenotypeResult:
        """Classificazione generica per geni non specificamente supportati"""
        genotype_lower = genotype.lower()
        
        if 'poor' in genotype_lower:
            phenotype = MetabolizerPhenotype.POOR
        elif 'intermediate' in genotype_lower:
            phenotype = MetabolizerPhenotype.INTERMEDIATE
        elif 'ultra' in genotype_lower or 'rapid' in genotype_lower:
            phenotype = MetabolizerPhenotype.ULTRA_RAPID
        elif 'normal' in genotype_lower or 'wt' in genotype_lower:
            phenotype = MetabolizerPhenotype.NORMAL
        else:
            phenotype = MetabolizerPhenotype.UNKNOWN
        
        return PhenotypeResult(
            gene=gene,
            phenotype=phenotype,
            genotype=genotype,
            notes=["Classificazione generica - verificare manualmente"]
        )
    
    def _parse_diplotype(self, genotype: str) -> List[str]:
        """Estrae i due alleli da un diplotipo es. '*1/*4' -> ['*1', '*4']"""
        import re
        
        # Cerca pattern *X/*Y
        match = re.search(r'(\*\d+[A-Z]?)\s*/\s*(\*\d+[A-Z]?)', genotype)
        if match:
            return [match.group(1), match.group(2)]
        
        # Cerca singolo allele ripetuto (homozygous)
        match = re.search(r'(\*\d+[A-Z]?)', genotype)
        if match and ('hom' in genotype.lower() or genotype.count(match.group(1)) >= 2):
            return [match.group(1), match.group(1)]
        
        # Eterozigote con wild-type implicito
        if match:
            return ['*1', match.group(1)]
        
        return []


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("METABOLIZER CLASSIFIER - TEST")
    print("=" * 60)
    
    classifier = MetabolizerClassifier()
    
    test_cases = [
        ("CYP2D6", "*1/*4"),
        ("CYP2D6", "*4/*4"),
        ("CYP2D6", "*1/*1"),
        ("DPYD", "*1/*2A heterozygous"),
        ("DPYD", "*2A/*2A"),
        ("UGT1A1", "*1/*28"),
        ("UGT1A1", "*28/*28"),
        ("TPMT", "*1/*3A"),
        ("G6PD", "A- deficient"),
    ]
    
    for gene, genotype in test_cases:
        result = classifier.classify(gene, genotype)
        print(f"\n{gene} {genotype}:")
        print(f"   Fenotipo: {result.phenotype.value} ({result.phenotype.abbreviation})")
        if result.activity_score is not None:
            print(f"   Activity Score: {result.activity_score}")
        if result.notes:
            for note in result.notes:
                print(f"   Note: {note}")
    
    print("\n✅ MetabolizerClassifier pronto!")
