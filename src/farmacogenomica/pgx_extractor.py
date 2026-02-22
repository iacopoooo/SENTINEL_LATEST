"""
PGX EXTRACTOR - SENTINEL FARMACOGENOMICA
=========================================
Estrae varianti farmacogenomiche dai dati NGS del paziente.
Supporta formato SENTINEL baseline e VCF.

Le varianti PGx spesso NON sono nel pannello oncologico standard.
Questo modulo:
1. Cerca varianti PGx se presenti nei dati
2. Identifica quali geni PGx critici NON sono stati testati
3. Suggerisce test PGx aggiuntivi quando necessario
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariantSource(Enum):
    """Fonte della variante"""
    NGS_PANEL = "ngs_panel"
    LIQUID_BIOPSY = "liquid_biopsy"
    DEDICATED_PGX = "dedicated_pgx"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


@dataclass
class PGxVariant:
    """Variante farmacogenomica estratta"""
    gene: str
    variant: str  # es. "*2A", "rs3918290", "c.1905+1G>A"
    genotype: Optional[str] = None  # es. "*1/*2A", "het", "hom"
    vaf: Optional[float] = None  # Variant Allele Frequency se disponibile
    source: VariantSource = VariantSource.UNKNOWN
    confidence: float = 1.0  # 0-1
    raw_value: Optional[str] = None  # Valore originale dal report


@dataclass
class ExtractionResult:
    """Risultato estrazione PGx"""
    variants_found: List[PGxVariant]
    genes_tested: Set[str]
    genes_not_tested: Set[str]  # Geni critici non nel pannello
    recommendations: List[str]  # Suggerimenti per test aggiuntivi
    warnings: List[str]


class PGxExtractor:
    """
    Estrae varianti PGx dai dati paziente SENTINEL.
    """
    
    # Geni PGx critici per oncologia
    CRITICAL_PGX_GENES = {
        'DPYD': ['*2A', '*13', 'c.2846A>T', 'HapB3', 'rs3918290', 'rs55886062', 'rs67376798'],
        'UGT1A1': ['*28', '*6', '*27', 'rs8175347', 'rs4148323'],
        'CYP2D6': ['*3', '*4', '*5', '*6', '*10', '*17', '*41'],
        'TPMT': ['*2', '*3A', '*3B', '*3C'],
        'NUDT15': ['*2', '*3', '*4', '*5', '*6'],
        'G6PD': ['A-', 'Mediterranean', 'Canton', 'deficiency'],
        'CYP3A4': ['*1B', '*22'],
        'CYP2C19': ['*2', '*3', '*17'],
        'CYP2C8': ['*2', '*3', '*4'],
        'MTHFR': ['C677T', 'A1298C']
    }
    
    # Pattern per riconoscere varianti nei testi
    VARIANT_PATTERNS = {
        'star_allele': r'\*(\d+[A-Z]?)',  # *2A, *28
        'rs_number': r'(rs\d+)',  # rs3918290
        'hgvs_c': r'(c\.\d+[ACGT]>[ACGT])',  # c.1905+1G>A
        'hgvs_p': r'(p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2})',  # p.Gly71Arg
    }
    
    # Farmaci che richiedono test PGx specifici
    DRUG_PGX_REQUIREMENTS = {
        '5-fu': ['DPYD'],
        '5-fluorouracil': ['DPYD'],
        'fluorouracil': ['DPYD'],
        'capecitabina': ['DPYD'],
        'capecitabine': ['DPYD'],
        'xeloda': ['DPYD'],
        'irinotecano': ['UGT1A1'],
        'irinotecan': ['UGT1A1'],
        'camptosar': ['UGT1A1'],
        'tamoxifene': ['CYP2D6'],
        'tamoxifen': ['CYP2D6'],
        'nolvadex': ['CYP2D6'],
        'mercaptopurina': ['TPMT', 'NUDT15'],
        'mercaptopurine': ['TPMT', 'NUDT15'],
        '6-mp': ['TPMT', 'NUDT15'],
        'azatioprina': ['TPMT', 'NUDT15'],
        'azathioprine': ['TPMT', 'NUDT15'],
        'rasburicase': ['G6PD'],
        'elitek': ['G6PD'],
        'fasturtec': ['G6PD'],
    }
    
    def __init__(self):
        self.all_critical_genes = set(self.CRITICAL_PGX_GENES.keys())
    
    def extract_from_sentinel(self, patient_data: Dict[str, Any]) -> ExtractionResult:
        """
        Estrae varianti PGx dai dati paziente SENTINEL.
        
        Args:
            patient_data: Dict con struttura {"baseline": {...}, "visits": [...]}
            
        Returns:
            ExtractionResult con varianti trovate e raccomandazioni
        """
        variants_found = []
        genes_tested = set()
        warnings = []
        
        baseline = patient_data.get('baseline', patient_data)
        genetics = baseline.get('genetics', {})
        
        # 1. Cerca varianti PGx nei dati genetici
        for key, value in genetics.items():
            if value is None or str(value).lower() in ['unknown', 'none', 'wt', 'na', '']:
                continue
            
            # Normalizza chiave
            gene_name = self._normalize_gene_name(key)
            
            # Check se è un gene PGx
            if gene_name in self.CRITICAL_PGX_GENES:
                genes_tested.add(gene_name)
                
                # Estrai variante
                variant = self._extract_variant_from_value(str(value), gene_name)
                if variant:
                    variants_found.append(variant)

        # 2. Cerca in campi specifici PGx se presenti (FUORI dal loop genetics!)
        pgx_data = baseline.get('pharmacogenomics', {})
        if pgx_data:
            for gene, value in pgx_data.items():
                gene_upper = gene.upper()
                if gene_upper in self.CRITICAL_PGX_GENES:
                    genes_tested.add(gene_upper)
                    variant = self._extract_variant_from_value(str(value), gene_upper)
                    if variant:
                        variant.source = VariantSource.DEDICATED_PGX
                        variants_found.append(variant)

        # 3. Cerca in pgx_profile (formato SENTINEL new_patient.py) - FUORI dal loop!
        pgx_profile = baseline.get('pgx_profile', {})
        if pgx_profile:
            for gene, genotype in pgx_profile.items():
                if genotype is None:
                    continue
                gene_upper = gene.upper()
                if gene_upper in self.CRITICAL_PGX_GENES:
                    genes_tested.add(gene_upper)
                    variant = PGxVariant(
                        gene=gene_upper,
                        variant=str(genotype).split('/')[0] if '/' in str(genotype) else str(genotype),
                        genotype=str(genotype),
                        source=VariantSource.DEDICATED_PGX,
                        confidence=1.0,
                        raw_value=str(genotype)
                    )
                    variants_found.append(variant)
        
        # 3. Determina quali geni critici NON sono stati testati
        genes_not_tested = self.all_critical_genes - genes_tested
        
        # 4. Genera raccomandazioni basate sulla terapia
        recommendations = self._generate_recommendations(
            baseline.get('current_therapy', ''),
            genes_tested,
            genes_not_tested
        )
        
        # 5. Warnings per situazioni critiche
        if genes_not_tested:
            therapy = baseline.get('current_therapy', '').lower()
            for drug, required_genes in self.DRUG_PGX_REQUIREMENTS.items():
                if drug in therapy:
                    missing = [g for g in required_genes if g in genes_not_tested]
                    if missing:
                        warnings.append(
                            f"⚠️ CRITICO: Terapia '{drug}' richiede test {', '.join(missing)} non disponibile"
                        )
        
        return ExtractionResult(
            variants_found=variants_found,
            genes_tested=genes_tested,
            genes_not_tested=genes_not_tested,
            recommendations=recommendations,
            warnings=warnings
        )
    
    def _normalize_gene_name(self, key: str) -> str:
        """Normalizza nome gene dalla chiave del dizionario"""
        # Rimuovi suffissi comuni
        clean = key.upper()
        for suffix in ['_STATUS', '_MUTATION', '_VARIANT', '_GENOTYPE']:
            clean = clean.replace(suffix, '')
        return clean.strip()
    
    def _extract_variant_from_value(self, value: str, gene: str) -> Optional[PGxVariant]:
        """Estrae informazioni variante dal valore"""
        if not value or value.lower() in ['wt', 'wild-type', 'normal', 'none', 'negative']:
            return None
        
        variant_id = None
        genotype = None
        
        # Cerca star allele (*2A, *28)
        star_match = re.search(r'\*(\d+[A-Z]?)', value)
        if star_match:
            variant_id = f"*{star_match.group(1)}"
        
        # Cerca rs number
        rs_match = re.search(r'(rs\d+)', value, re.IGNORECASE)
        if rs_match:
            variant_id = rs_match.group(1)
        
        # Cerca HGVS
        hgvs_match = re.search(r'(c\.\d+[+-]?\d*[ACGT]+>[ACGT]+)', value, re.IGNORECASE)
        if hgvs_match:
            variant_id = hgvs_match.group(1)
        
        # Se non trovato pattern specifico, usa valore raw
        if not variant_id:
            variant_id = value
        
        # Cerca genotipo (het, hom, *1/*2)
        if 'het' in value.lower():
            genotype = 'heterozygous'
        elif 'hom' in value.lower():
            genotype = 'homozygous'
        
        geno_match = re.search(r'(\*\d+[A-Z]?/\*\d+[A-Z]?)', value)
        if geno_match:
            genotype = geno_match.group(1)
        
        return PGxVariant(
            gene=gene,
            variant=variant_id,
            genotype=genotype,
            source=VariantSource.NGS_PANEL,
            raw_value=value
        )
    
    def _generate_recommendations(self, 
                                   therapy: str, 
                                   genes_tested: Set[str],
                                   genes_not_tested: Set[str]) -> List[str]:
        """Genera raccomandazioni per test PGx"""
        recommendations = []
        therapy_lower = therapy.lower() if therapy else ""
        
        # Check farmaci che richiedono test specifici
        for drug, required_genes in self.DRUG_PGX_REQUIREMENTS.items():
            if drug in therapy_lower:
                for gene in required_genes:
                    if gene in genes_not_tested:
                        recommendations.append(
                            f"🧬 Test {gene} RACCOMANDATO prima di {drug.upper()}"
                        )
        
        # Raccomandazioni generali per geni critici
        critical_not_tested = genes_not_tested.intersection({'DPYD', 'UGT1A1', 'G6PD'})
        if critical_not_tested and not recommendations:
            recommendations.append(
                f"ℹ️ Considera test farmacogenomico per: {', '.join(critical_not_tested)}"
            )
        
        return recommendations
    
    def check_drug_compatibility(self, 
                                  drug: str, 
                                  patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica se un farmaco è sicuro per il paziente basandosi sui dati PGx.
        
        Args:
            drug: Nome del farmaco
            patient_data: Dati paziente SENTINEL
            
        Returns:
            Dict con status e raccomandazioni
        """
        result = {
            'drug': drug,
            'safe': True,
            'pgx_tested': False,
            'required_genes': [],
            'missing_genes': [],
            'alerts': [],
            'recommendation': "Nessuna controindicazione PGx nota"
        }
        
        drug_lower = drug.lower()
        
        # Trova geni richiesti per questo farmaco
        for drug_pattern, genes in self.DRUG_PGX_REQUIREMENTS.items():
            if drug_pattern in drug_lower:
                result['required_genes'] = genes
                break
        
        if not result['required_genes']:
            result['recommendation'] = "Nessun test PGx specifico richiesto per questo farmaco"
            return result
        
        # Estrai dati PGx paziente
        extraction = self.extract_from_sentinel(patient_data)
        
        # Verifica quali geni sono stati testati
        for gene in result['required_genes']:
            if gene in extraction.genes_tested:
                result['pgx_tested'] = True
            else:
                result['missing_genes'].append(gene)
        
        if result['missing_genes']:
            result['safe'] = False  # Non possiamo garantire sicurezza
            result['alerts'].append(
                f"Test {', '.join(result['missing_genes'])} non disponibile - "
                f"impossibile escludere rischio PGx"
            )
            result['recommendation'] = (
                f"Eseguire test {', '.join(result['missing_genes'])} "
                f"PRIMA di iniziare {drug}"
            )
        
        # Check varianti trovate
        for variant in extraction.variants_found:
            if variant.gene in result['required_genes']:
                # Qui si dovrebbe cross-reference con PharmGKB
                # Per ora flag come "richiede revisione"
                result['alerts'].append(
                    f"Variante {variant.gene} {variant.variant} rilevata - verificare implicazioni"
                )
        
        return result
    
    def get_required_tests_for_therapy(self, therapy: str) -> List[str]:
        """
        Ritorna lista di test PGx raccomandati per una terapia.
        """
        tests = set()
        therapy_lower = therapy.lower()
        
        for drug, genes in self.DRUG_PGX_REQUIREMENTS.items():
            if drug in therapy_lower:
                tests.update(genes)
        
        return list(tests)


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PGX EXTRACTOR - TEST")
    print("=" * 60)
    
    extractor = PGxExtractor()
    
    # Test con paziente esempio
    test_patient = {
        'baseline': {
            'patient_id': 'TEST-001',
            'current_therapy': 'FOLFOX (5-FU + Oxaliplatino)',
            'genetics': {
                'tp53_status': 'mutated',
                'kras_mutation': 'G12C',
                'dpyd_status': '*1/*2A heterozygous',  # PGx variante!
                'ugt1a1_status': '*1/*28'
            }
        }
    }
    
    print("\n📋 Test paziente con FOLFOX:")
    result = extractor.extract_from_sentinel(test_patient)
    
    print(f"\n   Varianti PGx trovate: {len(result.variants_found)}")
    for v in result.variants_found:
        print(f"      - {v.gene}: {v.variant} ({v.genotype})")
    
    print(f"\n   Geni testati: {result.genes_tested}")
    print(f"   Geni NON testati: {result.genes_not_tested}")
    
    print(f"\n   Raccomandazioni:")
    for rec in result.recommendations:
        print(f"      {rec}")
    
    print(f"\n   Warnings:")
    for warn in result.warnings:
        print(f"      {warn}")
    
    print("\n" + "=" * 60)
    print("💊 Test compatibilità farmaco:")
    compat = extractor.check_drug_compatibility("5-FU", test_patient)
    print(f"   Farmaco: {compat['drug']}")
    print(f"   Safe: {compat['safe']}")
    print(f"   Geni richiesti: {compat['required_genes']}")
    print(f"   Geni mancanti: {compat['missing_genes']}")
    print(f"   Alerts: {compat['alerts']}")
    
    print("\n✅ PGxExtractor pronto!")
