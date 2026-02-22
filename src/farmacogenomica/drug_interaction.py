"""
DRUG INTERACTION ENGINE - SENTINEL FARMACOGENOMICA
===================================================
Calcola rischio tossicità/inefficacia per combinazione paziente-farmaco.
Integra dati PGx con database PharmGKB per generare risk score.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .pharmgkb_database import PharmGKBDatabase, DrugGeneInteraction, EvidenceLevel, InteractionType
from .pgx_extractor import PGxExtractor, PGxVariant
from .metabolizer_classifier import MetabolizerClassifier, MetabolizerPhenotype, PhenotypeResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Livello di rischio complessivo"""
    CONTRAINDICATED = "contraindicated"  # Non usare
    HIGH = "high"                        # Rischio elevato, richiede azione
    MODERATE = "moderate"                # Monitoraggio stretto
    LOW = "low"                          # Rischio minimo
    UNKNOWN = "unknown"                  # Dati insufficienti
    SAFE = "safe"                        # Nessun rischio PGx noto
    
    @property
    def color(self) -> str:
        colors = {
            RiskLevel.CONTRAINDICATED: "#B71C1C",  # Rosso scuro
            RiskLevel.HIGH: "#D32F2F",              # Rosso
            RiskLevel.MODERATE: "#FF9800",          # Arancione
            RiskLevel.LOW: "#FFC107",               # Giallo
            RiskLevel.UNKNOWN: "#9E9E9E",           # Grigio
            RiskLevel.SAFE: "#4CAF50"               # Verde
        }
        return colors.get(self, "#9E9E9E")
    
    @property
    def priority(self) -> int:
        """Priorità per ordinamento (più basso = più urgente)"""
        priorities = {
            RiskLevel.CONTRAINDICATED: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MODERATE: 3,
            RiskLevel.LOW: 4,
            RiskLevel.UNKNOWN: 5,
            RiskLevel.SAFE: 6
        }
        return priorities.get(self, 5)


@dataclass
class DrugRiskAssessment:
    """Valutazione rischio per un singolo farmaco"""
    drug: str
    risk_level: RiskLevel
    toxicity_risk: int  # 0-100
    efficacy_risk: int  # 0-100 (rischio inefficacia)
    
    # Dettagli
    gene_interactions: List[DrugGeneInteraction] = field(default_factory=list)
    phenotypes: List[PhenotypeResult] = field(default_factory=list)
    
    # Raccomandazioni
    primary_recommendation: str = ""
    alternative_drugs: List[str] = field(default_factory=list)
    dosing_adjustment: Optional[str] = None
    monitoring_required: List[str] = field(default_factory=list)
    
    # Metadati
    evidence_level: Optional[EvidenceLevel] = None
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def requires_immediate_action(self) -> bool:
        return self.risk_level in [RiskLevel.CONTRAINDICATED, RiskLevel.HIGH]


@dataclass
class PatientPGxProfile:
    """Profilo PGx completo del paziente"""
    patient_id: str
    phenotypes: Dict[str, PhenotypeResult]  # gene -> phenotype
    variants: List[PGxVariant]
    genes_tested: List[str]
    genes_not_tested: List[str]
    overall_risk: RiskLevel = RiskLevel.UNKNOWN


class DrugInteractionEngine:
    """
    Engine per calcolo interazioni farmaco-gene.
    Combina tutti i componenti PGx per generare valutazioni rischio.
    """
    
    def __init__(self, database: Optional[PharmGKBDatabase] = None):
        """
        Args:
            database: Database PharmGKB (crea nuovo se non fornito)
        """
        self.database = database or PharmGKBDatabase()
        self.extractor = PGxExtractor()
        self.classifier = MetabolizerClassifier()
    
    def analyze_patient(self, patient_data: Dict[str, Any]) -> PatientPGxProfile:
        """
        Analizza profilo PGx completo del paziente.
        
        Args:
            patient_data: Dati paziente SENTINEL
            
        Returns:
            PatientPGxProfile con tutti i fenotipi
        """
        baseline = patient_data.get('baseline', patient_data)
        patient_id = baseline.get('patient_id', 'Unknown')
        
        # Estrai varianti
        extraction = self.extractor.extract_from_sentinel(patient_data)
        
        # Classifica fenotipi per ogni variante trovata
        phenotypes = {}
        for variant in extraction.variants_found:
            genotype = variant.genotype or variant.variant
            result = self.classifier.classify(variant.gene, genotype)
            phenotypes[variant.gene] = result
        
        return PatientPGxProfile(
            patient_id=patient_id,
            phenotypes=phenotypes,
            variants=extraction.variants_found,
            genes_tested=list(extraction.genes_tested),
            genes_not_tested=list(extraction.genes_not_tested)
        )
    
    def assess_drug_risk(self, 
                         drug: str, 
                         patient_data: Dict[str, Any]) -> DrugRiskAssessment:
        """
        Valuta rischio di un farmaco per un paziente.
        
        Args:
            drug: Nome del farmaco
            patient_data: Dati paziente SENTINEL
            
        Returns:
            DrugRiskAssessment con rischio e raccomandazioni
        """
        # Analizza profilo paziente
        profile = self.analyze_patient(patient_data)
        
        # Cerca interazioni nel database
        interactions = self.database.get_interactions_for_drug(drug)
        
        if not interactions:
            return DrugRiskAssessment(
                drug=drug,
                risk_level=RiskLevel.SAFE,
                toxicity_risk=0,
                efficacy_risk=0,
                primary_recommendation=f"Nessuna interazione PGx nota per {drug}",
                confidence=0.8
            )
        
        # Filtra interazioni rilevanti per questo paziente
        relevant_interactions = []
        matched_phenotypes = []
        
        for inter in interactions:
            gene = inter.gene
            
            # Check se abbiamo dati PGx per questo gene
            if gene in profile.phenotypes:
                phenotype = profile.phenotypes[gene]
                
                # Match fenotipo con interazione
                if self._phenotype_matches_interaction(phenotype, inter):
                    relevant_interactions.append(inter)
                    matched_phenotypes.append(phenotype)
            
            elif gene in profile.genes_not_tested:
                # Gene critico non testato
                if inter.is_life_threatening:
                    relevant_interactions.append(inter)
        
        # Calcola rischio complessivo
        return self._calculate_risk_assessment(
            drug=drug,
            interactions=relevant_interactions,
            phenotypes=matched_phenotypes,
            profile=profile
        )
    
    def _phenotype_matches_interaction(self, 
                                        phenotype: PhenotypeResult,
                                        interaction: DrugGeneInteraction) -> bool:
        """Verifica se il fenotipo del paziente corrisponde all'interazione"""
        
        # Match per tipo di metabolizzatore
        pheno_type = phenotype.phenotype
        inter_pheno = interaction.phenotype.lower()
        
        if pheno_type == MetabolizerPhenotype.POOR and 'poor' in inter_pheno:
            return True
        if pheno_type == MetabolizerPhenotype.INTERMEDIATE and 'intermediate' in inter_pheno:
            return True
        if pheno_type == MetabolizerPhenotype.ULTRA_RAPID and 'ultra' in inter_pheno:
            return True
        
        # Match per variante specifica
        if phenotype.genotype:
            inter_variant = interaction.variant.lower()
            geno_lower = phenotype.genotype.lower()
            
            # Cerca overlap
            if any(v in geno_lower for v in inter_variant.split('/')):
                return True
        
        return False
    
    def _calculate_risk_assessment(self,
                                    drug: str,
                                    interactions: List[DrugGeneInteraction],
                                    phenotypes: List[PhenotypeResult],
                                    profile: PatientPGxProfile) -> DrugRiskAssessment:
        """Calcola assessment finale"""
        
        if not interactions:
            return DrugRiskAssessment(
                drug=drug,
                risk_level=RiskLevel.SAFE,
                toxicity_risk=0,
                efficacy_risk=0,
                primary_recommendation="Nessuna interazione PGx rilevante trovata"
            )
        
        # Trova interazione più critica
        max_tox = 0
        max_eff = 0
        is_contraindicated = False
        requires_action = False
        best_evidence = EvidenceLevel.UNKNOWN
        recommendations = []
        alternatives = []
        dosing = None
        monitoring = []
        warnings = []
        
        for inter in interactions:
            # Aggiorna rischi massimi
            if inter.toxicity_risk > max_tox:
                max_tox = inter.toxicity_risk
            if inter.efficacy_risk > max_eff:
                max_eff = inter.efficacy_risk
            
            # Check controindicazione
            if inter.interaction_type == InteractionType.CONTRAINDICATION:
                is_contraindicated = True
            
            if inter.requires_immediate_action:
                requires_action = True
            
            # Migliore evidenza
            if inter.evidence_level.priority < best_evidence.priority:
                best_evidence = inter.evidence_level
            
            # Raccomandazioni
            recommendations.append(inter.recommendation)
            
            # Dosing
            if inter.interaction_type == InteractionType.DOSING:
                dosing = inter.recommendation
            
            # Monitoring
            if inter.interaction_type == InteractionType.MONITORING:
                monitoring.append(inter.effect)
            
            # Life-threatening warning
            if inter.is_life_threatening:
                warnings.append(f"⚠️ CRITICO: {inter.gene} - {inter.effect}")
        
        # Check geni critici non testati
        for gene in profile.genes_not_tested:
            critical_inters = [i for i in self.database.get_interactions_for_gene(gene)
                             if i.drug.lower() in drug.lower() and i.is_life_threatening]
            if critical_inters:
                warnings.append(f"⚠️ Test {gene} NON disponibile - rischio non escludibile")
        
        # Determina risk level
        if is_contraindicated:
            risk_level = RiskLevel.CONTRAINDICATED
        elif max_tox >= 80 or requires_action:
            risk_level = RiskLevel.HIGH
        elif max_tox >= 50:
            risk_level = RiskLevel.MODERATE
        elif max_tox >= 20 or max_eff >= 50:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.SAFE
        
        # Raccomandazione primaria
        if is_contraindicated:
            primary_rec = f"❌ {drug} CONTROINDICATO - {recommendations[0] if recommendations else 'Usare alternativa'}"
        elif risk_level == RiskLevel.HIGH:
            primary_rec = f"⚠️ ALTO RISCHIO - {recommendations[0] if recommendations else 'Richiede aggiustamento'}"
        else:
            primary_rec = recommendations[0] if recommendations else "Monitoraggio standard"
        
        return DrugRiskAssessment(
            drug=drug,
            risk_level=risk_level,
            toxicity_risk=max_tox,
            efficacy_risk=max_eff,
            gene_interactions=interactions,
            phenotypes=phenotypes,
            primary_recommendation=primary_rec,
            alternative_drugs=alternatives,
            dosing_adjustment=dosing,
            monitoring_required=monitoring,
            evidence_level=best_evidence,
            warnings=warnings
        )
    
    def assess_therapy_regimen(self, 
                               therapy: str,
                               patient_data: Dict[str, Any]) -> List[DrugRiskAssessment]:
        """
        Valuta un regime terapeutico completo (es. "FOLFOX").
        
        Args:
            therapy: Nome regime o lista farmaci
            patient_data: Dati paziente
            
        Returns:
            Lista di DrugRiskAssessment per ogni farmaco
        """
        # Espandi regimi comuni
        drugs = self._expand_regimen(therapy)
        
        assessments = []
        for drug in drugs:
            assessment = self.assess_drug_risk(drug, patient_data)
            assessments.append(assessment)
        
        # Ordina per rischio (più alto prima)
        assessments.sort(key=lambda x: x.risk_level.priority)
        
        return assessments

    def _expand_regimen(self, therapy: str) -> List[str]:
        """Espande nomi regime in singoli farmaci"""

        # Mapping regimi comuni
        regimens = {
            'folfox': ['5-FU', 'Leucovorin', 'Oxaliplatino'],
            'folfiri': ['5-FU', 'Leucovorin', 'Irinotecano'],
            'folfoxiri': ['5-FU', 'Leucovorin', 'Oxaliplatino', 'Irinotecano'],
            'folfirinox': ['5-FU', 'Leucovorin', 'Oxaliplatino', 'Irinotecano'],
            'xelox': ['Capecitabina', 'Oxaliplatino'],
            'capox': ['Capecitabina', 'Oxaliplatino'],
            'xeliri': ['Capecitabina', 'Irinotecano'],
            'r-chop': ['Rituximab', 'Ciclofosfamide', 'Doxorubicina', 'Vincristina', 'Prednisone'],
            'abvd': ['Doxorubicina', 'Bleomicina', 'Vinblastina', 'Dacarbazina'],
        }

        therapy_lower = therapy.lower()

        # Cerca regime noto
        for regimen, drugs in regimens.items():
            if regimen in therapy_lower:
                return drugs

        # Se non è un regime, considera come singolo farmaco
        # o lista separata da + o ,
        if '+' in therapy:
            return [d.strip() for d in therapy.split('+')]
        elif ',' in therapy:
            return [d.strip() for d in therapy.split(',')]

        # NUOVO: Estrai nome farmaco da formato "NomeFarmaco (descrizione)"
        # Es: "Lorlatinib (ALK 3rd-gen)" -> "Lorlatinib"
        if '(' in therapy:
            drug_name = therapy.split('(')[0].strip()
            return [drug_name]

        return [therapy]
    
    def get_safe_alternatives(self, 
                              drug: str,
                              patient_data: Dict[str, Any]) -> List[str]:
        """
        Suggerisce alternative più sicure per un farmaco ad alto rischio.
        """
        # Questo è un placeholder - in produzione avrebbe un database di alternative
        alternatives = {
            '5-fu': ['Raltitrexed', 'Gemcitabina'],
            'capecitabina': ['Raltitrexed', 'Gemcitabina'],
            'irinotecano': ['Topotecan (con cautela)'],
            'tamoxifene': ['Anastrozolo', 'Letrozolo', 'Exemestane'],
            'mercaptopurina': ['Alternative da valutare con ematologo'],
            'rasburicase': ['Allopurinolo', 'Febuxostat']
        }
        
        drug_lower = drug.lower()
        for key, alts in alternatives.items():
            if key in drug_lower:
                return alts
        
        return []


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DRUG INTERACTION ENGINE - TEST")
    print("=" * 60)
    
    engine = DrugInteractionEngine()
    
    # Paziente test con deficit DPYD
    test_patient = {
        'baseline': {
            'patient_id': 'TEST-PGX-001',
            'current_therapy': 'FOLFOX',
            'genetics': {
                'tp53_status': 'mutated',
                'dpyd_status': '*1/*2A heterozygous',
                'ugt1a1_status': '*1/*28'
            }
        }
    }
    
    print("\n👤 Profilo PGx paziente:")
    profile = engine.analyze_patient(test_patient)
    print(f"   ID: {profile.patient_id}")
    print(f"   Geni testati: {profile.genes_tested}")
    print(f"   Geni NON testati: {profile.genes_not_tested}")
    for gene, pheno in profile.phenotypes.items():
        print(f"   {gene}: {pheno.phenotype.value}")
    
    print("\n💊 Assessment 5-FU:")
    assessment = engine.assess_drug_risk("5-FU", test_patient)
    print(f"   Risk Level: {assessment.risk_level.value}")
    print(f"   Toxicity Risk: {assessment.toxicity_risk}%")
    print(f"   Recommendation: {assessment.primary_recommendation}")
    for warn in assessment.warnings:
        print(f"   {warn}")
    
    print("\n💊 Assessment regime FOLFOX:")
    assessments = engine.assess_therapy_regimen("FOLFOX", test_patient)
    for a in assessments:
        print(f"   {a.drug}: {a.risk_level.value} (Tox: {a.toxicity_risk}%)")
    
    print("\n✅ DrugInteractionEngine pronto!")
