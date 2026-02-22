"""
PGX RECOMMENDER - SENTINEL FARMACOGENOMICA
===========================================
Genera raccomandazioni cliniche basate su linee guida CPIC e DPWG.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .pharmgkb_database import EvidenceLevel
from .metabolizer_classifier import MetabolizerPhenotype, PhenotypeResult
from .drug_interaction import DrugRiskAssessment, RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    OPTIONAL = "optional"


@dataclass
class ClinicalRecommendation:
    drug: str
    gene: str
    phenotype: str
    action: str
    action_detail: str
    dose_adjustment: Optional[str] = None
    dose_percentage: Optional[int] = None
    alternatives: List[str] = field(default_factory=list)
    monitoring: List[str] = field(default_factory=list)
    monitoring_frequency: Optional[str] = None
    strength: RecommendationStrength = RecommendationStrength.MODERATE
    guideline_source: str = "CPIC"
    evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_2A
    black_box_warning: bool = False
    fda_label: bool = False
    pmid: Optional[str] = None


class PGxRecommender:
    def __init__(self):
        self.recommendations_db = self._build_recommendations_db()
    
    def _build_recommendations_db(self) -> Dict:
        db = {}
        
        # DPYD + 5-FU
        db['5-fu'] = {
            'DPYD_PM': ClinicalRecommendation(
                drug="5-FU", gene="DPYD", phenotype="Poor Metabolizer",
                action="avoid",
                action_detail="CONTROINDICATO. Non somministrare fluoropirimidine.",
                alternatives=["Raltitrexed", "Gemcitabina"],
                strength=RecommendationStrength.STRONG,
                evidence_level=EvidenceLevel.LEVEL_1A,
                black_box_warning=True, fda_label=True
            ),
            'DPYD_IM': ClinicalRecommendation(
                drug="5-FU", gene="DPYD", phenotype="Intermediate Metabolizer",
                action="reduce_dose",
                action_detail="Ridurre dose iniziale del 50%. Titolare in base a tolleranza.",
                dose_adjustment="50% della dose standard", dose_percentage=50,
                monitoring=["Emocromo settimanale", "Mucositi", "Diarrea"],
                strength=RecommendationStrength.STRONG,
                evidence_level=EvidenceLevel.LEVEL_1A, fda_label=True
            )
        }
        db['capecitabina'] = db['5-fu']
        
        # UGT1A1 + Irinotecano
        db['irinotecano'] = {
            'UGT1A1_PM': ClinicalRecommendation(
                drug="Irinotecano", gene="UGT1A1", phenotype="Poor Metabolizer (*28/*28)",
                action="reduce_dose",
                action_detail="Ridurre dose iniziale del 30%. Alto rischio neutropenia.",
                dose_adjustment="70% della dose standard", dose_percentage=70,
                monitoring=["Emocromo prima di ogni ciclo", "Diarrea", "Bilirubina"],
                strength=RecommendationStrength.STRONG,
                evidence_level=EvidenceLevel.LEVEL_1A, fda_label=True
            ),
            'UGT1A1_IM': ClinicalRecommendation(
                drug="Irinotecano", gene="UGT1A1", phenotype="Intermediate Metabolizer (*1/*28)",
                action="monitor",
                action_detail="Dose standard con monitoraggio stretto.",
                monitoring=["Emocromo", "Diarrea"],
                strength=RecommendationStrength.MODERATE,
                evidence_level=EvidenceLevel.LEVEL_2A
            )
        }
        
        # CYP2D6 + Tamoxifene
        db['tamoxifene'] = {
            'CYP2D6_PM': ClinicalRecommendation(
                drug="Tamoxifene", gene="CYP2D6", phenotype="Poor Metabolizer",
                action="alternative",
                action_detail="Considerare inibitore aromatasi (se post-menopausa).",
                alternatives=["Anastrozolo", "Letrozolo", "Exemestane"],
                strength=RecommendationStrength.MODERATE,
                evidence_level=EvidenceLevel.LEVEL_1B
            )
        }
        
        # TPMT + Mercaptopurina
        db['mercaptopurina'] = {
            'TPMT_PM': ClinicalRecommendation(
                drug="Mercaptopurina", gene="TPMT", phenotype="Poor Metabolizer",
                action="reduce_dose",
                action_detail="Ridurre al 10% della dose, 3 volte/settimana.",
                dose_percentage=10,
                monitoring=["Emocromo 2x/settimana"],
                strength=RecommendationStrength.STRONG,
                evidence_level=EvidenceLevel.LEVEL_1A, fda_label=True
            )
        }
        
        # G6PD + Rasburicase
        db['rasburicase'] = {
            'G6PD_PM': ClinicalRecommendation(
                drug="Rasburicase", gene="G6PD", phenotype="G6PD Deficient",
                action="avoid",
                action_detail="CONTROINDICATO. Rischio emolisi severa/fatale.",
                alternatives=["Allopurinolo", "Febuxostat"],
                strength=RecommendationStrength.STRONG,
                evidence_level=EvidenceLevel.LEVEL_1A,
                black_box_warning=True, fda_label=True
            )
        }
        
        return db
    
    def get_recommendation(self, drug: str, gene: str, phenotype: MetabolizerPhenotype) -> Optional[ClinicalRecommendation]:
        drug_lower = drug.lower()
        if drug_lower not in self.recommendations_db:
            return None
        
        drug_recs = self.recommendations_db[drug_lower]
        key = f"{gene}_{phenotype.abbreviation}"
        
        if key in drug_recs:
            return drug_recs[key]
        
        for rec_key, rec in drug_recs.items():
            if gene in rec_key and phenotype.abbreviation in rec_key:
                return rec
        
        return None
    
    def format_recommendation_text(self, rec: ClinicalRecommendation) -> str:
        icon = "❌" if rec.action == "avoid" else "⚠️" if rec.action == "reduce_dose" else "ℹ️"
        lines = [
            f"{icon} {rec.drug} + {rec.gene} ({rec.phenotype})",
            f"Raccomandazione: {rec.action_detail}"
        ]
        if rec.dose_adjustment:
            lines.append(f"Dose: {rec.dose_adjustment}")
        if rec.alternatives:
            lines.append(f"Alternative: {', '.join(rec.alternatives)}")
        if rec.monitoring:
            lines.append(f"Monitoraggio: {', '.join(rec.monitoring)}")
        lines.append(f"Fonte: {rec.guideline_source} | Evidenza: {rec.evidence_level.value}")
        if rec.black_box_warning:
            lines.append("⬛ BLACK BOX WARNING FDA")
        return "\n".join(lines)
    
    def format_recommendation_html(self, rec: ClinicalRecommendation) -> str:
        color = "#B71C1C" if rec.action == "avoid" else "#E65100" if rec.action == "reduce_dose" else "#1565C0"
        icon = "🚫" if rec.action == "avoid" else "⚠️" if rec.action == "reduce_dose" else "ℹ️"
        
        html = f"""
        <div style="border: 2px solid {color}; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h4 style="color: {color};">{icon} {rec.drug} + {rec.gene} ({rec.phenotype})</h4>
            <p><strong>Raccomandazione:</strong> {rec.action_detail}</p>
        """
        if rec.dose_adjustment:
            html += f"<p><strong>Dose:</strong> {rec.dose_adjustment}</p>"
        if rec.alternatives:
            html += f"<p><strong>Alternative:</strong> {', '.join(rec.alternatives)}</p>"
        if rec.monitoring:
            html += f"<p><strong>Monitoraggio:</strong> {', '.join(rec.monitoring)}</p>"
        html += f"""
            <small>{rec.guideline_source} | {rec.evidence_level.value}
            {"| ⬛ BLACK BOX" if rec.black_box_warning else ""}</small>
        </div>
        """
        return html


if __name__ == "__main__":
    print("PGX RECOMMENDER - TEST")
    recommender = PGxRecommender()
    
    rec = recommender.get_recommendation("5-FU", "DPYD", MetabolizerPhenotype.POOR)
    if rec:
        print(recommender.format_recommendation_text(rec))
    
    print("\n✅ PGxRecommender pronto!")
