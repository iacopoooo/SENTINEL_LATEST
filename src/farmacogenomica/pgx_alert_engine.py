"""
PGX ALERT ENGINE - SENTINEL FARMACOGENOMICA
============================================
Filtra e prioritizza alert PGx per evitare alert fatigue.
Solo alert clinicamente actionable con evidenza ≥2A.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from .pharmgkb_database import PharmGKBDatabase, EvidenceLevel
from .drug_interaction import DrugInteractionEngine, DrugRiskAssessment, RiskLevel, PatientPGxProfile
from .pgx_recommender import PGxRecommender, ClinicalRecommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Priorità alert"""
    CRITICAL = 1    # Azione immediata richiesta (life-threatening)
    HIGH = 2        # Azione richiesta prima del trattamento
    MODERATE = 3    # Considerare modifica
    LOW = 4         # Informativo
    
    @property
    def color(self) -> str:
        colors = {
            AlertPriority.CRITICAL: "#B71C1C",
            AlertPriority.HIGH: "#D32F2F",
            AlertPriority.MODERATE: "#FF9800",
            AlertPriority.LOW: "#2196F3"
        }
        return colors.get(self, "#9E9E9E")
    
    @property
    def icon(self) -> str:
        icons = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MODERATE: "⚡",
            AlertPriority.LOW: "ℹ️"
        }
        return icons.get(self, "•")


@dataclass
class PGxAlert:
    """Singolo alert farmacogenomico"""
    alert_id: str
    priority: AlertPriority
    title: str
    message: str
    
    # Dettagli clinici
    drug: str
    gene: str
    phenotype: str
    risk_type: str  # "toxicity", "efficacy", "both"
    
    # Raccomandazione
    recommendation: Optional[ClinicalRecommendation] = None
    action_required: str = ""
    
    # Metadati
    evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_2A
    source: str = "SENTINEL PGx"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Flag
    is_life_threatening: bool = False
    requires_immediate_action: bool = False
    acknowledged: bool = False


@dataclass
class AlertSummary:
    """Riepilogo alert per paziente"""
    patient_id: str
    total_alerts: int
    critical_count: int
    high_count: int
    moderate_count: int
    low_count: int
    alerts: List[PGxAlert]
    
    # Status complessivo
    has_contraindications: bool = False
    therapy_safe: bool = True
    requires_review: bool = False
    
    # Terapia analizzata
    therapy_analyzed: str = ""
    genes_tested: List[str] = field(default_factory=list)
    genes_not_tested: List[str] = field(default_factory=list)


class PGxAlertEngine:
    """
    Engine per generazione e filtro alert PGx.
    """
    
    # Soglia minima evidenza per generare alert
    MIN_EVIDENCE_LEVEL = EvidenceLevel.LEVEL_2A
    
    # Soglia rischio per generare alert
    MIN_TOXICITY_RISK = 30
    MIN_EFFICACY_RISK = 50
    
    def __init__(self):
        self.database = PharmGKBDatabase()
        self.interaction_engine = DrugInteractionEngine(self.database)
        self.recommender = PGxRecommender()
        self._alert_counter = 0
    
    def _generate_alert_id(self) -> str:
        self._alert_counter += 1
        return f"PGX-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
    
    def analyze_patient_therapy(self, 
                                 patient_data: Dict[str, Any],
                                 therapy: Optional[str] = None) -> AlertSummary:
        """
        Analizza paziente e terapia, genera alert filtrati.
        
        Args:
            patient_data: Dati paziente SENTINEL
            therapy: Terapia da analizzare (default: current_therapy dal baseline)
            
        Returns:
            AlertSummary con tutti gli alert rilevanti
        """
        baseline = patient_data.get('baseline', patient_data)
        patient_id = baseline.get('patient_id', 'Unknown')
        
        # Determina terapia
        if therapy is None:
            therapy = baseline.get('current_therapy', '')
        
        if not therapy:
            return AlertSummary(
                patient_id=patient_id,
                total_alerts=0,
                critical_count=0, high_count=0, moderate_count=0, low_count=0,
                alerts=[],
                therapy_safe=True,
                therapy_analyzed="Nessuna terapia specificata"
            )
        
        # Analizza profilo PGx
        profile = self.interaction_engine.analyze_patient(patient_data)
        
        # Valuta ogni farmaco nel regime
        assessments = self.interaction_engine.assess_therapy_regimen(therapy, patient_data)
        
        # Genera alert
        alerts = []
        for assessment in assessments:
            drug_alerts = self._generate_alerts_for_assessment(assessment, profile)
            alerts.extend(drug_alerts)
        
        # Aggiungi alert per geni non testati
        missing_gene_alerts = self._generate_missing_gene_alerts(profile, therapy)
        alerts.extend(missing_gene_alerts)
        
        # Filtra per evidenza
        alerts = self._filter_by_evidence(alerts)
        
        # Ordina per priorità
        alerts.sort(key=lambda x: x.priority.value)
        
        # Calcola conteggi
        critical = sum(1 for a in alerts if a.priority == AlertPriority.CRITICAL)
        high = sum(1 for a in alerts if a.priority == AlertPriority.HIGH)
        moderate = sum(1 for a in alerts if a.priority == AlertPriority.MODERATE)
        low = sum(1 for a in alerts if a.priority == AlertPriority.LOW)
        
        # Determina status
        has_contraindications = any(a.is_life_threatening for a in alerts)
        therapy_safe = critical == 0 and high == 0
        requires_review = len(alerts) > 0
        
        return AlertSummary(
            patient_id=patient_id,
            total_alerts=len(alerts),
            critical_count=critical,
            high_count=high,
            moderate_count=moderate,
            low_count=low,
            alerts=alerts,
            has_contraindications=has_contraindications,
            therapy_safe=therapy_safe,
            requires_review=requires_review,
            therapy_analyzed=therapy,
            genes_tested=profile.genes_tested,
            genes_not_tested=profile.genes_not_tested
        )
    
    def _generate_alerts_for_assessment(self, 
                                         assessment: DrugRiskAssessment,
                                         profile: PatientPGxProfile) -> List[PGxAlert]:
        """Genera alert per un singolo drug assessment"""
        alerts = []
        
        # Skip se rischio troppo basso
        if assessment.toxicity_risk < self.MIN_TOXICITY_RISK and \
           assessment.efficacy_risk < self.MIN_EFFICACY_RISK:
            return alerts
        
        # Determina priorità
        if assessment.risk_level == RiskLevel.CONTRAINDICATED:
            priority = AlertPriority.CRITICAL
        elif assessment.risk_level == RiskLevel.HIGH:
            priority = AlertPriority.HIGH
        elif assessment.risk_level == RiskLevel.MODERATE:
            priority = AlertPriority.MODERATE
        else:
            priority = AlertPriority.LOW
        
        # Genera alert per ogni interazione rilevante
        for inter in assessment.gene_interactions:
            # Cerca raccomandazione
            rec = None
            for phenotype in assessment.phenotypes:
                if phenotype.gene == inter.gene:
                    rec = self.recommender.get_recommendation(
                        assessment.drug, inter.gene, phenotype.phenotype
                    )
                    break
            
            # Determina tipo rischio
            if inter.toxicity_risk > 0 and inter.efficacy_risk > 0:
                risk_type = "both"
            elif inter.toxicity_risk > 0:
                risk_type = "toxicity"
            else:
                risk_type = "efficacy"
            
            alert = PGxAlert(
                alert_id=self._generate_alert_id(),
                priority=priority,
                title=f"{inter.gene}: {inter.phenotype}",
                message=inter.effect,
                drug=assessment.drug,
                gene=inter.gene,
                phenotype=inter.phenotype,
                risk_type=risk_type,
                recommendation=rec,
                action_required=inter.recommendation,
                evidence_level=inter.evidence_level,
                is_life_threatening=inter.is_life_threatening,
                requires_immediate_action=inter.requires_immediate_action
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_missing_gene_alerts(self, 
                                       profile: PatientPGxProfile,
                                       therapy: str) -> List[PGxAlert]:
        """Genera alert per geni critici non testati"""
        alerts = []
        therapy_lower = therapy.lower()
        
        # Mapping farmaco -> geni critici
        critical_mappings = {
            '5-fu': ['DPYD'],
            'capecitabina': ['DPYD'],
            'irinotecano': ['UGT1A1'],
            'tamoxifene': ['CYP2D6'],
            'mercaptopurina': ['TPMT', 'NUDT15'],
            'azatioprina': ['TPMT', 'NUDT15'],
            'rasburicase': ['G6PD'],
        }
        
        for drug, genes in critical_mappings.items():
            if drug in therapy_lower:
                for gene in genes:
                    if gene in profile.genes_not_tested:
                        # Check se è life-threatening
                        interactions = self.database.get_interactions_for_gene(gene)
                        is_critical = any(i.is_life_threatening for i in interactions 
                                         if drug in i.drug.lower())
                        
                        alert = PGxAlert(
                            alert_id=self._generate_alert_id(),
                            priority=AlertPriority.HIGH if is_critical else AlertPriority.MODERATE,
                            title=f"Test {gene} Non Disponibile",
                            message=f"Il test {gene} è raccomandato prima di {drug.upper()} ma non è stato eseguito.",
                            drug=drug.upper(),
                            gene=gene,
                            phenotype="Non testato",
                            risk_type="unknown",
                            action_required=f"Eseguire test {gene} prima di iniziare {drug.upper()}",
                            evidence_level=EvidenceLevel.LEVEL_1A if is_critical else EvidenceLevel.LEVEL_2A,
                            is_life_threatening=is_critical,
                            requires_immediate_action=is_critical
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _filter_by_evidence(self, alerts: List[PGxAlert]) -> List[PGxAlert]:
        """Filtra alert per livello evidenza minimo"""
        return [a for a in alerts 
                if a.evidence_level.priority <= self.MIN_EVIDENCE_LEVEL.priority]
    
    def format_alert_card(self, alert: PGxAlert) -> str:
        """Formatta alert come card HTML"""
        return f"""
        <div style="border-left: 4px solid {alert.priority.color}; 
                    padding: 15px; margin: 10px 0; 
                    background-color: rgba(0,0,0,0.2); border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {alert.priority.color};">
                    {alert.priority.icon} {alert.title}
                </h4>
                <span style="background: {alert.priority.color}; color: white; 
                            padding: 2px 8px; border-radius: 10px; font-size: 12px;">
                    {alert.priority.name}
                </span>
            </div>
            <p style="margin: 10px 0;"><strong>{alert.drug}</strong>: {alert.message}</p>
            <p style="margin: 5px 0; color: #FFD54F;"><strong>Azione:</strong> {alert.action_required}</p>
            <small style="color: #888;">
                {alert.gene} | {alert.evidence_level.value} | {alert.source}
            </small>
        </div>
        """
    
    def format_summary_html(self, summary: AlertSummary) -> str:
        """Formatta riepilogo come HTML"""
        
        # Status badge
        if summary.has_contraindications:
            status_color = "#B71C1C"
            status_text = "🚨 CONTROINDICAZIONI PRESENTI"
        elif not summary.therapy_safe:
            status_color = "#D32F2F"
            status_text = "⚠️ RICHIEDE REVISIONE"
        elif summary.requires_review:
            status_color = "#FF9800"
            status_text = "⚡ ALERT PRESENTI"
        else:
            status_color = "#4CAF50"
            status_text = "✅ NESSUN ALERT PGx"
        
        html = f"""
        <div style="border: 2px solid {status_color}; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h3 style="color: {status_color}; margin-top: 0;">{status_text}</h3>
            <p>Terapia analizzata: <strong>{summary.therapy_analyzed}</strong></p>
            
            <div style="display: flex; gap: 20px; margin: 15px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 24px; color: #B71C1C;">{summary.critical_count}</div>
                    <small>Critici</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; color: #D32F2F;">{summary.high_count}</div>
                    <small>Alti</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; color: #FF9800;">{summary.moderate_count}</div>
                    <small>Moderati</small>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 24px; color: #2196F3;">{summary.low_count}</div>
                    <small>Info</small>
                </div>
            </div>
        """
        
        if summary.genes_not_tested:
            html += f"""
            <div style="background: #FF980033; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong>⚠️ Geni non testati:</strong> {', '.join(summary.genes_not_tested)}
            </div>
            """
        
        html += "</div>"
        
        # Alert cards
        for alert in summary.alerts:
            html += self.format_alert_card(alert)
        
        return html


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PGX ALERT ENGINE - TEST")
    print("=" * 60)
    
    engine = PGxAlertEngine()
    
    # Test paziente con DPYD deficit
    test_patient = {
        'baseline': {
            'patient_id': 'ALERT-TEST-001',
            'current_therapy': 'FOLFOX (5-FU + Oxaliplatino)',
            'genetics': {
                'dpyd_status': '*1/*2A heterozygous',
                'ugt1a1_status': '*28/*28'
            }
        }
    }
    
    print("\n🔬 Analisi paziente con FOLFOX:")
    summary = engine.analyze_patient_therapy(test_patient)
    
    print(f"\n   Paziente: {summary.patient_id}")
    print(f"   Terapia: {summary.therapy_analyzed}")
    print(f"   Alert totali: {summary.total_alerts}")
    print(f"   - Critici: {summary.critical_count}")
    print(f"   - Alti: {summary.high_count}")
    print(f"   - Moderati: {summary.moderate_count}")
    print(f"   Therapy safe: {summary.therapy_safe}")
    print(f"   Controindicazioni: {summary.has_contraindications}")
    
    print("\n📋 Alert generati:")
    for alert in summary.alerts:
        print(f"\n   {alert.priority.icon} [{alert.priority.name}] {alert.title}")
        print(f"      Drug: {alert.drug}")
        print(f"      Message: {alert.message[:60]}...")
        print(f"      Action: {alert.action_required[:60]}...")
    
    print("\n✅ PGxAlertEngine pronto!")
