"""
SENTINEL FARMACOGENOMICA MODULE
================================
Predizione tossicità e inefficacia farmaci basata su varianti genetiche.
Integra PharmGKB, linee guida CPIC/DPWG per raccomandazioni actionable.

Geni chiave oncologia:
- DPYD → 5-FU, Capecitabina (tossicità fatale)
- UGT1A1 → Irinotecano (neutropenia severa)
- CYP2D6 → Tamoxifene (inefficacia)
- TPMT/NUDT15 → Mercaptopurina, Azatioprina
- CYP3A4 → TKI (metabolismo variabile)

Components:
- pharmgkb_database: Database interazioni gene-farmaco
- pgx_extractor: Estrazione varianti PGx da dati NGS
- metabolizer_classifier: Classificazione fenotipo metabolizzatore
- drug_interaction: Calcolo rischio tossicità/inefficacia
- pgx_recommender: Raccomandazioni CPIC/DPWG
- pgx_alert_engine: Filtro alert clinicamente rilevanti
"""

from .pharmgkb_database import PharmGKBDatabase
from .pgx_extractor import PGxExtractor
from .metabolizer_classifier import MetabolizerClassifier
from .drug_interaction import DrugInteractionEngine
from .pgx_recommender import PGxRecommender
from .pgx_alert_engine import PGxAlertEngine

__version__ = "1.0.0"
__all__ = [
    "PharmGKBDatabase",
    "PGxExtractor",
    "MetabolizerClassifier",
    "DrugInteractionEngine",
    "PGxRecommender",
    "PGxAlertEngine"
]
