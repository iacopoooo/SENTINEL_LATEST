"""
PROMETHEUS — Pre-cancer Epistatic Discovery Engine for SENTINEL.
================================================================
Scansiona il database pazienti e scopre interazioni epistatiche
nascoste (gene×biomarker, gene×gene, triple) che i singoli marker
non catturano. Le regole scoperte vengono iniettate in ORACLE.

Moduli:
  - feature_engineering: Estrae features dai JSON pazienti
  - epistatic_engine:    Discovery a 4 fasi (A/B/C/D) + FDR
  - oracle_bridge:       Ponte regole → Evidence per ORACLE
"""

__version__ = "1.0.0"
