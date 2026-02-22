"""
PDF INGESTION CONFIG - Configurazione pipeline
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path

class OCREngine(Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    HYBRID = "hybrid"

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    NONE = "none"

@dataclass
class IngestionConfig:
    ocr_engine: OCREngine = OCREngine.TESSERACT
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-3-haiku-20240307"
    confidence_threshold: float = 0.7
    output_dir: Path = field(default_factory=lambda: Path("data/patients"))
    upload_dir: Path = field(default_factory=lambda: Path("data/uploads"))
    
LAB_PATTERNS = {
    "ldh": r"LDH[:\s]*(\d+(?:\.\d+)?)",
    "cea": r"CEA[:\s]*(\d+(?:\.\d+)?)",
    "tmb": r"TMB[:\s]*(\d+(?:\.\d+)?)",
}

GENE_PATTERNS = ["TP53", "KRAS", "EGFR", "ALK", "BRAF", "PIK3CA", "STK11", "KEAP1", "MET", "ERBB2", "ROS1", "RET", "NTRK"]
