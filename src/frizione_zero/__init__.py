"""
SENTINEL FRIZIONE ZERO - PDF Ingestion Module
==============================================
Caricamento automatico referti NGS in formato PDF.
Pipeline: PDF → OCR → LLM → Validazione → SENTINEL JSON
"""

from .pdf_ingestion_config import IngestionConfig, OCREngine, LLMProvider
from .pdf_text_extractor import PDFTextExtractor, ExtractionResult, ExtractionMethod
from .llm_data_extractor import LLMDataExtractor, ExtractedNGSData
from .sentinel_schema_mapper import SentinelSchemaMapper, SentinelPatient
from .validation_engine import ValidationEngine, ValidationResult, ValidationIssue, ValidationSeverity
from .patient_database import PatientDatabase
from .ingestion_pipeline import IngestionPipeline, IngestionResult, IngestionStatus

__version__ = "1.0.0"
__all__ = [
    # Config
    "IngestionConfig", 
    "OCREngine", 
    "LLMProvider",
    # Extractor
    "PDFTextExtractor", 
    "ExtractionResult",
    "ExtractionMethod",
    # LLM
    "LLMDataExtractor", 
    "ExtractedNGSData",
    # Mapper
    "SentinelSchemaMapper", 
    "SentinelPatient",
    # Validation
    "ValidationEngine", 
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Database
    "PatientDatabase",
    # Pipeline
    "IngestionPipeline", 
    "IngestionResult",
    "IngestionStatus",
]
