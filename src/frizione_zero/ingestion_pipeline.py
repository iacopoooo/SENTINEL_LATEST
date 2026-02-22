"""
INGESTION PIPELINE - Pipeline completa PDF → SENTINEL
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .pdf_text_extractor import PDFTextExtractor, ExtractionResult
from .llm_data_extractor import LLMDataExtractor, ExtractedNGSData
from .sentinel_schema_mapper import SentinelSchemaMapper, SentinelPatient
from .validation_engine import ValidationEngine, ValidationResult
from .patient_database import PatientDatabase

class IngestionStatus(Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    PARSING = "parsing"
    VALIDATING = "validating"
    SAVING = "saving"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class IngestionResult:
    status: IngestionStatus
    patient_id: Optional[str] = None
    patient_data: Optional[Dict[str, Any]] = None
    validation: Optional[ValidationResult] = None
    extraction_confidence: float = 0.0
    processing_time_ms: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class IngestionPipeline:
    def __init__(self, data_dir: Path = None, llm_provider: str = "anthropic"):
        self.data_dir = Path(data_dir) if data_dir else Path("data/patients")
        self.extractor = PDFTextExtractor()
        self.llm_extractor = LLMDataExtractor(provider=llm_provider)
        self.mapper = SentinelSchemaMapper()
        self.validator = ValidationEngine()
        self.database = PatientDatabase(self.data_dir)
    
    def process_pdf(self, pdf_path: Path, patient_id: Optional[str] = None, 
                    merge_existing: bool = True) -> IngestionResult:
        """Pipeline completa: PDF → Estrazione → Parsing → Validazione → Salvataggio"""
        start_time = datetime.now()
        result = IngestionResult(status=IngestionStatus.PENDING)
        
        try:
            # 1. Estrazione testo
            result.status = IngestionStatus.EXTRACTING
            extraction = self.extractor.extract(pdf_path)
            
            if not extraction.text or len(extraction.text.strip()) < 50:
                result.status = IngestionStatus.FAILED
                result.errors.append("Testo insufficiente estratto dal PDF")
                return result
            
            result.warnings.extend(extraction.warnings)
            
            # 2. Parsing con LLM
            result.status = IngestionStatus.PARSING
            ngs_data = self.llm_extractor.extract(extraction.text)
            result.extraction_confidence = ngs_data.confidence
            
            # 3. Mapping a SENTINEL
            sentinel_patient = self.mapper.map_to_sentinel(ngs_data, patient_id)
            patient_data = sentinel_patient.to_sentinel_json()
            
            # Aggiungi metadati
            patient_data["baseline"]["source_pdf"] = str(pdf_path.name)
            patient_data["baseline"]["extraction_method"] = extraction.method.value
            patient_data["baseline"]["extraction_confidence"] = ngs_data.confidence
            
            result.patient_id = sentinel_patient.patient_id
            result.patient_data = patient_data
            
            # 4. Validazione
            result.status = IngestionStatus.VALIDATING
            validation = self.validator.validate(patient_data)
            result.validation = validation
            
            for issue in validation.warnings:
                result.warnings.append(f"{issue.field}: {issue.message}")
            
            if not validation.is_valid:
                for issue in validation.errors:
                    result.errors.append(f"{issue.field}: {issue.message}")
                # Continua comunque se ci sono solo errori minori
                if len(validation.errors) > 3:
                    result.status = IngestionStatus.FAILED
                    return result
            
            # 5. Salvataggio
            result.status = IngestionStatus.SAVING
            
            if merge_existing and self.database.patient_exists(sentinel_patient.patient_id):
                patient_data = self.database.merge_with_existing(
                    sentinel_patient.patient_id, patient_data
                )
                result.patient_data = patient_data
            
            self.database.save_patient(patient_data, overwrite=True)
            
            # Success
            result.status = IngestionStatus.SUCCESS
            
        except FileNotFoundError as e:
            result.status = IngestionStatus.FAILED
            result.errors.append(f"File non trovato: {e}")
        except Exception as e:
            result.status = IngestionStatus.FAILED
            result.errors.append(f"Errore pipeline: {str(e)}")
        
        # Tempo di processing
        result.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return result
    
    def process_batch(self, pdf_paths: List[Path]) -> List[IngestionResult]:
        """Processa multipli PDF"""
        return [self.process_pdf(p) for p in pdf_paths]
    
    def get_processing_stats(self, results: List[IngestionResult]) -> Dict[str, Any]:
        """Statistiche batch processing"""
        return {
            "total": len(results),
            "success": len([r for r in results if r.status == IngestionStatus.SUCCESS]),
            "failed": len([r for r in results if r.status == IngestionStatus.FAILED]),
            "avg_confidence": sum(r.extraction_confidence for r in results) / len(results) if results else 0,
            "avg_time_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0,
        }
