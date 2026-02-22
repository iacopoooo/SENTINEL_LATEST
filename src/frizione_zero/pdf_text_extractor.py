"""
PDF TEXT EXTRACTOR - Estrazione testo da PDF
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path

class ExtractionMethod(Enum):
    NATIVE = "native"
    OCR = "ocr"
    HYBRID = "hybrid"

@dataclass
class ExtractionResult:
    text: str
    method: ExtractionMethod
    confidence: float
    pages: int
    warnings: list

class PDFTextExtractor:
    def __init__(self, ocr_engine: str = "tesseract"):
        self.ocr_engine = ocr_engine
    
    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Estrae testo da PDF, auto-detect metodo migliore"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF non trovato: {pdf_path}")
        
        # Prova estrazione nativa
        text, is_native = self._extract_native(pdf_path)
        
        if is_native and len(text.strip()) > 100:
            return ExtractionResult(
                text=text, method=ExtractionMethod.NATIVE,
                confidence=0.95, pages=1, warnings=[]
            )
        
        # Fallback OCR
        text = self._extract_ocr(pdf_path)
        return ExtractionResult(
            text=text, method=ExtractionMethod.OCR,
            confidence=0.75, pages=1, warnings=["Used OCR extraction"]
        )
    
    def _extract_native(self, pdf_path: Path) -> tuple:
        """Estrazione testo nativo con PyMuPDF"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text, True
        except ImportError:
            return "", False
        except Exception:
            return "", False
    
    def _extract_ocr(self, pdf_path: Path) -> str:
        """Estrazione OCR con Tesseract"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            images = convert_from_path(pdf_path)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img, lang='ita+eng')
            return text
        except ImportError:
            return "[OCR non disponibile - installa pytesseract e pdf2image]"
        except Exception as e:
            return f"[Errore OCR: {e}]"
