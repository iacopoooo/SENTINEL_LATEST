"""
LLM DATA EXTRACTOR - Parsing intelligente con LLM
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import re

@dataclass
class ExtractedNGSData:
    patient_id: Optional[str] = None
    sample_date: Optional[str] = None
    sample_type: Optional[str] = None
    genes: Dict[str, Dict] = field(default_factory=dict)
    biomarkers: Dict[str, float] = field(default_factory=dict)
    therapy_suggestions: List[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0

class LLMDataExtractor:
    EXTRACTION_PROMPT = '''Sei un esperto di oncologia. Estrai i dati dal seguente referto NGS in formato JSON.

REFERTO:
{text}

Rispondi SOLO con JSON valido nel formato:
{{
  "patient_id": "string o null",
  "sample_date": "YYYY-MM-DD o null",
  "sample_type": "tissue/liquid_biopsy/unknown",
  "genes": {{
    "GENE_NAME": {{
      "status": "mutated/wild-type/amplified/deleted",
      "mutation": "es. p.G12C",
      "vaf": numero o null,
      "clinical_significance": "pathogenic/likely_pathogenic/vus/benign"
    }}
  }},
  "biomarkers": {{
    "tmb_score": numero o null,
    "msi_status": "MSI-H/MSS/unknown",
    "pd_l1_tps": numero o null
  }},
  "therapy_suggestions": ["lista farmaci suggeriti"]
}}'''

    def __init__(self, provider: str = "anthropic", model: str = "claude-3-haiku-20240307"):
        self.provider = provider
        self.model = model
    
    def extract(self, text: str) -> ExtractedNGSData:
        """Estrae dati strutturati dal testo usando LLM o fallback regex"""
        # Prima prova LLM
        try:
            result = self._extract_with_llm(text)
            if result and result.confidence > 0.5:
                return result
        except Exception:
            pass
        
        # Fallback regex
        return self._extract_with_regex(text)
    
    def _extract_with_llm(self, text: str) -> Optional[ExtractedNGSData]:
        """Estrazione con LLM (Anthropic)"""
        try:
            import anthropic
            client = anthropic.Anthropic()
            
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": self.EXTRACTION_PROMPT.format(text=text[:8000])}]
            )
            
            json_text = response.content[0].text
            # Pulisci JSON
            json_text = re.sub(r'^```json\s*', '', json_text)
            json_text = re.sub(r'\s*```$', '', json_text)
            
            data = json.loads(json_text)
            
            return ExtractedNGSData(
                patient_id=data.get("patient_id"),
                sample_date=data.get("sample_date"),
                sample_type=data.get("sample_type", "unknown"),
                genes=data.get("genes", {}),
                biomarkers=data.get("biomarkers", {}),
                therapy_suggestions=data.get("therapy_suggestions", []),
                raw_text=text,
                confidence=0.85
            )
        except Exception:
            return None
    
    def _extract_with_regex(self, text: str) -> ExtractedNGSData:
        """Fallback estrazione con regex"""
        genes = {}
        biomarkers = {}
        
        # Pattern comuni per geni
        gene_patterns = {
            "TP53": r"TP53[:\s]*(mutated|mut|wild.?type|wt|positive|negative)",
            "KRAS": r"KRAS[:\s]*(G12C|G12D|G12V|mutated|mut|wild.?type|wt)",
            "EGFR": r"EGFR[:\s]*(mutated|mut|wild.?type|wt|L858R|T790M|exon\s*\d+)",
            "ALK": r"ALK[:\s]*(positive|negative|rearranged|fusion)",
            "BRAF": r"BRAF[:\s]*(V600E|mutated|mut|wild.?type|wt)",
        }
        
        for gene, pattern in gene_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status_raw = match.group(1).lower()
                if any(x in status_raw for x in ['mut', 'positive', 'g12', 'v600', 'l858', 't790']):
                    status = "mutated"
                    mutation = match.group(1) if match.group(1).upper().startswith(('G12', 'V600', 'L858', 'T790')) else None
                else:
                    status = "wild-type"
                    mutation = None
                genes[gene] = {"status": status, "mutation": mutation}
        
        # Biomarkers
        tmb_match = re.search(r"TMB[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if tmb_match:
            biomarkers["tmb_score"] = float(tmb_match.group(1))
        
        pdl1_match = re.search(r"PD.?L1[:\s]*(\d+(?:\.\d+)?)\s*%?", text, re.IGNORECASE)
        if pdl1_match:
            biomarkers["pd_l1_tps"] = float(pdl1_match.group(1))
        
        msi_match = re.search(r"(MSI.?H|MSS|microsatellite.?stable|microsatellite.?instab)", text, re.IGNORECASE)
        if msi_match:
            biomarkers["msi_status"] = "MSI-H" if "H" in msi_match.group(1).upper() or "instab" in msi_match.group(1).lower() else "MSS"
        
        return ExtractedNGSData(
            genes=genes,
            biomarkers=biomarkers,
            raw_text=text,
            confidence=0.5 if genes else 0.2
        )
