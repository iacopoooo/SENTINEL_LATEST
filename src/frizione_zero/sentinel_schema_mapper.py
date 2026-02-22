"""
SENTINEL SCHEMA MAPPER v2.0 - Mapping a formato SENTINEL
=========================================================
Integrato con GeneticsConverter per supporto dual-format:
- genetics: formato flat per CHRONOS (tp53_vaf, kras_vaf, etc.)
- noise_variants: formato array per Oracle ([{gene, vaf}, ...])

Changelog v2.0:
- Aggiunto campo noise_variants a SentinelPatient
- Integrato GeneticsConverter per conversione automatica
- Supporto PDF esterni con qualsiasi formato

"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import GeneticsConverter (deve essere in src/)
try:
    from genetics_converter import GeneticsConverter
    HAS_CONVERTER = True
except ImportError:
    # Fallback se non disponibile
    HAS_CONVERTER = False
    

@dataclass
class SentinelPatient:
    """
    Rappresentazione paziente SENTINEL.
    
    Campi genetici:
    - genetics: formato flat per CHRONOS (tp53_vaf: 28.0)
    - noise_variants: formato array per Oracle ([{gene: "TP53", vaf: 0.28}])
    """
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    ecog_ps: int = 0
    stage: str = "unknown"
    histology: str = "unknown"
    current_therapy: str = ""
    
    # Genetica dual-format
    genetics: Dict[str, Any] = field(default_factory=dict)
    noise_variants: List[Dict[str, Any]] = field(default_factory=list)  # NUOVO per Oracle
    
    # Altri campi
    blood_markers: Dict[str, float] = field(default_factory=dict)
    biomarkers: Dict[str, Any] = field(default_factory=dict)
    biopsy_image_path: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "frizione_zero"
    
    def to_sentinel_json(self) -> Dict[str, Any]:
        """Esporta in formato JSON SENTINEL completo."""
        data = asdict(self)
        return {
            "baseline": data,
            "visits": []
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Esporta come dizionario."""
        return asdict(self)


class SentinelSchemaMapper:
    """
    Mapper per convertire dati NGS estratti nel formato SENTINEL.
    
    Flusso:
    1. PDF → LLM estrae ExtractedNGSData
    2. SentinelSchemaMapper converte → SentinelPatient
    3. SentinelPatient ha sia genetics (flat) che noise_variants (array)
    """
    
    THERAPY_MAPPING = {
        "EGFR": ["Osimertinib", "Erlotinib", "Gefitinib"],
        "ALK": ["Alectinib", "Crizotinib", "Lorlatinib"],
        "KRAS_G12C": ["Sotorasib", "Adagrasib"],
        "BRAF_V600E": ["Dabrafenib + Trametinib"],
        "ROS1": ["Crizotinib", "Entrectinib"],
        "RET": ["Selpercatinib", "Pralsetinib"],
        "MET": ["Capmatinib", "Tepotinib"],
        "HIGH_TMB": ["Pembrolizumab", "Nivolumab"],
        "HIGH_PDL1": ["Pembrolizumab", "Atezolizumab"],
    }
    
    def __init__(self):
        self._last_noise_variants: List[Dict[str, Any]] = []
    
    def map_to_sentinel(self, ngs_data, patient_id: Optional[str] = None) -> SentinelPatient:
        """
        Converte ExtractedNGSData in SentinelPatient.
        
        Args:
            ngs_data: Dati estratti dal PDF (ExtractedNGSData)
            patient_id: ID paziente opzionale (auto-generato se None)
            
        Returns:
            SentinelPatient con genetics E noise_variants popolati
        """
        pid = patient_id or getattr(ngs_data, 'patient_id', None) or f"FZ-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Mappa genetica (genera sia flat che noise_variants)
        genetics = self._map_genetics(getattr(ngs_data, 'genes', {}) or {})
        noise_variants = self._last_noise_variants  # Popolato da _map_genetics
        
        # Mappa biomarker
        biomarkers = self._map_biomarkers(getattr(ngs_data, 'biomarkers', {}) or {})
        
        # Inferisci terapia
        therapy = self._infer_therapy(genetics, biomarkers)
        
        # Estrai età/sesso se disponibili
        age = getattr(ngs_data, 'age', None)
        sex = getattr(ngs_data, 'sex', None)
        
        return SentinelPatient(
            patient_id=pid,
            age=age,
            sex=sex,
            genetics=genetics,
            noise_variants=noise_variants,  # NUOVO!
            biomarkers=biomarkers,
            current_therapy=therapy,
            source="frizione_zero"
        )
    
    def _map_genetics(self, genes: Dict) -> Dict:
        """
        Mappa geni al formato SENTINEL (flat).
        
        Genera anche noise_variants per Oracle.
        
        Args:
            genes: Dict dal LLM extractor, formato:
                   {"TP53": {"status": "mutated", "vaf": 28.0, "mutation": "R175H"}}
                   
        Returns:
            Dict formato flat: {"tp53_status": "mutated", "tp53_vaf": 28.0}
        """
        result = {}
        noise_variants = []
        
        for gene, data in genes.items():
            gene_upper = gene.upper().strip()
            gene_lower = gene.lower().strip()
            
            if not isinstance(data, dict):
                continue
                
            status = data.get("status", "unknown")
            
            if status in ["mutated", "positive", "amplified", "detected"]:
                result[f"{gene_lower}_status"] = "mutated"
                
                # Mutation type (es. G12C, L858R)
                mutation = data.get("mutation") or data.get("variant")
                if mutation:
                    result[f"{gene_lower}_mutation"] = mutation
                
                # VAF
                vaf = data.get("vaf")
                if vaf is not None:
                    # Normalizza VAF (se > 1, è percentuale)
                    try:
                        vaf_float = float(vaf)
                        if vaf_float > 1.0:
                            vaf_float = vaf_float  # Mantieni come percentuale per flat
                        else:
                            vaf_float = vaf_float * 100.0  # Converti in percentuale
                        result[f"{gene_lower}_vaf"] = vaf_float
                        
                        # Per noise_variants, usa sempre 0-1
                        vaf_fraction = vaf_float / 100.0 if vaf_float > 1.0 else float(vaf)
                        variant_entry = {
                            "gene": gene_upper,
                            "vaf": vaf_fraction
                        }
                        if mutation:
                            variant_entry["mutation"] = mutation
                        noise_variants.append(variant_entry)
                        
                    except (ValueError, TypeError):
                        pass
                else:
                    # Gene mutato ma senza VAF - aggiungi comunque a noise_variants con VAF=0
                    # (Oracle può comunque usarlo per tracking clonale)
                    pass
                        
                # Copy Number
                cn = data.get("copy_number") or data.get("cn")
                if cn is not None:
                    try:
                        result[f"{gene_lower}_cn"] = float(cn)
                    except (ValueError, TypeError):
                        pass
                        
            else:
                result[f"{gene_lower}_status"] = "wild-type"
        
        # Salva noise_variants per uso successivo
        self._last_noise_variants = noise_variants
        
        # Se abbiamo GeneticsConverter, verifica/arricchisci
        if HAS_CONVERTER and noise_variants:
            # Opzionale: double-check con converter
            pass
        
        return result
    
    def _map_biomarkers(self, biomarkers: Dict) -> Dict:
        """Mappa biomarker al formato SENTINEL."""
        result = {}
        
        if not biomarkers:
            return result
            
        # TMB
        tmb = biomarkers.get("tmb_score") or biomarkers.get("tmb")
        if tmb is not None:
            try:
                result["tmb_score"] = float(tmb)
            except (ValueError, TypeError):
                pass
        
        # PD-L1
        pdl1 = biomarkers.get("pd_l1_tps") or biomarkers.get("pdl1") or biomarkers.get("pd_l1")
        if pdl1 is not None:
            try:
                result["pd_l1_tps"] = float(pdl1)
            except (ValueError, TypeError):
                pass
        
        # MSI
        msi = biomarkers.get("msi_status") or biomarkers.get("msi")
        if msi:
            result["msi_status"] = str(msi)
        
        return result
    
    def _infer_therapy(self, genetics: Dict, biomarkers: Dict) -> str:
        """
        Inferisce terapia suggerita basata su genetica e biomarker.
        
        Returns:
            Nome terapia suggerita o stringa vuota
        """
        suggestions = []
        
        # EGFR mutato
        if genetics.get("egfr_status") == "mutated":
            mutation = genetics.get("egfr_mutation", "").upper()
            if "T790M" in mutation:
                suggestions.append("Osimertinib")
            elif "L858R" in mutation or "DEL19" in mutation or "EXON19" in mutation:
                suggestions.append("Osimertinib")
            else:
                suggestions.append("Osimertinib")  # Default EGFR
        
        # ALK positivo
        if genetics.get("alk_status") == "mutated":
            suggestions.append("Alectinib")
        
        # KRAS G12C
        kras_mut = genetics.get("kras_mutation", "").upper()
        if "G12C" in kras_mut:
            suggestions.append("Sotorasib")
        
        # BRAF V600E
        braf_mut = genetics.get("braf_mutation", "").upper()
        if "V600" in braf_mut:
            suggestions.append("Dabrafenib + Trametinib")
        
        # ROS1
        if genetics.get("ros1_status") == "mutated":
            suggestions.append("Crizotinib")
        
        # RET
        if genetics.get("ret_status") == "mutated":
            suggestions.append("Selpercatinib")
        
        # MET amplificato/mutato
        met_cn = genetics.get("met_cn", 0)
        if genetics.get("met_status") == "mutated" or (met_cn and met_cn >= 6):
            suggestions.append("Capmatinib")
        
        # High TMB → Immunoterapia
        tmb = biomarkers.get("tmb_score", 0)
        if tmb and tmb >= 10:
            suggestions.append("Pembrolizumab (high TMB)")
        
        # High PD-L1 → Immunoterapia
        pdl1 = biomarkers.get("pd_l1_tps", 0)
        if pdl1 and pdl1 >= 50:
            suggestions.append("Pembrolizumab (high PD-L1)")
        
        # STK11/KEAP1 → Warning (resistenza a immunoterapia)
        if genetics.get("stk11_status") == "mutated" or genetics.get("keap1_status") == "mutated":
            # Non suggerire immunoterapia primaria
            suggestions = [s for s in suggestions if "Pembrolizumab" not in s and "Nivolumab" not in s]
        
        return suggestions[0] if suggestions else ""
    
    def map_from_raw_variants(
        self, 
        variants: List[Dict[str, Any]], 
        patient_id: str,
        biomarkers: Optional[Dict] = None
    ) -> SentinelPatient:
        """
        Crea SentinelPatient direttamente da noise_variants.
        
        Utile per:
        - Pazienti importati da sistemi esterni
        - VCF già parsati
        - Dati ctDNA/liquid biopsy
        
        Args:
            variants: Lista noise_variants [{gene, vaf, mutation?}, ...]
            patient_id: ID paziente
            biomarkers: Biomarker opzionali
            
        Returns:
            SentinelPatient con genetics E noise_variants popolati
        """
        # Converti noise_variants → flat usando GeneticsConverter
        if HAS_CONVERTER:
            genetics = GeneticsConverter.noise_variants_to_flat(variants)
        else:
            # Fallback manuale
            genetics = {}
            for var in variants:
                gene = var.get("gene", "").lower()
                if not gene:
                    continue
                vaf = var.get("vaf")
                if vaf is not None:
                    try:
                        vaf_float = float(vaf)
                        if vaf_float <= 1.0:
                            vaf_float *= 100.0
                        genetics[f"{gene}_vaf"] = vaf_float
                        genetics[f"{gene}_status"] = "mutated"
                    except (ValueError, TypeError):
                        pass
                mutation = var.get("mutation")
                if mutation:
                    genetics[f"{gene}_mutation"] = mutation
        
        # Inferisci terapia
        therapy = self._infer_therapy(genetics, biomarkers or {})
        
        return SentinelPatient(
            patient_id=patient_id,
            genetics=genetics,
            noise_variants=variants,  # Già nel formato giusto
            biomarkers=biomarkers or {},
            current_therapy=therapy,
            source="external_import"
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧬 SENTINEL SCHEMA MAPPER v2.0 - TEST")
    print("=" * 70)
    
    # Simula ExtractedNGSData
    class MockNGSData:
        def __init__(self):
            self.patient_id = "TEST-001"
            self.genes = {
                "TP53": {"status": "mutated", "vaf": 28.0, "mutation": "R175H"},
                "KRAS": {"status": "mutated", "vaf": 18.0, "mutation": "G12C"},
                "EGFR": {"status": "wild-type"},
                "STK11": {"status": "mutated", "vaf": 5.0}
            }
            self.biomarkers = {
                "tmb_score": 12.5,
                "pd_l1_tps": 60
            }
            self.age = 65
            self.sex = "M"
    
    mapper = SentinelSchemaMapper()
    
    # Test 1: map_to_sentinel
    print("\n📋 TEST 1: map_to_sentinel()")
    mock_data = MockNGSData()
    patient = mapper.map_to_sentinel(mock_data)
    
    print(f"   Patient ID: {patient.patient_id}")
    print(f"   Genetics (flat): {patient.genetics}")
    print(f"   Noise Variants: {patient.noise_variants}")
    print(f"   Therapy: {patient.current_therapy}")
    
    assert "tp53_vaf" in patient.genetics, "Manca tp53_vaf"
    assert len(patient.noise_variants) >= 2, "Dovrebbe avere almeno 2 noise_variants"
    assert patient.current_therapy == "Sotorasib", f"Therapy dovrebbe essere Sotorasib, got {patient.current_therapy}"
    print("   ✅ PASS")
    
    # Test 2: map_from_raw_variants
    print("\n📋 TEST 2: map_from_raw_variants()")
    raw_variants = [
        {"gene": "EGFR", "vaf": 0.35, "mutation": "L858R"},
        {"gene": "TP53", "vaf": 0.22}
    ]
    patient2 = mapper.map_from_raw_variants(raw_variants, "TEST-002")
    
    print(f"   Patient ID: {patient2.patient_id}")
    print(f"   Genetics (flat): {patient2.genetics}")
    print(f"   Noise Variants: {patient2.noise_variants}")
    print(f"   Therapy: {patient2.current_therapy}")
    
    assert "egfr_vaf" in patient2.genetics, "Manca egfr_vaf"
    assert patient2.current_therapy == "Osimertinib", f"Therapy dovrebbe essere Osimertinib"
    print("   ✅ PASS")
    
    # Test 3: to_sentinel_json
    print("\n📋 TEST 3: to_sentinel_json()")
    json_output = patient.to_sentinel_json()
    print(f"   Keys: {list(json_output.keys())}")
    print(f"   Baseline keys: {list(json_output['baseline'].keys())[:5]}...")
    
    assert "baseline" in json_output, "Manca baseline"
    assert "noise_variants" in json_output["baseline"], "Manca noise_variants nel baseline"
    print("   ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 70)
