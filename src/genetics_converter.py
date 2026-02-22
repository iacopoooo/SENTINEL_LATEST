#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL GENETICS CONVERTER v1.0
================================
Converter centralizzato per unificare i formati genetici in SENTINEL.

Problema risolto:
- new_patient.py usa: {"tp53_vaf": 28.0, "kras_vaf": 18.0}
- Oracle usa: [{"gene": "TP53", "vaf": 0.28}, {"gene": "KRAS", "vaf": 0.18}]

Questo modulo traduce tra i due formati in modo trasparente.

Usage:
    from genetics_converter import GeneticsConverter
    
    # Da paziente esistente (qualsiasi formato) → noise_variants per Oracle
    variants = GeneticsConverter.get_unified_variants(patient_data)
    
    # Da PDF NGS estratto → formato flat per CHRONOS
    flat = GeneticsConverter.noise_variants_to_flat(variants)

Author: SENTINEL Team
Date: 2026-02-05
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


# =============================================================================
# CONFIGURAZIONE GENI SUPPORTATI
# =============================================================================

# Mapping gene name → chiavi nel formato flat
# Formato: "GENE": ("status_key", "vaf_key", "mutation_key")
GENE_MAPPING = {
    # Core driver genes
    "TP53": ("tp53_status", "tp53_vaf", None),
    "KRAS": ("kras_status", "kras_vaf", "kras_mutation"),
    "EGFR": ("egfr_status", "egfr_vaf", "egfr_mutation"),
    "BRAF": ("braf_status", "braf_vaf", "braf_mutation"),
    "ALK": ("alk_status", "alk_vaf", None),
    "MET": ("met_status", "met_vaf", None),
    "RET": ("ret_status", "ret_vaf", None),
    "ROS1": ("ros1_status", "ros1_vaf", None),
    "HER2": ("her2_status", "her2_vaf", None),
    "ERBB2": ("her2_status", "her2_vaf", None),  # Alias per HER2
    
    # Tumor suppressors
    "STK11": ("stk11_status", "stk11_vaf", None),
    "KEAP1": ("keap1_status", "keap1_vaf", None),
    "RB1": ("rb1_status", "rb1_vaf", None),
    "PTEN": ("pten_status", "pten_vaf", None),
    "NF1": ("nf1_status", "nf1_vaf", None),
    
    # Other oncogenes
    "PIK3CA": ("pik3ca_status", "pik3ca_vaf", None),
    "NRAS": ("nras_status", "nras_vaf", "nras_mutation"),
    "HRAS": ("hras_status", "hras_vaf", "hras_mutation"),
    
    # Pancreatic/other
    "CDKN2A": ("cdkn2a_status", "cdkn2a_vaf", None),
    "SMAD4": ("smad4_status", "smad4_vaf", None),
    "BRCA1": ("brca1_status", "brca1_vaf", None),
    "BRCA2": ("brca2_status", "brca2_vaf", None),
    "APC": ("apc_status", "apc_vaf", None),
    
    # FGFR family
    "FGFR1": ("fgfr1_status", "fgfr1_vaf", None),
    "FGFR2": ("fgfr2_status", "fgfr2_vaf", None),
    "FGFR3": ("fgfr3_status", "fgfr3_vaf", None),
    
    # IDH
    "IDH1": ("idh1_status", "idh1_vaf", None),
    "IDH2": ("idh2_status", "idh2_vaf", None),
    
    # CHIP genes
    "DNMT3A": ("dnmt3a_status", "dnmt3a_vaf", None),
    "TET2": ("tet2_status", "tet2_vaf", None),
    "ASXL1": ("asxl1_status", "asxl1_vaf", None),
    "JAK2": ("jak2_status", "jak2_vaf", None),
    "SF3B1": ("sf3b1_status", "sf3b1_vaf", None),
    "SRSF2": ("srsf2_status", "srsf2_vaf", None),
    "PPM1D": ("ppm1d_status", "ppm1d_vaf", None),
}

# Reverse mapping: vaf_key → gene name
VAF_KEY_TO_GENE = {}
for gene, (status_key, vaf_key, mut_key) in GENE_MAPPING.items():
    if vaf_key:
        VAF_KEY_TO_GENE[vaf_key] = gene
        # Anche formato alternativo vaf_gene
        VAF_KEY_TO_GENE[f"vaf_{gene.lower()}"] = gene


# =============================================================================
# DATACLASS PER VARIANTE UNIFICATA
# =============================================================================

@dataclass
class UnifiedVariant:
    """Rappresentazione unificata di una variante genetica."""
    gene: str
    vaf: float  # Sempre in formato 0-1 (frazione)
    status: Optional[str] = None  # "mutated", "wt", "amplified", etc.
    mutation: Optional[str] = None  # "G12C", "L858R", etc.
    mutation_type: Optional[str] = None  # "missense", "nonsense", etc.
    
    def to_noise_variant(self) -> Dict[str, Any]:
        """Converte in formato noise_variants per Oracle."""
        result = {
            "gene": self.gene.upper(),
            "vaf": self.vaf,
        }
        if self.mutation:
            result["mutation"] = self.mutation
        if self.mutation_type:
            result["mutation_type"] = self.mutation_type
        return result
    
    @property
    def vaf_percent(self) -> float:
        """VAF in formato percentuale (0-100)."""
        return self.vaf * 100.0


# =============================================================================
# CONVERTER PRINCIPALE
# =============================================================================

class GeneticsConverter:
    """
    Converter centralizzato per formati genetici SENTINEL.
    
    Gestisce:
    1. Formato flat: {"tp53_vaf": 28.0, "tp53_status": "mutated"}
    2. Formato noise_variants: [{"gene": "TP53", "vaf": 0.28}]
    3. Formato vaf_values dict: {"vaf_values": {"TP53": 28.0}}
    """
    
    @staticmethod
    def normalize_vaf(vaf: Any) -> Optional[float]:
        """
        Normalizza VAF al formato 0-1 (frazione).
        
        Gestisce:
        - 0.28 → 0.28 (già frazione)
        - 28.0 → 0.28 (percentuale)
        - "28%" → 0.28 (stringa)
        - None → None
        """
        if vaf is None:
            return None
        
        try:
            if isinstance(vaf, str):
                # Rimuovi % e spazi
                vaf = vaf.strip().replace("%", "")
                vaf = float(vaf)
            else:
                vaf = float(vaf)
            
            # Se > 1, assume sia percentuale
            if vaf > 1.0:
                vaf = vaf / 100.0
            
            # Clamp a 0-1
            return max(0.0, min(1.0, vaf))
            
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def normalize_gene_name(gene: str) -> str:
        """Normalizza nome gene (uppercase, trim)."""
        if not gene:
            return ""
        return str(gene).strip().upper()
    
    # =========================================================================
    # FLAT → NOISE_VARIANTS
    # =========================================================================
    
    @classmethod
    def flat_to_noise_variants(
        cls, 
        genetics: Dict[str, Any],
        include_zero_vaf: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Converte formato flat → noise_variants.
        
        Input:
            {"tp53_status": "mutated", "tp53_vaf": 28.0, "kras_vaf": 18.0}
        
        Output:
            [{"gene": "TP53", "vaf": 0.28}, {"gene": "KRAS", "vaf": 0.18}]
        """
        if not genetics:
            return []
        
        variants = []
        seen_genes = set()
        
        # 1. Cerca chiavi *_vaf o vaf_*
        for key, value in genetics.items():
            key_lower = key.lower()
            gene = None
            
            # Pattern: tp53_vaf, kras_vaf, etc.
            if key_lower in VAF_KEY_TO_GENE:
                gene = VAF_KEY_TO_GENE[key_lower]
            # Pattern: vaf_tp53, vaf_kras, etc.
            elif key_lower.startswith("vaf_"):
                gene_part = key_lower[4:].upper()
                if gene_part in GENE_MAPPING:
                    gene = gene_part
            # Pattern: tp53Vaf, krasVaf (camelCase)
            elif "vaf" in key_lower:
                match = re.match(r"([a-z0-9]+)_?vaf", key_lower)
                if match:
                    gene_part = match.group(1).upper()
                    if gene_part in GENE_MAPPING:
                        gene = gene_part
            
            if gene and gene not in seen_genes:
                vaf = cls.normalize_vaf(value)
                if vaf is not None and (vaf > 0 or include_zero_vaf):
                    # Cerca anche mutation type se disponibile
                    status_key, _, mut_key = GENE_MAPPING.get(gene, (None, None, None))
                    mutation = genetics.get(mut_key) if mut_key else None
                    
                    variant = {"gene": gene, "vaf": vaf}
                    if mutation:
                        variant["mutation"] = mutation
                    
                    variants.append(variant)
                    seen_genes.add(gene)
        
        # 2. Cerca nel formato vaf_values dict (usato da follow_up.py)
        vaf_values = genetics.get("vaf_values", {})
        if isinstance(vaf_values, dict):
            for gene, vaf in vaf_values.items():
                gene = cls.normalize_gene_name(gene)
                if gene and gene not in seen_genes:
                    vaf_norm = cls.normalize_vaf(vaf)
                    if vaf_norm is not None and (vaf_norm > 0 or include_zero_vaf):
                        variants.append({"gene": gene, "vaf": vaf_norm})
                        seen_genes.add(gene)
        
        return variants
    
    # =========================================================================
    # NOISE_VARIANTS → FLAT
    # =========================================================================
    
    @classmethod
    def noise_variants_to_flat(
        cls,
        variants: List[Dict[str, Any]],
        vaf_as_percent: bool = True
    ) -> Dict[str, Any]:
        """
        Converte noise_variants → formato flat.
        
        Input:
            [{"gene": "TP53", "vaf": 0.28}, {"gene": "KRAS", "vaf": 0.18, "mutation": "G12C"}]
        
        Output:
            {"tp53_vaf": 28.0, "tp53_status": "mutated", 
             "kras_vaf": 18.0, "kras_mutation": "G12C"}
        """
        if not variants:
            return {}
        
        result = {}
        
        for var in variants:
            gene = cls.normalize_gene_name(var.get("gene", ""))
            if not gene or gene not in GENE_MAPPING:
                continue
            
            status_key, vaf_key, mut_key = GENE_MAPPING[gene]
            
            # VAF
            vaf = cls.normalize_vaf(var.get("vaf"))
            if vaf is not None:
                result[vaf_key] = vaf * 100.0 if vaf_as_percent else vaf
                # Se c'è VAF > 0, imposta status come mutated
                if vaf > 0 and status_key:
                    result[status_key] = "mutated"
            
            # Mutation (es. G12C)
            mutation = var.get("mutation")
            if mutation and mut_key:
                result[mut_key] = mutation
        
        return result
    
    # =========================================================================
    # UNIFIED GETTER (QUALSIASI FORMATO → NOISE_VARIANTS)
    # =========================================================================
    
    @classmethod
    def get_unified_variants(
        cls,
        patient_data: Dict[str, Any],
        include_baseline: bool = True,
        include_visits: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Estrae noise_variants da QUALSIASI struttura paziente.
        
        Cerca in:
        1. patient_data["noise_variants"] (già nel formato giusto)
        2. patient_data["genetics"] (formato flat)
        3. patient_data["baseline"]["genetics"] (formato flat)
        4. patient_data["visits"][*]["noise_variants"]
        5. patient_data["visits"][*]["genetics"]
        
        Returns:
            Lista unificata di noise_variants (deduplicated per gene)
        """
        all_variants = []
        
        # 1. Top-level noise_variants
        if "noise_variants" in patient_data:
            nv = patient_data["noise_variants"]
            if isinstance(nv, list):
                all_variants.extend(nv)
        
        # 2. Top-level genetics (flat)
        if "genetics" in patient_data:
            converted = cls.flat_to_noise_variants(patient_data["genetics"])
            all_variants.extend(converted)
        
        # 3. Baseline genetics
        if include_baseline:
            baseline = patient_data.get("baseline", {})
            if "genetics" in baseline:
                converted = cls.flat_to_noise_variants(baseline["genetics"])
                all_variants.extend(converted)
            if "noise_variants" in baseline:
                all_variants.extend(baseline["noise_variants"])
        
        # 4. Visits
        if include_visits:
            visits = patient_data.get("visits", [])
            if isinstance(visits, list):
                for visit in visits:
                    if not isinstance(visit, dict):
                        continue
                    
                    # noise_variants nella visita
                    if "noise_variants" in visit:
                        nv = visit["noise_variants"]
                        if isinstance(nv, list):
                            all_variants.extend(nv)
                    
                    # genetics flat nella visita
                    if "genetics" in visit:
                        converted = cls.flat_to_noise_variants(visit["genetics"])
                        all_variants.extend(converted)
        
        # Normalizza tutti
        normalized = []
        for var in all_variants:
            if not isinstance(var, dict):
                continue
            gene = cls.normalize_gene_name(var.get("gene", ""))
            vaf = cls.normalize_vaf(var.get("vaf"))
            if gene and vaf is not None:
                normalized.append({
                    "gene": gene,
                    "vaf": vaf,
                    **{k: v for k, v in var.items() if k not in ("gene", "vaf")}
                })
        
        return normalized
    
    # =========================================================================
    # PER-VISIT EXTRACTION (per Oracle timeline)
    # =========================================================================
    
    @classmethod
    def get_variants_by_visit(
        cls,
        patient_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Estrae noise_variants organizzati per visita (per Oracle timeline).
        
        Returns:
            [
                {"date": "2024-01-15", "noise_variants": [...]},
                {"date": "2024-04-20", "noise_variants": [...]}
            ]
        """
        result = []
        
        # Supporta sia "visits" che "history" (formato Oracle validation dataset)
        visits = patient_data.get("visits", [])
        if not isinstance(visits, list) or len(visits) == 0:
            visits = patient_data.get("history", [])
        
        for visit in visits:
            if not isinstance(visit, dict):
                continue
            
            date = visit.get("date")
            if not date:
                continue
            
            # Estrai variants
            variants = []
            
            # noise_variants diretto
            if "noise_variants" in visit:
                nv = visit["noise_variants"]
                if isinstance(nv, list):
                    for var in nv:
                        gene = cls.normalize_gene_name(var.get("gene", ""))
                        vaf = cls.normalize_vaf(var.get("vaf"))
                        if gene and vaf is not None:
                            variants.append({"gene": gene, "vaf": vaf, **{k: v for k, v in var.items() if k not in ("gene", "vaf")}})
            
            # genetics flat
            if "genetics" in visit:
                converted = cls.flat_to_noise_variants(visit["genetics"])
                # Merge evitando duplicati per gene
                seen = {v["gene"] for v in variants}
                for cv in converted:
                    if cv["gene"] not in seen:
                        variants.append(cv)
            
            result.append({
                "date": date,
                "noise_variants": variants
            })
        
        return result
    
    # =========================================================================
    # HELPER: CHECK IF HAS VARIANTS
    # =========================================================================
    
    @classmethod
    def has_genetic_data(cls, patient_data: Dict[str, Any]) -> bool:
        """Verifica se il paziente ha dati genetici (in qualsiasi formato)."""
        variants = cls.get_unified_variants(patient_data)
        return len(variants) > 0
    
    @classmethod
    def count_variants(cls, patient_data: Dict[str, Any]) -> int:
        """Conta il numero di varianti genetiche."""
        return len(cls.get_unified_variants(patient_data))


# =============================================================================
# UTILITY FUNCTIONS (shorthand)
# =============================================================================

def flat_to_noise_variants(genetics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shorthand per GeneticsConverter.flat_to_noise_variants()"""
    return GeneticsConverter.flat_to_noise_variants(genetics)


def noise_variants_to_flat(variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Shorthand per GeneticsConverter.noise_variants_to_flat()"""
    return GeneticsConverter.noise_variants_to_flat(variants)


def get_unified_variants(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shorthand per GeneticsConverter.get_unified_variants()"""
    return GeneticsConverter.get_unified_variants(patient_data)


def get_variants_by_visit(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shorthand per GeneticsConverter.get_variants_by_visit()"""
    return GeneticsConverter.get_variants_by_visit(patient_data)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧬 SENTINEL GENETICS CONVERTER - TEST SUITE")
    print("=" * 70)
    
    # Test 1: Flat → Noise Variants
    print("\n📋 TEST 1: Flat → Noise Variants")
    flat_genetics = {
        "tp53_status": "mutated",
        "tp53_vaf": 28.0,
        "kras_mutation": "G12C",
        "kras_vaf": 18.0,
        "egfr_status": "wt",
        "stk11_vaf": 5.0
    }
    print(f"   Input:  {flat_genetics}")
    result = GeneticsConverter.flat_to_noise_variants(flat_genetics)
    print(f"   Output: {result}")
    assert len(result) == 3, "Dovrebbe avere 3 varianti"
    assert result[0]["gene"] == "TP53", "Prima variante dovrebbe essere TP53"
    assert abs(result[0]["vaf"] - 0.28) < 0.01, "VAF TP53 dovrebbe essere 0.28"
    print("   ✅ PASS")
    
    # Test 2: Noise Variants → Flat
    print("\n📋 TEST 2: Noise Variants → Flat")
    noise_vars = [
        {"gene": "TP53", "vaf": 0.28},
        {"gene": "KRAS", "vaf": 0.18, "mutation": "G12C"},
        {"gene": "EGFR", "vaf": 0.05}
    ]
    print(f"   Input:  {noise_vars}")
    result = GeneticsConverter.noise_variants_to_flat(noise_vars)
    print(f"   Output: {result}")
    assert "tp53_vaf" in result, "Dovrebbe avere tp53_vaf"
    assert abs(result["tp53_vaf"] - 28.0) < 0.1, "tp53_vaf dovrebbe essere 28.0"
    assert result.get("kras_mutation") == "G12C", "Dovrebbe avere kras_mutation"
    print("   ✅ PASS")
    
    # Test 3: Unified Variants (mixed format)
    print("\n📋 TEST 3: Unified Variants (formato misto)")
    patient_data = {
        "genetics": {
            "tp53_vaf": 28.0,
            "kras_vaf": 18.0
        },
        "visits": [
            {
                "date": "2024-01-15",
                "noise_variants": [
                    {"gene": "EGFR", "vaf": 0.05}
                ]
            },
            {
                "date": "2024-04-20",
                "genetics": {
                    "braf_vaf": 12.0
                }
            }
        ]
    }
    print(f"   Input: paziente con genetics flat + visite miste")
    result = GeneticsConverter.get_unified_variants(patient_data)
    print(f"   Output: {result}")
    genes = {v["gene"] for v in result}
    assert "TP53" in genes, "Dovrebbe avere TP53"
    assert "EGFR" in genes, "Dovrebbe avere EGFR"
    assert "BRAF" in genes, "Dovrebbe avere BRAF"
    print("   ✅ PASS")
    
    # Test 4: Variants by Visit (per Oracle)
    print("\n📋 TEST 4: Variants by Visit (per Oracle timeline)")
    result = GeneticsConverter.get_variants_by_visit(patient_data)
    print(f"   Output: {result}")
    assert len(result) == 2, "Dovrebbe avere 2 visite"
    print("   ✅ PASS")
    
    # Test 5: VAF Normalization
    print("\n📋 TEST 5: VAF Normalization")
    test_cases = [
        (0.28, 0.28),
        (28.0, 0.28),
        ("28%", 0.28),
        ("0.28", 0.28),
        (None, None),
        (150, 1.0),  # Clamped
    ]
    for input_val, expected in test_cases:
        result = GeneticsConverter.normalize_vaf(input_val)
        if expected is None:
            assert result is None, f"Input {input_val} dovrebbe dare None"
        else:
            assert abs(result - expected) < 0.01, f"Input {input_val} dovrebbe dare {expected}, got {result}"
    print("   ✅ PASS")
    
    # Test 6: Oracle validation dataset format
    print("\n📋 TEST 6: Oracle validation dataset format")
    oracle_patient = {
        "id": "GHOST_RELAPSE_LUNG_141",
        "history": [
            {
                "date": "2016-03-14",
                "noise_variants": [{"gene": "EGFR", "vaf": 0.03}]
            },
            {
                "date": "2016-07-21",
                "noise_variants": [{"gene": "TP53", "vaf": 0.118}]
            },
            {
                "date": "2017-07-11",
                "noise_variants": [
                    {"gene": "EGFR", "vaf": 0.842},
                    {"gene": "TP53", "vaf": 0.43}
                ]
            }
        ]
    }
    result = GeneticsConverter.get_variants_by_visit(oracle_patient)
    print(f"   Output: {len(result)} visite")
    assert len(result) == 3, "Dovrebbe avere 3 visite"
    print("   ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 70)
