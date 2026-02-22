"""
# PRIMA ERA COSI
import os

class SentinelGenomics:
    def __init__(self):
        self.target_panel = ['TP53', 'KRAS', 'PIK3CA', 'EGFR', 'MET', 'BRAF']

    def parse_vcf(self, vcf_path):

       # Legge un file VCF grezzo e restituisce un dizionario pulito per Sentinel.

        if not os.path.exists(vcf_path):
            return {"error": "File not found"}

        extracted_data = {
            "tp53_status": "wt",
            "kras_mutation": "wt",
            "pik3ca_status": "wt",
            "met_cn": "1.0",  # Default
            "raw_mutations_found": []
        }

        print(f"🔍 ANALISI SEQUENZIAMENTO: {os.path.basename(vcf_path)}...")

        with open(vcf_path, 'r') as f:
            for line in f:
                # Salta le righe di commento (Header)
                if line.startswith("#"):
                    continue

                # Analizza la riga della variante
                parts = line.split('\t')
                if len(parts) < 8: continue

                info_column = parts[7]  # La colonna INFO contiene i dati del gene

                # Cerca i nostri geni target nella colonna INFO
                for target in self.target_panel:
                    if f"GENE={target}" in info_column:
                        # Abbiamo trovato una mutazione in un gene target!
                        print(f"   ⚠️  MUTAZIONE RILEVATA: {target}")
                        extracted_data["raw_mutations_found"].append(target)

                        # Mapping specifico per Sentinel
                        if target == 'TP53':
                            extracted_data['tp53_status'] = 'mutated'
                        elif target == 'PIK3CA':
                            extracted_data['pik3ca_status'] = 'mutated'
                        elif target == 'KRAS':
                            # Cerca di estrarre il tipo esatto (es. G12D) se presente
                            # Simulazione grezza di parsing stringa
                            if "G12C" in line:
                                extracted_data['kras_mutation'] = "G12C"
                            elif "G12D" in line:
                                extracted_data['kras_mutation'] = "G12D"
                            else:
                                extracted_data['kras_mutation'] = "Other"

        return extracted_data


# Test rapido se lanciato direttamente
if __name__ == "__main__":
    parser = SentinelGenomics()

    # Testiamo sul paziente malato generato prima
    path = "../data/genomics/RAW_DATA_PATIENT_002.vcf"
    if os.path.exists(path):
        result = parser.parse_vcf(path)
        print("\n📊 DATI ESTRATTI PER SENTINEL JSON:")
        print(result)
    else:
        print("⚠️ Esegui prima tools/generate_mock_vcf.py!")

        """

# !/usr/bin/env python3
"""
SENTINEL VCF PARSER v2.0 - Production Grade
============================================
Improvements:
- Supporto VCF standard (VCFv4.x)
- Estrazione VAF (Variant Allele Frequency)
- Parsing variant type (SNV, INDEL, CNV)
- Extended gene panel (20+ genes)
- Validazione formato
- Germline vs Somatic detection
- Logging strutturato
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariantType(Enum):
    """Tipo di variante genomica"""
    SNV = "SNV"  # Single Nucleotide Variant
    INDEL = "INDEL"  # Insertion/Deletion
    CNV = "CNV"  # Copy Number Variant
    FUSION = "FUSION"  # Gene Fusion
    UNKNOWN = "UNKNOWN"


class VariantOrigin(Enum):
    """Origine variante"""
    SOMATIC = "somatic"  # Tumor-specific
    GERMLINE = "germline"  # Inherited
    UNKNOWN = "unknown"


@dataclass
class Variant:
    """Rappresenta una singola variante"""
    gene: str
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    variant_type: VariantType
    variant_origin: VariantOrigin
    vaf: Optional[float] = None  # Variant Allele Frequency
    depth: Optional[int] = None  # Read depth
    protein_change: Optional[str] = None  # p.G12C, p.L858R, etc.
    consequence: Optional[str] = None  # missense, frameshift, etc.
    raw_info: str = ""


@dataclass
class GenomicProfile:
    """Profilo genomico completo del paziente"""
    patient_id: str
    variants: List[Variant] = field(default_factory=list)

    # SENTINEL-specific mappings
    tp53_status: str = "wt"
    kras_mutation: str = "wt"
    egfr_mutation: str = "wt"
    pik3ca_status: str = "wt"
    stk11_status: str = "wt"
    keap1_status: str = "wt"
    rb1_status: str = "wt"
    braf_mutation: str = "wt"
    alk_status: str = "wt"
    met_status: str = "wt"
    her2_status: str = "wt"

    # Copy number
    met_cn: float = 2.0
    her2_cn: float = 2.0
    egfr_cn: float = 2.0

    # Metadata
    tmb: float = 0.0  # Tumor Mutational Burden
    total_variants: int = 0
    somatic_variants: int = 0
    germline_variants: int = 0


class SentinelGenomics:
    """
    SENTINEL VCF Parser v2.0

    Supporta:
    - VCF standard (v4.x)
    - Formati custom (MSK-IMPACT, Foundation, Guardant)
    - Estrazione VAF, depth, protein change
    - Panel esteso (20+ geni oncology)
    """

    VERSION = "2.0"

    # Extended gene panel per NSCLC
    TARGET_PANEL = [
        # Driver genes
        'EGFR', 'KRAS', 'BRAF', 'MET', 'ALK', 'ROS1', 'RET', 'NTRK1', 'NTRK2', 'NTRK3',
        # Resistance genes
        'TP53', 'RB1', 'PIK3CA', 'PTEN', 'STK11', 'KEAP1',
        # Amplifications
        'HER2', 'ERBB2', 'FGFR1', 'MYC',
        # Others
        'NF1', 'CDKN2A', 'SMAD4', 'APC'
    ]

    # KRAS variant patterns (comprehensive)
    KRAS_VARIANTS = {
        'G12C', 'G12D', 'G12V', 'G12A', 'G12S', 'G12R',
        'G13C', 'G13D', 'G13V',
        'Q61H', 'Q61L', 'Q61R',
        'A146T', 'K117N'
    }

    # EGFR variant patterns
    EGFR_VARIANTS = {
        'L858R', 'T790M', 'C797S',
        'Exon19Del', 'Exon20Ins',
        'G719A', 'G719C', 'G719S',
        'L861Q', 'S768I'
    }

    def __init__(self, patient_id: str = "UNKNOWN"):
        self.patient_id = patient_id
        self.profile = GenomicProfile(patient_id=patient_id)

    # =========================================================================
    # CORE PARSING
    # =========================================================================

    def parse_vcf(self, vcf_path: str) -> Dict:
        """
        Parse VCF file e restituisce profilo genomico

        Returns:
            Dict compatibile con SENTINEL JSON format
        """
        if not os.path.exists(vcf_path):
            logger.error(f"VCF file not found: {vcf_path}")
            return {"error": "File not found"}

        logger.info(f"🧬 PARSING VCF: {os.path.basename(vcf_path)}")

        # Validate VCF format
        if not self._validate_vcf(vcf_path):
            logger.error("Invalid VCF format")
            return {"error": "Invalid VCF format"}

        # Parse variants
        variants = self._parse_variants(vcf_path)
        self.profile.variants = variants
        self.profile.total_variants = len(variants)

        # Count somatic vs germline
        self.profile.somatic_variants = sum(1 for v in variants if v.variant_origin == VariantOrigin.SOMATIC)
        self.profile.germline_variants = sum(1 for v in variants if v.variant_origin == VariantOrigin.GERMLINE)

        # Calculate TMB (somatic non-synonymous per Mb)
        # Assume 30 Mb exome coverage for MSK-IMPACT
        self.profile.tmb = round(self.profile.somatic_variants / 30.0, 2)

        # Map to SENTINEL format
        self._map_to_sentinel()

        # Generate output
        return self._format_output()

    def _validate_vcf(self, vcf_path: str) -> bool:
        """Valida formato VCF"""
        try:
            with open(vcf_path, 'r') as f:
                # Check header
                first_line = f.readline()
                if not first_line.startswith('##fileformat=VCF'):
                    logger.warning("Missing VCF header, attempting parse anyway")

                # Check columns
                for line in f:
                    if line.startswith('#CHROM'):
                        # Standard VCF columns
                        required = ['CHROM', 'POS', 'REF', 'ALT', 'INFO']
                        if all(col in line for col in required):
                            return True
                        break

                # If no header, assume custom format
                logger.warning("Non-standard VCF, using permissive parsing")
                return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def _parse_variants(self, vcf_path: str) -> List[Variant]:
        """Parse tutte le varianti dal VCF"""
        variants = []

        with open(vcf_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip headers
                if line.startswith('#'):
                    continue

                try:
                    variant = self._parse_variant_line(line, line_num)
                    if variant:
                        variants.append(variant)
                except Exception as e:
                    logger.warning(f"Line {line_num} parse error: {e}")
                    continue

        logger.info(f"✓ Parsed {len(variants)} variants")
        return variants

    def _parse_variant_line(self, line: str, line_num: int) -> Optional[Variant]:
        """Parse singola riga VCF"""
        parts = line.strip().split('\t')

        if len(parts) < 8:
            return None

        # Standard VCF columns
        chrom = parts[0]
        pos = int(parts[1]) if parts[1].isdigit() else 0
        ref = parts[3]
        alt = parts[4]
        info = parts[7]

        # Extract gene from INFO
        gene = self._extract_gene(info)
        if not gene or gene not in self.TARGET_PANEL:
            return None

        # Extract variant details
        variant_type = self._determine_variant_type(ref, alt, info)
        vaf = self._extract_vaf(info)
        depth = self._extract_depth(info)
        protein_change = self._extract_protein_change(info)
        consequence = self._extract_consequence(info)
        origin = self._determine_origin(info)

        variant = Variant(
            gene=gene,
            chromosome=chrom,
            position=pos,
            ref_allele=ref,
            alt_allele=alt,
            variant_type=variant_type,
            variant_origin=origin,
            vaf=vaf,
            depth=depth,
            protein_change=protein_change,
            consequence=consequence,
            raw_info=info
        )

        logger.debug(f"✓ {gene} {protein_change} (VAF: {vaf}%)")

        return variant

    # =========================================================================
    # EXTRACTION HELPERS
    # =========================================================================

    def _extract_gene(self, info: str) -> Optional[str]:
        """Estrae nome gene da INFO"""
        # Try multiple patterns
        patterns = [
            r'GENE=([A-Z0-9]+)',  # Standard
            r'Gene=([A-Z0-9]+)',  # Case variant
            r'gene=([A-Z0-9]+)',  # Lowercase
            r'SYMBOL=([A-Z0-9]+)',  # Some formats
            r';([A-Z0-9]+)(?:;|$)'  # Gene as single field
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                return match.group(1).upper()

        return None

    def _extract_vaf(self, info: str) -> Optional[float]:
        """Estrae Variant Allele Frequency"""
        # Common patterns
        patterns = [
            r'VAF=([0-9.]+)',  # VAF=0.35
            r'AF=([0-9.]+)',  # AF=0.35
            r'FREQ=([0-9.]+)%?',  # FREQ=35
            r'AlleleFreq=([0-9.]+)',  # AlleleFreq=0.35
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                vaf = float(match.group(1))
                # Normalize to 0-100 scale
                if vaf <= 1.0:
                    vaf *= 100
                return round(vaf, 2)

        return None

    def _extract_depth(self, info: str) -> Optional[int]:
        """Estrae read depth"""
        patterns = [
            r'DP=(\d+)',  # DP=150
            r'DEPTH=(\d+)',  # DEPTH=150
            r'COV=(\d+)',  # COV=150
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                return int(match.group(1))

        return None

    def _extract_protein_change(self, info: str) -> Optional[str]:
        """Estrae cambio proteico (p.G12C, p.L858R, etc.)"""
        patterns = [
            r'HGVSP=([^;]+)',  # HGVSP=p.G12C
            r'AAChange=([^;]+)',  # AAChange=G12C
            r'ProteinChange=([^;]+)',  # ProteinChange=p.G12C
            r'p\.([A-Z]\d+[A-Z*])',  # p.G12C inline
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                change = match.group(1)
                # Normalize format
                if not change.startswith('p.'):
                    change = f"p.{change}"
                return change

        return None

    def _extract_consequence(self, info: str) -> Optional[str]:
        """Estrae consequence type (missense, frameshift, etc.)"""
        patterns = [
            r'Consequence=([^;]+)',
            r'EFFECT=([^;]+)',
            r'VarType=([^;]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                return match.group(1).lower()

        # Infer from other info
        if 'frameshift' in info.lower():
            return 'frameshift'
        if 'missense' in info.lower():
            return 'missense'
        if 'nonsense' in info.lower() or 'stop_gained' in info.lower():
            return 'nonsense'

        return None

    def _determine_variant_type(self, ref: str, alt: str, info: str) -> VariantType:
        """Determina tipo di variante"""
        # Check INFO field first
        if 'CNV' in info or 'Copy' in info:
            return VariantType.CNV
        if 'Fusion' in info or 'FUSION' in info:
            return VariantType.FUSION

        # Check allele lengths
        if len(ref) == 1 and len(alt) == 1:
            return VariantType.SNV
        elif len(ref) != len(alt):
            return VariantType.INDEL

        return VariantType.UNKNOWN

    def _determine_origin(self, info: str) -> VariantOrigin:
        """Determina origine variante (somatic vs germline)"""
        if 'SOMATIC' in info.upper() or 'TUMOR' in info.upper():
            return VariantOrigin.SOMATIC
        elif 'GERMLINE' in info.upper():
            return VariantOrigin.GERMLINE

        # Default: assume somatic for cancer VCF
        return VariantOrigin.SOMATIC

    # =========================================================================
    # SENTINEL MAPPING
    # =========================================================================

    def _map_to_sentinel(self):
        """Mappa varianti al formato SENTINEL"""
        for variant in self.profile.variants:
            gene = variant.gene
            protein = variant.protein_change or ""

            # TP53
            if gene == 'TP53':
                self.profile.tp53_status = 'mutated'

            # KRAS
            elif gene == 'KRAS':
                # Try to extract specific variant
                for kras_var in self.KRAS_VARIANTS:
                    if kras_var in protein:
                        self.profile.kras_mutation = kras_var
                        break
                else:
                    self.profile.kras_mutation = 'Other'

            # EGFR
            elif gene == 'EGFR':
                for egfr_var in self.EGFR_VARIANTS:
                    if egfr_var in protein:
                        self.profile.egfr_mutation = egfr_var
                        break
                else:
                    self.profile.egfr_mutation = 'Other'

            # PIK3CA
            elif gene == 'PIK3CA':
                self.profile.pik3ca_status = 'mutated'

            # STK11
            elif gene == 'STK11':
                self.profile.stk11_status = 'mutated'

            # KEAP1
            elif gene == 'KEAP1':
                self.profile.keap1_status = 'mutated'

            # RB1
            elif gene == 'RB1':
                self.profile.rb1_status = 'loss'

            # BRAF
            elif gene == 'BRAF':
                if 'V600E' in protein:
                    self.profile.braf_mutation = 'V600E'
                else:
                    self.profile.braf_mutation = 'Other'

            # ALK
            elif gene == 'ALK':
                if variant.variant_type == VariantType.FUSION:
                    self.profile.alk_status = 'fusion'
                else:
                    self.profile.alk_status = 'mutated'

            # MET
            elif gene == 'MET':
                self.profile.met_status = 'mutated'
                # If CNV, extract copy number
                if variant.variant_type == VariantType.CNV:
                    cn = self._extract_copy_number(variant.raw_info)
                    if cn:
                        self.profile.met_cn = cn

            # HER2/ERBB2
            elif gene in ['HER2', 'ERBB2']:
                self.profile.her2_status = 'mutated'
                if variant.variant_type == VariantType.CNV:
                    cn = self._extract_copy_number(variant.raw_info)
                    if cn:
                        self.profile.her2_cn = cn

    def _extract_copy_number(self, info: str) -> Optional[float]:
        """Estrae copy number da INFO"""
        patterns = [
            r'CN=([0-9.]+)',
            r'CopyNumber=([0-9.]+)',
            r'COPY=([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, info)
            if match:
                return float(match.group(1))

        return None

    def _format_output(self) -> Dict:
        """Formatta output per SENTINEL"""
        return {
            # Basic info
            "patient_id": self.profile.patient_id,
            "total_variants": self.profile.total_variants,
            "somatic_variants": self.profile.somatic_variants,
            "germline_variants": self.profile.germline_variants,
            "tmb": self.profile.tmb,

            # Gene status (SENTINEL format)
            "tp53_status": self.profile.tp53_status,
            "kras_mutation": self.profile.kras_mutation,
            "egfr_mutation": self.profile.egfr_mutation,
            "pik3ca_status": self.profile.pik3ca_status,
            "stk11_status": self.profile.stk11_status,
            "keap1_status": self.profile.keap1_status,
            "rb1_status": self.profile.rb1_status,
            "braf_mutation": self.profile.braf_mutation,
            "alk_status": self.profile.alk_status,
            "met_status": self.profile.met_status,
            "her2_status": self.profile.her2_status,

            # Copy numbers
            "met_cn": self.profile.met_cn,
            "her2_cn": self.profile.her2_cn,
            "egfr_cn": self.profile.egfr_cn,

            # Raw variants (for detailed analysis)
            "raw_mutations_found": [v.gene for v in self.profile.variants],
            "detailed_variants": [
                {
                    "gene": v.gene,
                    "protein_change": v.protein_change,
                    "vaf": v.vaf,
                    "depth": v.depth,
                    "type": v.variant_type.value,
                    "origin": v.variant_origin.value
                }
                for v in self.profile.variants
            ]
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vcf_parser.py <vcf_file> [patient_id]")
        sys.exit(1)

    vcf_file = sys.argv[1]
    patient_id = sys.argv[2] if len(sys.argv) > 2 else "TEST"

    parser = SentinelGenomics(patient_id=patient_id)
    result = parser.parse_vcf(vcf_file)

    # Pretty print
    import json

    print("\n" + "=" * 70)
    print("SENTINEL GENOMICS PARSER v2.0")
    print("=" * 70)
    print(json.dumps(result, indent=2))