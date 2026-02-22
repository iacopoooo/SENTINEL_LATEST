#!/usr/bin/env python3
"""
SENTINEL - AlphaGenome Integration Module v2
=============================================
Analyzes non-coding variants, splicing alterations, and regulatory elements
using Google DeepMind's AlphaGenome model.

Features:
- Splicing variant effect prediction (MET exon 14 skipping, etc.)
- Promoter/enhancer variant analysis
- Gene expression impact prediction
- Proper parsing of AlphaGenome TrackData output

Usage:
    from alphagenome_integration import AlphaGenomeAnalyzer
    
    ag = AlphaGenomeAnalyzer(api_key="YOUR_KEY")
    result = ag.analyze_splicing_variant("MET", "chr7", 116771994, "G", "A")
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import AlphaGenome
try:
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    ALPHAGENOME_AVAILABLE = True
    logger.info("✓ AlphaGenome Module Loaded")
except ImportError as e:
    ALPHAGENOME_AVAILABLE = False
    logger.warning(f"⚠️ AlphaGenome not available: {e}")

# Paths
BASE_DIR = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
KEY_FILE = BASE_DIR / 'alphagnome_key.txt'
# Rimosso path assoluto hardcoded - usa variabile d'ambiente ALPHAGENOME_API_KEY

# Supported sequence lengths by AlphaGenome
SUPPORTED_LENGTHS = [16384, 131072, 524288, 1048576]
DEFAULT_CONTEXT_SIZE = 131072  # 128kb - good balance of context and speed

# Tissue ontology - only confirmed supported terms
LUNG_ONTOLOGY_TERMS = ['UBERON:0002048']  # Lung

# Oncology-relevant genomic regions for splicing analysis
SPLICING_HOTSPOTS = {
    'MET': {
        'exon14_skip': {
            'chromosome': 'chr7',
            'region': (116771000, 116772500),
            'description': 'MET exon 14 skipping - activating alteration',
            'clinical': 'ACTIONABLE: Sensitive to MET inhibitors (capmatinib, tepotinib)',
            'exon': 'Exon 14',
        },
    },
    'EGFR': {
        'exon19_del': {
            'chromosome': 'chr7',
            'region': (55242400, 55242600),
            'description': 'EGFR exon 19 deletion region',
            'clinical': 'ACTIONABLE: Sensitive to EGFR TKIs (osimertinib)',
            'exon': 'Exon 19',
        },
        'exon20_ins': {
            'chromosome': 'chr7',
            'region': (55248900, 55249200),
            'description': 'EGFR exon 20 insertion region',
            'clinical': 'Resistant to 1st/2nd gen TKIs, sensitive to amivantamab/mobocertinib',
            'exon': 'Exon 20',
        },
    },
    'ALK': {
        'fusion_breakpoint': {
            'chromosome': 'chr2',
            'region': (29415000, 29450000),
            'description': 'ALK kinase domain - fusion breakpoint region',
            'clinical': 'ACTIONABLE: ALK fusions sensitive to ALK inhibitors (alectinib, lorlatinib)',
            'exon': 'Kinase domain',
        },
    },
    'ROS1': {
        'fusion_breakpoint': {
            'chromosome': 'chr6',
            'region': (117600000, 117750000),
            'description': 'ROS1 kinase domain - fusion breakpoint region',
            'clinical': 'ACTIONABLE: ROS1 fusions sensitive to crizotinib, entrectinib',
            'exon': 'Kinase domain',
        },
    },
    'RET': {
        'fusion_breakpoint': {
            'chromosome': 'chr10',
            'region': (43572000, 43625000),
            'description': 'RET kinase domain - fusion breakpoint region',
            'clinical': 'ACTIONABLE: RET fusions sensitive to selpercatinib, pralsetinib',
            'exon': 'Kinase domain',
        },
    },
}

# Gene coordinates (GRCh38/hg38)
GENE_COORDINATES = {
    'MET': {'chr': 'chr7', 'start': 116672196, 'end': 116798377},
    'EGFR': {'chr': 'chr7', 'start': 55019017, 'end': 55211628},
    'KRAS': {'chr': 'chr12', 'start': 25205246, 'end': 25250929},
    'BRAF': {'chr': 'chr7', 'start': 140719327, 'end': 140924929},
    'ALK': {'chr': 'chr2', 'start': 29192774, 'end': 29921566},
    'ROS1': {'chr': 'chr6', 'start': 117288300, 'end': 117425850},
    'RET': {'chr': 'chr10', 'start': 43077027, 'end': 43130351},
    'TP53': {'chr': 'chr17', 'start': 7668402, 'end': 7687538},
    'STK11': {'chr': 'chr19', 'start': 1205798, 'end': 1228431},
    'KEAP1': {'chr': 'chr19', 'start': 10486892, 'end': 10503355},
}


@dataclass
class SplicingResult:
    """Results from splicing analysis."""
    gene: str
    chromosome: str
    position: int
    ref: str
    alt: str
    splicing_impact: str  # HIGH, MEDIUM, LOW, NONE, UNKNOWN
    delta_splice_score: Optional[float]  # Max change in splice site score
    delta_psi: Optional[float]  # Estimated change in percent spliced in
    affected_exon: Optional[str]
    mechanism: str
    clinical_relevance: str
    expression_change: Optional[float]  # Fold change in expression
    confidence: str  # HIGH, MEDIUM, LOW
    splice_site_type: Optional[str]  # DONOR, ACCEPTOR, or None
    ref_max_score: Optional[float]
    alt_max_score: Optional[float]
    error: Optional[str] = None


@dataclass 
class RegulatoryResult:
    """Results from regulatory element analysis."""
    gene: str
    chromosome: str
    position: int
    ref: str
    alt: str
    element_type: str  # PROMOTER, ENHANCER, SILENCER, INSULATOR
    impact: str  # HIGH, MEDIUM, LOW
    expression_fold_change: Optional[float]
    affected_tissues: List[str] = field(default_factory=list)
    confidence: str = 'MEDIUM'
    error: Optional[str] = None


class AlphaGenomeAnalyzer:
    """
    AlphaGenome integration for SENTINEL.
    Analyzes splicing variants and regulatory elements.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize AlphaGenome analyzer."""
        self.api_key = api_key or self._load_api_key()
        self.model = None
        self._available = False
        
        if not ALPHAGENOME_AVAILABLE:
            logger.warning("AlphaGenome package not installed")
            return
        
        if not self.api_key:
            logger.warning("AlphaGenome API key not found")
            return
        
        try:
            self.model = dna_client.create(self.api_key)
            self._available = True
            logger.info("✓ AlphaGenome API Connected")
        except Exception as e:
            logger.error(f"Failed to connect to AlphaGenome API: {e}")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment variable or file."""
        # 1. Prova variabile d'ambiente
        env_key = os.getenv("ALPHAGENOME_API_KEY", "").strip()
        if env_key:
            logger.info("AlphaGenome API key loaded from environment variable")
            return env_key
        
        # 2. Fallback: leggi da file locale
        if KEY_FILE.exists():
            try:
                with open(KEY_FILE, 'r') as f:
                    key = f.read().strip()
                    if key:
                        logger.info(f"API key loaded from {KEY_FILE}")
                        return key
            except Exception as e:
                logger.warning(f"Failed to read key from {KEY_FILE}: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if AlphaGenome is ready."""
        return self._available and self.model is not None
    
    def _calculate_interval(self, position: int, context_size: int = DEFAULT_CONTEXT_SIZE) -> Tuple[int, int]:
        """Calculate interval that fits supported lengths."""
        half = context_size // 2
        start = max(1, position - half)
        end = start + context_size
        return start, end
    
    def analyze_splicing_variant(self, gene: str, chromosome: str, 
                                  position: int, ref: str, alt: str,
                                  context_size: int = DEFAULT_CONTEXT_SIZE) -> SplicingResult:
        """
        Analyze a variant's effect on splicing using AlphaGenome.
        
        Args:
            gene: Gene symbol
            chromosome: Chromosome (e.g., "chr7")
            position: Genomic position (1-based, GRCh38)
            ref: Reference allele
            alt: Alternate allele
            context_size: Size of genomic context (must be in SUPPORTED_LENGTHS)
        
        Returns:
            SplicingResult with splicing impact analysis
        """
        # Validate context size
        if context_size not in SUPPORTED_LENGTHS:
            context_size = DEFAULT_CONTEXT_SIZE
        
        if not self.is_available():
            return SplicingResult(
                gene=gene, chromosome=chromosome, position=position,
                ref=ref, alt=alt, splicing_impact='UNKNOWN',
                delta_splice_score=None, delta_psi=None, affected_exon=None,
                mechanism='AlphaGenome not available',
                clinical_relevance=self._get_clinical_relevance(gene, chromosome, position),
                expression_change=None, confidence='NONE',
                splice_site_type=None, ref_max_score=None, alt_max_score=None,
                error='AlphaGenome API not available'
            )
        
        try:
            # Calculate interval
            start, end = self._calculate_interval(position, context_size)
            
            interval = genome.Interval(
                chromosome=chromosome,
                start=start,
                end=end
            )
            
            variant = genome.Variant(
                chromosome=chromosome,
                position=position,
                reference_bases=ref,
                alternate_bases=alt
            )
            
            # Get predictions
            outputs = self.model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=LUNG_ONTOLOGY_TERMS,
                requested_outputs=[
                    dna_client.OutputType.RNA_SEQ,
                    dna_client.OutputType.SPLICE_SITES,
                ]
            )
            
            # Calculate position index within the interval
            pos_index = position - start
            
            # Analyze splicing impact
            splice_result = self._analyze_splice_scores(outputs, pos_index)
            
            # Analyze expression change
            expression_change = self._calculate_expression_change(outputs, pos_index)
            
            # Get clinical relevance
            clinical_relevance = self._get_clinical_relevance(gene, chromosome, position)
            
            # Get affected exon
            affected_exon = self._get_affected_exon(gene, chromosome, position)
            
            # Determine overall impact and mechanism
            splicing_impact, mechanism, confidence = self._determine_impact(
                splice_result, expression_change, gene, position
            )
            
            return SplicingResult(
                gene=gene,
                chromosome=chromosome,
                position=position,
                ref=ref,
                alt=alt,
                splicing_impact=splicing_impact,
                delta_splice_score=splice_result.get('delta_max'),
                delta_psi=splice_result.get('estimated_delta_psi'),
                affected_exon=affected_exon,
                mechanism=mechanism,
                clinical_relevance=clinical_relevance,
                expression_change=expression_change,
                confidence=confidence,
                splice_site_type=splice_result.get('site_type'),
                ref_max_score=splice_result.get('ref_max'),
                alt_max_score=splice_result.get('alt_max')
            )
            
        except Exception as e:
            logger.error(f"Splicing analysis failed: {e}")
            return SplicingResult(
                gene=gene, chromosome=chromosome, position=position,
                ref=ref, alt=alt, splicing_impact='ERROR',
                delta_splice_score=None, delta_psi=None, affected_exon=None,
                mechanism=str(e),
                clinical_relevance=self._get_clinical_relevance(gene, chromosome, position),
                expression_change=None, confidence='NONE',
                splice_site_type=None, ref_max_score=None, alt_max_score=None,
                error=str(e)
            )
    
    def _analyze_splice_scores(self, outputs, pos_index: int, window: int = 50) -> Dict:
        """
        Analyze splice site scores around the variant position.
        
        AlphaGenome SPLICE_SITES output has shape (length, 4):
        - Track 0: Donor site, positive strand
        - Track 1: Donor site, negative strand
        - Track 2: Acceptor site, positive strand
        - Track 3: Acceptor site, negative strand
        """
        result = {
            'delta_max': None,
            'estimated_delta_psi': None,
            'ref_max': None,
            'alt_max': None,
            'site_type': None,
        }
        
        try:
            ref_splice = outputs.reference.splice_sites
            alt_splice = outputs.alternate.splice_sites
            
            if ref_splice is None or alt_splice is None:
                return result
            
            ref_values = ref_splice.values  # Shape: (131072, 4)
            alt_values = alt_splice.values
            
            # Define window around variant
            start_idx = max(0, pos_index - window)
            end_idx = min(len(ref_values), pos_index + window)
            
            # Extract window
            ref_window = ref_values[start_idx:end_idx]
            alt_window = alt_values[start_idx:end_idx]
            
            # Calculate max scores across all tracks in window
            ref_max = float(np.max(ref_window))
            alt_max = float(np.max(alt_window))
            
            # Calculate delta
            delta_max = alt_max - ref_max
            
            # Determine which splice site type is most affected
            ref_donor = np.max(ref_window[:, :2])  # Tracks 0-1
            alt_donor = np.max(alt_window[:, :2])
            ref_acceptor = np.max(ref_window[:, 2:])  # Tracks 2-3
            alt_acceptor = np.max(alt_window[:, 2:])
            
            donor_delta = abs(alt_donor - ref_donor)
            acceptor_delta = abs(alt_acceptor - ref_acceptor)
            
            if donor_delta > acceptor_delta:
                site_type = 'DONOR'
            elif acceptor_delta > donor_delta:
                site_type = 'ACCEPTOR'
            else:
                site_type = None
            
            # Estimate delta PSI (simplified model)
            # High splice score change → likely splicing effect
            # Scores are typically 0-1, with >0.5 being strong splice sites
            if ref_max > 0.1 or alt_max > 0.1:  # Only if there's a real splice site
                estimated_delta_psi = delta_max * 100  # Convert to percentage
            else:
                estimated_delta_psi = None
            
            result = {
                'delta_max': delta_max,
                'estimated_delta_psi': estimated_delta_psi,
                'ref_max': ref_max,
                'alt_max': alt_max,
                'site_type': site_type,
                'donor_delta': float(donor_delta),
                'acceptor_delta': float(acceptor_delta),
            }
            
        except Exception as e:
            logger.warning(f"Splice score analysis error: {e}")
        
        return result
    
    def _calculate_expression_change(self, outputs, pos_index: int, window: int = 1000) -> Optional[float]:
        """Calculate expression fold change from RNA-seq predictions."""
        try:
            ref_rna = outputs.reference.rna_seq
            alt_rna = outputs.alternate.rna_seq
            
            if ref_rna is None or alt_rna is None:
                return None
            
            ref_values = ref_rna.values
            alt_values = alt_rna.values
            
            # Get window around variant
            start_idx = max(0, pos_index - window)
            end_idx = min(len(ref_values), pos_index + window)
            
            ref_window = ref_values[start_idx:end_idx]
            alt_window = alt_values[start_idx:end_idx]
            
            # Calculate mean expression
            ref_mean = float(np.mean(ref_window))
            alt_mean = float(np.mean(alt_window))
            
            if ref_mean > 0.001:  # Avoid division by zero
                fold_change = alt_mean / ref_mean
                return round(fold_change, 3)
            
            return None
            
        except Exception as e:
            logger.warning(f"Expression change calculation error: {e}")
            return None
    
    def _determine_impact(self, splice_result: Dict, expression_change: Optional[float],
                          gene: str, position: int) -> Tuple[str, str, str]:
        """Determine overall splicing impact, mechanism, and confidence."""
        
        delta_max = splice_result.get('delta_max')
        delta_psi = splice_result.get('estimated_delta_psi')
        site_type = splice_result.get('site_type')
        
        # Check for known hotspots first
        is_hotspot = False
        if gene.upper() in SPLICING_HOTSPOTS:
            for hotspot in SPLICING_HOTSPOTS[gene.upper()].values():
                start, end = hotspot['region']
                if start <= position <= end:
                    is_hotspot = True
                    break
        
        # Determine impact level
        if delta_max is None:
            impact = 'UNKNOWN'
            mechanism = 'Could not calculate splice site changes'
            confidence = 'LOW'
        elif abs(delta_max) > 0.3:
            impact = 'HIGH'
            if site_type == 'DONOR':
                mechanism = 'Major disruption of splice donor site'
            elif site_type == 'ACCEPTOR':
                mechanism = 'Major disruption of splice acceptor site'
            else:
                mechanism = 'Major splice site disruption'
            confidence = 'HIGH'
        elif abs(delta_max) > 0.1:
            impact = 'MEDIUM'
            mechanism = f'Moderate {site_type.lower() if site_type else "splice"} site alteration'
            confidence = 'MEDIUM'
        elif abs(delta_max) > 0.03:
            impact = 'LOW'
            mechanism = 'Minor splicing effect predicted'
            confidence = 'MEDIUM'
        else:
            impact = 'NONE'
            mechanism = 'No significant splicing impact predicted'
            confidence = 'HIGH'
        
        # Boost confidence for known hotspots
        if is_hotspot and impact in ['HIGH', 'MEDIUM']:
            confidence = 'HIGH'
            mechanism += ' (known splicing hotspot)'
        
        # Add expression context if available
        if expression_change is not None:
            if expression_change < 0.5:
                mechanism += f'; Expression reduced to {expression_change:.1%}'
            elif expression_change > 2.0:
                mechanism += f'; Expression increased {expression_change:.1f}x'
        
        return impact, mechanism, confidence
    
    def _get_clinical_relevance(self, gene: str, chromosome: str, position: int) -> str:
        """Get clinical relevance for known splicing hotspots."""
        if gene.upper() in SPLICING_HOTSPOTS:
            for hotspot_name, hotspot in SPLICING_HOTSPOTS[gene.upper()].items():
                if hotspot['chromosome'] == chromosome:
                    start, end = hotspot['region']
                    if start <= position <= end:
                        return hotspot['clinical']
        return 'Clinical significance requires expert review'
    
    def _get_affected_exon(self, gene: str, chromosome: str, position: int) -> Optional[str]:
        """Determine which exon is affected by the variant."""
        if gene.upper() in SPLICING_HOTSPOTS:
            for hotspot in SPLICING_HOTSPOTS[gene.upper()].values():
                if hotspot['chromosome'] == chromosome:
                    start, end = hotspot['region']
                    if start <= position <= end:
                        return hotspot.get('exon')
        return None
    
    def analyze_met_exon14(self, chromosome: str = 'chr7', 
                           position: int = None, 
                           ref: str = None, 
                           alt: str = None) -> SplicingResult:
        """
        Specialized analysis for MET exon 14 skipping variants.
        
        MET exon 14 skipping is a clinically actionable alteration in NSCLC,
        sensitive to MET inhibitors (capmatinib, tepotinib).
        """
        if position is None:
            position = 116771994  # Common MET exon 14 splice donor
        if ref is None:
            ref = 'G'
        if alt is None:
            alt = 'A'
        
        result = self.analyze_splicing_variant('MET', chromosome, position, ref, alt)
        
        # Enhance with MET-specific information for known splice sites
        if result.splicing_impact in ['HIGH', 'MEDIUM', 'UNKNOWN']:
            result.clinical_relevance = (
                'MET exon 14 skipping - ACTIONABLE. '
                'FDA-approved: capmatinib, tepotinib. '
                'ORR ~40-50% in MET ex14 altered NSCLC.'
            )
            if result.mechanism and 'hotspot' not in result.mechanism.lower():
                result.mechanism += ' (MET exon 14 splice region)'
        
        return result
    
    def analyze_regulatory_variant(self, gene: str, chromosome: str,
                                    position: int, ref: str, alt: str) -> RegulatoryResult:
        """Analyze a variant's effect on regulatory elements."""
        if not self.is_available():
            return RegulatoryResult(
                gene=gene, chromosome=chromosome, position=position,
                ref=ref, alt=alt, element_type='UNKNOWN',
                impact='UNKNOWN', expression_fold_change=None,
                confidence='NONE', error='AlphaGenome not available'
            )
        
        try:
            start, end = self._calculate_interval(position)
            
            interval = genome.Interval(chromosome=chromosome, start=start, end=end)
            variant = genome.Variant(
                chromosome=chromosome, position=position,
                reference_bases=ref, alternate_bases=alt
            )
            
            outputs = self.model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=LUNG_ONTOLOGY_TERMS,
                requested_outputs=[
                    dna_client.OutputType.RNA_SEQ,
                    dna_client.OutputType.DNASE,
                ]
            )
            
            pos_index = position - start
            fold_change = self._calculate_expression_change(outputs, pos_index)
            
            # Determine impact
            if fold_change is None:
                impact = 'UNKNOWN'
            elif fold_change < 0.5 or fold_change > 2.0:
                impact = 'HIGH'
            elif fold_change < 0.7 or fold_change > 1.5:
                impact = 'MEDIUM'
            else:
                impact = 'LOW'
            
            return RegulatoryResult(
                gene=gene,
                chromosome=chromosome,
                position=position,
                ref=ref,
                alt=alt,
                element_type='REGULATORY',
                impact=impact,
                expression_fold_change=fold_change,
                affected_tissues=['Lung'],
                confidence='MEDIUM'
            )
            
        except Exception as e:
            logger.error(f"Regulatory analysis failed: {e}")
            return RegulatoryResult(
                gene=gene, chromosome=chromosome, position=position,
                ref=ref, alt=alt, element_type='ERROR',
                impact='UNKNOWN', expression_fold_change=None,
                confidence='NONE', error=str(e)
            )
    
    def batch_analyze_variants(self, variants: List[Dict]) -> List[SplicingResult]:
        """Analyze multiple variants."""
        results = []
        for v in variants:
            result = self.analyze_splicing_variant(
                gene=v.get('gene', 'Unknown'),
                chromosome=v.get('chromosome', v.get('chr', '')),
                position=v.get('position', v.get('pos', 0)),
                ref=v.get('ref', v.get('reference', '')),
                alt=v.get('alt', v.get('alternate', ''))
            )
            results.append(result)
        return results


# Singleton instance
_instance: Optional[AlphaGenomeAnalyzer] = None


def get_instance() -> Optional[AlphaGenomeAnalyzer]:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        try:
            _instance = AlphaGenomeAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize AlphaGenome: {e}")
            return None
    return _instance


def analyze_splicing(gene: str, chromosome: str, position: int, 
                     ref: str, alt: str) -> Optional[SplicingResult]:
    """Quick function to analyze a splicing variant."""
    instance = get_instance()
    if instance:
        return instance.analyze_splicing_variant(gene, chromosome, position, ref, alt)
    return None


def is_available() -> bool:
    """Check if AlphaGenome is available."""
    return ALPHAGENOME_AVAILABLE


# ============================================================
# TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AlphaGenome Integration Module v2 - Test")
    print("=" * 60)
    
    print(f"\nAlphaGenome package available: {ALPHAGENOME_AVAILABLE}")
    
    ag = AlphaGenomeAnalyzer()
    print(f"AlphaGenome API connected: {ag.is_available()}")
    
    if ag.is_available():
        print("\n🧬 Testing MET Exon 14 Skipping Analysis...")
        
        result = ag.analyze_met_exon14()
        
        print(f"\nMET Exon 14 Analysis:")
        print(f"   Position: {result.chromosome}:{result.position}")
        print(f"   Variant: {result.ref}>{result.alt}")
        print(f"   Splicing Impact: {result.splicing_impact}")
        print(f"   Delta Splice Score: {result.delta_splice_score:.4f}" if result.delta_splice_score else "   Delta Splice Score: N/A")
        print(f"   Ref Max Score: {result.ref_max_score:.4f}" if result.ref_max_score else "   Ref Max Score: N/A")
        print(f"   Alt Max Score: {result.alt_max_score:.4f}" if result.alt_max_score else "   Alt Max Score: N/A")
        print(f"   Splice Site Type: {result.splice_site_type or 'N/A'}")
        print(f"   Expression Change: {result.expression_change:.2f}x" if result.expression_change else "   Expression Change: N/A")
        print(f"   Affected Exon: {result.affected_exon or 'N/A'}")
        print(f"   Mechanism: {result.mechanism}")
        print(f"   Clinical: {result.clinical_relevance}")
        print(f"   Confidence: {result.confidence}")
        
        if result.error:
            print(f"   Error: {result.error}")
        
        print("\n🧬 Testing EGFR Exon 19 Region...")
        result2 = ag.analyze_splicing_variant(
            gene='EGFR',
            chromosome='chr7',
            position=55242470,
            ref='G',
            alt='A'
        )
        
        print(f"\nEGFR Exon 19 Analysis:")
        print(f"   Splicing Impact: {result2.splicing_impact}")
        print(f"   Delta Splice Score: {result2.delta_splice_score:.4f}" if result2.delta_splice_score else "   Delta Splice Score: N/A")
        print(f"   Clinical: {result2.clinical_relevance}")
        print(f"   Confidence: {result2.confidence}")
        
    else:
        print("\n⚠️ AlphaGenome API not available")
        print(f"   Ensure API key is in: {KEY_FILE}")
    
    print("\n✅ Test complete!")
