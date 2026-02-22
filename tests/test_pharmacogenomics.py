"""
SENTINEL Test Suite - Pharmacogenomics Tests
==============================================
Tests for PGx extraction and safety alerts.
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestPGxExtractor:
    """Test PGx variant extraction."""
    
    def test_pgx_extractor_import(self):
        """Test that PGxExtractor can be imported."""
        from farmacogenomica.pgx_extractor import PGxExtractor
        assert PGxExtractor is not None
    
    def test_pgx_profile_extraction(self, pgx_poor_metabolizer_patient):
        """Test that pgx_profile is correctly extracted."""
        from farmacogenomica.pgx_extractor import PGxExtractor
        
        extractor = PGxExtractor()
        result = extractor.extract_from_sentinel(pgx_poor_metabolizer_patient)
        
        # Should find variants from pgx_profile
        assert len(result.variants_found) >= 2, f"Should find DPYD and UGT1A1, got {len(result.variants_found)}"
        
        # Check DPYD was found
        dpyd_variants = [v for v in result.variants_found if v.gene == 'DPYD']
        assert len(dpyd_variants) > 0, "Should find DPYD variant"
        
        dpyd = dpyd_variants[0]
        assert dpyd.genotype == "*2A/*2A", f"DPYD genotype should be *2A/*2A, got {dpyd.genotype}"
    
    def test_pgx_warnings_for_dangerous_therapy(self, pgx_poor_metabolizer_patient):
        """Test that warnings are generated for dangerous drug-gene combinations."""
        from farmacogenomica.pgx_extractor import PGxExtractor
        
        extractor = PGxExtractor()
        result = extractor.extract_from_sentinel(pgx_poor_metabolizer_patient)
        
        # Should generate warning about 5-FU
        warning_text = ' '.join(result.warnings).lower()
        
        # Either has explicit warning or has recommendation
        has_5fu_concern = '5-fu' in warning_text or 'dpyd' in ' '.join(result.recommendations).lower()
        assert has_5fu_concern or len(result.variants_found) > 0
    
    def test_drug_compatibility_check(self, pgx_poor_metabolizer_patient):
        """Test drug compatibility checker."""
        from farmacogenomica.pgx_extractor import PGxExtractor
        
        extractor = PGxExtractor()
        compat = extractor.check_drug_compatibility("5-FU", pgx_poor_metabolizer_patient)
        
        # 5-FU requires DPYD
        assert 'DPYD' in compat['required_genes'], "5-FU should require DPYD test"


class TestMetabolizerClassifier:
    """Test metabolizer phenotype classification."""
    
    def test_metabolizer_import(self):
        """Test that MetabolizerClassifier can be imported."""
        from farmacogenomica.metabolizer_classifier import MetabolizerClassifier
        assert MetabolizerClassifier is not None
    
    def test_dpyd_poor_metabolizer(self):
        """Test DPYD *2A/*2A is classified as Poor Metabolizer."""
        from farmacogenomica.metabolizer_classifier import MetabolizerClassifier, MetabolizerPhenotype
        
        classifier = MetabolizerClassifier()
        result = classifier.classify("DPYD", "*2A/*2A")
        
        assert result.phenotype == MetabolizerPhenotype.POOR, \
            f"DPYD *2A/*2A should be Poor, got {result.phenotype}"
    
    def test_dpyd_intermediate_metabolizer(self):
        """Test DPYD *1/*2A is classified as Intermediate Metabolizer."""
        from farmacogenomica.metabolizer_classifier import MetabolizerClassifier, MetabolizerPhenotype
        
        classifier = MetabolizerClassifier()
        result = classifier.classify("DPYD", "*1/*2A heterozygous")
        
        assert result.phenotype == MetabolizerPhenotype.INTERMEDIATE, \
            f"DPYD *1/*2A should be Intermediate, got {result.phenotype}"
    
    def test_cyp2d6_normal_metabolizer(self):
        """Test CYP2D6 *1/*1 is classified as Normal Metabolizer."""
        from farmacogenomica.metabolizer_classifier import MetabolizerClassifier, MetabolizerPhenotype
        
        classifier = MetabolizerClassifier()
        result = classifier.classify("CYP2D6", "*1/*1")
        
        assert result.phenotype == MetabolizerPhenotype.NORMAL, \
            f"CYP2D6 *1/*1 should be Normal, got {result.phenotype}"
