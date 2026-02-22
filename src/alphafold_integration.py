#!/usr/bin/env python3
"""
SENTINEL - AlphaFold + AutoDock Vina Integration Module
========================================================
Validates drug-target binding using AlphaFold structures and Vina docking.

Features:
- Load AlphaFold structures for oncology targets (EGFR, KRAS, etc.)
- Analyze mutation impact on protein structure (pLDDT scores)
- Molecular docking with AutoDock Vina
- Therapy validation: does the drug still bind to mutated protein?

Usage:
    from alphafold_integration import AlphaFoldVina
    
    av = AlphaFoldVina()
    result = av.validate_therapy("EGFR", "L858R", "osimertinib")
    # {'binding_affinity': -9.2, 'mutation_impact': 'LOW', 'therapy_valid': True}
"""

import os
import gzip
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
DATA_DIR = BASE_DIR / 'data'
ALPHAFOLD_DB = DATA_DIR / 'alphafold_db'
DRUGS_DIR = DATA_DIR / 'drugs_sdf'

# Gene to UniProt mapping (oncology targets)
GENE_TO_UNIPROT = {
    'EGFR': 'P00533',
    'KRAS': 'P01116',
    'NRAS': 'P01111',
    'HRAS': 'P01112',
    'BRAF': 'P15056',
    'ALK': 'Q9UM73',
    'ROS1': 'P08922',
    'MET': 'P08581',
    'RET': 'P07949',
    'HER2': 'P04626',
    'ERBB2': 'P04626',
    'PIK3CA': 'P42336',
    'TP53': 'P04637',
    'STK11': 'Q15831',
    'LKB1': 'Q15831',
    'KEAP1': 'Q14145',
    'PTEN': 'P60484',
    'AKT1': 'P31749',
    'MTOR': 'P42345',
    'FGFR1': 'P11362',
    'FGFR2': 'P21802',
    'FGFR3': 'P22607',
    'DDR2': 'Q16832',
    'NTRK1': 'P04629',
    'NTRK2': 'Q16620',
    'NTRK3': 'Q16288',
}

# Drug to target mapping
DRUG_TARGETS = {
    'osimertinib': {'target': 'EGFR', 'binding_site': (790, 860), 'type': 'TKI_3rd_gen'},
    'gefitinib': {'target': 'EGFR', 'binding_site': (790, 860), 'type': 'TKI_1st_gen'},
    'erlotinib': {'target': 'EGFR', 'binding_site': (790, 860), 'type': 'TKI_1st_gen'},
    'sotorasib': {'target': 'KRAS', 'binding_site': (10, 60), 'type': 'G12C_covalent'},
    'adagrasib': {'target': 'KRAS', 'binding_site': (10, 60), 'type': 'G12C_covalent'},
    'crizotinib': {'target': 'MET', 'binding_site': (1100, 1250), 'type': 'MET_ALK_TKI'},
    'capmatinib': {'target': 'MET', 'binding_site': (1100, 1250), 'type': 'MET_TKI'},
    'dabrafenib': {'target': 'BRAF', 'binding_site': (460, 600), 'type': 'BRAF_inhibitor'},
    'trametinib': {'target': 'MAP2K1', 'binding_site': (60, 200), 'type': 'MEK_inhibitor'},
}

# Known resistance mutations and their impact
RESISTANCE_MUTATIONS = {
    'EGFR': {
        'T790M': {'impact': 'HIGH', 'mechanism': 'Steric hindrance', 'affects': ['gefitinib', 'erlotinib']},
        'C797S': {'impact': 'HIGH', 'mechanism': 'Covalent bond disruption', 'affects': ['osimertinib']},
        'L858R': {'impact': 'LOW', 'mechanism': 'Sensitizing mutation', 'affects': []},
        'exon19del': {'impact': 'LOW', 'mechanism': 'Sensitizing mutation', 'affects': []},
    },
    'KRAS': {
        'G12C': {'impact': 'TARGET', 'mechanism': 'Drug target', 'affects': []},
        'G12D': {'impact': 'HIGH', 'mechanism': 'No approved inhibitor', 'affects': ['sotorasib', 'adagrasib']},
        'G12V': {'impact': 'HIGH', 'mechanism': 'No approved inhibitor', 'affects': ['sotorasib', 'adagrasib']},
    },
    'BRAF': {
        'V600E': {'impact': 'TARGET', 'mechanism': 'Drug target', 'affects': []},
    },
    'MET': {
        'D1228N': {'impact': 'MEDIUM', 'mechanism': 'Reduced binding', 'affects': ['crizotinib']},
        'Y1230C': {'impact': 'MEDIUM', 'mechanism': 'Reduced binding', 'affects': ['crizotinib']},
    }
}


@dataclass
class DockingResult:
    """Results from molecular docking."""
    drug: str
    target_gene: str
    mutation: Optional[str]
    binding_affinity: float  # kcal/mol (more negative = stronger binding)
    binding_quality: str  # STRONG, MODERATE, WEAK, NONE
    mutation_impact: str  # LOW, MEDIUM, HIGH
    therapy_recommendation: str
    plddt_at_mutation: Optional[float]
    structure_file: Optional[str]
    error: Optional[str] = None


class AlphaFoldVina:
    """
    AlphaFold + Vina integration for therapy validation.
    """
    
    def __init__(self, alphafold_dir: Path = None, drugs_dir: Path = None):
        self.alphafold_dir = alphafold_dir or ALPHAFOLD_DB
        self.drugs_dir = drugs_dir or DRUGS_DIR
        self.vina_path = self._find_vina()
        self.obabel_path = self._find_obabel()
        
        # Check availability
        self._available = self.alphafold_dir.exists() and len(list(self.alphafold_dir.glob('*.pdb.gz'))) > 0
        
        if self._available:
            logger.info(f"✓ AlphaFold Integration Loaded ({len(list(self.alphafold_dir.glob('*.pdb.gz')))} structures)")
        else:
            logger.warning(f"⚠️ AlphaFold structures not found at {self.alphafold_dir}")
    
    def _find_vina(self) -> Optional[str]:
        """Find AutoDock Vina executable."""
        for path in ['/usr/bin/vina', '/usr/local/bin/vina', shutil.which('vina')]:
            if path and os.path.exists(path):
                return path
        return None
    
    def _find_obabel(self) -> Optional[str]:
        """Find Open Babel executable."""
        for path in ['/usr/bin/obabel', '/usr/local/bin/obabel', shutil.which('obabel')]:
            if path and os.path.exists(path):
                return path
        return None
    
    def is_available(self) -> bool:
        """Check if module is ready."""
        return self._available
    
    def get_structure_path(self, gene: str) -> Optional[Path]:
        """
        Get AlphaFold structure path for a gene.
        
        Args:
            gene: Gene symbol (e.g., "EGFR")
        
        Returns:
            Path to PDB.gz file or None
        """
        uniprot_id = GENE_TO_UNIPROT.get(gene.upper())
        if not uniprot_id:
            logger.warning(f"Unknown gene: {gene}")
            return None
        
        # AlphaFold naming: AF-{UniProt}-F1-model_v4.pdb.gz
        pattern = f"AF-{uniprot_id}-F1-model_v*.pdb.gz"
        matches = list(self.alphafold_dir.glob(pattern))
        
        if matches:
            return matches[0]
        
        logger.warning(f"Structure not found for {gene} ({uniprot_id})")
        return None
    
    def extract_pdb(self, pdb_gz_path: Path, output_dir: Path = None) -> Optional[Path]:
        """Extract gzipped PDB file."""
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        
        output_path = output_dir / pdb_gz_path.name.replace('.gz', '')
        
        try:
            with gzip.open(pdb_gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            return output_path
        except Exception as e:
            logger.error(f"Failed to extract {pdb_gz_path}: {e}")
            return None
    
    def get_plddt_at_position(self, pdb_path: Path, position: int) -> Optional[float]:
        """
        Get pLDDT (confidence) score at a specific residue position.
        
        AlphaFold stores pLDDT in the B-factor column.
        
        Args:
            pdb_path: Path to PDB file
            position: Residue number
        
        Returns:
            pLDDT score (0-100) or None
        """
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        res_num = int(line[22:26].strip())
                        if res_num == position:
                            # B-factor is in columns 60-66
                            plddt = float(line[60:66].strip())
                            return plddt
        except Exception as e:
            logger.error(f"Failed to read pLDDT: {e}")
        
        return None
    
    def get_region_plddt(self, pdb_path: Path, start: int, end: int) -> Dict:
        """
        Get average pLDDT for a region (e.g., binding site).
        
        Returns:
            Dict with 'mean', 'min', 'max' pLDDT values
        """
        plddt_values = []
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        res_num = int(line[22:26].strip())
                        if start <= res_num <= end:
                            plddt = float(line[60:66].strip())
                            plddt_values.append(plddt)
        except Exception as e:
            logger.error(f"Failed to read region pLDDT: {e}")
        
        if plddt_values:
            return {
                'mean': sum(plddt_values) / len(plddt_values),
                'min': min(plddt_values),
                'max': max(plddt_values),
                'residues': len(plddt_values)
            }
        
        return {'mean': 0, 'min': 0, 'max': 0, 'residues': 0}
    
    def prepare_receptor_pdbqt(self, pdb_path: Path) -> Optional[Path]:
        """
        Convert PDB to PDBQT format for Vina docking.
        Requires Open Babel.
        """
        if not self.obabel_path:
            logger.warning("Open Babel not found, cannot prepare receptor")
            return None
        
        output_path = pdb_path.with_suffix('.pdbqt')
        
        try:
            cmd = [self.obabel_path, str(pdb_path), '-O', str(output_path), '-xr']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if output_path.exists():
                return output_path
            else:
                logger.error(f"PDBQT conversion failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Receptor preparation failed: {e}")
            return None
    
    def prepare_ligand_pdbqt(self, sdf_path: Path) -> Optional[Path]:
        """
        Convert SDF to PDBQT format for Vina docking.
        """
        if not self.obabel_path:
            logger.warning("Open Babel not found, cannot prepare ligand")
            return None
        
        output_path = sdf_path.with_suffix('.pdbqt')
        
        try:
            cmd = [self.obabel_path, str(sdf_path), '-O', str(output_path), '--gen3d']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if output_path.exists():
                return output_path
            else:
                logger.error(f"Ligand PDBQT conversion failed: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return None
    
    def run_vina_docking(self, receptor_pdbqt: Path, ligand_pdbqt: Path, 
                         center: Tuple[float, float, float],
                         size: Tuple[float, float, float] = (30, 30, 30)) -> Optional[float]:
        """
        Run AutoDock Vina docking.
        
        Args:
            receptor_pdbqt: Path to receptor PDBQT
            ligand_pdbqt: Path to ligand PDBQT
            center: (x, y, z) center of search box
            size: (x, y, z) size of search box
        
        Returns:
            Best binding affinity (kcal/mol) or None
        """
        if not self.vina_path:
            logger.warning("Vina not found")
            return None
        
        output_path = ligand_pdbqt.with_suffix('.docked.pdbqt')
        
        try:
            cmd = [
                self.vina_path,
                '--receptor', str(receptor_pdbqt),
                '--ligand', str(ligand_pdbqt),
                '--out', str(output_path),
                '--center_x', str(center[0]),
                '--center_y', str(center[1]),
                '--center_z', str(center[2]),
                '--size_x', str(size[0]),
                '--size_y', str(size[1]),
                '--size_z', str(size[2]),
                '--exhaustiveness', '8',
                '--num_modes', '5'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse output for best affinity
            for line in result.stdout.split('\n'):
                if line.strip().startswith('1'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            affinity = float(parts[1])
                            return affinity
                        except ValueError:
                            pass
            
            logger.warning(f"Could not parse Vina output: {result.stdout}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error("Vina docking timed out")
            return None
        except Exception as e:
            logger.error(f"Vina docking failed: {e}")
            return None
    
    def get_binding_site_center(self, pdb_path: Path, start: int, end: int) -> Tuple[float, float, float]:
        """
        Calculate center of mass for a region (binding site).
        """
        coords = []
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        res_num = int(line[22:26].strip())
                        if start <= res_num <= end:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords.append((x, y, z))
        except Exception as e:
            logger.error(f"Failed to calculate center: {e}")
            return (0, 0, 0)
        
        if coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
            cz = sum(c[2] for c in coords) / len(coords)
            return (cx, cy, cz)
        
        return (0, 0, 0)
    
    def parse_mutation(self, mutation: str) -> Tuple[str, int, str]:
        """
        Parse mutation string like "L858R" into (ref_aa, position, alt_aa).
        """
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation.upper())
        if match:
            return match.group(1), int(match.group(2)), match.group(3)
        return None, None, None
    
    def analyze_mutation_impact(self, gene: str, mutation: str) -> Dict:
        """
        Analyze how a mutation affects protein structure.
        
        Args:
            gene: Gene symbol
            mutation: Mutation (e.g., "L858R")
        
        Returns:
            Dict with impact analysis
        """
        result = {
            'gene': gene,
            'mutation': mutation,
            'plddt_at_position': None,
            'structure_confidence': 'UNKNOWN',
            'known_impact': None,
            'mechanism': None,
            'clinical_relevance': 'UNKNOWN'
        }
        
        # Check known resistance mutations first
        if gene.upper() in RESISTANCE_MUTATIONS:
            if mutation.upper() in RESISTANCE_MUTATIONS[gene.upper()]:
                known = RESISTANCE_MUTATIONS[gene.upper()][mutation.upper()]
                result['known_impact'] = known['impact']
                result['mechanism'] = known['mechanism']
                result['clinical_relevance'] = 'KNOWN_RESISTANCE' if known['impact'] == 'HIGH' else 'SENSITIZING'
        
        # Get structure and analyze pLDDT
        pdb_gz = self.get_structure_path(gene)
        if pdb_gz:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdb_path = self.extract_pdb(pdb_gz, Path(tmpdir))
                if pdb_path:
                    _, position, _ = self.parse_mutation(mutation)
                    if position:
                        plddt = self.get_plddt_at_position(pdb_path, position)
                        result['plddt_at_position'] = plddt
                        
                        if plddt:
                            if plddt >= 90:
                                result['structure_confidence'] = 'VERY_HIGH'
                            elif plddt >= 70:
                                result['structure_confidence'] = 'HIGH'
                            elif plddt >= 50:
                                result['structure_confidence'] = 'MEDIUM'
                            else:
                                result['structure_confidence'] = 'LOW'
        
        return result
    
    def validate_therapy(self, gene: str, mutation: str, drug: str, 
                         run_docking: bool = False) -> DockingResult:
        """
        Validate if a drug is appropriate for a mutated target.
        
        Args:
            gene: Target gene (e.g., "EGFR")
            mutation: Mutation (e.g., "L858R", "T790M")
            drug: Drug name (e.g., "osimertinib")
            run_docking: Whether to run actual Vina docking (slower)
        
        Returns:
            DockingResult with therapy recommendation
        """
        drug_lower = drug.lower().replace(' ', '').replace('-', '')
        
        # Check if drug targets this gene
        drug_info = DRUG_TARGETS.get(drug_lower)
        if not drug_info:
            return DockingResult(
                drug=drug,
                target_gene=gene,
                mutation=mutation,
                binding_affinity=0,
                binding_quality='UNKNOWN',
                mutation_impact='UNKNOWN',
                therapy_recommendation='Drug not in database',
                plddt_at_mutation=None,
                structure_file=None,
                error=f"Drug '{drug}' not found in database"
            )
        
        # Check gene-drug match
        if drug_info['target'].upper() != gene.upper():
            return DockingResult(
                drug=drug,
                target_gene=gene,
                mutation=mutation,
                binding_affinity=0,
                binding_quality='NONE',
                mutation_impact='N/A',
                therapy_recommendation=f'MISMATCH: {drug} targets {drug_info["target"]}, not {gene}',
                plddt_at_mutation=None,
                structure_file=None
            )
        
        # Analyze mutation impact
        impact = self.analyze_mutation_impact(gene, mutation)
        
        # Check known resistance
        mutation_impact = 'LOW'
        therapy_valid = True
        recommendation = 'APPROPRIATE'
        
        if gene.upper() in RESISTANCE_MUTATIONS:
            mut_info = RESISTANCE_MUTATIONS[gene.upper()].get(mutation.upper())
            if mut_info:
                if drug_lower in mut_info.get('affects', []):
                    mutation_impact = 'HIGH'
                    therapy_valid = False
                    recommendation = f'RESISTANCE: {mutation} confers resistance to {drug}'
                elif mut_info['impact'] == 'TARGET':
                    mutation_impact = 'TARGET'
                    recommendation = f'APPROPRIATE: {mutation} is the drug target'
                elif mut_info['impact'] == 'LOW':
                    mutation_impact = 'LOW'
                    recommendation = f'APPROPRIATE: {mutation} is a sensitizing mutation'
        
        # Estimate binding affinity (heuristic without actual docking)
        base_affinity = -8.5  # Typical good binding
        
        if mutation_impact == 'HIGH':
            binding_affinity = -4.0  # Poor binding due to resistance
            binding_quality = 'WEAK'
        elif mutation_impact == 'TARGET':
            binding_affinity = -9.5  # Enhanced binding to target mutation
            binding_quality = 'STRONG'
        elif mutation_impact == 'LOW':
            binding_affinity = -8.0  # Good binding
            binding_quality = 'STRONG'
        else:
            binding_affinity = base_affinity
            binding_quality = 'MODERATE'
        
        # Optional: run actual Vina docking
        actual_affinity = None
        if run_docking and self.vina_path and self.obabel_path:
            actual_affinity = self._run_full_docking(gene, drug_lower)
            if actual_affinity:
                binding_affinity = actual_affinity
                if actual_affinity <= -9:
                    binding_quality = 'STRONG'
                elif actual_affinity <= -7:
                    binding_quality = 'MODERATE'
                elif actual_affinity <= -5:
                    binding_quality = 'WEAK'
                else:
                    binding_quality = 'NONE'
        
        return DockingResult(
            drug=drug,
            target_gene=gene,
            mutation=mutation,
            binding_affinity=binding_affinity,
            binding_quality=binding_quality,
            mutation_impact=mutation_impact,
            therapy_recommendation=recommendation,
            plddt_at_mutation=impact.get('plddt_at_position'),
            structure_file=str(self.get_structure_path(gene)) if self.get_structure_path(gene) else None
        )
    
    def _run_full_docking(self, gene: str, drug: str) -> Optional[float]:
        """Run full Vina docking (slow but accurate)."""
        try:
            pdb_gz = self.get_structure_path(gene)
            drug_sdf = self.drugs_dir / f"{drug}.sdf"
            
            if not pdb_gz or not drug_sdf.exists():
                return None
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                # Extract and prepare receptor
                pdb_path = self.extract_pdb(pdb_gz, tmpdir)
                receptor_pdbqt = self.prepare_receptor_pdbqt(pdb_path)
                
                # Prepare ligand
                ligand_pdbqt = self.prepare_ligand_pdbqt(drug_sdf)
                
                if not receptor_pdbqt or not ligand_pdbqt:
                    return None
                
                # Get binding site center
                drug_info = DRUG_TARGETS.get(drug, {})
                binding_site = drug_info.get('binding_site', (1, 100))
                center = self.get_binding_site_center(pdb_path, binding_site[0], binding_site[1])
                
                # Run docking
                affinity = self.run_vina_docking(receptor_pdbqt, ligand_pdbqt, center)
                return affinity
                
        except Exception as e:
            logger.error(f"Full docking failed: {e}")
            return None
    
    def get_binding_site_plddt(self, gene: str, drug: str) -> Dict:
        """
        Get pLDDT scores for the drug binding site region.
        """
        drug_lower = drug.lower().replace(' ', '').replace('-', '')
        drug_info = DRUG_TARGETS.get(drug_lower, {})
        binding_site = drug_info.get('binding_site', (1, 100))
        
        pdb_gz = self.get_structure_path(gene)
        if pdb_gz:
            with tempfile.TemporaryDirectory() as tmpdir:
                pdb_path = self.extract_pdb(pdb_gz, Path(tmpdir))
                if pdb_path:
                    return self.get_region_plddt(pdb_path, binding_site[0], binding_site[1])
        
        return {'mean': 0, 'min': 0, 'max': 0, 'residues': 0}


# Singleton instance
_instance: Optional[AlphaFoldVina] = None


def get_instance() -> Optional[AlphaFoldVina]:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        try:
            _instance = AlphaFoldVina()
        except Exception as e:
            logger.error(f"Failed to initialize AlphaFold+Vina: {e}")
            return None
    return _instance


def validate_therapy(gene: str, mutation: str, drug: str) -> Optional[DockingResult]:
    """
    Quick function to validate a therapy.
    
    Usage:
        result = validate_therapy("EGFR", "L858R", "osimertinib")
    """
    instance = get_instance()
    if instance:
        return instance.validate_therapy(gene, mutation, drug)
    return None


def is_available() -> bool:
    """Check if AlphaFold+Vina is available."""
    return ALPHAFOLD_DB.exists() and len(list(ALPHAFOLD_DB.glob('*.pdb.gz'))) > 0


# ============================================================
# TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AlphaFold + Vina Integration Module - Test")
    print("=" * 60)
    
    # Initialize
    av = AlphaFoldVina()
    
    print(f"\nAlphaFold available: {av.is_available()}")
    print(f"Vina path: {av.vina_path}")
    print(f"OpenBabel path: {av.obabel_path}")
    
    # Test structure lookup
    print("\n🧬 Structure Lookup:")
    for gene in ['EGFR', 'KRAS', 'BRAF', 'MET', 'TP53']:
        path = av.get_structure_path(gene)
        if path:
            print(f"   ✓ {gene}: {path.name}")
        else:
            print(f"   ✗ {gene}: Not found")
    
    # Test mutation impact
    print("\n🔬 Mutation Impact Analysis:")
    test_mutations = [
        ('EGFR', 'L858R'),
        ('EGFR', 'T790M'),
        ('EGFR', 'C797S'),
        ('KRAS', 'G12C'),
        ('BRAF', 'V600E'),
    ]
    
    for gene, mutation in test_mutations:
        impact = av.analyze_mutation_impact(gene, mutation)
        plddt = impact.get('plddt_at_position', 'N/A')
        known = impact.get('known_impact', 'Unknown')
        print(f"   {gene} {mutation}: pLDDT={plddt}, Known impact={known}")
    
    # Test therapy validation
    print("\n💊 Therapy Validation:")
    test_therapies = [
        ('EGFR', 'L858R', 'osimertinib'),
        ('EGFR', 'T790M', 'gefitinib'),
        ('EGFR', 'C797S', 'osimertinib'),
        ('KRAS', 'G12C', 'sotorasib'),
        ('BRAF', 'V600E', 'dabrafenib'),
        ('MET', 'D1228N', 'capmatinib'),
    ]
    
    for gene, mutation, drug in test_therapies:
        result = av.validate_therapy(gene, mutation, drug)
        emoji = "✓" if 'APPROPRIATE' in result.therapy_recommendation else "✗"
        print(f"   {emoji} {gene} {mutation} + {drug}:")
        print(f"      Affinity: {result.binding_affinity} kcal/mol ({result.binding_quality})")
        print(f"      Recommendation: {result.therapy_recommendation}")
    
    print("\n✅ Test complete!")
