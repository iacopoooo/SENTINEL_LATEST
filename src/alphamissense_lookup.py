#!/usr/bin/env python3
"""
SENTINEL - AlphaMissense Integration Module
============================================
Classifica mutazioni missense usando il database AlphaMissense di DeepMind.

Features:
- Database SQLite indicizzato per query O(1)
- Lookup per gene+variant (es. "TP53 R273H")
- Lookup per coordinate genomiche (es. chr17:7577121 G>A)
- Cache in memoria per mutazioni frequenti
- Integrazione diretta con Ferrari Engine

Usage:
    from alphamissense_lookup import AlphaMissenseLookup

    am = AlphaMissenseLookup()
    result = am.classify("TP53", "R273H")
    # {'pathogenicity': 0.98, 'class': 'likely_pathogenic', 'confidence': 'HIGH'}
"""

import sqlite3
import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from functools import lru_cache
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
DATA_DIR = BASE_DIR / 'data' / 'alphamissense_db'
TSV_FILE = DATA_DIR / 'AlphaMissense_hg38.tsv'
DB_FILE = DATA_DIR / 'alphamissense.db'

# Gene name to UniProt ID mapping (oncology-focused, espandibile)
GENE_TO_UNIPROT = {
    # Lung cancer drivers
    'EGFR': 'P00533',
    'KRAS': 'P01116',
    'TP53': 'P04637',
    'ALK': 'Q9UM73',
    'ROS1': 'P08922',
    'BRAF': 'P15056',
    'MET': 'P08581',
    'RET': 'P07949',
    'HER2': 'P04626',
    'ERBB2': 'P04626',
    'NRAS': 'P01111',
    'PIK3CA': 'P42336',
    'STK11': 'Q15831',
    'LKB1': 'Q15831',
    'KEAP1': 'Q14145',
    'NF1': 'P21359',
    'RB1': 'P06400',
    'PTEN': 'P60484',
    'APC': 'P25054',
    'SMAD4': 'Q13485',
    'CDKN2A': 'P42771',
    'ATM': 'Q13315',
    'BRCA1': 'P38398',
    'BRCA2': 'P51587',
    'PALB2': 'Q86YC2',
    'ARID1A': 'O14497',
    'NOTCH1': 'P46531',
    'FBXW7': 'Q969H0',
    'CTNNB1': 'P35222',
    'IDH1': 'O75874',
    'IDH2': 'P48735',
    'FGFR1': 'P11362',
    'FGFR2': 'P21802',
    'FGFR3': 'P22607',
    'DDR2': 'Q16832',
    'MAP2K1': 'Q02750',
    'MEK1': 'Q02750',
    'NTRK1': 'P04629',
    'NTRK2': 'Q16620',
    'NTRK3': 'Q16288',
}


class AlphaMissenseLookup:
    """
    Fast lookup for AlphaMissense pathogenicity scores.

    Uses SQLite with indexes for O(1) lookups.
    First run creates the database (~10-15 min), subsequent runs are instant.
    """

    def __init__(self, db_path: Path = None, tsv_path: Path = None):
        self.db_path = db_path or DB_FILE
        self.tsv_path = tsv_path or TSV_FILE
        self.conn = None
        self._initialized = False

        # Check if database exists, if not create it
        if not self.db_path.exists():
            if self.tsv_path.exists():
                logger.info("Database not found. Creating from TSV (this takes ~10-15 minutes)...")
                self._create_database()
            else:
                logger.error(f"TSV file not found: {self.tsv_path}")
                raise FileNotFoundError(f"AlphaMissense TSV not found at {self.tsv_path}")

        self._connect()

    def _connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._initialized = True
        logger.info(f"Connected to AlphaMissense database: {self.db_path}")

    def _create_database(self):
        """Create SQLite database from TSV file."""
        logger.info("Creating AlphaMissense SQLite database...")
        logger.info("This is a one-time operation and will take ~10-15 minutes...")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chrom TEXT,
                pos INTEGER,
                ref TEXT,
                alt TEXT,
                uniprot_id TEXT,
                transcript_id TEXT,
                protein_variant TEXT,
                am_pathogenicity REAL,
                am_class TEXT
            )
        ''')

        # Read and insert TSV
        row_count = 0
        batch = []
        batch_size = 100000

        with open(self.tsv_path, 'r') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue

                parts = line.strip().split('\t')
                if len(parts) >= 10:
                    batch.append((
                        parts[0],  # chrom
                        int(parts[1]),  # pos
                        parts[2],  # ref
                        parts[3],  # alt
                        parts[5],  # uniprot_id
                        parts[6],  # transcript_id
                        parts[7],  # protein_variant
                        float(parts[8]),  # am_pathogenicity
                        parts[9]  # am_class
                    ))

                    row_count += 1

                    if len(batch) >= batch_size:
                        cursor.executemany('''
                            INSERT INTO variants 
                            (chrom, pos, ref, alt, uniprot_id, transcript_id, protein_variant, am_pathogenicity, am_class)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', batch)
                        conn.commit()
                        logger.info(f"  Processed {row_count:,} variants...")
                        batch = []

        # Insert remaining
        if batch:
            cursor.executemany('''
                INSERT INTO variants 
                (chrom, pos, ref, alt, uniprot_id, transcript_id, protein_variant, am_pathogenicity, am_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
            conn.commit()

        logger.info(f"Total variants loaded: {row_count:,}")

        # Create indexes for fast lookup
        logger.info("Creating indexes (this may take a few minutes)...")

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uniprot_variant ON variants(uniprot_id, protein_variant)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chrom_pos ON variants(chrom, pos)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_protein_variant ON variants(protein_variant)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uniprot ON variants(uniprot_id)')

        conn.commit()
        conn.close()

        logger.info("✅ AlphaMissense database created successfully!")

    def classify(self, gene: str, variant: str) -> Optional[Dict]:
        """
        Classify a mutation by gene name and protein variant.

        Args:
            gene: Gene symbol (e.g., "TP53", "EGFR")
            variant: Protein variant (e.g., "R273H", "L858R")

        Returns:
            Dict with pathogenicity info or None if not found
        """
        if not self._initialized:
            return None

        # Normalize variant format
        variant = self._normalize_variant(variant)

        # Try by UniProt ID first
        uniprot_id = GENE_TO_UNIPROT.get(gene.upper())

        cursor = self.conn.cursor()

        if uniprot_id:
            cursor.execute('''
                SELECT * FROM variants 
                WHERE uniprot_id = ? AND protein_variant = ?
                LIMIT 1
            ''', (uniprot_id, variant))
        else:
            # Fallback: search by variant only (less precise)
            cursor.execute('''
                SELECT * FROM variants 
                WHERE protein_variant = ?
                LIMIT 1
            ''', (variant,))

        row = cursor.fetchone()

        if row:
            return self._format_result(gene, variant, row)

        return None

    def classify_by_coordinates(self, chrom: str, pos: int, ref: str, alt: str) -> Optional[Dict]:
        """
        Classify a mutation by genomic coordinates.

        Args:
            chrom: Chromosome (e.g., "chr17" or "17")
            pos: Position (1-based)
            ref: Reference allele
            alt: Alternate allele

        Returns:
            Dict with pathogenicity info or None if not found
        """
        if not self._initialized:
            return None

        # Normalize chromosome format
        if not chrom.startswith('chr'):
            chrom = f'chr{chrom}'

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM variants 
            WHERE chrom = ? AND pos = ? AND ref = ? AND alt = ?
            LIMIT 1
        ''', (chrom, pos, ref, alt))

        row = cursor.fetchone()

        if row:
            return self._format_result(None, row['protein_variant'], row)

        return None

    def classify_batch(self, mutations: List[Tuple[str, str]]) -> List[Optional[Dict]]:
        """
        Classify multiple mutations at once.

        Args:
            mutations: List of (gene, variant) tuples

        Returns:
            List of results (same order as input)
        """
        return [self.classify(gene, variant) for gene, variant in mutations]

    def _normalize_variant(self, variant: str) -> str:
        """Normalize variant format (e.g., 'p.R273H' -> 'R273H')."""
        variant = variant.strip()

        # Remove 'p.' prefix if present
        if variant.lower().startswith('p.'):
            variant = variant[2:]

        # Uppercase
        variant = variant.upper()

        return variant

    def _format_result(self, gene: str, variant: str, row: sqlite3.Row) -> Dict:
        """Format database row into result dict."""
        pathogenicity = row['am_pathogenicity']
        am_class = row['am_class']

        # Determine confidence level
        if pathogenicity >= 0.9 or pathogenicity <= 0.1:
            confidence = 'HIGH'
        elif pathogenicity >= 0.7 or pathogenicity <= 0.3:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Clinical interpretation
        if am_class == 'likely_pathogenic':
            clinical = 'PATHOGENIC - Likely disease-causing'
        elif am_class == 'likely_benign':
            clinical = 'BENIGN - Likely neutral'
        else:
            clinical = 'VUS - Uncertain significance'

        return {
            'gene': gene or row['uniprot_id'],
            'variant': variant,
            'uniprot_id': row['uniprot_id'],
            'transcript': row['transcript_id'],
            'chrom': row['chrom'],
            'pos': row['pos'],
            'ref': row['ref'],
            'alt': row['alt'],
            'pathogenicity': pathogenicity,
            'class': am_class,
            'confidence': confidence,
            'clinical_interpretation': clinical,
            # For SENTINEL integration
            'is_pathogenic': am_class == 'likely_pathogenic',
            'is_benign': am_class == 'likely_benign',
            'is_vus': am_class == 'ambiguous',
            'ferrari_boost': self._calculate_ferrari_boost(pathogenicity, am_class)
        }

    def _calculate_ferrari_boost(self, pathogenicity: float, am_class: str) -> float:
        """
        Calculate boost factor for Ferrari Engine based on pathogenicity.

        Returns multiplier (1.0 = no change, >1.0 = increase risk, <1.0 = decrease)
        """
        if am_class == 'likely_pathogenic':
            # High pathogenicity → boost resistance risk
            return 1.0 + (pathogenicity * 0.3)  # Max +30%
        elif am_class == 'likely_benign':
            # Benign → slight reduction
            return 1.0 - ((1 - pathogenicity) * 0.2)  # Max -20%
        else:
            # Ambiguous → no change
            return 1.0

    def get_stats(self) -> Dict:
        """Get database statistics."""
        if not self._initialized:
            return {}

        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM variants')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM variants WHERE am_class = "likely_pathogenic"')
        pathogenic = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM variants WHERE am_class = "likely_benign"')
        benign = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM variants WHERE am_class = "ambiguous"')
        ambiguous = cursor.fetchone()[0]

        return {
            'total_variants': total,
            'likely_pathogenic': pathogenic,
            'likely_benign': benign,
            'ambiguous': ambiguous,
            'pathogenic_percent': round(pathogenic / total * 100, 1) if total > 0 else 0
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self._initialized = False


# Singleton instance for easy import
_instance: Optional[AlphaMissenseLookup] = None


def get_instance() -> Optional[AlphaMissenseLookup]:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        try:
            _instance = AlphaMissenseLookup()
        except Exception as e:
            logger.error(f"Failed to initialize AlphaMissense: {e}")
            return None
    return _instance


def classify_mutation(gene: str, variant: str) -> Optional[Dict]:
    """
    Quick function to classify a single mutation.

    Usage:
        result = classify_mutation("TP53", "R273H")
    """
    instance = get_instance()
    if instance:
        return instance.classify(gene, variant)
    return None


def classify_mutations_batch(mutations: List[Tuple[str, str]]) -> List[Optional[Dict]]:
    """
    Quick function to classify multiple mutations.

    Usage:
        results = classify_mutations_batch([("TP53", "R273H"), ("EGFR", "L858R")])
    """
    instance = get_instance()
    if instance:
        return instance.classify_batch(mutations)
    return [None] * len(mutations)


def is_available() -> bool:
    """Check if AlphaMissense database is available."""
    return DB_FILE.exists() or TSV_FILE.exists()


# ============================================================
# TEST / DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AlphaMissense Lookup Module - Test")
    print("=" * 60)

    # Check files
    print(f"\nTSV file exists: {TSV_FILE.exists()}")
    print(f"DB file exists: {DB_FILE.exists()}")

    if not TSV_FILE.exists() and not DB_FILE.exists():
        print("\n❌ No data files found!")
        print(f"   Expected TSV at: {TSV_FILE}")
        exit(1)

    # Initialize
    print("\nInitializing AlphaMissense lookup...")
    am = AlphaMissenseLookup()

    # Stats
    stats = am.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total variants: {stats.get('total_variants', 0):,}")
    print(f"   Likely pathogenic: {stats.get('likely_pathogenic', 0):,} ({stats.get('pathogenic_percent', 0)}%)")
    print(f"   Likely benign: {stats.get('likely_benign', 0):,}")
    print(f"   Ambiguous: {stats.get('ambiguous', 0):,}")

    # Test queries
    print("\n🧬 Test Queries:")

    test_mutations = [
        ("TP53", "R273H"),
        ("EGFR", "L858R"),
        ("KRAS", "G12C"),
        ("BRAF", "V600E"),
        ("PIK3CA", "E545K"),
        ("MET", "D1228N"),
    ]

    for gene, variant in test_mutations:
        result = am.classify(gene, variant)
        if result:
            emoji = "🔴" if result['is_pathogenic'] else "🟢" if result['is_benign'] else "🟡"
            print(
                f"   {emoji} {gene} {variant}: {result['pathogenicity']:.3f} ({result['class']}) - Ferrari boost: {result['ferrari_boost']:.2f}x")
        else:
            print(f"   ⚪ {gene} {variant}: Not found in database")

    am.close()
    print("\n✅ Test complete!")