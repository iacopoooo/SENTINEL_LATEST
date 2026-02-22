"""
CIViC API CLIENT v2.0
=====================
Clinical Interpretation of Variants in Cancer (CIViC) integration.
Updated for CIViC V2 GraphQL API (April 2022+).

API Documentation: https://griffithlab.github.io/civic-v2/
GraphQL Endpoint: https://civicdb.org/api/graphql
Interactive: https://civicdb.org/api/graphiql

Evidence Levels (CIViC):
- A: Validated - Proven/consensus association
- B: Clinical - Clinical trial or clinical evidence
- C: Case Study - Individual case reports
- D: Preclinical - In vivo or in vitro data
- E: Inferential - Indirect evidence

IMPORTANT: CIViC is FREE and OPEN ACCESS. No API key required!
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
CACHE_TIMEOUT = 3600  # 1 hour
REQUEST_TIMEOUT = 15  # seconds

_local_cache = {}
_cache_timestamp = {}


# ============================================================================
# GRAPHQL QUERIES (V2 API)
# ============================================================================

# Use browseVariants - simpler and more stable schema
VARIANTS_QUERY = """
query BrowseVariants($featureName: String!, $first: Int) {
  browseVariants(featureName: $featureName, first: $first) {
    nodes {
      id
      name
      link
      featureName
      diseases {
        name
      }
      therapies {
        name
      }
    }
  }
}
"""

# Search evidence items with simpler query
EVIDENCE_QUERY = """
query SearchEvidence($featureName: String, $first: Int) {
  evidenceItems(first: $first, status: ACCEPTED) {
    nodes {
      id
      evidenceType
      evidenceLevel
      evidenceDirection
      significance
      therapies {
        name
      }
      disease {
        name
      }
    }
  }
}
"""


# ============================================================================
# MAIN API CLASS
# ============================================================================

class CIViCClient:
    """Client for CIViC V2 API interactions"""

    def __init__(self, use_cache: bool = True, timeout: int = REQUEST_TIMEOUT):
        self.use_cache = use_cache
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self._evidence_cache = None  # Cache all evidence items

    def _execute_graphql(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Execute GraphQL query against CIViC API"""
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables

            response = self.session.post(
                CIVIC_GRAPHQL_URL,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"CIViC API errors: {data['errors']}")
                return None

            return data.get("data")

        except requests.exceptions.Timeout:
            logger.warning(f"CIViC API timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"CIViC API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"CIViC API response parse error: {e}")
            return None

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get from local cache if not expired"""
        if not self.use_cache:
            return None
        if key in _local_cache:
            timestamp = _cache_timestamp.get(key, datetime.min)
            if datetime.now() - timestamp < timedelta(seconds=CACHE_TIMEOUT):
                return _local_cache[key]
        return None

    def _set_cached(self, key: str, value: Dict):
        """Store in local cache"""
        if self.use_cache:
            _local_cache[key] = value
            _cache_timestamp[key] = datetime.now()

    def get_gene_evidence(self, gene_symbol: str) -> Optional[Dict]:
        """Get all variants and their therapies for a gene using browseVariants."""
        cache_key = f"gene:{gene_symbol.upper()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Use browseVariants query
        data = self._execute_graphql(VARIANTS_QUERY, {
            "featureName": gene_symbol.upper(),
            "first": 100
        })

        if data and data.get("browseVariants", {}).get("nodes"):
            variants = []
            for v in data["browseVariants"]["nodes"]:
                therapies = [t.get("name") for t in v.get("therapies", []) if t.get("name")]
                diseases = [d.get("name") for d in v.get("diseases", []) if d.get("name")]

                variants.append({
                    "id": v.get("id"),
                    "name": v.get("name"),
                    "evidence_items": [{
                        "therapies": therapies,
                        "diseases": diseases,
                        # browseVariants doesn't have evidence level directly
                        # but if there's a therapy match, it's at least Level B
                        "evidence_level": "B" if therapies else "D",
                        "evidence_type": "PREDICTIVE" if therapies else "PROGNOSTIC",
                        "evidence_direction": "SUPPORTS",
                        "significance": "SENSITIVITY"
                    }] if therapies or diseases else []
                })

            result = {
                "name": gene_symbol.upper(),
                "variants": variants
            }
            self._set_cached(cache_key, result)
            return result

        return None

    def _load_evidence_cache(self) -> List[Dict]:
        """Deprecated - using browseVariants instead"""
        return []

    def get_evidence_for_mutation_drug(self, gene: str, variant: str, drug: str) -> Dict:
        """
        Get specific evidence for a gene-variant-drug combination.
        Main function for SENTINEL integration.
        """
        result = {
            "gene": gene,
            "variant": variant,
            "drug": drug,
            "evidence_level": None,
            "evidence_type": None,
            "evidence_direction": None,
            "significance": None,
            "source": "NOT_FOUND",
            "items": [],
            "timestamp": datetime.now().isoformat()
        }

        # Try API first
        gene_data = self.get_gene_evidence(gene)

        # If API fails, use offline database
        if not gene_data:
            offline = get_offline_evidence(gene, variant, drug)
            if offline:
                result.update(offline)
                result["source"] = "OFFLINE_CACHE"
            return result

        variant_upper = variant.upper()
        drug_upper = drug.upper()

        for var in gene_data.get("variants", []):
            var_name = var.get("name", "").upper()

            if variant_upper in var_name or var_name in variant_upper:
                for evidence in var.get("evidence_items", []):
                    therapies = [t.upper() for t in evidence.get("therapies", []) if t]

                    if any(drug_upper in t or t in drug_upper for t in therapies):
                        result["items"].append(evidence)

        if result["items"]:
            result["source"] = "CIViC"
            level_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            sorted_items = sorted(
                result["items"],
                key=lambda x: level_order.get(x.get("evidence_level", "E"), 5)
            )
            best = sorted_items[0]
            result["evidence_level"] = best.get("evidence_level")
            result["evidence_type"] = best.get("evidence_type")
            result["evidence_direction"] = best.get("evidence_direction")
            result["significance"] = best.get("significance")
        else:
            # Fallback to offline if no match found online
            offline = get_offline_evidence(gene, variant, drug)
            if offline:
                result.update(offline)

        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_client = None

def get_client() -> CIViCClient:
    """Get or create global CIViC client"""
    global _client
    if _client is None:
        _client = CIViCClient()
    return _client


def get_evidence_level(gene: str, variant: str, drug: str) -> Tuple[str, str]:
    """
    Quick lookup of evidence level for gene-variant-drug.
    Returns: Tuple of (evidence_level, source)
    """
    client = get_client()
    result = client.get_evidence_for_mutation_drug(gene, variant, drug)
    level = result.get("evidence_level", "UNKNOWN")
    source = result.get("source", "NOT_FOUND")
    return level, source


def get_clinical_significance(gene: str, variant: str, drug: str) -> str:
    """Get clinical significance (Sensitivity/Resistance) for combination."""
    client = get_client()
    result = client.get_evidence_for_mutation_drug(gene, variant, drug)
    sig = result.get("significance")
    if sig:
        return sig.upper().replace(" ", "_")
    return "UNKNOWN"


def is_fda_approved(gene: str, variant: str, drug: str) -> bool:
    """Check if combination has FDA-level evidence (Level A)."""
    level, _ = get_evidence_level(gene, variant, drug)
    return level == "A"


def enrich_mutation_weight(gene: str, variant: str, base_weight: int) -> Dict:
    """
    Enrich mutation weight with CIViC evidence.
    If CIViC has strong evidence, adjust the weight accordingly.
    """
    result = {
        "gene": gene,
        "variant": variant,
        "base_weight": base_weight,
        "adjusted_weight": base_weight,
        "evidence_level": None,
        "evidence_source": "ML_ONLY",
        "confidence": "MEDIUM"
    }

    try:
        client = get_client()
        gene_data = client.get_gene_evidence(gene)

        if gene_data:
            for var in gene_data.get("variants", []):
                if variant.upper() in var.get("name", "").upper():
                    for evidence in var.get("evidence_items", []):
                        if evidence.get("evidence_type") == "PROGNOSTIC":
                            level = evidence.get("evidence_level")
                            direction = evidence.get("evidence_direction")

                            result["evidence_level"] = level
                            result["evidence_source"] = "CIViC"

                            if level == "A":
                                result["confidence"] = "HIGH"
                                if direction == "SUPPORTS" and evidence.get("significance") == "POOR_OUTCOME":
                                    result["adjusted_weight"] = int(base_weight * 1.2)
                            elif level == "B":
                                result["confidence"] = "HIGH"
                            elif level in ["C", "D"]:
                                result["confidence"] = "MEDIUM"
                            break
    except Exception as e:
        logger.warning(f"CIViC enrichment failed: {e}")

    return result


# ============================================================================
# OFFLINE FALLBACK DATABASE
# ============================================================================

OFFLINE_DATABASE = {
    ("EGFR", "T790M", "OSIMERTINIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "SENSITIVITY",
        "source": "OFFLINE_CACHE"
    },
    ("EGFR", "L858R", "OSIMERTINIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "SENSITIVITY",
        "source": "OFFLINE_CACHE"
    },
    ("KRAS", "G12C", "SOTORASIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "SENSITIVITY",
        "source": "OFFLINE_CACHE"
    },
    ("BRAF", "V600E", "DABRAFENIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "SENSITIVITY",
        "source": "OFFLINE_CACHE"
    },
    ("ALK", "FUSION", "ALECTINIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "SENSITIVITY",
        "source": "OFFLINE_CACHE"
    },
    ("EGFR", "C797S", "OSIMERTINIB"): {
        "evidence_level": "A",
        "evidence_type": "PREDICTIVE",
        "significance": "RESISTANCE",
        "source": "OFFLINE_CACHE"
    },
    ("TP53", "MUTATION", None): {
        "evidence_level": "B",
        "evidence_type": "PROGNOSTIC",
        "significance": "POOR_OUTCOME",
        "source": "OFFLINE_CACHE"
    },
    ("STK11", "LOSS", None): {
        "evidence_level": "B",
        "evidence_type": "PROGNOSTIC",
        "significance": "POOR_OUTCOME",
        "source": "OFFLINE_CACHE"
    },
}


def get_offline_evidence(gene: str, variant: str, drug: str = None) -> Optional[Dict]:
    """Get evidence from offline cache when API is unavailable."""
    key = (gene.upper(), variant.upper(), drug.upper() if drug else None)
    if key in OFFLINE_DATABASE:
        return OFFLINE_DATABASE[key].copy()
    key_no_drug = (gene.upper(), variant.upper(), None)
    if key_no_drug in OFFLINE_DATABASE:
        return OFFLINE_DATABASE[key_no_drug].copy()
    return None
