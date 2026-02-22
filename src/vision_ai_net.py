"""
VISION AI NET v3.0 - HUMAN-IN-THE-LOOP
======================================
Modulo di Computer Vision per analisi biopsie.

IMPORTANT CHANGE in v3.0:
- NO MORE RANDOM/SIMULATED VALUES
- Requires pathologist input OR real AI model
- Clear distinction between human scores and AI predictions
- Audit trail for who provided the score

Modes:
1. MANUAL: Pathologist enters score directly
2. ASSISTED: AI suggests score, pathologist confirms/overrides
3. AUTO: Fully automated (requires validated AI model)

Default mode is MANUAL for safety.
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class VisionMode(Enum):
    MANUAL = "MANUAL"
    ASSISTED = "ASSISTED"
    AUTO = "AUTO"

CURRENT_MODE = VisionMode.MANUAL
AI_MODEL_AVAILABLE = False

# Audit log
_audit_log = []
_assessments_cache = {}


# ============================================================================
# MAIN FUNCTION (Backwards Compatible)
# ============================================================================

def analyze_biopsy(image_path: str) -> Optional[Dict]:
    """
    Analyze biopsy image.

    In v3.0 MANUAL mode:
    - Returns cached assessment if available
    - Returns None if no pathologist assessment exists
    - Does NOT generate random values

    Args:
        image_path: Path to biopsy image

    Returns:
        Dict with visual_risk, chaos_score, etc. or None
    """
    if not image_path or image_path == "None":
        return None

    # Check if we have a cached pathologist assessment
    if image_path in _assessments_cache:
        assessment = _assessments_cache[image_path]
        logger.info(f"📋 Using cached pathologist assessment for {os.path.basename(image_path)}")
        return assessment

    # In MANUAL mode, we don't generate fake scores
    if CURRENT_MODE == VisionMode.MANUAL:
        logger.warning(f"⚠️ No pathologist assessment for: {os.path.basename(image_path)}")
        logger.warning("   Use enter_pathologist_assessment() to provide scores")
        logger.warning("   Returning None - Vision AI will not be used")
        return None

    # In AUTO mode, would use real AI model
    if CURRENT_MODE == VisionMode.AUTO:
        if AI_MODEL_AVAILABLE:
            return _run_ai_model(image_path)
        else:
            logger.error("❌ AUTO mode but no AI model available!")
            return None

    return None


def enter_pathologist_assessment(
    image_path: str,
    visual_risk: float,
    chaos_score: float,
    cellularity: str,
    classification: str,
    mitosis_count: int,
    assessor_id: str = None,
    assessor_name: str = None,
    grade: str = None,
    notes: str = None
) -> Dict:
    """
    Enter pathologist assessment manually.

    Args:
        image_path: Path to biopsy image
        visual_risk: 0-100 risk score
        chaos_score: 0-10 nuclear atypia score
        cellularity: "Low", "Medium", or "High"
        classification: Histological classification
        mitosis_count: Mitoses per 10 HPF
        assessor_id: Pathologist ID (for audit)
        assessor_name: Pathologist name
        grade: Tumor grade (G1/G2/G3)
        notes: Additional notes

    Returns:
        Assessment dict (same format as analyze_biopsy)
    """
    # Validate inputs
    if not 0 <= visual_risk <= 100:
        raise ValueError(f"visual_risk must be 0-100, got {visual_risk}")
    if not 0 <= chaos_score <= 10:
        raise ValueError(f"chaos_score must be 0-10, got {chaos_score}")
    if cellularity not in ["Low", "Medium", "High"]:
        raise ValueError(f"cellularity must be Low/Medium/High, got {cellularity}")

    assessment = {
        "visual_risk": float(visual_risk),
        "chaos_score": float(chaos_score),
        "cellularity": cellularity,
        "class": classification,
        "mitosis_count": int(mitosis_count),
        "grade": grade,
        "source": "PATHOLOGIST",
        "assessor_id": assessor_id,
        "assessor_name": assessor_name,
        "assessment_date": datetime.now().isoformat(),
        "notes": notes
    }

    # Cache it
    _assessments_cache[image_path] = assessment

    # Audit log
    _audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "type": "PATHOLOGIST_ASSESSMENT",
        "image_path": image_path,
        "visual_risk": visual_risk,
        "assessor": assessor_name or assessor_id or "Unknown"
    })

    logger.info(f"✅ Pathologist assessment recorded for {os.path.basename(image_path)}")
    logger.info(f"   Visual Risk: {visual_risk}%")
    logger.info(f"   Classification: {classification}")

    return assessment


def enter_assessment_batch(assessments: list) -> int:
    """
    Enter multiple pathologist assessments at once.

    Args:
        assessments: List of dicts with same params as enter_pathologist_assessment

    Returns:
        Number of assessments entered
    """
    count = 0
    for a in assessments:
        try:
            enter_pathologist_assessment(**a)
            count += 1
        except Exception as e:
            logger.error(f"Failed to enter assessment: {e}")
    return count


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_assessment(image_path: str) -> Optional[Dict]:
    """Get cached assessment for an image"""
    return _assessments_cache.get(image_path)


def list_assessments() -> list:
    """List all cached assessments"""
    return list(_assessments_cache.items())


def clear_assessments():
    """Clear all cached assessments"""
    global _assessments_cache
    _assessments_cache = {}
    logger.info("Cleared all cached assessments")


def get_audit_log() -> list:
    """Get audit log of all assessments"""
    return _audit_log.copy()


def set_mode(mode: VisionMode):
    """Set Vision AI mode"""
    global CURRENT_MODE
    CURRENT_MODE = mode
    logger.info(f"Vision AI mode set to: {mode.value}")


def get_mode() -> VisionMode:
    """Get current mode"""
    return CURRENT_MODE


# ============================================================================
# AI MODEL INTERFACE (For future use)
# ============================================================================

def _run_ai_model(image_path: str) -> Optional[Dict]:
    """
    Run AI model on image.

    This is a placeholder for real AI integration.
    When implemented, would:
    1. Load image
    2. Preprocess (resize, normalize)
    3. Extract features with UNI/CTransPath
    4. Run classification head
    5. Return structured output
    """
    logger.error("AI model not implemented - use MANUAL mode")
    return None


def register_ai_model(model_path: str, model_version: str):
    """
    Register an AI model for AUTO mode.

    Args:
        model_path: Path to trained model weights
        model_version: Version string for audit
    """
    global AI_MODEL_AVAILABLE, AI_MODEL_PATH, AI_MODEL_VERSION

    if os.path.exists(model_path):
        AI_MODEL_AVAILABLE = True
        AI_MODEL_PATH = model_path
        AI_MODEL_VERSION = model_version
        logger.info(f"✅ AI model registered: {model_version}")
        logger.info(f"   Path: {model_path}")
    else:
        logger.error(f"❌ Model file not found: {model_path}")


# ============================================================================
# RISK CLASSIFICATION HELPERS
# ============================================================================

def classify_risk(visual_risk: float) -> str:
    """Classify risk level from visual_risk score"""
    if visual_risk >= 80:
        return "CRITICAL"
    elif visual_risk >= 60:
        return "HIGH"
    elif visual_risk >= 40:
        return "MODERATE"
    elif visual_risk >= 20:
        return "LOW"
    else:
        return "MINIMAL"


def suggest_review_priority(assessment: Dict) -> str:
    """Suggest review priority based on assessment"""
    risk = assessment.get("visual_risk", 0)
    mitosis = assessment.get("mitosis_count", 0)
    chaos = assessment.get("chaos_score", 0)

    # High priority if any critical indicator
    if risk >= 80 or mitosis >= 20 or chaos >= 8:
        return "URGENT"
    elif risk >= 60 or mitosis >= 10 or chaos >= 6:
        return "HIGH"
    elif risk >= 40 or mitosis >= 5:
        return "ROUTINE"
    else:
        return "LOW"
