"""
CLINICAL NOTES LLM v2.0 - Google AI Studio (API Key)
====================================================
Versione semplificata che usa google-generativeai invece di Vertex AI.
Bypassa i problemi di permessi/regioni di GCP.
"""


import os
import json
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Carica variabili d'ambiente da .env
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Key da variabile d'ambiente (NON hardcoded)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# MODIFICA CRUCIALE: Usiamo l'alias stabile che punta alla versione Flash corrente
# Questo risolve l'errore "limit: 0" dei modelli sperimentali
MODEL_ID = "gemini-flash-latest"

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

EXTRACTION_PROMPT = """Sei un assistente medico esperto. Analizza la nota clinica ed estrai un JSON rigoroso.

NOTA CLINICA:
{notes}

REGOLE: 
- NON usare "Risposta Completa" o "CR" a meno che non sia esplicitamente scritto nelle note.
- NON menzionare CEA se non è nelle note originali.
- Mantieni fedeltà alle note originali senza inferire conclusioni cliniche.

SCHEMA JSON RICHIESTO:
{{
    "adverse_events": ["lista stringhe, es: 'nausea G2'"],
    "compliance": "HIGH/MEDIUM/LOW/UNKNOWN",
    "compliance_issues": "descrizione o null",
    "symptoms": ["lista sintomi"],
    "urgency_flags": ["lista urgenze o array vuoto"],
    "therapy_changes": "descrizione o null",
    "extracted_ecog": integer_0_to_4_or_null,
    "ecog_reasoning": "stringa o null",
    "weight_change": "aumento/diminuzione/stabile/null",
    "next_actions": ["lista azioni"]
}}
"""

SUMMARY_PROMPT = "Riassumi in 2 frasi per un oncologo: {notes}"


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def _configure_genai():
    if not GOOGLE_API_KEY or "INCOLLA" in GOOGLE_API_KEY:
        logger.error("❌ API KEY mancante - impostare GOOGLE_API_KEY nel file .env")
        return False
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return True
    except Exception as e:
        logger.error(f"❌ Errore config: {e}")
        return False


def _call_gemini_safe(model, prompt, max_retries=3):
    """
    Funzione wrapper che gestisce l'errore 429 (Rate Limit)
    aspettando automaticamente prima di riprovare.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = 10 * (attempt + 1)  # Backoff esponenziale: 10s, 20s, 30s
                logger.warning(
                    f"⚠️ Quota raggiunta. Attendo {wait_time} secondi... (Tentativo {attempt + 1}/{max_retries})")
                print(f"⏳ Traffico intenso, attendo {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Se è un altro errore, lancialo subito
                logger.error(f"❌ Errore Gemini irrecuperabile: {e}")
                return None
    return None


def extract_from_notes(notes: str) -> Dict:
    if not notes or len(notes.strip()) < 5: return {}
    if not _configure_genai(): return {}

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_ID,
            generation_config={"response_mime_type": "application/json"}
        )

        # Filtri di sicurezza al minimo per contesto medico
        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Chiamata protetta con retry
        response = _call_gemini_safe(model, EXTRACTION_PROMPT.format(notes=notes))

        if response and response.text:
            return json.loads(response.text)
        return {}

    except Exception as e:
        logger.error(f"Errore parsing: {e}")
        return {}


def summarize_notes(notes: str) -> str:
    if not notes: return ""
    if not _configure_genai(): return "Errore API"

    try:
        model = genai.GenerativeModel(MODEL_ID)
        response = _call_gemini_safe(model, SUMMARY_PROMPT.format(notes=notes))
        if response:
            return response.text.strip()
        return "Servizio momentaneamente non disponibile"
    except Exception:
        return "Impossibile riassumere"


# ============================================================================
# SENTINEL ADAPTERS
# ============================================================================

def is_available() -> bool:
    return bool(GOOGLE_API_KEY) and "INCOLLA" not in GOOGLE_API_KEY


def enrich_visit_data(visit_data: Dict) -> Dict:
    notes = visit_data.get("notes", "")
    if not notes: return visit_data

    # Estrazione
    data = extract_from_notes(notes)

    if data:
        # Mapping Adverse Events
        if data.get("adverse_events"):
            ae_list = []
            for ae_str in data["adverse_events"]:
                grade = 1
                if "G2" in ae_str:
                    grade = 2
                elif "G3" in ae_str:
                    grade = 3
                elif "G4" in ae_str:
                    grade = 4
                ae_list.append({"term": ae_str, "grade": f"G{grade}"})
            visit_data["adverse_events"] = ae_list

        if data.get("compliance"): visit_data["compliance"] = data["compliance"]
        if data.get("extracted_ecog") is not None: visit_data["ecog_ps"] = data["extracted_ecog"]

    # Summary separato
    visit_data["llm_notes_summary"] = summarize_notes(notes)

    return visit_data


# ============================================================================
# TEST RAPIDO
# ============================================================================
if __name__ == "__main__":
    test_note = "Paziente con forte nausea G3. ECOG peggiorato a 2."
    print(f"Test connessione AI Studio ({MODEL_ID})...")
    res = extract_from_notes(test_note)
    print(json.dumps(res, indent=2))