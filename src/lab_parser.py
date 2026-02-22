"""
SENTINEL Lab Text Parser
=========================
Parsa testo libero con valori di laboratorio e li converte in strutture dati SENTINEL.

Usato da:
- Streamlit Clinical Safety page
- Test suite
"""

import re
from typing import Dict, Any
from datetime import datetime

# =============================================================================
# LAB ALIASES → NORMALIZED NAMES
# =============================================================================

LAB_ALIASES: Dict[str, str] = {
    # Neutrofili / ANC
    'neutrofili': 'neutrophils', 'neutrophil': 'neutrophils', 'neutrophils': 'neutrophils',
    'anc': 'neutrophils', 'neu': 'neutrophils', 'neut': 'neutrophils',
    'granulociti neutrofili': 'neutrophils', 'polymorphonuclear': 'neutrophils',
    # Linfociti
    'linfociti': 'lymphocytes', 'lymphocytes': 'lymphocytes', 'lymph': 'lymphocytes',
    'lin': 'lymphocytes', 'lym': 'lymphocytes',
    # Piastrine
    'piastrine': 'platelets', 'platelets': 'platelets', 'plt': 'platelets',
    'thrombocytes': 'platelets', 'trombociti': 'platelets',
    # Emoglobina
    'emoglobina': 'hemoglobin', 'hemoglobin': 'hemoglobin', 'hb': 'hemoglobin',
    'hgb': 'hemoglobin', 'haemoglobin': 'hemoglobin',
    # Potassio
    'potassio': 'potassium', 'potassium': 'potassium', 'k': 'potassium',
    'k+': 'potassium',
    # Sodio
    'sodio': 'sodium', 'sodium': 'sodium', 'na': 'sodium', 'na+': 'sodium',
    # Calcio
    'calcio': 'calcium', 'calcium': 'calcium', 'ca': 'calcium', 'ca++': 'calcium',
    'ca2+': 'calcium',
    # Glucosio / Glicemia
    'glucosio': 'glucose', 'glucose': 'glucose', 'glicemia': 'glucose',
    'glycemia': 'glucose', 'glu': 'glucose',
    # Creatinina
    'creatinina': 'creatinine', 'creatinine': 'creatinine', 'cr': 'creatinine',
    'crea': 'creatinine',
    # LDH
    'ldh': 'ldh', 'lattato deidrogenasi': 'ldh', 'lactate dehydrogenase': 'ldh',
    # INR
    'inr': 'inr',
    # Lattato
    'lattato': 'lactate', 'lactate': 'lactate', 'acido lattico': 'lactate',
    # Leucociti / WBC
    'leucociti': 'leukocytes', 'leukocytes': 'leukocytes', 'wbc': 'leukocytes',
    'globuli bianchi': 'leukocytes', 'white blood cells': 'leukocytes',
    # Albumina
    'albumina': 'albumin', 'albumin': 'albumin', 'alb': 'albumin',
    # CEA
    'cea': 'cea',
    # CRP / PCR
    'pcr': 'crp', 'crp': 'crp', 'proteina c reattiva': 'crp',
    'c-reactive protein': 'crp',
    # Temperatura
    'temperatura': 'temperature', 'temperature': 'temperature',
    'temp': 'temperature', 'tc': 'temperature',
    'febbre': 'temperature',
    # Uricemia
    'acido urico': 'uric_acid', 'uric acid': 'uric_acid', 'uricemia': 'uric_acid',
    # Fosfato
    'fosfato': 'phosphate', 'phosphate': 'phosphate', 'fosforo': 'phosphate',
    'phosphorus': 'phosphate',
    # Magnesio
    'magnesio': 'magnesium', 'magnesium': 'magnesium', 'mg2+': 'magnesium',
    # Peso
    'peso': 'weight', 'weight': 'weight',
    # Pressione sistolica
    'pressione sistolica': 'bp_systolic', 'sistolica': 'bp_systolic',
    'pas': 'bp_systolic', 'sbp': 'bp_systolic',
    # Frequenza respiratoria
    'frequenza respiratoria': 'respiratory_rate', 'fr': 'respiratory_rate',
    'respiratory rate': 'respiratory_rate', 'rr': 'respiratory_rate',
}

# Sorted longest-first for greedy matching
_SORTED_ALIASES = sorted(LAB_ALIASES.keys(), key=len, reverse=True)


# =============================================================================
# REFERENCE RANGES & DISPLAY NAMES
# =============================================================================

REFERENCE_RANGES = {
    'neutrophils': (1500, 8000, '/µL'),
    'lymphocytes': (1000, 4000, '/µL'),
    'platelets': (150000, 400000, '/µL'),
    'hemoglobin': (12.0, 17.5, 'g/dL'),
    'potassium': (3.5, 5.0, 'mEq/L'),
    'sodium': (136, 145, 'mEq/L'),
    'calcium': (8.5, 10.5, 'mg/dL'),
    'glucose': (70, 100, 'mg/dL'),
    'creatinine': (0.6, 1.2, 'mg/dL'),
    'ldh': (120, 250, 'U/L'),
    'albumin': (3.5, 5.5, 'g/dL'),
    'leukocytes': (4000, 11000, '/µL'),
    'crp': (0, 5, 'mg/L'),
    'inr': (0.8, 1.2, ''),
    'lactate': (0.5, 2.0, 'mmol/L'),
    'temperature': (36.0, 37.5, '°C'),
    'cea': (0, 5, 'ng/mL'),
    'uric_acid': (2.5, 7.0, 'mg/dL'),
    'phosphate': (2.5, 4.5, 'mg/dL'),
    'magnesium': (1.7, 2.2, 'mg/dL'),
}

DISPLAY_NAMES = {
    'neutrophils': 'Neutrofili', 'lymphocytes': 'Linfociti',
    'platelets': 'Piastrine', 'hemoglobin': 'Emoglobina',
    'potassium': 'Potassio (K+)', 'sodium': 'Sodio (Na)',
    'calcium': 'Calcio (Ca)', 'glucose': 'Glicemia',
    'creatinine': 'Creatinina', 'ldh': 'LDH',
    'albumin': 'Albumina', 'leukocytes': 'Leucociti (WBC)',
    'crp': 'PCR', 'inr': 'INR', 'lactate': 'Lattato',
    'temperature': 'Temperatura', 'cea': 'CEA',
    'uric_acid': 'Acido Urico', 'phosphate': 'Fosfato',
    'magnesium': 'Magnesio', 'weight': 'Peso',
    'bp_systolic': 'PA Sistolica', 'respiratory_rate': 'Freq. Resp.',
}


# =============================================================================
# PARSER
# =============================================================================

def parse_lab_text(text: str) -> Dict[str, float]:
    """
    Parsa testo libero con valori di laboratorio e li restituisce come dict normalizzato.
    
    Supporta formati:
      - "Neutrofili: 300 cells/µL"
      - "LDH 480 U/L"
      - "K+ 6.8, Na 128"
      - "WBC    12500"
      - "Emoglobina = 6.2 g/dL"
    
    Returns:
        Dict mapping normalized lab names to float values
    """
    results: Dict[str, float] = {}
    
    text_lower = text.lower().strip()
    if not text_lower:
        return results
    
    # Replace comma-as-decimal (digit,digit) with dot BEFORE splitting
    # This preserves "K+ 6,8" as "K+ 6.8" while still splitting "K+ 6.8, Na 128"
    text_lower = re.sub(r'(\d),(\d)', r'\1.\2', text_lower)
    
    # Split into lines
    lines = text_lower.replace(';', '\n').replace(',', '\n').split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for alias in _SORTED_ALIASES:
            escaped = re.escape(alias)
            # Pattern: alias followed by separator and number
            pattern = rf'(?:^|[\s,;])({escaped})\s*[:=\s]\s*([0-9]+(?:[.,][0-9]+)?)'
            match = re.search(pattern, line)
            
            if not match:
                # Try with alias at start of line
                pattern2 = rf'^({escaped})\s*[:=\s]\s*([0-9]+(?:[.,][0-9]+)?)'
                match = re.search(pattern2, line)
            
            if match:
                lab_name = LAB_ALIASES[alias]
                value_str = match.group(2).replace(',', '.')
                try:
                    value = float(value_str)
                    if lab_name not in results:
                        results[lab_name] = value
                except ValueError:
                    continue
    
    return results


def build_patient_data(
    labs: Dict[str, float],
    patient_id: str = "QUICK_CHECK",
    age: int = 65,
    sex: str = "M",
    therapy: str = "",
    histology: str = "NSCLC"
) -> Dict[str, Any]:
    """
    Costruisce il patient_data dict nel formato SENTINEL a partire dai lab values parsati.
    """
    clinical_keys = {'temperature', 'weight', 'bp_systolic', 'respiratory_rate'}
    
    blood_markers = {k: v for k, v in labs.items() if k not in clinical_keys}
    clinical_status = {}
    
    if 'temperature' in labs:
        clinical_status['temperature'] = labs['temperature']
    if 'weight' in labs:
        clinical_status['weight_kg'] = labs['weight']
    if 'bp_systolic' in labs:
        clinical_status['bp_systolic'] = labs['bp_systolic']
    if 'respiratory_rate' in labs:
        clinical_status['respiratory_rate'] = labs['respiratory_rate']
    
    patient_data = {
        "baseline": {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "histology": histology,
        },
        "visits": [
            {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "blood_markers": blood_markers,
                "clinical_status": clinical_status,
            }
        ]
    }
    
    if therapy:
        patient_data["baseline"]["current_therapy"] = therapy
        patient_data["baseline"]["medications"] = [therapy]
    
    return patient_data
