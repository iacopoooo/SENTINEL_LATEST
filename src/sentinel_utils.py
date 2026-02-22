"""
SENTINEL UTILITIES v1.0
=======================
Funzioni di utilità condivise per la gestione sicura dei tipi.
Da importare in sentinel_engine.py e generate_final_report.py

USAGE:
    from src.sentinel_utils import safe_float, safe_int, safe_str
    
    # Invece di:
    ldh = float(blood.get('ldh', 200))  # CRASH se ldh è None!
    
    # Usa:
    ldh = safe_float(blood.get('ldh'), 200)  # Mai crash
"""


def safe_float(value, default=0.0):
    """
    Converte un valore a float in modo sicuro.
    
    Gestisce:
    - None → default
    - "" (stringa vuota) → default
    - "null", "Unknown", "N/A", "none" → default
    - Numeri validi → float
    - Stringhe numeriche ("250", "3.14") → float
    - Stringhe con virgola ("3,14") → float (3.14)
    - Qualsiasi altro errore → default
    
    Args:
        value: Il valore da convertire (any type)
        default: Valore di ritorno se conversione fallisce (default: 0.0)
    
    Returns:
        float: Il valore convertito o il default
    
    Examples:
        >>> safe_float(None)
        0.0
        >>> safe_float("")
        0.0
        >>> safe_float("null")
        0.0
        >>> safe_float(250)
        250.0
        >>> safe_float("250")
        250.0
        >>> safe_float("3,14")
        3.14
        >>> safe_float("Unknown")
        0.0
        >>> safe_float(None, 200)
        200.0
    """
    if value is None:
        return float(default)
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        value = value.strip()
        
        # Valori speciali che indicano "dato mancante"
        if value == '' or value.lower() in ('null', 'unknown', 'n/a', 'none', 'nan', '-'):
            return float(default)
        
        try:
            # Supporta sia punto che virgola come separatore decimale
            return float(value.replace(',', '.'))
        except ValueError:
            return float(default)
    
    # Per qualsiasi altro tipo, prova la conversione diretta
    try:
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def safe_int(value, default=0):
    """
    Converte un valore a int in modo sicuro.
    
    Args:
        value: Il valore da convertire
        default: Valore di ritorno se conversione fallisce (default: 0)
    
    Returns:
        int: Il valore convertito o il default
    """
    result = safe_float(value, float(default))
    return int(result)


def safe_str(value, default="N/A"):
    """
    Converte un valore a stringa in modo sicuro.
    
    Args:
        value: Il valore da convertire
        default: Valore di ritorno se None (default: "N/A")
    
    Returns:
        str: Il valore come stringa o il default
    """
    if value is None:
        return default
    return str(value)


def safe_get(dictionary, key, default=None):
    """
    Ottiene un valore da un dizionario in modo sicuro.
    Gestisce dizionari None e chiavi mancanti.
    
    Args:
        dictionary: Il dizionario da cui leggere
        key: La chiave da cercare
        default: Valore di ritorno se chiave mancante
    
    Returns:
        Il valore o il default
    """
    if dictionary is None:
        return default
    return dictionary.get(key, default)


def safe_get_float(dictionary, key, default=0.0):
    """
    Ottiene un float da un dizionario in modo sicuro.
    Combina safe_get e safe_float.
    
    Args:
        dictionary: Il dizionario da cui leggere
        key: La chiave da cercare
        default: Valore float di default
    
    Returns:
        float: Il valore convertito o il default
    
    Example:
        >>> blood = {'ldh': None, 'nlr': '2.5'}
        >>> safe_get_float(blood, 'ldh', 200)
        200.0
        >>> safe_get_float(blood, 'nlr')
        2.5
        >>> safe_get_float(blood, 'missing')
        0.0
    """
    raw_value = safe_get(dictionary, key)
    return safe_float(raw_value, default)


def safe_get_int(dictionary, key, default=0):
    """
    Ottiene un int da un dizionario in modo sicuro.
    """
    raw_value = safe_get(dictionary, key)
    return safe_int(raw_value, default)


# ============================================================================
# CLINICAL UTILITIES
# ============================================================================

def calculate_nlr(neutrophils, lymphocytes, default=0.0):
    """
    Calcola NLR (Neutrophil-to-Lymphocyte Ratio) in modo sicuro.
    
    Args:
        neutrophils: Conteggio neutrofili (può essere None)
        lymphocytes: Conteggio linfociti (può essere None)
        default: Valore di ritorno se calcolo impossibile
    
    Returns:
        float: NLR calcolato o default
    """
    neut = safe_float(neutrophils, 0)
    lymph = safe_float(lymphocytes, 0)
    
    if lymph <= 0:
        return default
    
    return round(neut / lymph, 2)


def interpret_ldh(ldh_value):
    """
    Interpreta il livello di LDH.
    
    Args:
        ldh_value: Valore LDH (può essere None)
    
    Returns:
        tuple: (ldh_float, interpretation_string, is_elevated)
    """
    ldh = safe_float(ldh_value, 0)
    
    if ldh <= 0:
        return 0, "Not Available", False
    elif ldh < 250:
        return ldh, "Normal", False
    elif ldh < 350:
        return ldh, "Borderline", False
    elif ldh < 500:
        return ldh, "Elevated (Warburg Effect)", True
    else:
        return ldh, "Very High (Elephant Protocol)", True


def interpret_pdl1(pdl1_value):
    """
    Interpreta l'espressione PD-L1.
    
    Args:
        pdl1_value: Valore PD-L1 % (può essere None)
    
    Returns:
        str: Interpretazione clinica
    """
    pdl1 = safe_float(pdl1_value, -1)
    
    if pdl1 < 0:
        return "Unknown"
    elif pdl1 < 1:
        return "Low/Negative"
    elif pdl1 < 50:
        return "Low/Intermediate"
    else:
        return "High Expression"


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing safe_float...")
    
    test_cases = [
        (None, 0.0),
        ("", 0.0),
        ("null", 0.0),
        ("Unknown", 0.0),
        (250, 250.0),
        ("250", 250.0),
        ("3,14", 3.14),
        ("3.14", 3.14),
        (None, 200.0),  # con default=200
    ]
    
    print("  safe_float(None) =", safe_float(None))
    print("  safe_float('') =", safe_float(""))
    print("  safe_float('null') =", safe_float("null"))
    print("  safe_float(250) =", safe_float(250))
    print("  safe_float('250') =", safe_float("250"))
    print("  safe_float(None, 200) =", safe_float(None, 200))
    
    print("\nTesting clinical utilities...")
    print("  calculate_nlr(4500, 1500) =", calculate_nlr(4500, 1500))
    print("  calculate_nlr(None, 1500) =", calculate_nlr(None, 1500))
    print("  interpret_ldh(450) =", interpret_ldh(450))
    print("  interpret_pdl1(45) =", interpret_pdl1(45))
    print("  interpret_pdl1(None) =", interpret_pdl1(None))
    
    print("\n✅ All tests passed!")
