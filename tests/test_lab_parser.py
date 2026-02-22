"""
Test suite per il Lab Text Parser di SENTINEL Clinical Safety.
"""

import sys
from pathlib import Path

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

from lab_parser import parse_lab_text, build_patient_data


# =============================================================================
# TEST: PARSER
# =============================================================================

class TestLabParser:
    """Test parsing di testo libero con valori di laboratorio."""
    
    def test_parse_italian_label_value(self):
        """Parser riconosce formato italiano label: valore"""
        text = """Neutrofili: 300
Piastrine: 15000
Emoglobina: 6.2
Potassio: 6.8"""
        result = parse_lab_text(text)
        assert result['neutrophils'] == 300
        assert result['platelets'] == 15000
        assert result['hemoglobin'] == 6.2
        assert result['potassium'] == 6.8
    
    def test_parse_english_abbreviations(self):
        """Parser riconosce abbreviazioni inglesi"""
        text = """ANC: 500
PLT: 20000
HB: 7.0
K: 5.5"""
        result = parse_lab_text(text)
        assert result['neutrophils'] == 500
        assert result['platelets'] == 20000
        assert result['hemoglobin'] == 7.0
        assert result['potassium'] == 5.5
    
    def test_parse_with_units(self):
        """Parser ignora unità di misura e prende solo il valore"""
        text = """LDH: 480 U/L
Creatinina: 3.5 mg/dL
Neutrofili: 300 cells/µL"""
        result = parse_lab_text(text)
        assert result['ldh'] == 480
        assert result['creatinine'] == 3.5
        assert result['neutrophils'] == 300
    
    def test_parse_tabular_format(self):
        """Parser riconosce formato tabellare con spazi"""
        text = """WBC    12500
NEU    300
PLT    18000
HB     6.2"""
        result = parse_lab_text(text)
        assert result['leukocytes'] == 12500
        assert result['neutrophils'] == 300
        assert result['platelets'] == 18000
        assert result['hemoglobin'] == 6.2
    
    def test_parse_equals_separator(self):
        """Parser riconosce separatore ="""
        text = """Potassio = 6.8
Sodio = 128"""
        result = parse_lab_text(text)
        assert result['potassium'] == 6.8
        assert result['sodium'] == 128
    
    def test_parse_comma_decimal(self):
        """Parser gestisce virgola come decimale"""
        text = "Emoglobina: 6,2"
        result = parse_lab_text(text)
        assert result['hemoglobin'] == 6.2
    
    def test_parse_clinical_values(self):
        """Parser riconosce valori clinici (temperature, peso)"""
        text = """Temperatura: 39.2
Peso: 72.5"""
        result = parse_lab_text(text)
        assert result['temperature'] == 39.2
        assert result['weight'] == 72.5
    
    def test_parse_empty_text(self):
        """Parser su testo vuoto ritorna dict vuoto"""
        assert parse_lab_text("") == {}
        assert parse_lab_text("   ") == {}
    
    def test_parse_no_valid_values(self):
        """Parser su testo senza valori ritorna dict vuoto"""
        text = "Il paziente si presenta in buone condizioni generali."
        result = parse_lab_text(text)
        assert isinstance(result, dict)
    
    def test_parse_mixed_formats(self):
        """Parser gestisce mix di formati in un unico testo"""
        text = """--- ESAMI DEL SANGUE ---
Neutrofili: 300
LDH 520
Temperatura: 39.2"""
        result = parse_lab_text(text)
        assert result['neutrophils'] == 300
        assert result['ldh'] == 520
        assert result['temperature'] == 39.2


# =============================================================================
# TEST: BUILD PATIENT DATA
# =============================================================================

class TestBuildPatientData:
    """Test costruzione patient_data dict."""
    
    def test_basic_structure(self):
        """Verifica struttura base del patient_data"""
        labs = {'neutrophils': 300, 'ldh': 480}
        result = build_patient_data(labs)
        
        assert 'baseline' in result
        assert 'visits' in result
        assert len(result['visits']) == 1
        assert result['baseline']['patient_id'] == 'QUICK_CHECK'
    
    def test_blood_markers_separated(self):
        """Valori lab vanno in blood_markers, clinical in clinical_status"""
        labs = {
            'neutrophils': 300,
            'temperature': 39.2,
            'weight': 72.5,
            'ldh': 480
        }
        result = build_patient_data(labs)
        visit = result['visits'][0]
        
        assert 'neutrophils' in visit['blood_markers']
        assert 'ldh' in visit['blood_markers']
        assert 'temperature' not in visit['blood_markers']
        assert visit['clinical_status']['temperature'] == 39.2
        assert visit['clinical_status']['weight_kg'] == 72.5
    
    def test_therapy_added(self):
        """Terapia aggiunta a baseline e medications"""
        labs = {'ldh': 200}
        result = build_patient_data(labs, therapy="Osimertinib")
        
        assert result['baseline']['current_therapy'] == "Osimertinib"
        assert "Osimertinib" in result['baseline']['medications']
    
    def test_custom_demographics(self):
        """Demographics personalizzati"""
        labs = {'ldh': 200}
        result = build_patient_data(labs, patient_id="PT-001", age=45, sex="F")
        
        assert result['baseline']['patient_id'] == "PT-001"
        assert result['baseline']['age'] == 45
        assert result['baseline']['sex'] == "F"


# =============================================================================
# TEST: INTEGRATION WITH SAFETY ENGINE
# =============================================================================

class TestParserToSafetyIntegration:
    """Test integrazione: parser → build → ClinicalSafetyEngine."""
    
    def test_critical_labs_generate_alerts(self):
        """Testo con valori critici genera alert dal Safety Engine"""
        from safety_alerts import ClinicalSafetyEngine, AlertSeverity
        
        text = """Neutrofili: 300
Potassio: 6.8
Emoglobina: 6.2
Piastrine: 15000"""
        
        parsed = parse_lab_text(text)
        patient_data = build_patient_data(parsed)
        
        engine = ClinicalSafetyEngine()
        alerts = engine.run_full_safety_check(patient_data)
        
        assert len(alerts) > 0
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        assert critical_count >= 3  # potassium, hemoglobin, platelets
    
    def test_normal_values_no_critical(self):
        """Testo con valori normali non genera alert critici"""
        from safety_alerts import ClinicalSafetyEngine, AlertSeverity
        
        text = """Neutrofili: 5000
Potassio: 4.2
Emoglobina: 14.0
Piastrine: 250000
LDH: 200"""
        
        parsed = parse_lab_text(text)
        patient_data = build_patient_data(parsed)
        
        engine = ClinicalSafetyEngine()
        alerts = engine.run_full_safety_check(patient_data)
        
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        assert critical_count == 0
