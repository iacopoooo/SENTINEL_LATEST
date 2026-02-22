"""
PATIENT DATABASE - Persistenza JSON pazienti
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class PatientDatabase:
    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/patients")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_patient(self, patient_data: Dict[str, Any], overwrite: bool = False) -> Path:
        """Salva paziente come JSON"""
        baseline = patient_data.get("baseline", patient_data)
        patient_id = baseline.get("patient_id")
        
        if not patient_id:
            raise ValueError("patient_id mancante")
        
        json_path = self.data_dir / f"{patient_id}.json"
        
        if json_path.exists() and not overwrite:
            raise FileExistsError(f"Paziente {patient_id} esiste già. Usa overwrite=True o merge.")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def load_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Carica paziente da JSON"""
        json_path = self.data_dir / f"{patient_id}.json"
        
        if not json_path.exists():
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def patient_exists(self, patient_id: str) -> bool:
        """Verifica se paziente esiste"""
        return (self.data_dir / f"{patient_id}.json").exists()
    
    def list_patients(self) -> List[str]:
        """Lista tutti i pazienti"""
        return [f.stem for f in self.data_dir.glob("*.json")]
    
    def merge_with_existing(self, patient_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge nuovi dati con paziente esistente"""
        existing = self.load_patient(patient_id)
        
        if not existing:
            return new_data
        
        # Merge baseline (new sovrascrive)
        if "baseline" in new_data:
            existing_baseline = existing.get("baseline", {})
            new_baseline = new_data.get("baseline", {})
            
            # Deep merge genetics
            if "genetics" in new_baseline:
                existing_genetics = existing_baseline.get("genetics", {})
                existing_genetics.update(new_baseline["genetics"])
                new_baseline["genetics"] = existing_genetics
            
            existing_baseline.update(new_baseline)
            existing["baseline"] = existing_baseline
        
        # Append visits
        if "visits" in new_data and new_data["visits"]:
            existing_visits = existing.get("visits", [])
            existing_visits.extend(new_data["visits"])
            existing["visits"] = existing_visits
        
        # Update metadata
        existing["baseline"]["updated_at"] = datetime.now().isoformat()
        
        return existing
    
    def delete_patient(self, patient_id: str) -> bool:
        """Elimina paziente"""
        json_path = self.data_dir / f"{patient_id}.json"
        if json_path.exists():
            json_path.unlink()
            return True
        return False
