import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/iacopo/Scrivania/SENTINEL_TRIAL_2/SENTINEL_TRIAL")
sys.path.insert(0, str(BASE_DIR / "src"))

from sentinel_engine import VetoSystem

with open(BASE_DIR / "data/patients/patient_208.json", "r") as f:
    patient_data = json.load(f)

baseline_data = patient_data.get("baseline", patient_data)
print("baseline therapy:", baseline_data.get("current_therapy"))
print("tmb:", baseline_data.get("tmb") or baseline_data.get("tmb_score") or baseline_data.get("genetics", {}).get("tmb"))
veto_sys = VetoSystem()
vetos = veto_sys.check_therapy(baseline_data)
print("Vetos:", vetos)

