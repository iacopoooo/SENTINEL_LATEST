import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/iacopo/Scrivania/SENTINEL_TRIAL_2/SENTINEL_TRIAL")
sys.path.insert(0, str(BASE_DIR / "src"))

from sentinel_engine import VetoSystem

with open(BASE_DIR / "data/patients/P-0000208.json", "r") as f:
    patient_data = json.load(f)

baseline_data = patient_data.get("baseline", patient_data)
from pprint import pprint

try:
    veto_sys = VetoSystem()
    vetos = veto_sys.check_therapy(baseline_data)
    for v in vetos:
        print(v)
except Exception as e:
    import traceback
    traceback.print_exc()

