import numpy as np
import json
import random
from datetime import datetime, timedelta

# =========================
# GLOBAL CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

START_YEAR = 2015

def random_dates(start_year=2015, n_visits=6, dropout_prob=0.1):
    dates = []
    current = datetime(start_year, 1, 1)
    for _ in range(n_visits):
        current += timedelta(days=random.randint(120, 180))
        if random.random() > dropout_prob:  # visita saltata
            dates.append(current.strftime("%Y-%m-%d"))
    return dates


# =========================
# CORE GENERATOR
# =========================
def generate_dataset(n_patients=120):
    dataset = []

    # =====================================================
    # GROUP 1 — TRUE HEALTHY (40%)
    # =====================================================
    for i in range(int(n_patients * 0.40)):
        visits = []
        dates = random_dates()

        base_glu = np.random.normal(85, 5)
        base_wgt = np.random.normal(75, 8)

        for d in dates:
            visits.append({
                "date": d,
                "blood": {
                    "glucose": round(base_glu + np.random.normal(0, 3), 1),
                    "ldh": round(np.random.normal(160, 15), 1),
                    "crp": round(abs(np.random.normal(0.8, 0.4)), 1),
                    "neutrophils": round(np.random.normal(4500, 600), 0)
                },
                "clinical": {
                    "weight": round(base_wgt + np.random.normal(0, 1), 1)
                },
                "noise_variants": []
            })

        dataset.append({
            "id": f"HEALTHY_{i:03d}",
            "sex": random.choice(["M", "F"]),
            "age_2015": random.randint(40, 75),
            "history": visits,
            "diagnosis_date": None,
            "truth": "HEALTHY"
        })

    # =====================================================
    # GROUP 2 — PANCREATIC METABOLIC DRIFT (20%)
    # =====================================================
    for i in range(int(n_patients * 0.20)):
        visits = []
        dates = random_dates()

        glucose = np.random.normal(88, 4)
        weight = np.random.normal(82, 7)

        for idx, d in enumerate(dates):
            if idx > 0:
                glucose += np.random.normal(2.5, 0.8)
                weight -= np.random.normal(1.3, 0.4)

            visits.append({
                "date": d,
                "blood": {
                    "glucose": round(glucose, 1),
                    "ldh": round(np.random.normal(170, 10), 1),
                },
                "clinical": {
                    "weight": round(weight, 1)
                },
                "noise_variants": []
            })

        dataset.append({
            "id": f"PANCREAS_{i:03d}",
            "sex": random.choice(["M", "F"]),
            "age_2015": random.randint(50, 80),
            "history": visits,
            "diagnosis_date": "2018-06-01",
            "truth": "PANCREATIC_CANCER"
        })

    # =====================================================
    # GROUP 3 — PREDIABETES (FALSE POSITIVE TRAP) (15%)
    # =====================================================
    for i in range(int(n_patients * 0.15)):
        visits = []
        dates = random_dates()

        glucose = np.random.normal(92, 4)
        weight = np.random.normal(80, 6)

        for d in dates:
            glucose += np.random.normal(2.0, 1.0)  # sale
            visits.append({
                "date": d,
                "blood": {
                    "glucose": round(glucose, 1),
                    "ldh": round(np.random.normal(165, 12), 1),
                },
                "clinical": {
                    "weight": round(weight + np.random.normal(0, 1), 1)  # peso stabile
                },
                "noise_variants": []
            })

        dataset.append({
            "id": f"PREDIAB_{i:03d}",
            "sex": random.choice(["M", "F"]),
            "age_2015": random.randint(45, 75),
            "history": visits,
            "diagnosis_date": None,
            "truth": "NO_CANCER_PREDIABETES"
        })

    # =====================================================
    # GROUP 4 — WEIGHT LOSS (DIET / STRESS) (10%)
    # =====================================================
    for i in range(int(n_patients * 0.10)):
        visits = []
        dates = random_dates()

        glucose = np.random.normal(88, 3)
        weight = np.random.normal(85, 6)

        for d in dates:
            weight -= np.random.normal(1.2, 0.6)
            visits.append({
                "date": d,
                "blood": {
                    "glucose": round(glucose + np.random.normal(0, 2), 1),
                    "ldh": round(np.random.normal(160, 15), 1),
                },
                "clinical": {
                    "weight": round(weight, 1)
                },
                "noise_variants": []
            })

        dataset.append({
            "id": f"DIET_{i:03d}",
            "sex": random.choice(["M", "F"]),
            "age_2015": random.randint(30, 65),
            "history": visits,
            "diagnosis_date": None,
            "truth": "NO_CANCER_WEIGHT_LOSS"
        })

    # =====================================================
    # GROUP 5 — GHOST ctDNA RELAPSE (15%)
    # =====================================================
    for i in range(int(n_patients * 0.15)):
        visits = []
        dates = random_dates()

        vafs = [0, 0.05, 0.1, 0.2, 0.6, 2.5]
        genes = ["TP53", "KRAS", "EGFR"]

        for idx, d in enumerate(dates):
            noise = []
            if idx >= 2 and random.random() > 0.25:  # dropout ctDNA
                noise.append({
                    "gene": random.choice(genes),
                    "vaf": round(vafs[idx], 3)
                })

            visits.append({
                "date": d,
                "blood": {
                    "glucose": 90,
                    "ldh": 180 + (idx * 18 if idx > 3 else 0),
                },
                "clinical": {
                    "weight": 72
                },
                "noise_variants": noise
            })

        dataset.append({
            "id": f"GHOST_{i:03d}",
            "sex": random.choice(["M", "F"]),
            "age_2015": random.randint(55, 80),
            "history": visits,
            "diagnosis_date": "2017-09-01",
            "truth": "MOLECULAR_RELAPSE"
        })

    return dataset


# =========================
# SAVE FILE
# =========================
if __name__ == "__main__":
    data = generate_dataset(120)
    with open("dataset_oracle_stress_v2.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Dataset generato: dataset_oracle_stress_v2.json")
    print("   - Veri sani")
    print("   - Pancreas metabolico (early drift)")
    print("   - Falsi positivi (prediabete, dieta)")
    print("   - Ghost ctDNA intermittente")
    print("   - Missing data + gap temporali")
