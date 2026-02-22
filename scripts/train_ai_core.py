#!/usr/bin/env python3
"""
SENTINEL AI CORE - RETROSPECTIVE TRAINING (v1.0)
================================================
1. Genera Dataset Storico (2020-2025) con esiti noti (Ground Truth).
2. Addestra Modello Machine Learning (Random Forest).
3. Testa accuratezza su dati storici (Backtesting).
4. Applica il modello ai pazienti attuali (Inference).
"""

import pandas as pd
import numpy as np
import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- CONFIGURAZIONE ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'patients'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Colori console
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_RESET = "\033[0m"


# --- 1. GENERAZIONE DATASET STORICO (SIMULAZIONE) ---
def generate_historical_data(n_patients=500):
    print(f"{C_CYAN}[HISTORY] Generazione database retrospettivo ({n_patients} casi)...{C_RESET}")

    data = []
    for i in range(n_patients):
        # Generazione Features Casuali
        age = random.randint(40, 85)
        sex = random.choice([0, 1])  # 0:F, 1:M
        ecog = random.choice([0, 1, 2, 3])
        ldh = int(np.random.normal(300, 150))  # Distribuzione normale attorno a 300
        ldh = max(100, ldh)  # Minimo 100

        neutrophils = random.randint(2000, 8000)
        lymphocytes = random.randint(500, 3000)
        nlr = round(neutrophils / lymphocytes, 2)

        kras = random.choice([0, 1])  # 1: Mutated
        tp53 = random.choice([0, 1])

        # --- LOGICA DEL DESTINO (GROUND TRUTH) ---
        # Qui nascondiamo i pattern che l'AI deve scoprire da sola
        death_score = 0

        # Pattern 1: Effetto Warburg (LDH)
        if ldh > 450: death_score += 4

        # Pattern 2: Resistenza Immunitaria (NLR)
        if nlr > 4.0: death_score += 3

        # Pattern 3: Genetica Aggressiva
        if tp53 == 1 and kras == 1: death_score += 3

        # Pattern 4: Clinica
        if ecog >= 2: death_score += 2

        # Random Noise (La medicina non è matematica perfetta)
        death_score += random.uniform(-2, 2)

        # Esito: 0 = Vivo (Responder), 1 = Deceduto/Progressione (Non-Responder)
        outcome = 1 if death_score > 4 else 0

        data.append({
            'age': age, 'sex': sex, 'ecog': ecog,
            'ldh': ldh, 'nlr': nlr,
            'kras': kras, 'tp53': tp53,
            'outcome': outcome  # TARGET
        })

    df = pd.DataFrame(data)
    print(f"{C_GREEN}[OK] Dataset generato. Esempio:{C_RESET}")
    print(df.head(3))
    print("-" * 50)
    return df


# --- 2. TRAINING & BACKTESTING ---
def train_and_validate():
    df = generate_historical_data(1000)

    # Separazione Features (X) e Target (y)
    X = df.drop('outcome', axis=1)
    y = df['outcome']

    # Split: 80% Train (Impara), 20% Test (Esame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"{C_CYAN}[TRAINING] Addestramento Random Forest in corso...{C_RESET}")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validazione
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{C_YELLOW}>>> ACCURATEZZA MODELLO: {acc * 100:.1f}%{C_RESET}")
    print(" (L'AI ha imparato correttamente che LDH e NLR sono cruciali?)")

    # Salva il cervello
    joblib.dump(model, MODEL_DIR / 'sentinel_brain_v1.pkl')
    return model


# --- 3. INFERENCE SUI PAZIENTI ATTUALI ---
def predict_current_patients(model):
    print(f"\n{C_CYAN}[LIVE] Analisi Pazienti Attuali (Dal JSON)...{C_RESET}")

    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("Nessun paziente trovato.")
        return

    print(f"{'ID PAZIENTE':<20} | {'LDH':<6} | {'NLR':<5} | {'PREDIZIONE AI':<20} | {'CONFIDENZA'}")
    print("-" * 80)

    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                base = data['baseline']
                blood = base.get('blood_markers', {})

                # Preparazione Dati per l'AI (Deve avere lo stesso formato del training)
                # Mappatura dati
                ecog = base.get('ecog_ps', 0)
                ldh = blood.get('ldh', 200)
                nlr = blood.get('nlr', 1.0)
                kras = 1 if base.get('kras_mutation') not in ['wt', 'Other', None] else 0
                tp53 = 1 if base.get('tp53_status') == 'mutated' else 0
                sex = 1 if base.get('sex') == 'M' else 0
                age = base.get('age', 60)

                # Input Array
                input_data = pd.DataFrame([{
                    'age': age, 'sex': sex, 'ecog': ecog,
                    'ldh': ldh, 'nlr': nlr,
                    'kras': kras, 'tp53': tp53
                }])

                # Predizione
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][prediction]

                outcome_str = f"{C_RED}HIGH RISK (Death){C_RESET}" if prediction == 1 else f"{C_GREEN}RESPONDER (Survive){C_RESET}"

                print(
                    f"{base['patient_id']:<20} | {str(ldh):<6} | {str(nlr):<5} | {outcome_str:<28} | {proba * 100:.0f}%")

        except Exception as e:
            print(f"Errore su {f.name}: {e}")


if __name__ == "__main__":
    # Controllo Librerie
    try:
        import sklearn
    except ImportError:
        print(f"{C_RED}[ERR] Manca 'scikit-learn'. Esegui: pip install scikit-learn pandas{C_RESET}")
        exit()

    brain = train_and_validate()
    predict_current_patients(brain)