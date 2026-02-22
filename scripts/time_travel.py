import json
import os
import sys
import random
from datetime import datetime, timedelta

# Setup Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'patients')


def load_patient(patient_id):
    path = os.path.join(DATA_DIR, f"{patient_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def save_new_visit(original_id, data, months_passed):
    # Crea un nuovo ID per la visita (es. SENT-2026-0012_V1)
    new_id = f"{original_id}_V1"
    data['baseline']['patient_id'] = new_id

    # Aggiorna la data
    try:
        current_date = datetime.strptime("2026-01-11", "%Y-%m-%d")  # Data fissa simulazione
    except:
        current_date = datetime.now()

    future_date = current_date + timedelta(days=30 * months_passed)
    data['last_update'] = future_date.strftime("%Y-%m-%d")

    path = os.path.join(DATA_DIR, f"{new_id}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    return new_id, path


def simulate_evolution(patient_data, intervention_type):
    """
    Logica di evoluzione clonale basata sulla terapia scelta.
    intervention_type: 'standard' (Chemo) o 'precision' (Combo Elefante)
    """
    base = patient_data.get('baseline', {})

    # Fattori di Rischio Iniziali
    has_tp53 = base.get('tp53_status') == 'mutated'
    has_met = float(base.get('met_cn') or 0) >= 4.0
    has_pik3ca = base.get('pik3ca_status') == 'mutated'

    print(f"\n[SIMULATION ENGINE] Analyzing Evolution for {base.get('patient_id')}...")
    print(f"   > Drivers: TP53={has_tp53}, MET={has_met}, PIK3CA={has_pik3ca}")
    print(f"   > Intervention: {intervention_type.upper()}")

    # --- SCENARIO 1: TERAPIA SBAGLIATA (Standard Chemo su profilo High Risk) ---
    if intervention_type == 'standard' and (has_tp53 or has_met or has_pik3ca):
        print("   > ⚠️ RILEVATA PRESSIONE SELETTIVA INEFFICACE.")

        # 1. Peggioramento Clinico
        base['ecog_ps'] = min(int(base.get('ecog_ps', 0)) + 1, 4)

        # 2. Esplosione MET (Resistenza acquisita)
        if has_met:
            new_cn = float(base.get('met_cn')) + 4.0  # Da 6.0 a 10.0!
            base['met_cn'] = str(new_cn)
            print(f"   > 🧬 MET Amplification Worsened: {new_cn}")
        elif random.random() > 0.6:  # 40% chance di sviluppare MET ex-novo
            base['met_cn'] = "5.5"
            print("   > 🧬 NEW RESISTANCE: MET Amplification detected")

        # 3. Trasformazione Fenotipica (Se c'è TP53)
        if has_tp53 and random.random() > 0.5:
            base['rb1_status'] = 'loss'
            base['histology'] = 'SCLC-Transformation (High Grade)'
            print("   > ☢️ CRITICAL EVENT: Histological Transformation to SCLC")

        base['current_therapy'] = "Chemotherapy (Failing)"
        outcome = "PROGRESSION"

    # --- SCENARIO 2: TERAPIA CORRETTA (Combo Elefante) ---
    elif intervention_type == 'precision':
        print("   > ✅ TARGETING MOLECOLARE E METABOLICO ATTIVO.")

        # 1. Risposta Clinica
        base['ecog_ps'] = max(int(base.get('ecog_ps', 1)) - 1, 0)

        # 2. Soppressione Cloni
        if has_met:
            base['met_cn'] = "2.1"  # Normalizzazione
            print("   > 📉 MET Clone Suppressed (CN normalized)")

        if has_pik3ca:
            base['pik3ca_status'] = 'wt'  # Clone PIK3CA non più rilevabile
            print("   > 📉 PIK3CA Clone Eradicated (Undetectable)")

        base['current_therapy'] = "Elephant Combo Maintenance"
        outcome = "PARTIAL RESPONSE / STABLE"

    # --- SCENARIO 3: NEUTRO ---
    else:
        outcome = "STABLE DISEASE"

    patient_data['baseline'] = base
    return outcome


# --- INTERFACCIA UTENTE ---
def main():
    print("============================================================")
    print(" 🕰️  SENTINEL TIME MACHINE - CLONAL EVOLUTION SIMULATOR")
    print("============================================================")

    # 1. Selezione Paziente
    patient_id = input("Inserisci ID Paziente (es. SENT-2026-0012): ").strip()
    data = load_patient(patient_id)

    if not data:
        print("❌ Paziente non trovato.")
        return

    print(f"Paziente caricato: {patient_id}")
    print("Simulazione: +6 Mesi dalla diagnosi.")

    # 2. Scelta Terapeutica
    print("\nQuale terapia è stata somministrata in questi 6 mesi?")
    print("  [1] Standard Care (Chemioterapia / Solo Immunoterapia)")
    print("  [2] Sentinel Protocol (Combo: Metformina + Target Specifico)")

    choice = input("Scelta [1/2]: ").strip()

    if choice == '1':
        intervention = 'standard'
    elif choice == '2':
        intervention = 'precision'
    else:
        print("Scelta non valida.")
        return

    # 3. Esecuzione Simulazione
    outcome = simulate_evolution(data, intervention)

    # 4. Salvataggio
    new_id, new_path = save_new_visit(patient_id, data, 6)

    print("\n============================================================")
    print(f" RISULTATO SIMULAZIONE: {outcome}")
    print("============================================================")
    print(f"✅ Nuova cartella clinica generata: {new_id}")
    print(f"📂 File salvato: {new_path}")
    print("-> Vai alla Dashboard per vedere l'evoluzione.")


if __name__ == "__main__":
    main()