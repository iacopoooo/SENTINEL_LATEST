#!/usr/bin/env python3
"""
SENTINEL TIME TRAVEL ENGINE (v8.0)
==================================
Simula la sopravvivenza (Kaplan-Meier simulata)
basandosi su:
1. Genetica (TP53, KRAS)
2. Effetto Warburg (LDH Alto) -> Protocollo Elefante
3. Stato Clinico (ECOG)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import sys

# Setup Percorsi relativo alla posizione di questo script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'patients'


def calculate_survival_curve(patient_data):
    """
    Genera una curva di sopravvivenza (24 mesi) basata sui dati reali del JSON.
    """
    base = patient_data['baseline']
    pid = base['patient_id']

    # 1. Recupero Fattori di Rischio
    tp53 = base.get('tp53_status') == 'mutated'
    kras = base.get('kras_mutation') not in ['wt', 'Other', None]
    ecog = base.get('ecog_ps', 0)

    # Recupero Sangue (Gestione sicura se manca)
    blood = base.get('blood_markers', {})
    ldh = blood.get('ldh', 200)
    if ldh is None: ldh = 200

    # 2. Logica di Simulazione
    months = np.arange(0, 25, 1)  # 24 mesi

    # SCENARIO A: PROTOCOLLO ELEFANTE (Warburg Effect)
    # Se LDH > 400, il paziente è metabolicamente attivo ma riceve Metformina
    if ldh > 400:
        # La curva scende inizialmente, poi si stabilizza grazie alla terapia metabolica
        curve = 100 * np.exp(-0.08 * months)
        # Boost Elefante: Dal mese 6 la curva smette di scendere
        curve[6:] = curve[6] * (1 - 0.01 * (months[6:] - 6))  # Declino lentissimo

        label = f"{pid} (ELEPHANT PROTOCOL)"
        color = 'orange'
        style = '-'

    # SCENARIO B: CRITICO (Genetica + Clinica pessima)
    elif (tp53 and kras) or ecog >= 2:
        # Crollo rapido
        curve = 100 * np.exp(-0.15 * months)
        label = f"{pid} (CRITICAL RISK)"
        color = 'red'
        style = '--'

    # SCENARIO C: STABILE (Osimertinib/Target efficace)
    else:
        # Sopravvivenza lunga
        curve = 100 * np.exp(-0.02 * months)
        label = f"{pid} (STABLE / TARGET)"
        color = 'green'
        style = '-'

    return months, curve, label, color, style


def run_simulation():
    print("\n⏳ [TIME TRAVEL] Caricamento Motore Predittivo...")
    print(f"📂 Lettura dati da: {DATA_DIR}")
    time.sleep(1)

    json_files = list(DATA_DIR.glob("*.json"))

    if not json_files:
        print("❌ Nessun paziente trovato nel database!")
        return

    # Setup Grafico
    plt.figure(figsize=(12, 7))
    plt.title(f"SENTINEL PREDICTIVE AI: 24-MONTH SURVIVAL PROJECTION", fontsize=14, fontweight='bold')
    plt.xlabel("Mesi dalla diagnosi")
    plt.ylabel("Probabilità di Sopravvivenza (%)")

    count = 0
    for f in json_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                months, curve, label, color, style = calculate_survival_curve(data)

                plt.plot(months, curve, label=label, color=color, linestyle=style, linewidth=2.5)
                count += 1
                print(f"   -> Calcolo curva per: {label}")
        except Exception as e:
            print(f"   [ERR] Errore lettura file {f.name}: {e}")

    if count > 0:
        plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label="Soglia Critica (50%)")
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)

        print(f"\n✅ Simulazione completata per {count} pazienti.")
        print("📊 Apertura finestra grafica...")
        plt.show()
    else:
        print("❌ Nessun dato valido da graficare.")


if __name__ == "__main__":
    run_simulation()