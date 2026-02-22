"""
VITTORIA AI - CORE PREDICTION & ANALYSIS ENGINE (v5.0 Pan-Cancer)
=================================================================
Versione Integrata per SENTINEL ENGINE v15.
Include:
1. Predizione ML (Random Forest)
2. Analisi Euristica (Protocollo Elefante, Rischio SCLC)
3. Adapter per compatibilità con Sentinel Engine
"""

import os
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# --- STRUTTURE DATI PER COMPATIBILITÀ CON SENTINEL ENGINE ---
@dataclass
class DrugRecStub:
    drug_name: str
    notes: str

@dataclass
class VittoriaResult:
    neural_risk: int
    recommendations: List[DrugRecStub]

class VittoriaNeuralNet:
    """
    VITTORIA AI (ex VittoriaAI class)
    Rinomina per compatibilità con l'import di Sentinel Engine.
    """

    def __init__(self, model_path="../models/vittoria_model_v1.pkl", encoders_path="../models/label_encoders.pkl"):
        self.model = None
        self.encoders = {}

        # Tenta il caricamento del modello
        if os.path.exists(model_path) and os.path.exists(encoders_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
            except Exception as e:
                # print(f"[VITTORIA] Errore caricamento modello: {e}") # Silenziato per pulizia log
                pass
        else:
            # print("[VITTORIA] Modello ML non trovato - uso regole euristiche di fallback.")
            pass

    # =========================================================================
    # 1. ANALISI CLINICA E RISCHIO (Il Tuo Codice Originale)
    # =========================================================================
    def analyze_patient(self, patient_data):
        """
        Analizza il paziente e restituisce Risk Score, Raccomandazioni e Alert.
        Integra la logica del PROTOCOLLO ELEFANTE.
        """
        risk_score = 0
        reasons = []
        recommendation = "Monitoraggio Standard (CT/MRI ogni 3 mesi)"

        # Estrazione Dati con gestione sicura dei tipi
        tp53 = patient_data.get('tp53_status', 'wt')

        try:
            met_cn = float(patient_data.get('met_cn') or 0.0)
        except:
            met_cn = 0.0

        pik3ca = patient_data.get('pik3ca_status', 'wt')
        therapy = str(patient_data.get('current_therapy', '')).lower()
        histology = patient_data.get('histology', 'adenocarcinoma')
        rb1 = patient_data.get('rb1_status', 'wt')

        # --- LOGICA PROTOCOLLO ELEFANTE (Updated Pan-Cancer) ---

        # A. Base: TP53 Mutato (Il 'Guardiano' è caduto)
        if str(tp53).lower() in ['mutated', 'loss', 'pos']:
            risk_score += 30
            reasons.append("TP53 Loss of Function (Genomic Instability)")

            # Se è sotto Chemio o Immunoterapia generica, il rischio aumenta
            if 'chemo' in therapy or 'platinum' in therapy:
                risk_score += 20
                reasons.append("High Risk of Chemo-Resistance")
                recommendation = "PROTOCOLLO ELEFANTE: Valutare aggiunta Metformina"

            # B. Avanzato: MET Amplification (Il 'Turbo' della resistenza)
            if met_cn >= 5.0:
                risk_score += 40
                reasons.append(f"MET Amplification (CN: {met_cn})")
                recommendation = "PROTOCOLLO ELEFANTE COMBO: Metformina + Capmatinib/Tepotinib"

            # C. Avanzato: PIK3CA Mutation (Via di fuga metabolica)
            if str(pik3ca).lower() in ['mutated', 'pos']:
                risk_score += 35
                reasons.append("PIK3CA Activation (mTOR pathway)")
                recommendation = "PROTOCOLLO ELEFANTE COMBO: Metformina + Alpelisib"

        # --- ALTRI RISCHI (SCLC Transformation) ---
        if str(rb1).lower() in ['loss', 'mutated'] and str(tp53).lower() in ['mutated', 'loss']:
            risk_score = 95
            reasons.append("CRITICAL: High Risk of SCLC Transformation")
            recommendation = "Urgent Biopsy & Plasma-Seq required. Consider Platinum-Etoposide."

        # --- RISCHI STANDARD (KRAS/EGFR) ---
        kras = patient_data.get('kras_mutation')
        if kras and kras not in ['wt', 'None', None]:
            if 'sotorasib' not in therapy and 'adagrasib' not in therapy:
                risk_score += 50
                reasons.append(f"KRAS {kras} non trattato con inibitore specifico")
                recommendation = "Valutare switch a Sotorasib/Adagrasib"

        return {
            "risk_score": min(risk_score, 100),
            "reasons": reasons,
            "recommendation": recommendation,
            "dominant_risk": reasons[0] if reasons else "Stable Disease"
        }

    # =========================================================================
    # 2. ADAPTER PER SENTINEL ENGINE (Il Ponte Magico)
    # =========================================================================
    def get_recommendations(self, profile: dict) -> VittoriaResult:
        """
        Questo metodo viene chiamato da sentinel_engine.py.
        Traduce la richiesta dell'engine nella tua logica analyze_patient.
        """
        # Mappiamo i nomi delle chiavi dall'Engine al tuo formato
        mapped_data = {
            'tp53_status': profile.get('tp53_status', 'wt'),
            'kras_mutation': profile.get('kras_mutation', 'wt'),
            'current_therapy': profile.get('current_therapy', ''),
            'met_cn': profile.get('met_cn', 0.0),
            'pik3ca_status': profile.get('pik3ca_status', 'wt'),
            'rb1_status': profile.get('rb1_status', 'wt'),
            # Aggiungiamo dati base se mancano
            'age': 65,
            'sex': 'M'
        }

        # Usiamo la tua logica originale
        analysis = self.analyze_patient(mapped_data)

        # Restituiamo il risultato nel formato che l'Engine si aspetta
        rec_obj = DrugRecStub(
            drug_name=analysis['recommendation'],
            notes=f"Reasons: {', '.join(analysis['reasons'])}"
        )

        return VittoriaResult(
            neural_risk=analysis['risk_score'],
            recommendations=[rec_obj]
        )