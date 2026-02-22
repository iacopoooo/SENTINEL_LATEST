# SENTINEL Trial v3.0

Sistema di Early Warning Bayesiano Avanzato per Resistenza NSCLC EGFR+

---

## 🆕 Novità v3.0

### Correlation Priors
I meccanismi di resistenza non sono più trattati come indipendenti:

- **EMT Cluster**: EMT ↔ MET ↔ AXL (co-attivazione frequente)
- **Genomic Instability**: TP53 ↔ RB1 ↔ SCLC transformation
- **Bypass Signaling**: MET ↔ HER2 ↔ PIK3CA (redundancy)
- **On-target**: C797S ↔ T790M (mutually exclusive in cis)

### Temporal Dependencies
L'ordine temporale modifica le probabilità:

- **Early resistance** (< 6 mesi): C797S, T790M più probabili
- **Late resistance** (> 12 mesi): Transformation, EMT più probabili

### Pattern Recognition Automatico
- `rapid_vaf_stable_imaging`: Clone emergente pre-clinico
- `pd_stable_vaf`: Possibile transformation
- `c797s_with_emt`: Resistenza multifattoriale

---

## 🆕 VITTORIA-NSCLC Integration (v3.5)

### Cos'è VITTORIA 3.0?
Sistema di raccomandazione farmacologica **unificato** che combina:

1. **Motore Statistico** (XGBoost trainato su MSK dataset)
   - Clustering data-driven (K=3)
   - AUC validato con 5-fold CV
   - Probabilità IO realistiche

2. **Database Farmaci** (15+ farmaci)
   - Probabilità per meccanismo
   - Boost/penalità specifici
   - Esclusione terapia attuale

3. **Integrazione SENTINEL**
   - Meccanismo di resistenza
   - Correlation priors

### I 3 Cluster (da XGBoost + K-Means)
| Cluster | Nome | Caratteristica | Risposta IO |
|---------|------|----------------|-------------|
| 0 | Standard | TMB medio | 45-55% |
| 1 | Sensitive | TMB alto, no STK11/KEAP1 | 70-80% |
| 2 | Resistant | STK11/KEAP1+ | 35-45% |

### Training del Modello
```bash
# Richiede dataset MSK
python scripts/train_vittoria_model.py --dataset IO_dataset_v01_MSK.csv
```

### Analisi Integrata
```bash
python scripts/integrated_analysis.py SENT-2026-0001
```

### Console Interattiva
```bash
python -c "from src.vittoria_3_0 import interactive_console; interactive_console()"
```

Output esempio:
```
FASE 1: SENTINEL v3.0 - Rilevamento Resistenza
   Meccanismo: MET_amplification (89%)
   
FASE 2: VITTORIA 3.0 - Selezione Farmaco
   Cluster: 0 - Standard
   Prob. Risposta IO: 52%
   
RACCOMANDAZIONI:
   1. Capmatinib           68%   MET inhibitor (FDA approved)
   2. Tepotinib            65%   MET inhibitor
   3. Savolitinib          60%   MET inhibitor
   
   ✅ RACCOMANDAZIONE: Capmatinib
   ⛔ EVITARE: Osimertinib (terapia attuale), Immunoterapia monoterapia
```

---

## 📁 Struttura Progetto

```
SENTINEL_TRIAL/
├── config/                     # Configurazione
│   └── trial_config.json
├── data/                       # Dati
│   ├── patients/               # JSON pazienti
│   ├── templates/              # Template Excel
│   ├── exports/
│   └── backups/
├── docs/                       # Documentazione
├── logs/                       # Audit logs
├── reports/                    # Report generati
│   ├── daily/
│   ├── weekly/
│   └── audit/
├── scripts/                    # Script operativi
│   ├── new_patient.py
│   ├── add_visit.py
│   └── analyze_patient.py
├── src/                        # Codice sorgente
│   ├── sentinel_v2_5.py
│   ├── evidence_mapper.py
│   └── patient_manager.py
├── validation/                 # Test
└── README.md
```

---

## 🚀 Quick Start

### 1. Nuovo Paziente
```bash
python scripts/new_patient.py          # Interattivo completo
python scripts/new_patient.py --quick  # Registrazione rapida
```

### 2. Modifica Paziente
```bash
python scripts/edit_patient.py                  # Seleziona da lista
python scripts/edit_patient.py SENT-2026-0001   # Paziente specifico
```

### 3. Aggiungi Visita
```bash
python scripts/add_visit.py SENT-2026-0001 --analyze
```

### 4. Analizza Paziente
```bash
python scripts/analyze_patient.py SENT-2026-0001
```

### 5. Dashboard e Report
```bash
python scripts/dashboard.py                     # Dashboard interattiva
python scripts/dashboard.py --weekly            # Report settimanale
python scripts/dashboard.py --evolution SENT-2026-0001  # Evoluzione paziente
python scripts/dashboard.py --export            # Export HTML
```

### 6. Sistema Alert
```bash
python scripts/alert_system.py                  # Check alert
python scripts/alert_system.py --save           # Salva report
```

### 7. Simulazione Scenari
```bash
python scripts/simulate_visit.py SENT-2026-0001         # Interattivo
python scripts/simulate_visit.py SENT-2026-0001 --batch # Tutti scenari
python scripts/simulate_visit.py --list                 # Lista scenari
```

### 8. Import/Export
```bash
python scripts/import_export.py --export excel  # Export Excel
python scripts/import_export.py --export stats  # Export statistiche
python scripts/import_export.py --backup        # Backup completo
python scripts/import_export.py --templates     # Crea template
```

### 9. Validazione Clinica
```bash
python scripts/clinical_validation.py --register SENT-2026-0001  # Registra outcome
python scripts/clinical_validation.py --report   # Report validazione
python scripts/clinical_validation.py --list     # Lista pazienti
```

### 10. Valida Sistema
```bash
python validation/run_tests.py
```

---

## ⌨️ Navigazione

Durante l'inserimento dati, digita `<` per tornare al campo precedente:

```
Sesso
----------------------------------------
  1. M
  2. F

  [<< Torna a: INDIETRO (digita '<')]

Scegli [1-2]: <
  [<< Torno a: Eta']
```

---

## 📊 Funzionalità Avanzate

### Dashboard
- Overview tutti i pazienti con rischi
- Report settimanale aggregato  
- Evoluzione probabilità nel tempo
- Export HTML/PDF

### Alert System
- Monitoraggio automatico soglie
- Lista urgenze ordinata per priorità
- Notifiche pazienti critici
- Check visite scadute

### Simulatore
- Test scenari "what-if"
- Confronto outcome possibili
- Previsione impatto trattamenti

### Validazione Clinica
- Registrazione outcome reali
- Calcolo sensibilità/specificità
- Curve ROC e AUC
- Report validazione modello

---

## 📋 Workflow

```
PAZIENTE ARRIVA
      │
      ▼
┌─────────────────────────┐
│ 1. new_patient.py       │
│    Registra baseline    │
└─────────────────────────┘
      │
      ▼
  data/patients/SENT-2026-XXXX.json
      │
      ▼
┌─────────────────────────┐
│ 2. add_visit.py         │
│    Ogni visita clinica  │
└─────────────────────────┘
      │
      ▼
┌─────────────────────────┐
│ 3. analyze_patient.py   │
│    SENTINEL calcola     │
│    rischio resistenza   │
└─────────────────────────┘
      │
      ▼
  reports/daily/SENT-2026-XXXX_*.txt
```

---

## 📊 Interpretazione Risultati

| Livello | Probabilità | Azione |
|---------|-------------|--------|
| 🚨 CRITICAL | ≥75% | Switch immediato |
| ⚠️ HIGH | 50-75% | Considerare switch |
| 📊 MEDIUM | 30-50% | Aumentare monitoring |
| 📉 LOW | 15-30% | Continuare, osservare |
| ✅ MINIMAL | <15% | Continuare terapia |

---

## ⚠️ Limitazioni

1. **Meccanismi indipendenti** - Correlazioni biologiche non modellate
2. **Sistema di SUPPORTO** - La decisione spetta al clinico
3. **Meccanismi multipli** - Identifica solo il dominante

---

## 📜 Versione

- SENTINEL Engine: v2.5
- Data: Gennaio 2026

---

## ⚖️ Disclaimer

Sistema approvato per uso in trial clinico.
Qualsiasi modifica richiede approvazione del comitato etico.

# 🧬 SENTINEL v18.0 - Neuro-Symbolic Clinical Decision Support System

![Version](https://img.shields.io/badge/Version-18.0_Release-blue.svg)
![Status](https://img.shields.io/badge/Status-Clinical_Ready-success.svg)
![Architecture](https://img.shields.io/badge/Architecture-Neuro--Symbolic_AI-purple.svg)

**SENTINEL** è un'architettura avanzata di Intelligenza Artificiale per l'oncologia di precisione. Progettato per operare in ambienti ad alta complessità clinica, il sistema fonde il **Machine Learning Predittivo** con un **Motore Deterministico di Regole Biologiche (VetoSystem)**, eliminando il rischio di "allucinazioni" statistiche e garantendo una sicurezza prescrittiva assoluta.

## 🚀 Executive Summary
SENTINEL analizza trasversalmente genomica (NGS/ctDNA), biomarcatori ematici, cinetica tumorale e farmacogenomica (PGx) per simulare l'evoluzione clinica del paziente. Il sistema intercetta errori terapeutici fatali (es. mismatch immunoterapici, tossicità genetiche) al "Giorno 0" e calcola traiettorie correttive in tempo reale.

## 🧠 L'Architettura Neuro-Simbolica (Core Modules)

SENTINEL si basa su 6 motori interconnessi:

### 1. PROMETHEUS (Epistatic Discovery Engine)
Il pianificatore strategico. Analizza le correlazioni epistatiche e proietta il rischio a 5 anni.
* **Neuro-Symbolic Integration:** Assorbe i divieti assoluti dal motore clinico (VetoSystem). Se viene rilevato un errore letale (es. Immunoterapia su tumore con TMB < 5), il rischio viene matematicamente portato a livelli critici (99/100).
* **Output:** Genera uno *Strategic Action Plan* ordinato per priorità per correggere la traiettoria clinica del paziente.

### 2. ORACLE (Digital Twin Simulator)
Il motore diagnostico e prognostico puro. Addestrato su curve di sopravvivenza reali (Kaplan-Meier), calcola il tempo esatto alla progressione (PFS) e traccia l'impatto di mutazioni primarie e secondarie.

### 3. PROPHET (Temporal & Kinematics Engine)
Il motore cinematico. Non si limita a leggere i valori del sangue, ma ne calcola **Velocità e Accelerazione** nel tempo (es. impennate di LDH o CRP). 
* *Safety Fallback:* Richiede rigorosamente almeno 2 visite per estrapolare la derivata temporale. In caso di singola visita, si affida a soglie assolute di gravità per impedire falsi sensi di sicurezza.

### 4. CHRONOS (Clonal Evolution Tracker)
Sorveglianza longitudinale del ctDNA. Traccia la frequenza allelica delle varianti (VAF) nel sangue periferico, disegnando l'emersione di cloni resistenti (es. *EGFR C797S*, *MET amp*) prima ancora che la progressione sia visibile radiologicamente.

### 5. ADAPTIVE THERAPY & ELEPHANT PROTOCOL
Motore posologico dinamico. Modula l'intensità della terapia in base alla risposta:
* Suggerisce **Drug Holidays** in caso di *Deep Complete Response* prolungata.
* Scala a **Best Supportive Care (BSC)** intercettando indici di fragilità estrema (ECOG >= 3), evitando accanimenti terapeutici tossici.

### 6. FARMACOGENOMICA (PGx Safety Guard)
Analizza polimorfismi enzimatici germinali (es. *DPYD*, *UGT1A1*) e li incrocia con i farmaci prescritti, bloccando in millisecondi la somministrazione di chemioterapici in pazienti a rischio di tossicità letale (Grado 4/5).

---

## 🔒 Sicurezza e Gestione dei Dati (Split Code/Data Pattern)
Nel rispetto delle normative sulla privacy dei dati sanitari e dei limiti architetturali di Git:
* **Questo repository contiene esclusivamente il codice sorgente (il "cervello").**
* I dataset primari, i pesi dei modelli ML completi e i log clinici reali (oltre 30GB) sono gestiti separatamente su storage sicuri e **NON** sono tracciati in questo repository.
* La cartella `data/` richiede il popolamento manuale in ambiente di deployment per consentire l'avvio della pipeline.

## 📄 Generazione Report
Il sistema culmina nell'esportazione di un PDF longitudinale multipagina per il Tumor Board, arricchito da una sintesi testuale generata da un LLM ancorato strettamente ai dati biologici estratti, per prevenire qualsiasi allucinazione AI.

---
*Sviluppato per redefinire i confini del supporto decisionale clinico.*

