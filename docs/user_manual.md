# SENTINEL Trial - Manuale Utente

## Indice
1. [Introduzione](#introduzione)
2. [Requisiti](#requisiti)
3. [Workflow Operativo](#workflow-operativo)
4. [Comandi](#comandi)
5. [Interpretazione Risultati](#interpretazione-risultati)
6. [Troubleshooting](#troubleshooting)

---

## Introduzione

SENTINEL v2.5 è un sistema di Early Warning per la predizione della resistenza ai farmaci in pazienti con NSCLC EGFR+.

**NON è un sistema diagnostico.** È uno strumento di SUPPORTO decisionale. La decisione clinica finale spetta SEMPRE al medico.

---

## Requisiti

- Python 3.10+
- Librerie: pandas, numpy, scipy, openpyxl
- Accesso alla cartella SENTINEL_TRIAL

---

## Workflow Operativo

### 1. Nuovo Paziente
```bash
python scripts/new_patient.py
```

### 2. Aggiungere Visita
```bash
python scripts/add_visit.py SENT-2026-0001 --analyze
```

### 3. Analisi On-Demand
```bash
python scripts/analyze_patient.py SENT-2026-0001
```

---

## Comandi

| Comando | Descrizione |
|---------|-------------|
| `python scripts/new_patient.py` | Registra nuovo paziente |
| `python scripts/new_patient.py --quick` | Registrazione rapida |
| `python scripts/new_patient.py --templates` | Crea template Excel |
| `python scripts/add_visit.py ID` | Aggiungi visita |
| `python scripts/add_visit.py ID --analyze` | Visita + analisi |
| `python scripts/analyze_patient.py ID` | Analizza paziente |
| `python scripts/analyze_patient.py ID --override` | Con override clinico |

---

## Interpretazione Risultati

### Livelli di Rischio

| Livello | Probabilità | Azione |
|---------|-------------|--------|
| CRITICAL | ≥75% | Switch terapia immediato |
| HIGH | 50-75% | Considerare switch/combo |
| MEDIUM | 30-50% | Aumentare monitoraggio |
| LOW | 15-30% | Continuare, osservare |
| MINIMAL | <15% | Continuare terapia |

### Alert

Gli alert vengono generati quando un meccanismo supera il 50% di probabilità.

---

## Troubleshooting

**Errore: Patient not found**
- Verifica che il paziente sia registrato in `data/patients/`

**Errore: No evidences**
- Verifica che i dati della visita siano completi

**Report non generato**
- Controlla permessi sulla cartella `reports/`

---

## Contatti

- Supporto tecnico: sentinel.support@example.com
- Principal Investigator: [Nome]
