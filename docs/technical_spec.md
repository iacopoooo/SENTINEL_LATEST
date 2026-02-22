# SENTINEL v2.5 - Specifiche Tecniche

## Architettura

```
Input (Dati Clinici)
       │
       ▼
┌─────────────────┐
│ Evidence Mapper │  Converte dati clinici → evidenze SENTINEL
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Bayesian Engine │  Aggiorna probabilità con Teorema di Bayes
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Risk Assessor   │  Classifica rischio e genera raccomandazioni
└─────────────────┘
       │
       ▼
Output (Report + Alert)
```

---

## Algoritmo Bayesiano

### Formula (Odds Form)
```
posterior_odds = prior_odds × LR₁ × LR₂ × ... × LRₙ
posterior_prob = posterior_odds / (1 + posterior_odds)
```

### Likelihood Ratios (LR)
- LR > 1: Aumenta probabilità
- LR < 1: Diminuisce probabilità
- LR = 1: Nessun effetto

---

## Meccanismi di Resistenza Modellati

| Meccanismo | Prior | Evidenze Chiave |
|------------|-------|-----------------|
| C797S_mutation | 15% | ctDNA_C797S_trace/confirmed |
| MET_amplification | 15% | MET_cn_gain_low/medium/high |
| SCLC_transformation | 5% | TP53_RB1_double_loss, Synaptophysin |
| HER2_amplification | 5% | HER2_cn_gain_low/high |
| EMT_phenotype | 5% | Vimentin_high, E_cadherin_loss |
| PIK3CA_mutation | 5% | ctDNA_PIK3CA_detected |

---

## Mutual Exclusivity

Evidenze della stessa categoria si escludono (solo la più alta conta):
- MET_cn_gain: low < medium < high
- VAF_trend: decreasing < stable < rising_mild < rising_moderate < rising_rapid
- Clinical response: CR < PR < SD < PD

---

## Configurazione

File: `config/trial_config.json`

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| high_risk_threshold | 0.50 | Soglia per HIGH risk |
| critical_risk_threshold | 0.75 | Soglia per CRITICAL |
| time_decay_rate | 0.98 | Decay settimanale |
| max_cumulative_lr | 500 | Cap su LR cumulativo |

---

## Versione

- Engine: v2.5
- Data: Gennaio 2026
