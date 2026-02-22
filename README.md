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
