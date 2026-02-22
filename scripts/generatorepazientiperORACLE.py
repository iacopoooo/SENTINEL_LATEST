#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL SYNTHETIC DATA GENERATOR v3.1 (Enterprise + Stress Test)
=================================================================
Generatore di coorti sintetiche per stress-test clinico di Oracle/Prophet.

Upgrade vs v3.0:
- week_offset realistico calcolato dalle date
- diagnosis_date + diagnosis_window per fenotipi tumorali
- missingness per-marker (lab values mancanti)
- infection/toxicity spikes (confounder realistico per LDH/CRP/Neutrofili)
- rounding-safe allocation: total_n rispettato sempre
- output schema compatibile Sentinel + metadati utili

Author: Sentinel Team
Date: 2026-02-05
"""

from __future__ import annotations

import json
import random
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sentinel_synth")

SEED = 42
SIMULATION_START_YEAR = 2015
OUTPUT_FILENAME = "dataset_oracle_enterprise_v3_1.json"

random.seed(SEED)
np.random.seed(SEED)

DATE_FMT = "%Y-%m-%d"

# =============================================================================
# 2. DATA MODELS (Domain Entities)
# =============================================================================

@dataclass
class Variant:
    gene: str
    vaf: float
    mutation_type: str = "missense"

@dataclass
class BloodMarkers:
    glucose: Optional[float]
    ldh: Optional[float]
    crp: Optional[float]
    neutrophils: Optional[float]
    lymphocytes: Optional[float] = None
    albumin: Optional[float] = None

@dataclass
class ClinicalMetrics:
    weight: Optional[float]
    bmi: Optional[float] = None

@dataclass
class Visit:
    date: str
    week_offset: int
    blood: BloodMarkers
    clinical: ClinicalMetrics
    noise_variants: List[Variant] = field(default_factory=list)
    events: List[str] = field(default_factory=list)  # es. ["infection_spike"]

@dataclass
class Patient:
    id: str
    sex: str
    age_baseline: int
    phenotype_label: str
    history: List[Visit]
    ground_truth: str

    # opzionali utili per valutazione Oracle/lead-time
    diagnosis_date: Optional[str] = None
    diagnosis_window: Optional[str] = None  # es. "2017-06-01..2018-12-01"
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sex": self.sex,
            "age_2015": self.age_baseline,
            "truth": self.ground_truth,
            "phenotype": self.phenotype_label,
            "diagnosis_date": self.diagnosis_date,
            "diagnosis_window": self.diagnosis_window,
            "notes": self.notes,
            "history": [
                {
                    "date": v.date,
                    "week_offset": v.week_offset,
                    "blood": asdict(v.blood),
                    "clinical": asdict(v.clinical),
                    "noise_variants": [asdict(var) for var in v.noise_variants],
                    "events": list(v.events),
                }
                for v in self.history
            ],
        }

# =============================================================================
# 3. SIMULATION ENGINES (Logic Core)
# =============================================================================

class TimeEngine:
    """Generazione timeline visite con dropout."""
    @staticmethod
    def generate_schedule(
        start_year: int,
        n_visits_target: int,
        dropout_rate: float,
        min_days: int = 120,
        max_days: int = 180,
    ) -> List[datetime]:
        dates: List[datetime] = []
        current = datetime(start_year, 1, 1)
        for _ in range(n_visits_target):
            current += timedelta(days=random.randint(min_days, max_days))
            if random.random() > dropout_rate:
                dates.append(current)
        # garantiamo almeno 3 visite per avere senso temporale
        if len(dates) < 3:
            # aggiungi visite extra senza dropout
            while len(dates) < 3:
                current += timedelta(days=random.randint(min_days, max_days))
                dates.append(current)
        return dates

class BioEngine:
    """Generazione valori biologici con drift + rumore + missingness + spikes."""
    @staticmethod
    def apply_drift(baseline: float, visit_idx: int, drift_per_visit: float, noise_std: float) -> float:
        return float(baseline + (drift_per_visit * visit_idx) + np.random.normal(0, noise_std))

    @staticmethod
    def maybe_missing(value: float, miss_prob: float) -> Optional[float]:
        if random.random() < miss_prob:
            return None
        return value

    @staticmethod
    def clamp(value: float, min_v: float, max_v: float) -> float:
        return float(max(min_v, min(max_v, value)))

    @staticmethod
    def generate_ctdna_trajectory(
        n_visits: int,
        start_idx: int,
        detection_prob: float,
        genes_pool: Tuple[str, ...] = ("TP53", "KRAS", "EGFR"),
        vaf_template: Tuple[float, ...] = (0.0, 0.0, 0.05, 0.1, 0.3, 0.8, 2.5, 5.0),
        tech_noise_std: float = 0.04,
    ) -> List[List[Variant]]:
        traj: List[List[Variant]] = []
        for i in range(n_visits):
            variants: List[Variant] = []
            if i >= start_idx:
                vaf_theoretical = vaf_template[i] if i < len(vaf_template) else vaf_template[-1] * 1.5

                # shedding intermittente + fallimento tecnico
                if vaf_theoretical > 0 and random.random() < detection_prob:
                    vaf_obs = max(0.01, float(vaf_theoretical + np.random.normal(0, tech_noise_std)))
                    vaf_obs = round(vaf_obs, 3)

                    # gene principale
                    gene1 = random.choice(genes_pool)
                    variants.append(Variant(gene=gene1, vaf=vaf_obs))

                    # secondo clone a volte (più realistico)
                    if vaf_obs > 0.5 and random.random() < 0.45:
                        gene2 = random.choice(tuple(g for g in genes_pool if g != gene1))
                        variants.append(Variant(gene=gene2, vaf=round(vaf_obs * np.random.uniform(0.3, 0.75), 3)))
            traj.append(variants)
        return traj

class EventEngine:
    """Eventi clinici confondenti: infezione, infiammazione, tossicità."""
    @staticmethod
    def sample_spike_events(
        n_visits: int,
        spike_prob: float,
        max_spikes: int = 2,
    ) -> Dict[int, List[str]]:
        # mappa: index_visita -> eventi
        events: Dict[int, List[str]] = {}
        if random.random() > spike_prob:
            return events

        # scegli 1-2 visite random (non tutte)
        k = 1 if random.random() < 0.7 else 2
        k = min(k, max_spikes, n_visits)
        idxs = sorted(random.sample(range(n_visits), k=k))

        for idx in idxs:
            # mix eventi
            ev = []
            if random.random() < 0.7:
                ev.append("infection_spike")
            if random.random() < 0.4:
                ev.append("inflammation_spike")
            if random.random() < 0.25:
                ev.append("toxicity_spike")
            if not ev:
                ev = ["infection_spike"]
            events[idx] = ev
        return events

    @staticmethod
    def apply_spike_to_markers(
        ldh: float,
        crp: float,
        neut: float,
        events: List[str],
    ) -> Tuple[float, float, float]:
        ldh2, crp2, neut2 = ldh, crp, neut

        if "infection_spike" in events:
            crp2 += np.random.uniform(6, 22)
            neut2 += np.random.uniform(1500, 6000)
            ldh2 += np.random.uniform(20, 120)

        if "inflammation_spike" in events:
            crp2 += np.random.uniform(3, 15)
            ldh2 += np.random.uniform(10, 80)

        if "toxicity_spike" in events:
            ldh2 += np.random.uniform(30, 150)
            # neutropenia a volte
            if random.random() < 0.35:
                neut2 -= np.random.uniform(1200, 3500)

        return float(ldh2), float(crp2), float(neut2)

# =============================================================================
# 4. PHENOTYPE DEFINITIONS
# =============================================================================

@dataclass
class PhenotypeConfig:
    name: str
    truth_label: str
    count_ratio: float

    # baseline distributions (mean, std)
    base_glucose: Tuple[float, float]
    base_weight: Tuple[float, float]
    base_ldh: Tuple[float, float]

    # drift per visit
    drift_glucose: float
    drift_weight: float
    drift_ldh: float

    # typical noise std
    noise_glucose: float = 3.0
    noise_weight: float = 0.8
    noise_ldh: float = 12.0

    # missingness probabilities
    miss_glucose: float = 0.03
    miss_weight: float = 0.02
    miss_ldh: float = 0.03
    miss_crp: float = 0.06
    miss_neut: float = 0.05

    # ctDNA
    ctdna_active: bool = False
    ctdna_start_visit_idx: int = 999
    ctdna_detection_prob: float = 0.0

    # confounder spikes
    spike_prob: float = 0.10  # probabilità che il paziente abbia 1-2 spike eventi nella timeline

    # diagnosis timing (per Oracle lead time)
    diagnosis_year_range: Optional[Tuple[int, int]] = None  # es (2017, 2018)
    diagnosis_month_choices: Tuple[int, ...] = (3, 6, 9, 12)

PHENOTYPES: List[PhenotypeConfig] = [
    PhenotypeConfig(
        name="HEALTHY_CONTROL",
        truth_label="HEALTHY",
        count_ratio=0.40,
        base_glucose=(85, 5), base_weight=(75, 8), base_ldh=(160, 15),
        drift_glucose=0.0, drift_weight=0.0, drift_ldh=0.0,
        ctdna_active=False,
        spike_prob=0.07,
        # più missingness realistica
        miss_crp=0.12, miss_neut=0.10
    ),
    PhenotypeConfig(
        name="PANCREATIC_METABOLIC",
        truth_label="PANCREATIC_CANCER",
        count_ratio=0.20,
        base_glucose=(88, 4), base_weight=(82, 7), base_ldh=(170, 10),
        drift_glucose=2.6,  # drift metabolico
        drift_weight=-1.4,
        drift_ldh=4.0,      # lieve salita/infiammazione subclinica
        ctdna_active=False,
        spike_prob=0.18,
        diagnosis_year_range=(2017, 2018),
    ),
    PhenotypeConfig(
        name="PREDIABETES_CONFOUNDER",
        truth_label="NO_CANCER_PREDIABETES",
        count_ratio=0.15,
        base_glucose=(92, 4), base_weight=(82, 7), base_ldh=(165, 12),
        drift_glucose=2.2,
        drift_weight=0.2,   # peso stabile/lievemente su
        drift_ldh=1.0,      # piccola variabilità
        ctdna_active=False,
        spike_prob=0.12,
        diagnosis_year_range=None,
    ),
    PhenotypeConfig(
        name="WEIGHTLOSS_DIET_CONFOUNDER",
        truth_label="NO_CANCER_WEIGHTLOSS",
        count_ratio=0.10,
        base_glucose=(88, 3), base_weight=(90, 5), base_ldh=(160, 15),
        drift_glucose=-0.6,  # migliora col tempo (dieta)
        drift_weight=-1.6,   # perde peso ma NON è cancro
        drift_ldh=0.5,
        ctdna_active=False,
        spike_prob=0.22,     # stress/infezioni più comuni
        diagnosis_year_range=None,
    ),
    PhenotypeConfig(
        name="GHOST_RELAPSE_LUNG",
        truth_label="MOLECULAR_RELAPSE",
        count_ratio=0.15,
        base_glucose=(90, 5), base_weight=(72, 5), base_ldh=(180, 10),
        drift_glucose=0.0,
        drift_weight=-0.6,   # cachexia leggera
        drift_ldh=10.0,      # tende a salire verso la fine
        ctdna_active=True,
        ctdna_start_visit_idx=2,
        ctdna_detection_prob=0.75,
        spike_prob=0.18,
        diagnosis_year_range=(2016, 2017),
    ),
]

# =============================================================================
# 5. COHORT GENERATOR
# =============================================================================

class CohortGenerator:
    def __init__(self, output_file: str = OUTPUT_FILENAME):
        self.output_file = output_file
        self.patients: List[Patient] = []

    def generate(self, total_n: int, n_visits_target: int = 6, dropout_rate: float = 0.10):
        logger.info(f"🚀 Starting generation for {total_n} patients...")
        allocations = self._allocate_counts(total_n, PHENOTYPES)
        patient_counter = 0

        for cfg in PHENOTYPES:
            n_subgroup = allocations[cfg.name]
            logger.info(f"   generating {n_subgroup} cases for phenotype: {cfg.name}")
            for _ in range(n_subgroup):
                patient_counter += 1
                self.patients.append(
                    self._create_patient(
                        idx=patient_counter,
                        cfg=cfg,
                        n_visits_target=n_visits_target,
                        dropout_rate=dropout_rate,
                    )
                )

        random.shuffle(self.patients)
        logger.info("✅ Generation complete.")

    def _allocate_counts(self, total_n: int, configs: List[PhenotypeConfig]) -> Dict[str, int]:
        # allocazione robusta: floor + distribuzione resti
        raw = [(cfg.name, total_n * cfg.count_ratio) for cfg in configs]
        base = {name: int(np.floor(val)) for name, val in raw}
        assigned = sum(base.values())
        remaining = total_n - assigned

        # distribuisci rimanenti ai gruppi con frazione più alta
        fracs = sorted(
            [(name, (total_n * cfg.count_ratio) - base[name]) for name, cfg in [(c.name, c) for c in configs]],
            key=lambda x: x[1],
            reverse=True,
        )

        for i in range(remaining):
            base[fracs[i % len(fracs)][0]] += 1

        # sanity
        assert sum(base.values()) == total_n, "Allocation error: total mismatch"
        return base

    def _create_patient(self, idx: int, cfg: PhenotypeConfig, n_visits_target: int, dropout_rate: float) -> Patient:
        sex = random.choice(["M", "F"])
        age = random.randint(30, 80)

        # baseline individuale
        p_base_glu = float(np.random.normal(*cfg.base_glucose))
        p_base_wgt = float(np.random.normal(*cfg.base_weight))
        p_base_ldh = float(np.random.normal(*cfg.base_ldh))

        # timeline
        dates = TimeEngine.generate_schedule(SIMULATION_START_YEAR, n_visits_target, dropout_rate)
        dates = sorted(dates)
        n_visits = len(dates)

        # week_offset realistico
        t0 = dates[0]

        # ctDNA track (se attivo)
        ctdna_track: List[List[Variant]] = []
        if cfg.ctdna_active:
            ctdna_track = BioEngine.generate_ctdna_trajectory(
                n_visits=n_visits,
                start_idx=cfg.ctdna_start_visit_idx,
                detection_prob=cfg.ctdna_detection_prob,
            )
        else:
            ctdna_track = [[] for _ in range(n_visits)]

        # eventi confondenti
        spike_map = EventEngine.sample_spike_events(n_visits, spike_prob=cfg.spike_prob)

        # diagnosis metadata
        diagnosis_date, diagnosis_window = self._diagnosis_metadata(cfg, dates)

        visits: List[Visit] = []

        for i, d in enumerate(dates):
            # drift + noise
            glu = BioEngine.apply_drift(p_base_glu, i, cfg.drift_glucose, cfg.noise_glucose)
            wgt = BioEngine.apply_drift(p_base_wgt, i, cfg.drift_weight, cfg.noise_weight)
            ldh = BioEngine.apply_drift(p_base_ldh, i, cfg.drift_ldh, cfg.noise_ldh)

            # clamp realistico
            glu = BioEngine.clamp(glu, 55, 260)
            wgt = BioEngine.clamp(wgt, 40, 150)
            ldh = BioEngine.clamp(ldh, 80, 1400)

            # marker secondari baseline
            crp = float(abs(np.random.normal(1.2, 0.8)))
            neut = float(np.random.normal(5000, 1100))

            # apply spikes
            ev = spike_map.get(i, [])
            if ev:
                ldh, crp, neut = EventEngine.apply_spike_to_markers(ldh, crp, neut, ev)

            # clamp after spikes
            crp = BioEngine.clamp(crp, 0.0, 80.0)
            neut = BioEngine.clamp(neut, 800, 25000)

            # missingness per marker
            glu_o = BioEngine.maybe_missing(round(glu, 1), cfg.miss_glucose)
            wgt_o = BioEngine.maybe_missing(round(wgt, 1), cfg.miss_weight)
            ldh_o = BioEngine.maybe_missing(round(ldh, 1), cfg.miss_ldh)
            crp_o = BioEngine.maybe_missing(round(crp, 1), cfg.miss_crp)
            neut_o = BioEngine.maybe_missing(round(neut, 0), cfg.miss_neut)

            week_offset = int((d - t0).days // 7)

            variants = ctdna_track[i] if i < len(ctdna_track) else []

            visits.append(
                Visit(
                    date=d.strftime(DATE_FMT),
                    week_offset=week_offset,
                    blood=BloodMarkers(
                        glucose=glu_o,
                        ldh=ldh_o,
                        crp=crp_o,
                        neutrophils=neut_o,
                    ),
                    clinical=ClinicalMetrics(weight=wgt_o),
                    noise_variants=variants,
                    events=list(ev),
                )
            )

        pid = f"{cfg.name}_{idx:03d}"

        notes = {
            "seed": SEED,
            "n_visits": n_visits,
            "dropout_rate": dropout_rate,
            "spike_visits": {str(k): v for k, v in spike_map.items()} if spike_map else {},
        }

        return Patient(
            id=pid,
            sex=sex,
            age_baseline=age,
            phenotype_label=cfg.name,
            ground_truth=cfg.truth_label,
            history=visits,
            diagnosis_date=diagnosis_date,
            diagnosis_window=diagnosis_window,
            notes=notes,
        )

    def _diagnosis_metadata(self, cfg: PhenotypeConfig, dates: List[datetime]) -> Tuple[Optional[str], Optional[str]]:
        if not cfg.diagnosis_year_range:
            return None, None

        y1, y2 = cfg.diagnosis_year_range
        year = random.randint(y1, y2)
        month = random.choice(cfg.diagnosis_month_choices)
        day = 1
        diag = datetime(year, month, day)

        # finestra diagnosi: utile per valutare “lead time” (quando doveva accendersi Oracle)
        # esempio: finestre 6-18 mesi prima della diagnosi per early warning
        start_window = diag - timedelta(days=random.randint(180, 540))
        end_window = diag - timedelta(days=random.randint(60, 180))
        if start_window > end_window:
            start_window, end_window = end_window, start_window

        return diag.strftime(DATE_FMT), f"{start_window.strftime(DATE_FMT)}..{end_window.strftime(DATE_FMT)}"

    def save(self):
        path = Path(self.output_file)
        data = [p.to_dict() for p in self.patients]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Dataset saved to: {path.absolute()}")
        logger.info(f"📊 Total patients: {len(data)}")

# =============================================================================
# 6. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    generator = CohortGenerator(output_file=OUTPUT_FILENAME)
    generator.generate(total_n=150, n_visits_target=6, dropout_rate=0.10)
    generator.save()

    print("✅ Generated dataset with:")
    print("   - Realistic week_offset from dates")
    print("   - diagnosis_date + diagnosis_window for cancer phenotypes")
    print("   - missingness per marker (labs can be None)")
    print("   - infection/toxicity spikes (CRP/LDH/Neut confounding)")
    print("   - confounders (prediabetes / diet weight loss)")
    print("   - ctDNA intermittent shedding for relapse group")
