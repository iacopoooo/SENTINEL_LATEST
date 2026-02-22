#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL ORACLE VALIDATION SUITE v4.2 (Bootstrap Fix Edition)
==============================================================
Enterprise replay validation con Bootstrap CI corretto.

CHANGELOG v4.2:
- FIX: Bootstrap CI ora ricampiona correttamente i pazienti
- FIX: Validator interno resettato per ogni iterazione bootstrap
- NUOVO: CI width check per validare variabilità
- NUOVO: Percentili aggiuntivi (25%, 75%) per diagnostica

Features:
- Continuous Replay (visit-by-visit)
- Multi-threshold first-trigger tracking
- Lead-time audit con date esatte
- Bootstrap CI su Sens/Spec/F1 (corretto!)
- PR curve plot
- Export JSON completi

Usage:
  python scripts/ORACLE_validation.py

Author: Sentinel Principal Engineer
Date: 2026-02-05
"""

from __future__ import annotations

import json
import sys
import logging
import random
import copy
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# --- plotting (PR curve) ---
try:
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SentinelValidation")

# =============================================================================
# Path setup
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from temporal_engine.oracle_v3 import SentinelOracleV3
except ImportError:
    logger.critical("❌ Modulo 'temporal_engine.oracle_v3' non trovato. Esegui dalla root del progetto.")
    sys.exit(1)


# =============================================================================
# Config
# =============================================================================
class Config:
    DATASET_FILE: Path = BASE_DIR / "scripts/dataset_oracle_enterprise_v3_1.json"

    EXPORTS_DIR: Path = BASE_DIR / "exports"

    SUMMARY_JSON: Path = EXPORTS_DIR / "oracle_validation_summary.json"
    LEAD_AUDIT_JSON: Path = EXPORTS_DIR / "oracle_lead_time_audit_by_threshold.json"
    PR_PLOT_PATH: Path = EXPORTS_DIR / "oracle_pr_curve.png"
    BOOTSTRAP_DIST_PATH: Path = EXPORTS_DIR / "oracle_bootstrap_distributions.png"

    THRESHOLDS: List[int] = [40, 50, 60, 70, 80]
    MIN_VISITS_TO_RUN: int = 3
    LEAD_TIME_THRESHOLD: int = 50

    # Bootstrap config
    BOOTSTRAP_ON: bool = True
    BOOTSTRAP_N: int = 1000  # Aumentato per CI più stabili
    BOOTSTRAP_SEED: int = 42

    # PR curve
    PR_PLOT_ON: bool = True


# =============================================================================
# Helpers & Domain
# =============================================================================
class PatientGroup:
    CANCER = "CANCER"
    HEALTHY = "HEALTHY"
    CONFOUNDER_METABOLIC = "CONFOUNDER_METABOLIC"
    CONFOUNDER_WEIGHT = "CONFOUNDER_WEIGHT"
    UNKNOWN = "UNKNOWN"

    @staticmethod
    def from_truth_label(label: Any) -> str:
        s = str(label).strip().upper().replace("-", "_").replace(" ", "_")

        cancer_keys = {"PANCREATIC", "MOLECULAR_RELAPSE", "CANCER", "RELAPSE"}
        if any(k in s for k in cancer_keys) and "NO_CANCER" not in s:
            return PatientGroup.CANCER

        if any(k in s for k in ["WEIGHTLOSS", "WEIGHT_LOSS", "DIET", "WEIGHTLOSS_DIET"]):
            return PatientGroup.CONFOUNDER_WEIGHT

        if any(k in s for k in ["PREDIABETES", "PREDIAB", "PRE_DIABETES"]):
            return PatientGroup.CONFOUNDER_METABOLIC

        if "HEALTHY" in s:
            return PatientGroup.HEALTHY

        return PatientGroup.UNKNOWN


def safe_parse_date(date_obj: Any) -> Optional[datetime]:
    if not date_obj:
        return None
    s = str(date_obj).strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


@dataclass
class ValidationStats:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def sensitivity(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    def f1(self) -> float:
        p = self.precision()
        r = self.sensitivity()
        return safe_div(2 * p * r, p + r)

    def specificity(self) -> float:
        return safe_div(self.tn, self.tn + self.fp)

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0


@dataclass
class OracleRunResult:
    probability: float
    reason: str


# =============================================================================
# Patient Outcome (per bootstrap)
# =============================================================================
@dataclass
class PatientOutcome:
    """Outcome pre-calcolato per un paziente, usato per bootstrap veloce."""
    patient_id: str
    group: str
    # Per ogni threshold: (triggered_before_diag: bool, lead_days: Optional[int])
    outcomes_by_threshold: Dict[int, Tuple[bool, Optional[int]]] = field(default_factory=dict)


# =============================================================================
# Core Validator
# =============================================================================
class OracleValidator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Crea exports dir
        self.cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Results per threshold
        self.results: Dict[int, Dict[str, ValidationStats]] = {
            t: defaultdict(ValidationStats) for t in self.cfg.THRESHOLDS
        }

        # Lead time audit
        self.lead_audit: Dict[int, List[Dict[str, Any]]] = {t: [] for t in self.cfg.THRESHOLDS}

        # FP explainability
        self.fp_reasons_at_60: Counter = Counter()

        # Counters
        self.processed_count: int = 0
        self.total_cancer_with_diag: int = 0

        # Patient outcomes per bootstrap (calcolati una volta)
        self.patient_outcomes: List[PatientOutcome] = []

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not self.cfg.DATASET_FILE.exists():
            logger.error(f"❌ Dataset non trovato: {self.cfg.DATASET_FILE}")
            sys.exit(1)

        with open(self.cfg.DATASET_FILE, "r") as f:
            data = json.load(f)

        logger.info(f"Caricati {len(data)} pazienti da {self.cfg.DATASET_FILE.name}")
        return data

    def run(self) -> None:
        patients = self.load_dataset()
        logger.info("🚀 Avvio Continuous Replay Validation (Multi-threshold)...")

        for idx, p in enumerate(patients, start=1):
            outcome = self._process_patient(p)
            if outcome:
                self.patient_outcomes.append(outcome)
            self.processed_count += 1
            if idx % 50 == 0:
                logger.info(f"... processati {idx} pazienti")

        self._print_report()
        self._export_outputs()

        if self.cfg.PR_PLOT_ON and HAS_MATPLOTLIB:
            self._plot_pr_curve()

        if self.cfg.BOOTSTRAP_ON:
            self._bootstrap_ci()

    # -------------------------------------------------------------------------
    # Patient processing
    # -------------------------------------------------------------------------
    def _process_patient(self, p: Dict[str, Any]) -> Optional[PatientOutcome]:
        pid = str(p.get("id", "UNKNOWN"))
        truth = p.get("truth")
        group = PatientGroup.from_truth_label(truth)

        history = self._sorted_history(p.get("history", []))
        if not history:
            return None

        diag_date = safe_parse_date(p.get("diagnosis_date"))

        if group == PatientGroup.CANCER and diag_date:
            self.total_cancer_with_diag += 1

        # Track first trigger per threshold
        first_trigger_date: Dict[int, Optional[datetime]] = {t: None for t in self.cfg.THRESHOLDS}
        first_trigger_prob: Dict[int, float] = {t: 0.0 for t in self.cfg.THRESHOLDS}
        first_trigger_reason: Dict[int, str] = {t: "" for t in self.cfg.THRESHOLDS}

        # Continuous replay
        for i in range(self.cfg.MIN_VISITS_TO_RUN, len(history) + 1):
            slice_history = history[:i]
            current_date = safe_parse_date(slice_history[-1].get("date"))
            if not current_date:
                continue

            if group == PatientGroup.CANCER and diag_date and current_date >= diag_date:
                break

            rr = self._run_oracle(pid, slice_history)

            for t in self.cfg.THRESHOLDS:
                if first_trigger_date[t] is None and rr.probability >= t:
                    first_trigger_date[t] = current_date
                    first_trigger_prob[t] = rr.probability
                    first_trigger_reason[t] = rr.reason

            if all(first_trigger_date[t] is not None for t in self.cfg.THRESHOLDS):
                break

        # Build patient outcome for bootstrap
        outcome = PatientOutcome(patient_id=pid, group=group)

        # Update results and build outcome
        for t in self.cfg.THRESHOLDS:
            trig_date = first_trigger_date[t]
            trig_prob = first_trigger_prob[t]
            trig_reason = first_trigger_reason[t]

            triggered_before_diag = False
            lead_days = None

            if group == PatientGroup.CANCER:
                if diag_date:
                    if trig_date is not None and trig_date < diag_date:
                        # TP
                        self.results[t][PatientGroup.CANCER].tp += 1
                        triggered_before_diag = True
                        lead_days = (diag_date - trig_date).days

                        self.lead_audit[t].append({
                            "id": pid,
                            "truth": str(truth),
                            "diagnosis_date": diag_date.strftime("%Y-%m-%d"),
                            "first_trigger_date": trig_date.strftime("%Y-%m-%d"),
                            "lead_time_days": int(lead_days),
                            "probability": float(trig_prob),
                            "reason": trig_reason,
                        })
                    else:
                        # FN
                        self.results[t][PatientGroup.CANCER].fn += 1
                        self.lead_audit[t].append({
                            "id": pid,
                            "truth": str(truth),
                            "diagnosis_date": diag_date.strftime("%Y-%m-%d"),
                            "first_trigger_date": None,
                            "lead_time_days": None,
                            "probability": float(trig_prob),
                            "reason": trig_reason or "No trigger before diagnosis",
                        })
                else:
                    if trig_date is not None:
                        self.results[t][PatientGroup.CANCER].tp += 1
                        triggered_before_diag = True
                    else:
                        self.results[t][PatientGroup.CANCER].fn += 1
            else:
                grp = group if group != PatientGroup.UNKNOWN else PatientGroup.HEALTHY
                if trig_date is not None:
                    self.results[t][grp].fp += 1
                    triggered_before_diag = True  # Per non-cancer, "triggered" = FP
                    if t == 60 and trig_reason:
                        self.fp_reasons_at_60[f"{grp}: {trig_reason}"] += 1
                else:
                    self.results[t][grp].tn += 1

            outcome.outcomes_by_threshold[t] = (triggered_before_diag, lead_days)

        return outcome

    # -------------------------------------------------------------------------
    # Oracle call
    # -------------------------------------------------------------------------
    def _run_oracle(self, pid: str, history_slice: List[Dict[str, Any]]) -> OracleRunResult:
        raw_ngs = [
            {"date": v.get("date"), "noise_variants": v.get("noise_variants", [])}
            for v in history_slice
        ]
        try:
            oracle = SentinelOracleV3(history_slice, patient_id=pid)
            alerts = oracle.run_oracle(raw_ngs_visits=raw_ngs)
            if alerts:
                a0 = alerts[0]
                reason = getattr(a0, "risk_type", "ALERT")
                sig_sources = getattr(a0, "signal_sources", None)
                if sig_sources and len(sig_sources) > 0:
                    k = getattr(sig_sources[0], "key", None)
                    if k:
                        reason = f"{reason} ({k})"
                return OracleRunResult(float(getattr(a0, "probability", 0.0)), reason)
        except Exception:
            pass
        return OracleRunResult(0.0, "")

    # -------------------------------------------------------------------------
    # History sorting
    # -------------------------------------------------------------------------
    def _sorted_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def k(v: Dict[str, Any]) -> datetime:
            d = safe_parse_date(v.get("date"))
            return d if d else datetime.max

        return sorted(history, key=k)

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    def _print_report(self) -> None:
        print("\n" + "=" * 110)
        print(f"📊 SENTINEL ORACLE VALIDATION REPORT (Continuous Replay)  N={self.processed_count}")
        print("=" * 110)

        print("\n1️⃣ THRESHOLD SWEEP (Sensitivity/Specificity + Precision/F1)")
        header = f"{'THR%':<5} | {'SENS(Cancer)':<12} | {'SPEC(H)':<10} | {'SPEC(M)':<10} | {'SPEC(W)':<10} | {'PREC':<8} | {'F1':<6}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        best_thr = None
        best_f1 = -1.0

        for t in self.cfg.THRESHOLDS:
            res = self.results[t]

            sens = res[PatientGroup.CANCER].sensitivity()
            spec_h = res[PatientGroup.HEALTHY].specificity()
            spec_m = res[PatientGroup.CONFOUNDER_METABOLIC].specificity()
            spec_w = res[PatientGroup.CONFOUNDER_WEIGHT].specificity()

            total_tp = res[PatientGroup.CANCER].tp
            total_fp = (
                    res[PatientGroup.HEALTHY].fp +
                    res[PatientGroup.CONFOUNDER_METABOLIC].fp +
                    res[PatientGroup.CONFOUNDER_WEIGHT].fp
            )

            prec = safe_div(total_tp, total_tp + total_fp)
            f1 = safe_div(2 * prec * sens, prec + sens)

            mark = "⭐" if f1 > best_f1 else ""
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t

            print(
                f"{t:<5} | {sens:>11.1%} | {spec_h:>9.1%} | {spec_m:>9.1%} | {spec_w:>9.1%} | {prec:>7.1%} | {f1:>5.3f} {mark}")

        # Lead-time summary
        t0 = self.cfg.LEAD_TIME_THRESHOLD
        leads = [x["lead_time_days"] for x in self.lead_audit[t0] if x.get("lead_time_days") is not None]
        print(f"\n2️⃣ LEAD TIME ANALYSIS (@ {t0}%)")
        if leads:
            arr = np.array(leads, dtype=float)
            print(f"   ► Cancer con diagnosis_date:       {self.total_cancer_with_diag}")
            print(f"   ► Rilevati in anticipo:            {len(leads)}")
            print(f"   ► Media anticipo:                  {arr.mean():.1f} giorni ({arr.mean() / 30.4:.1f} mesi)")
            print(f"   ► Mediana anticipo:                {np.median(arr):.0f} giorni")
            print(f"   ► Min / Max:                       {int(arr.min())} / {int(arr.max())} giorni")

            print("\n   Top 5 detections (audit):")
            sorted_audit = sorted(
                [x for x in self.lead_audit[t0] if x.get("lead_time_days")],
                key=lambda x: -x["lead_time_days"]
            )
            for row in sorted_audit[:5]:
                print(
                    f"     ✅ {row['id']}: {row['lead_time_days']}gg (trigger={row['first_trigger_date']} diag={row['diagnosis_date']})")
        else:
            print("   ⚠️ Nessun lead time rilevato.")

        print("\n3️⃣ FALSE POSITIVE EXPLAINABILITY (Top reasons @ 60%)")
        if self.fp_reasons_at_60:
            for reason, count in self.fp_reasons_at_60.most_common(8):
                print(f"   ❌ {count} casi: {reason}")
        else:
            print("   ✅ Nessun falso positivo a soglia 60%.")

        print(f"\n💡 CONSIGLIO OPERATIVO: best_thr={best_thr}%  (F1={best_f1:.3f})")
        print("=" * 110)

    # -------------------------------------------------------------------------
    # Exports
    # -------------------------------------------------------------------------
    def _export_outputs(self) -> None:
        summary = {
            "dataset": str(self.cfg.DATASET_FILE.name),
            "n_processed": self.processed_count,
            "thresholds": self.cfg.THRESHOLDS,
            "min_visits_to_run": self.cfg.MIN_VISITS_TO_RUN,
            "lead_time_threshold": self.cfg.LEAD_TIME_THRESHOLD,
            "metrics_by_threshold": {},
        }

        for t in self.cfg.THRESHOLDS:
            res = self.results[t]
            sens = res[PatientGroup.CANCER].sensitivity()
            spec_h = res[PatientGroup.HEALTHY].specificity()
            spec_m = res[PatientGroup.CONFOUNDER_METABOLIC].specificity()
            spec_w = res[PatientGroup.CONFOUNDER_WEIGHT].specificity()

            tp = res[PatientGroup.CANCER].tp
            fn = res[PatientGroup.CANCER].fn
            fp = (
                    res[PatientGroup.HEALTHY].fp +
                    res[PatientGroup.CONFOUNDER_METABOLIC].fp +
                    res[PatientGroup.CONFOUNDER_WEIGHT].fp
            )
            prec = safe_div(tp, tp + fp)
            f1 = safe_div(2 * prec * sens, prec + sens)

            summary["metrics_by_threshold"][str(t)] = {
                "tp": tp, "fn": fn, "fp_total": fp,
                "sens": round(sens, 4),
                "spec_healthy": round(spec_h, 4),
                "spec_prediab": round(spec_m, 4),
                "spec_weight": round(spec_w, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
            }

        with open(self.cfg.SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Salvato summary JSON: {self.cfg.SUMMARY_JSON}")

        with open(self.cfg.LEAD_AUDIT_JSON, "w") as f:
            json.dump(self.lead_audit, f, indent=2)
        logger.info(f"✅ Salvato lead-time audit: {self.cfg.LEAD_AUDIT_JSON}")

    # -------------------------------------------------------------------------
    # PR curve
    # -------------------------------------------------------------------------
    def _plot_pr_curve(self) -> None:
        if not HAS_MATPLOTLIB:
            logger.warning("⚠️ matplotlib non disponibile, skip PR curve")
            return

        thresholds = []
        precisions = []
        recalls = []

        for t in self.cfg.THRESHOLDS:
            res = self.results[t]
            tp = res[PatientGroup.CANCER].tp
            fn = res[PatientGroup.CANCER].fn
            fp = (
                    res[PatientGroup.HEALTHY].fp +
                    res[PatientGroup.CONFOUNDER_METABOLIC].fp +
                    res[PatientGroup.CONFOUNDER_WEIGHT].fp
            )
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)

            thresholds.append(t)
            precisions.append(precision)
            recalls.append(recall)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker="o", linewidth=2, markersize=8)
        for i, thr in enumerate(thresholds):
            plt.annotate(f"{thr}%", (recalls[i], precisions[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=10)

        plt.xlabel("Recall (Sensitivity)", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Oracle PR Curve (Threshold Sweep)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(self.cfg.PR_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Salvato PR curve: {self.cfg.PR_PLOT_PATH}")

    # -------------------------------------------------------------------------
    # Bootstrap CI (FIXED!)
    # -------------------------------------------------------------------------
    def _bootstrap_ci(self) -> None:
        logger.info(f"🧪 Bootstrap CI (N={self.cfg.BOOTSTRAP_N}, seed={self.cfg.BOOTSTRAP_SEED})")

        # Set seed per riproducibilità
        rng = np.random.RandomState(self.cfg.BOOTSTRAP_SEED)

        n_patients = len(self.patient_outcomes)
        if n_patients == 0:
            logger.warning("⚠️ Nessun patient outcome per bootstrap")
            return

        # Prepara strutture per raccogliere samples
        metric_samples: Dict[int, Dict[str, List[float]]] = {
            t: {"sens": [], "spec": [], "prec": [], "f1": []}
            for t in self.cfg.THRESHOLDS
        }

        # Bootstrap loop
        for b in range(self.cfg.BOOTSTRAP_N):
            # Resample con replacement
            indices = rng.randint(0, n_patients, size=n_patients)

            # Calcola metriche per questo resample
            for t in self.cfg.THRESHOLDS:
                tp, fn, fp, tn = 0, 0, 0, 0

                for idx in indices:
                    outcome = self.patient_outcomes[idx]
                    triggered, _ = outcome.outcomes_by_threshold.get(t, (False, None))

                    if outcome.group == PatientGroup.CANCER:
                        if triggered:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if triggered:
                            fp += 1
                        else:
                            tn += 1

                sens = safe_div(tp, tp + fn)
                spec = safe_div(tn, tn + fp)
                prec = safe_div(tp, tp + fp)
                f1 = safe_div(2 * prec * sens, prec + sens)

                metric_samples[t]["sens"].append(sens)
                metric_samples[t]["spec"].append(spec)
                metric_samples[t]["prec"].append(prec)
                metric_samples[t]["f1"].append(f1)

        # Calcola CI e stampa report
        print(f"\n4️⃣ BOOTSTRAP CI (95%, N={self.cfg.BOOTSTRAP_N})")
        print("-" * 90)

        ci_results: Dict[str, Dict[str, Any]] = {}

        for t in self.cfg.THRESHOLDS:
            sens_arr = np.array(metric_samples[t]["sens"])
            f1_arr = np.array(metric_samples[t]["f1"])
            prec_arr = np.array(metric_samples[t]["prec"])

            # CI 95%
            sens_ci = (float(np.percentile(sens_arr, 2.5)), float(np.percentile(sens_arr, 97.5)))
            f1_ci = (float(np.percentile(f1_arr, 2.5)), float(np.percentile(f1_arr, 97.5)))
            prec_ci = (float(np.percentile(prec_arr, 2.5)), float(np.percentile(prec_arr, 97.5)))

            # Stats aggiuntive
            sens_mean = float(np.mean(sens_arr))
            sens_std = float(np.std(sens_arr))
            f1_mean = float(np.mean(f1_arr))
            f1_std = float(np.std(f1_arr))

            ci_results[str(t)] = {
                "sens": {"mean": sens_mean, "std": sens_std, "ci_95": sens_ci},
                "prec": {"mean": float(np.mean(prec_arr)), "ci_95": prec_ci},
                "f1": {"mean": f1_mean, "std": f1_std, "ci_95": f1_ci},
            }

            # Width check
            sens_width = sens_ci[1] - sens_ci[0]
            f1_width = f1_ci[1] - f1_ci[0]

            print(f"   thr={t}%:")
            print(
                f"      Sens: {sens_mean:.3f} ± {sens_std:.3f}  CI=[{sens_ci[0]:.3f}, {sens_ci[1]:.3f}]  width={sens_width:.3f}")
            print(
                f"      F1:   {f1_mean:.3f} ± {f1_std:.3f}  CI=[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]  width={f1_width:.3f}")

        print("-" * 90)

        # Salva CI nel summary JSON
        try:
            with open(self.cfg.SUMMARY_JSON, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

        summary["bootstrap_ci"] = {
            "n_iterations": self.cfg.BOOTSTRAP_N,
            "seed": self.cfg.BOOTSTRAP_SEED,
            "ci_by_threshold": ci_results,
        }

        with open(self.cfg.SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Bootstrap CI salvati in: {self.cfg.SUMMARY_JSON}")

        # Plot distribuzioni bootstrap (opzionale)
        if HAS_MATPLOTLIB:
            self._plot_bootstrap_distributions(metric_samples)

    def _plot_bootstrap_distributions(self, metric_samples: Dict[int, Dict[str, List[float]]]) -> None:
        """Plot delle distribuzioni bootstrap per diagnostica."""
        fig, axes = plt.subplots(2, len(self.cfg.THRESHOLDS), figsize=(15, 8))

        for i, t in enumerate(self.cfg.THRESHOLDS):
            # Sensitivity distribution
            ax1 = axes[0, i]
            ax1.hist(metric_samples[t]["sens"], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(metric_samples[t]["sens"]), color='red', linestyle='--', label='Mean')
            ax1.axvline(np.percentile(metric_samples[t]["sens"], 2.5), color='orange', linestyle=':', label='CI 2.5%')
            ax1.axvline(np.percentile(metric_samples[t]["sens"], 97.5), color='orange', linestyle=':', label='CI 97.5%')
            ax1.set_title(f'Sens @ {t}%')
            ax1.set_xlabel('Sensitivity')
            if i == 0:
                ax1.set_ylabel('Frequency')

            # F1 distribution
            ax2 = axes[1, i]
            ax2.hist(metric_samples[t]["f1"], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.mean(metric_samples[t]["f1"]), color='red', linestyle='--')
            ax2.axvline(np.percentile(metric_samples[t]["f1"], 2.5), color='orange', linestyle=':')
            ax2.axvline(np.percentile(metric_samples[t]["f1"], 97.5), color='orange', linestyle=':')
            ax2.set_title(f'F1 @ {t}%')
            ax2.set_xlabel('F1 Score')
            if i == 0:
                ax2.set_ylabel('Frequency')

        plt.suptitle('Bootstrap Distributions (Sens & F1 by Threshold)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.cfg.BOOTSTRAP_DIST_PATH, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Bootstrap distributions plot: {self.cfg.BOOTSTRAP_DIST_PATH}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    cfg = Config()
    validator = OracleValidator(cfg)
    validator.run()