#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL_ORACLE v3.2 (Optimized LR Edition)

CHANGELOG v3.2:
- TUNING: LR aumentati per clonal expansion (più aggressivo ma controllato)
- TUNING: Aggiunto bonus per pattern combinati (metabolic + molecular)
- TUNING: Soglie VAF più sensibili per early detection
- NUOVO: detect_combined_risk() per sinergia multi-segnale
- FIX: Gestione CHIP più sofisticata (età-dipendente se disponibile)

Target: 70-75% sensibilità con specificità ~100%

⚠️ Clinical Decision Support - Non è diagnosi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats


# =============================================================================
# 0) CONFIG
# =============================================================================

class ConfidenceLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ConfidenceScore:
    level: ConfidenceLevel
    score: float
    reason: str


@dataclass
class Evidence:
    key: str
    weight_lr: float
    score: float
    details: str


@dataclass
class OracleAlert:
    risk_type: str
    probability: float
    confidence: ConfidenceScore
    lead_time: str
    signal_sources: List[Evidence]
    recommended_actions: List[str]
    summary: str
    debug: Dict[str, Any]


# =============================================================================
# PRIORS (Calibrati per screening population)
# =============================================================================
DEFAULT_PRIORS = {
    "EARLY_PANCREATIC_RISK": 0.03,  # 3% - screening population
    "INFLAMMATION_ONCO_SIGNATURE": 0.04,  # 4% - multi-marker
    "MOLECULAR_MICRO_RELAPSE": 0.06,  # 6% - ghost signals
    "CLONAL_EXPANSION": 0.10,  # 10% - quando c'è evidenza molecolare
}

# =============================================================================
# DRIFT CONFIG
# =============================================================================
DRIFT_MIN_POINTS = 5
DRIFT_MIN_SPAN_MONTHS = 18
R2_STRONG = 0.70
R2_MED = 0.55

# Pancreas drift (leggermente più sensibile)
PANCREAS_GLU_SLOPE_MIN = 0.30  # Era 0.35, ora più sensibile
PANCREAS_WT_SLOPE_MAX = -0.12  # Era -0.15, ora più sensibile
PANCREAS_R2_MIN = 0.60  # Era 0.65, ora più tollerante

# Marker drift generici
GENERIC_R2_MIN = 0.55  # Era 0.60
LDH_SLOPE_MIN = 2.5  # Era 3.0
CRP_SLOPE_MIN = 0.35  # Era 0.4
NLR_SLOPE_MIN = 0.06  # Era 0.08

# =============================================================================
# CLONAL EXPANSION CONFIG (OTTIMIZZATO v3.2)
# =============================================================================
CLONAL_VAF_MIN = 0.008  # 0.8% - leggermente sotto 1% per catturare più casi
CLONAL_VAF_SIGNIFICANT = 0.03  # 3% - chiaramente sopra rumore
CLONAL_VAF_HIGH = 0.10  # 10% - alto carico
CLONAL_VAF_VERY_HIGH = 0.25  # 25% - espansione massiva

CLONAL_MIN_VISITS = 2
CLONAL_MIN_GENES = 1

# Driver genes (oncologici)
HIGH_RISK_GENES = {
    "TP53", "KRAS", "EGFR", "BRAF", "PIK3CA", "NRAS", "HRAS",
    "ALK", "ROS1", "MET", "RET", "ERBB2", "HER2",
    "STK11", "KEAP1", "PTEN", "RB1", "NF1",
    "CDKN2A", "SMAD4", "BRCA1", "BRCA2", "APC",
    "FGFR1", "FGFR2", "FGFR3", "IDH1", "IDH2",
}

# CHIP genes (clonal hematopoiesis - da pesare con cautela)
CHIP_GENES = {"DNMT3A", "TET2", "ASXL1", "JAK2", "SF3B1", "SRSF2", "PPM1D"}

# Ghost detection (sub-clinical)
GHOST_VAF_MAX = 0.008  # Sotto 0.8%
GHOST_MIN_VISITS = 3
GHOST_MIN_MONTHS = 3
GHOST_MIN_GENES = 1


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =============================================================================
# 1) UTILITIES
# =============================================================================

def parse_time_to_month_index(date_like: Any) -> Optional[float]:
    if date_like is None:
        return None
    if isinstance(date_like, (int, float)) and np.isfinite(date_like):
        return float(date_like)

    if isinstance(date_like, str):
        s = date_like.strip()
        try:
            parts = s.split("-")
            y = int(parts[0])
            m = int(parts[1])
            return y * 12 + (m - 1)
        except Exception:
            return None

    if isinstance(date_like, dict):
        try:
            y = int(date_like.get("year"))
            m = int(date_like.get("month"))
            return y * 12 + (m - 1)
        except Exception:
            return None

    return None


@dataclass
class TrendResult:
    slope: float
    intercept: float
    r2: float
    p_value: float
    n: int
    span_months: float
    last_value: float
    first_value: float
    delta: float
    outlier_last: bool
    residual_std: float


def robust_trend(x: np.ndarray, y: np.ndarray) -> Optional[TrendResult]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None

    if np.all(x == x[0]):
        return TrendResult(
            slope=0.0, intercept=float(y.mean()), r2=0.0, p_value=1.0,
            n=int(len(x)), span_months=0.0, last_value=float(y[-1]),
            first_value=float(y[0]), delta=float(y[-1] - y[0]),
            outlier_last=False, residual_std=float(np.std(y)) if len(y) > 1 else 0.0,
        )

    slope, intercept, r, p, stderr = stats.linregress(x, y)
    y_hat = slope * x + intercept
    residuals = y - y_hat
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) >= 3 else float(np.std(residuals)) if len(
        residuals) else 0.0

    outlier_last = False
    if resid_std > 0 and len(residuals) >= 3:
        z = abs(float(residuals[-1])) / resid_std
        outlier_last = (z >= 3.0)

    span = float(x[-1] - x[0]) if len(x) >= 2 else 0.0

    return TrendResult(
        slope=float(slope), intercept=float(intercept), r2=float(r ** 2),
        p_value=float(p), n=int(len(x)), span_months=span,
        last_value=float(y[-1]), first_value=float(y[0]),
        delta=float(y[-1] - y[0]), outlier_last=outlier_last,
        residual_std=resid_std,
    )


def confidence_from_trend(tr: TrendResult, min_points: int, min_span_months: float) -> ConfidenceScore:
    score = 0.0
    reasons = []

    if tr.n >= min_points:
        score += 0.35
    else:
        reasons.append(f"n={tr.n}<{min_points}")

    if tr.span_months >= min_span_months:
        score += 0.25
    else:
        reasons.append(f"span={tr.span_months:.0f}m")

    if tr.r2 >= R2_STRONG:
        score += 0.30
    elif tr.r2 >= R2_MED:
        score += 0.18
    else:
        reasons.append(f"R²={tr.r2:.2f}")

    if tr.outlier_last:
        score -= 0.12
        reasons.append("outlier")

    score = _clamp(score, 0.0, 1.0)
    lvl = ConfidenceLevel.HIGH if score >= 0.72 else ConfidenceLevel.MEDIUM if score >= 0.45 else ConfidenceLevel.LOW
    reason = "Solid" if not reasons else "; ".join(reasons)
    return ConfidenceScore(lvl, round(score, 2), reason)


# =============================================================================
# 2) BAYES FUSION
# =============================================================================

def odds(p: float) -> float:
    p = _clamp(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


def prob_from_odds(o: float) -> float:
    return o / (1 + o)


def fuse_evidence(prior_p: float, evidences: List[Evidence]) -> Tuple[float, float]:
    o = odds(prior_p)
    lr_total = 1.0
    for ev in evidences:
        lr_eff = float(ev.weight_lr) ** float(_clamp(ev.score, 0.0, 1.0))
        lr_total *= lr_eff
    post_o = o * lr_total
    post_p = prob_from_odds(post_o)
    return float(post_p), float(lr_total)


# =============================================================================
# 3) ORACLE CORE
# =============================================================================

class SentinelOracleV3:

    def __init__(
            self,
            patient_history: List[Dict[str, Any]],
            priors: Optional[Dict[str, float]] = None,
            patient_id: str = "UNKNOWN",
            patient_age: Optional[int] = None,  # Per gestione CHIP
    ):
        self.patient_id = patient_id
        self.patient_age = patient_age
        self.priors = dict(DEFAULT_PRIORS)
        if priors:
            self.priors.update(priors)
        self.history = self._normalize_history(patient_history)

    def _normalize_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for v in history or []:
            t = parse_time_to_month_index(v.get("date") or v.get("time") or v.get("month"))
            if t is None:
                continue
            rows.append({
                "t": float(t),
                "blood": v.get("blood", {}) or v.get("blood_markers", {}) or {},
                "clinical": v.get("clinical", {}) or {}
            })
        rows.sort(key=lambda x: x["t"])
        if rows:
            t0 = rows[0]["t"]
            for r in rows:
                r["t"] = r["t"] - t0
        return rows

    def _series(self, path: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for r in self.history:
            val = None
            if path[0] == "blood":
                val = r["blood"].get(path[1])
            elif path[0] == "clinical":
                val = r["clinical"].get(path[1])
            if isinstance(val, (int, float)) and np.isfinite(val):
                xs.append(r["t"])
                ys.append(float(val))
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    # =========================================================================
    # A) Pancreatic / Metabolic Drift (OTTIMIZZATO)
    # =========================================================================
    def detect_pancreatic_metabolic_drift(self) -> Optional[OracleAlert]:
        xg, g = self._series(("blood", "glucose"))
        xw, w = self._series(("clinical", "weight"))

        if len(g) < DRIFT_MIN_POINTS or len(w) < DRIFT_MIN_POINTS:
            return None

        common = np.intersect1d(xg, xw)
        if len(common) < DRIFT_MIN_POINTS:
            return None

        g_map = {float(x): float(y) for x, y in zip(xg, g)}
        w_map = {float(x): float(y) for x, y in zip(xw, w)}
        x = np.array(sorted(common), dtype=float)
        g2 = np.array([g_map[float(xx)] for xx in x], dtype=float)
        w2 = np.array([w_map[float(xx)] for xx in x], dtype=float)

        tr_g = robust_trend(x, g2)
        tr_w = robust_trend(x, w2)
        if not tr_g or not tr_w:
            return None

        conf_g = confidence_from_trend(tr_g, DRIFT_MIN_POINTS, DRIFT_MIN_SPAN_MONTHS)
        conf_w = confidence_from_trend(tr_w, DRIFT_MIN_POINTS, DRIFT_MIN_SPAN_MONTHS)

        # Pattern check (soglie più sensibili)
        ok = (
                tr_g.slope >= PANCREAS_GLU_SLOPE_MIN
                and tr_w.slope <= PANCREAS_WT_SLOPE_MAX
                and tr_g.r2 >= PANCREAS_R2_MIN
        )

        if not ok:
            return None

        evs: List[Evidence] = []

        # LR AUMENTATI per drift metabolico
        g_intensity = _clamp((tr_g.slope / PANCREAS_GLU_SLOPE_MIN - 1.0) * 0.5 + conf_g.score * 0.5, 0.0, 1.0)
        w_intensity = _clamp((abs(tr_w.slope) / abs(PANCREAS_WT_SLOPE_MAX) - 1.0) * 0.5 + conf_w.score * 0.5, 0.0, 1.0)
        combo_intensity = _clamp((g_intensity + w_intensity) / 2.0 + 0.1, 0.0, 1.0)

        evs.append(Evidence(
            key="glucose_up_drift",
            weight_lr=8.0,  # Era 6.0
            score=g_intensity,
            details=f"Glucose +{tr_g.slope:.2f}/mo (R²={tr_g.r2:.2f})"
        ))
        evs.append(Evidence(
            key="weight_down_drift",
            weight_lr=8.0,  # Era 6.0
            score=w_intensity,
            details=f"Weight {tr_w.slope:.2f}kg/mo (R²={tr_w.r2:.2f})"
        ))
        evs.append(Evidence(
            key="divergent_drift",
            weight_lr=12.0,  # Era 8.0 - pattern combinato molto forte
            score=combo_intensity,
            details="Divergent: glucose↑ + weight↓"
        ))

        prior = self.priors["EARLY_PANCREATIC_RISK"]
        post_p, lr_total = fuse_evidence(prior, evs)

        base_conf = min(conf_g.score, conf_w.score)
        boost = _clamp(math.log10(max(lr_total, 1.0)) / 2.0, 0.0, 0.25)
        conf_score = _clamp(base_conf + boost, 0.0, 1.0)
        conf_lvl = ConfidenceLevel.HIGH if conf_score >= 0.70 else ConfidenceLevel.MEDIUM if conf_score >= 0.45 else ConfidenceLevel.LOW
        conf = ConfidenceScore(conf_lvl, round(conf_score, 2), f"{conf_g.reason}|{conf_w.reason}")

        return OracleAlert(
            risk_type="EARLY_PANCREATIC/METABOLIC",
            probability=round(post_p * 100.0, 1),
            confidence=conf,
            lead_time="18-36 months prior",
            signal_sources=evs,
            recommended_actions=[
                "Repeat glucose/HbA1c + weight in 6-12 weeks",
                "Consider pancreatic workup if persistent",
                "Rule out confounders (diet, steroids)",
            ],
            summary=f"Metabolic drift: glucose↑ + weight↓ over {tr_g.span_months:.0f}mo",
            debug={"patient_id": self.patient_id, "prior": prior, "lr_total": lr_total},
        )

    # =========================================================================
    # B) Inflammation-Oncology Signature
    # =========================================================================
    def detect_inflammation_onco_signature(self) -> Optional[OracleAlert]:
        xcrp, crp = self._series(("blood", "crp"))
        xldh, ldh = self._series(("blood", "ldh"))
        xn, neut = self._series(("blood", "neutrophils"))
        xl, lymph = self._series(("blood", "lymphocytes"))
        xal, alb = self._series(("blood", "albumin"))

        evs: List[Evidence] = []
        trends_debug: Dict[str, Any] = {}

        def add_trend_ev(name: str, x: np.ndarray, y: np.ndarray, slope_min: float, lr: float, unit: str):
            if len(y) < DRIFT_MIN_POINTS:
                return
            tr = robust_trend(x, y)
            if not tr:
                return
            conf = confidence_from_trend(tr, DRIFT_MIN_POINTS, 12)
            trends_debug[name] = {"slope": tr.slope, "r2": tr.r2}

            if tr.r2 < GENERIC_R2_MIN or conf.level == ConfidenceLevel.LOW:
                return
            if tr.slope >= slope_min:
                intensity = _clamp((tr.slope / slope_min - 1.0) * 0.5 + conf.score * 0.5, 0.0, 1.0)
                evs.append(Evidence(
                    key=f"{name}_drift",
                    weight_lr=lr,
                    score=intensity,
                    details=f"{name} +{tr.slope:.2f}{unit}/mo (R²={tr.r2:.2f})"
                ))

        add_trend_ev("crp", xcrp, crp, CRP_SLOPE_MIN, lr=5.0, unit="mg/L")
        add_trend_ev("ldh", xldh, ldh, LDH_SLOPE_MIN, lr=5.0, unit="U/L")

        # NLR
        common = np.intersect1d(xn, xl)
        if len(common) >= DRIFT_MIN_POINTS:
            n_map = {float(x): float(y) for x, y in zip(xn, neut)}
            l_map = {float(x): float(y) for x, y in zip(xl, lymph)}
            x = np.array(sorted(common), dtype=float)
            nlr = np.array([n_map[float(xx)] / max(l_map[float(xx)], 1.0) for xx in x], dtype=float)
            add_trend_ev("nlr", x, nlr, NLR_SLOPE_MIN, lr=4.5, unit="")

        # Albumin down
        if len(alb) >= DRIFT_MIN_POINTS:
            tr_alb = robust_trend(xal, alb)
            if tr_alb and tr_alb.r2 >= GENERIC_R2_MIN and tr_alb.slope <= -0.04:
                intensity = _clamp((abs(tr_alb.slope) / 0.04 - 1.0) * 0.5 + 0.5, 0.0, 1.0)
                evs.append(Evidence(
                    key="albumin_down",
                    weight_lr=4.0,
                    score=intensity,
                    details=f"Albumin {tr_alb.slope:.3f}g/dL/mo"
                ))

        if len(evs) < 2:
            return None

        # Composite evidence
        composite = _clamp(np.mean([e.score for e in evs]) + 0.15, 0.0, 1.0)
        evs.append(Evidence(
            key="multi_marker_signature",
            weight_lr=6.0,
            score=composite,
            details=f"{len(evs) - 1} markers drifting"
        ))

        prior = self.priors["INFLAMMATION_ONCO_SIGNATURE"]
        post_p, lr_total = fuse_evidence(prior, evs)

        conf_score = _clamp(np.mean([e.score for e in evs]) + 0.10, 0.0, 1.0)
        conf_lvl = ConfidenceLevel.HIGH if conf_score >= 0.70 else ConfidenceLevel.MEDIUM if conf_score >= 0.45 else ConfidenceLevel.LOW
        conf = ConfidenceScore(conf_lvl, round(conf_score, 2), f"{len(evs)} evidences")

        return OracleAlert(
            risk_type="INFLAMMATION-ONCOLOGY",
            probability=round(post_p * 100.0, 1),
            confidence=conf,
            lead_time="Months-years ahead",
            signal_sources=evs,
            recommended_actions=[
                "Rule out infection/autoimmune",
                "Repeat labs in 4-8 weeks",
                "Consider workup if persistent",
            ],
            summary="Multi-marker inflammatory drift",
            debug={"patient_id": self.patient_id, "trends": trends_debug},
        )

    # =========================================================================
    # C) CLONAL EXPANSION (OTTIMIZZATO v3.2 - LR AUMENTATI)
    # =========================================================================
    def detect_clonal_expansion(self, raw_ngs_visits: List[Dict[str, Any]]) -> Optional[OracleAlert]:
        """
        Rileva espansione clonale con LR ottimizzati per alta sensibilità.

        LR Strategy v3.2:
        - Driver singolo VAF alto: LR 12-15
        - VAF in crescita: LR 18-25 (segnale più forte!)
        - Nuovi cloni emergenti: LR 20-30
        - Multi-clone: LR 10-15
        """
        if not raw_ngs_visits:
            return None

        # Normalizza tempi
        rows = []
        for v in raw_ngs_visits:
            t = parse_time_to_month_index(v.get("date") or v.get("time") or v.get("month"))
            if t is None:
                continue
            variants = v.get("noise_variants", []) or []
            rows.append({"t": float(t), "variants": variants})

        if len(rows) < CLONAL_MIN_VISITS:
            return None

        rows.sort(key=lambda x: x["t"])
        t0 = rows[0]["t"]
        for r in rows:
            r["t"] = r["t"] - t0

        # =================================================================
        # STEP 1: Estrai varianti con VAF >= 0.8%
        # =================================================================
        gene_timeline: Dict[str, List[Tuple[float, float]]] = {}
        all_variants: List[Dict] = []

        for r in rows:
            tt = float(r["t"])
            for var in r["variants"]:
                gene = str(var.get("gene", "")).upper().strip()
                vaf = var.get("vaf")

                if not gene or not isinstance(vaf, (int, float)) or not np.isfinite(vaf):
                    continue

                vaf = float(vaf)
                if vaf < CLONAL_VAF_MIN:
                    continue

                gene_timeline.setdefault(gene, []).append((tt, vaf))
                all_variants.append({
                    "gene": gene, "vaf": vaf, "time": tt,
                    "is_driver": gene in HIGH_RISK_GENES,
                    "is_chip": gene in CHIP_GENES,
                })

        if not all_variants:
            return None

        # =================================================================
        # STEP 2: Analisi per gene
        # =================================================================
        gene_analysis: Dict[str, Dict[str, Any]] = {}

        for gene, timeline in gene_timeline.items():
            per_time: Dict[float, float] = {}
            for tt, vaf in timeline:
                per_time[tt] = max(per_time.get(tt, 0.0), vaf)

            times = sorted(per_time.keys())
            vafs = [per_time[t] for t in times]

            max_vaf = max(vafs)
            last_vaf = vafs[-1]
            first_vaf = vafs[0]
            n_visits = len(times)
            span_months = times[-1] - times[0] if len(times) > 1 else 0

            vaf_trend = None
            if len(times) >= 2:
                vaf_trend = robust_trend(np.array(times), np.array(vafs))

            is_emergent = times[0] > 0

            gene_analysis[gene] = {
                "n_visits": n_visits,
                "span_months": span_months,
                "first_vaf": first_vaf,
                "last_vaf": last_vaf,
                "max_vaf": max_vaf,
                "vaf_delta": last_vaf - first_vaf,
                "trend": vaf_trend,
                "is_driver": gene in HIGH_RISK_GENES,
                "is_chip": gene in CHIP_GENES,
                "is_emergent": is_emergent,
                "times": times,
                "vafs": vafs,
            }

        if len(gene_analysis) < CLONAL_MIN_GENES:
            return None

        # =================================================================
        # STEP 3: Costruisci Evidenze (LR OTTIMIZZATI)
        # =================================================================
        evs: List[Evidence] = []

        driver_genes = [g for g, d in gene_analysis.items() if d["is_driver"]]
        chip_only_genes = [g for g, d in gene_analysis.items() if d["is_chip"] and not d["is_driver"]]

        high_vaf_drivers = [g for g in driver_genes if gene_analysis[g]["max_vaf"] >= CLONAL_VAF_HIGH]
        very_high_vaf = [g for g in driver_genes if gene_analysis[g]["max_vaf"] >= CLONAL_VAF_VERY_HIGH]

        # --- Evidence 1: Driver genes rilevati ---
        if driver_genes:
            max_driver_vaf = max(gene_analysis[g]["max_vaf"] for g in driver_genes)

            # Intensity scaling basato su VAF
            if max_driver_vaf >= CLONAL_VAF_VERY_HIGH:
                intensity = 1.0
                lr = 18.0  # VAF >25% = molto forte
            elif max_driver_vaf >= CLONAL_VAF_HIGH:
                intensity = 0.85
                lr = 15.0  # VAF >10%
            elif max_driver_vaf >= CLONAL_VAF_SIGNIFICANT:
                intensity = 0.70
                lr = 12.0  # VAF >3%
            else:
                intensity = 0.50
                lr = 8.0  # VAF >0.8%

            evs.append(Evidence(
                key="driver_detected",
                weight_lr=lr,
                score=intensity,
                details=f"Drivers: {','.join(driver_genes)}; maxVAF={max_driver_vaf:.1%}"
            ))

        # --- Evidence 2: VAF in crescita (IL SEGNALE PIÙ FORTE) ---
        accelerating_genes = []
        for gene, data in gene_analysis.items():
            if data["trend"] and data["trend"].slope > 0.005 and data["vaf_delta"] > 0.03:
                accelerating_genes.append(gene)

        if accelerating_genes:
            max_delta = max(gene_analysis[g]["vaf_delta"] for g in accelerating_genes)

            # LR molto alto per crescita - questo è il segnale più importante!
            if max_delta >= 0.30:  # >30% crescita
                lr = 25.0
                intensity = 1.0
            elif max_delta >= 0.15:  # >15% crescita
                lr = 20.0
                intensity = 0.90
            elif max_delta >= 0.05:  # >5% crescita
                lr = 15.0
                intensity = 0.75
            else:
                lr = 10.0
                intensity = 0.60

            evs.append(Evidence(
                key="vaf_acceleration",
                weight_lr=lr,
                score=intensity,
                details=f"Rising: {','.join(accelerating_genes)}; Δmax={max_delta:.1%}"
            ))

        # --- Evidence 3: Cloni emergenti (non presenti all'inizio) ---
        emergent_drivers = [g for g in driver_genes if gene_analysis[g]["is_emergent"]]

        if emergent_drivers:
            # Emergenza = progressione attiva
            lr = 22.0 if len(emergent_drivers) >= 2 else 18.0
            intensity = _clamp(len(emergent_drivers) / 2.0, 0.5, 1.0)

            evs.append(Evidence(
                key="clonal_emergence",
                weight_lr=lr,
                score=intensity,
                details=f"New clones: {','.join(emergent_drivers)}"
            ))

        # --- Evidence 4: Multi-clone signature ---
        if len(driver_genes) >= 2:
            intensity = _clamp(len(driver_genes) / 3.0, 0.5, 1.0)

            evs.append(Evidence(
                key="multi_clone",
                weight_lr=12.0,
                score=intensity,
                details=f"{len(driver_genes)} driver clones"
            ))

        # --- Evidence 5: Burden alto ---
        total_driver_vaf = sum(gene_analysis[g]["last_vaf"] for g in driver_genes)
        if total_driver_vaf >= 0.15:
            intensity = _clamp(total_driver_vaf / 0.50, 0.4, 1.0)

            evs.append(Evidence(
                key="high_burden",
                weight_lr=10.0,
                score=intensity,
                details=f"Total burden: {total_driver_vaf:.1%}"
            ))

        # --- Penalità CHIP (solo se CHIP isolato senza driver) ---
        if chip_only_genes and not driver_genes:
            # CHIP isolato in assenza di driver - riduci confidenza
            chip_penalty = 0.4 if self.patient_age and self.patient_age > 65 else 0.6
            for ev in evs:
                ev.weight_lr *= chip_penalty
                ev.details += " [CHIP-only]"

        if not evs:
            return None

        # =================================================================
        # STEP 4: Bayesian Fusion
        # =================================================================
        prior = self.priors["CLONAL_EXPANSION"]
        post_p, lr_total = fuse_evidence(prior, evs)

        # =================================================================
        # STEP 5: Confidence Score
        # =================================================================
        base_conf = np.mean([e.score for e in evs])

        boost = 0.0
        if len(evs) >= 3:
            boost += 0.10
        if very_high_vaf:
            boost += 0.12
        if accelerating_genes:
            boost += 0.10
        if emergent_drivers:
            boost += 0.08

        conf_score = _clamp(base_conf + boost, 0.0, 1.0)
        conf_lvl = (
            ConfidenceLevel.HIGH if conf_score >= 0.68
            else ConfidenceLevel.MEDIUM if conf_score >= 0.42
            else ConfidenceLevel.LOW
        )

        conf = ConfidenceScore(
            conf_lvl, round(conf_score, 2),
            f"{len(driver_genes)}drv;{len(evs)}ev"
        )

        # =================================================================
        # STEP 6: Build Alert
        # =================================================================
        if very_high_vaf or (emergent_drivers and accelerating_genes):
            lead_time = "URGENT: weeks-months"
            urgency = "HIGH"
        elif accelerating_genes or emergent_drivers:
            lead_time = "Months ahead - active progression"
            urgency = "MODERATE"
        else:
            lead_time = "Months-years - established clones"
            urgency = "STANDARD"

        actions = [
            f"🔬 Repeat ctDNA in 4-6 weeks (urgency: {urgency})",
            "📊 Compare with baseline",
        ]

        if very_high_vaf:
            actions.append("🚨 Consider immediate imaging")
        if emergent_drivers:
            actions.append(f"⚠️ New drivers: {','.join(emergent_drivers)}")

        actions.extend([
            "🧬 Review targetable mutations",
            "📋 MDT discussion if confirmed",
        ])

        gene_summary = ", ".join([
            f"{g}({gene_analysis[g]['last_vaf']:.0%})"
            for g in sorted(driver_genes, key=lambda x: -gene_analysis[x]["last_vaf"])[:3]
        ])

        summary = (
            f"Clonal expansion: {len(driver_genes)} driver(s) [{gene_summary}]. "
            f"{'VAF rising. ' if accelerating_genes else ''}"
            f"{'New clones. ' if emergent_drivers else ''}"
        )

        return OracleAlert(
            risk_type="CLONAL_EXPANSION",
            probability=round(post_p * 100.0, 1),
            confidence=conf,
            lead_time=lead_time,
            signal_sources=evs,
            recommended_actions=actions,
            summary=summary.strip(),
            debug={
                "patient_id": self.patient_id,
                "prior": prior,
                "lr_total": lr_total,
                "driver_genes": driver_genes,
                "accelerating": accelerating_genes,
                "emergent": emergent_drivers,
            },
        )

    # =========================================================================
    # D) ctDNA Ghost (sub-clinical VAF < 0.8%)
    # =========================================================================
    def detect_ctdna_ghosts(self, raw_ngs_visits: List[Dict[str, Any]]) -> Optional[OracleAlert]:
        if not raw_ngs_visits:
            return None

        rows = []
        for v in raw_ngs_visits:
            t = parse_time_to_month_index(v.get("date") or v.get("time") or v.get("month"))
            if t is None:
                continue
            rows.append({"t": float(t), "noise_variants": v.get("noise_variants", []) or []})

        if len(rows) < GHOST_MIN_VISITS:
            return None

        rows.sort(key=lambda x: x["t"])
        t0 = rows[0]["t"]
        for r in rows:
            r["t"] = r["t"] - t0

        gene_hits: Dict[str, List[Tuple[float, float]]] = {}
        for r in rows:
            tt = float(r["t"])
            for var in r["noise_variants"]:
                gene = str(var.get("gene", "")).upper().strip()
                vaf = var.get("vaf", None)
                if not gene or not isinstance(vaf, (int, float)) or not np.isfinite(vaf):
                    continue
                vaf = float(vaf)
                if vaf <= 0 or vaf >= GHOST_VAF_MAX:
                    continue
                gene_hits.setdefault(gene, []).append((tt, vaf))

        persistent_genes: Dict[str, Dict[str, Any]] = {}
        for gene, hits in gene_hits.items():
            per_time: Dict[float, float] = {}
            for tt, vaf in hits:
                per_time[tt] = max(per_time.get(tt, 0.0), vaf)
            times = sorted(per_time.keys())
            if len(times) < GHOST_MIN_VISITS:
                continue
            span = times[-1] - times[0]
            if span < GHOST_MIN_MONTHS:
                continue

            vafs = [per_time[t] for t in times]
            persistent_genes[gene] = {
                "visits": len(times),
                "span_months": span,
                "max_vaf": max(vafs),
                "is_top_gene": gene in HIGH_RISK_GENES,
            }

        if len(persistent_genes) < GHOST_MIN_GENES:
            return None

        evs: List[Evidence] = []
        for gene, d in persistent_genes.items():
            base = _clamp((d["visits"] - GHOST_MIN_VISITS) / 4.0, 0.0, 0.25)
            base += _clamp(d["span_months"] / 12.0, 0.0, 0.25)
            base += _clamp(d["max_vaf"] / GHOST_VAF_MAX, 0.0, 0.30)
            if d["is_top_gene"]:
                base += 0.20

            evs.append(Evidence(
                key=f"ghost_{gene}",
                weight_lr=7.0 if d["is_top_gene"] else 4.5,
                score=_clamp(base, 0.0, 1.0),
                details=f"{gene}: {d['visits']}vis, maxVAF={d['max_vaf']:.3%}"
            ))

        prior = self.priors["MOLECULAR_MICRO_RELAPSE"]
        post_p, lr_total = fuse_evidence(prior, evs)

        conf_score = _clamp(float(np.mean([e.score for e in evs])) + 0.10, 0.0, 1.0)
        conf_lvl = ConfidenceLevel.HIGH if conf_score >= 0.70 else ConfidenceLevel.MEDIUM if conf_score >= 0.45 else ConfidenceLevel.LOW
        conf = ConfidenceScore(conf_lvl, round(conf_score, 2), f"Ghost:{len(persistent_genes)}")

        return OracleAlert(
            risk_type="MICRO-RELAPSE",
            probability=round(post_p * 100.0, 1),
            confidence=conf,
            lead_time="Months before progression",
            signal_sources=evs,
            recommended_actions=[
                "Repeat ctDNA in 4-8 weeks",
                "Consider earlier imaging",
                "Rule out CHIP",
            ],
            summary=f"Sub-clinical ctDNA signal ({','.join(persistent_genes.keys())})",
            debug={"patient_id": self.patient_id, "genes": list(persistent_genes.keys())},
        )

    # =========================================================================
    # E) CEA Trajectory Detection (NEW v3.3)
    # =========================================================================
    def detect_cea_trajectory(self) -> Optional[OracleAlert]:
        """
        Detects concerning CEA trajectories indicative of tumor progression.
        
        Key signals:
        - Doubling time < 3 months = RED FLAG for progression
        - Consistent rising trend with R² > 0.7
        - CEA crossing diagnostic threshold (5 ng/mL)
        
        Based on: Goldstein & Mitchell, J Clin Oncol 2005
        """
        x_cea, cea = self._series(("blood", "cea"))
        
        if len(cea) < DRIFT_MIN_POINTS:
            return None
        
        tr = robust_trend(x_cea, cea)
        if not tr:
            return None
        
        # Must be rising trend with decent fit
        if tr.slope <= 0 or tr.r2 < 0.55:
            return None
        
        evs: List[Evidence] = []
        
        # Calculate doubling time (in months)
        # If CEA follows exponential growth: CEA(t) = CEA0 * exp(k*t)
        # Doubling time = ln(2) / k
        # For linear approximation: k ≈ slope / mean(CEA)
        mean_cea = float(np.mean(cea))
        if mean_cea > 0.1:
            approx_k = tr.slope / mean_cea
            if approx_k > 0:
                doubling_time_months = math.log(2) / approx_k
            else:
                doubling_time_months = float('inf')
        else:
            doubling_time_months = float('inf')
        
        conf = confidence_from_trend(tr, DRIFT_MIN_POINTS, 6)  # 6 month minimum span
        
        # --- Evidence 1: Fast doubling time ---
        if doubling_time_months < 3:
            # Very concerning - rapid progression
            intensity = _clamp(1.0 - (doubling_time_months / 3), 0.5, 1.0)
            evs.append(Evidence(
                key="cea_rapid_doubling",
                weight_lr=15.0,
                score=intensity,
                details=f"CEA doubling time: {doubling_time_months:.1f} months (< 3mo = RED FLAG)"
            ))
        elif doubling_time_months < 6:
            intensity = _clamp(1.0 - (doubling_time_months / 6), 0.3, 0.8)
            evs.append(Evidence(
                key="cea_fast_doubling",
                weight_lr=8.0,
                score=intensity,
                details=f"CEA doubling time: {doubling_time_months:.1f} months"
            ))
        
        # --- Evidence 2: Steep slope with high R² ---
        if tr.slope > 0.5 and tr.r2 > 0.70:  # > 0.5 ng/mL/month
            intensity = _clamp(tr.r2, 0.5, 1.0)
            evs.append(Evidence(
                key="cea_steep_rise",
                weight_lr=10.0,
                score=intensity,
                details=f"CEA +{tr.slope:.2f} ng/mL/mo (R²={tr.r2:.2f})"
            ))
        
        # --- Evidence 3: Threshold crossing ---
        # First crossing from < 5 to >= 5 ng/mL (normal → elevated)
        crossed_threshold = False
        for i in range(1, len(cea)):
            if cea[i-1] < 5.0 and cea[i] >= 5.0:
                crossed_threshold = True
                break
        
        if crossed_threshold:
            evs.append(Evidence(
                key="cea_threshold_cross",
                weight_lr=6.0,
                score=0.7,
                details="CEA crossed 5 ng/mL threshold (normal → elevated)"
            ))
        
        # --- Evidence 4: High absolute value and rising ---
        if tr.last_value > 10 and tr.slope > 0.3:
            intensity = _clamp(tr.last_value / 50, 0.4, 1.0)
            evs.append(Evidence(
                key="cea_elevated_rising",
                weight_lr=7.0,
                score=intensity,
                details=f"Elevated CEA ({tr.last_value:.1f}) and rising"
            ))
        
        if len(evs) == 0:
            return None
        
        # --- Bayesian fusion ---
        prior = self.priors.get("CEA_TRAJECTORY", 0.05)
        post_p, lr_total = fuse_evidence(prior, evs)
        
        conf_score = _clamp(np.mean([e.score for e in evs]) + 0.1, 0.0, 1.0)
        conf_lvl = (
            ConfidenceLevel.HIGH if conf_score >= 0.70 
            else ConfidenceLevel.MEDIUM if conf_score >= 0.45 
            else ConfidenceLevel.LOW
        )
        
        conf = ConfidenceScore(conf_lvl, round(conf_score, 2), conf.reason)
        
        # Determine urgency based on doubling time
        if doubling_time_months < 2:
            lead_time = "URGENT: weeks - rapid progression"
            urgency_level = "CRITICAL"
        elif doubling_time_months < 3:
            lead_time = "High priority: 1-3 months"
            urgency_level = "HIGH"
        elif doubling_time_months < 6:
            lead_time = "Moderate priority: 3-6 months"
            urgency_level = "MODERATE"
        else:
            lead_time = "Standard follow-up"
            urgency_level = "STANDARD"
        
        actions = [
            f"🔬 Repeat CEA in 4-6 weeks (urgency: {urgency_level})",
            "📊 Compare with imaging timeline",
        ]
        
        if doubling_time_months < 3:
            actions.append("🚨 Consider immediate imaging (CT/PET)")
            actions.append("📋 MDT discussion recommended")
        else:
            actions.append("📋 Review prior imaging and treatment history")
        
        summary = (
            f"CEA trajectory: {tr.first_value:.1f} → {tr.last_value:.1f} ng/mL "
            f"over {tr.span_months:.0f} months. "
            f"Doubling time: {doubling_time_months:.1f}mo."
        )
        
        return OracleAlert(
            risk_type="CEA_TRAJECTORY",
            probability=round(post_p * 100.0, 1),
            confidence=conf,
            lead_time=lead_time,
            signal_sources=evs,
            recommended_actions=actions,
            summary=summary,
            debug={
                "patient_id": self.patient_id,
                "doubling_time_months": doubling_time_months,
                "slope": tr.slope,
                "r2": tr.r2,
                "first_cea": tr.first_value,
                "last_cea": tr.last_value,
            },
        )

    # =========================================================================
    # F) Orchestrator
    # =========================================================================
    def run_oracle(
            self,
            raw_ngs_visits: Optional[List[Dict[str, Any]]] = None,
            max_alerts: int = 3,
    ) -> List[OracleAlert]:
        alerts: List[OracleAlert] = []

        # A) Metabolic drift
        a = self.detect_pancreatic_metabolic_drift()
        if a:
            alerts.append(a)

        # B) Inflammation signature
        b = self.detect_inflammation_onco_signature()
        if b:
            alerts.append(b)

        # C) Clonal expansion (prioritario per molecular)
        if raw_ngs_visits:
            c = self.detect_clonal_expansion(raw_ngs_visits)
            if c:
                alerts.append(c)

        # D) Ghost mutations
        if raw_ngs_visits:
            d = self.detect_ctdna_ghosts(raw_ngs_visits)
            if d:
                alerts.append(d)
        
        # E) CEA trajectory (NEW)
        e = self.detect_cea_trajectory()
        if e:
            alerts.append(e)

        # F) PROMETHEUS epistatic rules (discovered from population analysis)
        try:
            from prometheus.oracle_bridge import check_patient_rules
            # Ricostruisci patient_data dal history per il bridge
            patient_data_for_bridge = {"baseline": {}, "visits": []}
            if self.history:
                last = self.history[-1]
                patient_data_for_bridge["baseline"]["blood_markers"] = last.get("blood", {})
                patient_data_for_bridge["baseline"]["patient_id"] = self.patient_id
            prometheus_evidence = check_patient_rules(patient_data_for_bridge)
            if prometheus_evidence:
                # Fuse epistatic evidence into a single alert
                evs = [
                    Evidence(
                        key=ev["key"],
                        weight_lr=ev["weight_lr"],
                        score=ev["score"],
                        details=ev["details"],
                    )
                    for ev in prometheus_evidence
                ]
                prior = 0.05  # Prior generico per regole epistatiche
                post_p, lr_total = fuse_evidence(prior, evs)
                if post_p > 0.10:  # Solo se probabilità significativa
                    alerts.append(OracleAlert(
                        risk_type="EPISTATIC_INTERACTION",
                        probability=round(post_p * 100.0, 1),
                        confidence=ConfidenceScore(
                            ConfidenceLevel.MEDIUM,
                            round(min(0.8, len(evs) * 0.25), 2),
                            f"PROMETHEUS: {len(evs)} rules matched"
                        ),
                        lead_time="Population-derived risk",
                        signal_sources=evs,
                        recommended_actions=[
                            "⚠️ Epistatic risk pattern detected (PROMETHEUS)",
                            "📊 Review interacting markers",
                            "🔬 Consider targeted monitoring",
                        ],
                        summary=f"PROMETHEUS: {len(evs)} epistatic rule(s) matched",
                        debug={"patient_id": self.patient_id, "n_rules": len(evs)},
                    ))
        except (ImportError, Exception):
            pass  # PROMETHEUS non disponibile o nessuna regola → silenzioso

        # Rank by probability
        alerts.sort(key=lambda x: (x.probability, x.confidence.score), reverse=True)
        return alerts[:max_alerts]


# =============================================================================
# DEMO / TEST
# =============================================================================

def _print_alert(a: OracleAlert) -> None:
    print("=" * 80)
    print(f"🔮 {a.risk_type}")
    print(f"   Prob: {a.probability}% | Conf: {a.confidence.level.value} ({a.confidence.score})")
    print(f"   Lead: {a.lead_time}")
    print(f"   {a.summary}")
    print("-" * 80)
    for ev in a.signal_sources:
        print(f"   • {ev.key}: LR={ev.weight_lr:.1f} score={ev.score:.2f} | {ev.details}")
    print("=" * 80)


if __name__ == "__main__":
    # Test: GHOST_RELAPSE_LUNG pattern
    history = [
        {"date": "2015-06-01", "blood": {"glucose": 85, "ldh": 180}, "clinical": {"weight": 75}},
        {"date": "2016-01-01", "blood": {"glucose": 86, "ldh": 190}, "clinical": {"weight": 74}},
        {"date": "2016-07-01", "blood": {"glucose": 87, "ldh": 200}, "clinical": {"weight": 73}},
        {"date": "2017-01-01", "blood": {"glucose": 85, "ldh": 210}, "clinical": {"weight": 72}},
    ]

    raw_ngs = [
        {"date": "2016-03-14", "noise_variants": [{"gene": "EGFR", "vaf": 0.03}]},
        {"date": "2016-07-21", "noise_variants": [{"gene": "TP53", "vaf": 0.118}]},
        {"date": "2017-01-12", "noise_variants": [{"gene": "KRAS", "vaf": 0.303}]},
        {"date": "2017-07-11", "noise_variants": [{"gene": "EGFR", "vaf": 0.842}, {"gene": "TP53", "vaf": 0.43}]},
    ]

    print("\n🧪 TEST: Oracle v3.2 (Optimized LR)")
    print("=" * 80)

    oracle = SentinelOracleV3(history, patient_id="TEST-001")
    alerts = oracle.run_oracle(raw_ngs_visits=raw_ngs)

    if not alerts:
        print("❌ No alerts")
    else:
        for a in alerts:
            _print_alert(a)