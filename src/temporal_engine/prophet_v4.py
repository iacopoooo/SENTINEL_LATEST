#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SENTINEL PROPHET v4 + CONFIDENCE v2 + FUSION v2 (Clinical Guardrails Edition)

Obiettivi dei fix:
- Stabilità matematica (Sxx ~ 0, settimane duplicate, outlier singolo)
- Confidence più "clinica" (non penalizza follow-up a 6-12 settimane; penalizza buchi enormi)
- VAF: gestione low-range (floor), threshold crossing, persistence (2+ visite)
- Guardrail CRITICAL: richiede gravità assoluta o forecast lower bound + persistenza
- Gate infezione/tossicità: se marker suggeriscono confondenti, abbassa urgenza e richiede repeat

Dipendenze:
- numpy
- scipy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# =============================================================================
# 0) CONFIG CLINICA
# =============================================================================

ULN_DEFAULTS = {
    "ldh": 250.0,
    "crp": 5.0,            # mg/L (molto variabile per lab)
    "alt": 45.0,           # U/L
    "ast": 40.0,           # U/L
    "neutrophils": 7500.0, # /µL
}

# Soglie pratiche per la logica
VAF_NOISE_FLOOR = 0.5  # % (floor per normalizzazione e robustezza low-range)
VAF_CLINICAL_THRESH = 1.0  # % (soglia "clinicamente credibile" come default euristico)
VAF_PERSISTENCE_K = 2  # visite consecutive sopra soglia = persistenza

LDH_UP_WEEKLY_PCT = 5.0      # crescita %/settimana considerata "in salita"
LDH_ACCEL_WEEKLY_PCT = 2.0   # accelerazione %/settimana^2 considerata rilevante
VAF_UP_WEEKLY_PCT = 5.0      # crescita %/settimana considerata "in salita"

LDH_GRAVITY_RATIO = 1.5      # ratio_to_uln per dire "grave"
LDH_FORECAST_GRAVITY_RATIO = 2.0  # lower bound del forecast supera 2x ULN = molto preoccupante

MAX_GAP_WEEKS_PENALTY = 16.0  # buco enorme (>=16 settimane) -> confidence giù
SXX_EPS = 1e-9                # per evitare divisioni per ~0
OUTLIER_Z = 3.0               # outlier se |resid| > OUTLIER_Z * resid_std
SMALL_DIFF_FRAC = 0.01        # volatilità: ignora diff < 1% mean


# =============================================================================
# 1) CONFIDENCE ENGINE v2 (Clinically tuned)
# =============================================================================

class ConfidenceLevel(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class ConfidenceScore:
    level: ConfidenceLevel
    score: float  # 0..1
    reason: str


class ConfidenceEngineV2:
    """
    Valuta affidabilità dei dati senza penalizzare follow-up "normali" (6-12 settimane),
    ma penalizza:
      - pochi punti,
      - trend rumoroso,
      - buchi ENORMI,
      - zig-zag reale (non micro-noise),
      - settimane duplicate / densità malformata (gestita a monte con aggregazione).
    """

    @staticmethod
    def evaluate(t: np.ndarray, y: np.ndarray, r_squared: float) -> ConfidenceScore:
        n = len(t)
        if n < 2:
            return ConfidenceScore(ConfidenceLevel.LOW, 0.0, "Insufficient points (<2)")

        score = 0.0
        reasons: List[str] = []

        # 1) Quantità dati
        if n >= 6:
            score += 0.35
        elif n >= 4:
            score += 0.20
        else:
            reasons.append("Few points (<4)")

        # 2) R² (ma attenzione: con pochi punti può essere fuorviante)
        if n >= 6:
            if r_squared > 0.80:
                score += 0.35
            elif r_squared > 0.55:
                score += 0.20
            else:
                reasons.append("Noisy trend (R² low)")
        else:
            # con n piccolo, r² è meno affidabile: peso ridotto
            if r_squared > 0.85:
                score += 0.20
            elif r_squared > 0.60:
                score += 0.10
            else:
                reasons.append("Noisy trend (R² low)")

        # 3) Gaps (clinico): penalizza solo buchi enormi
        gaps = np.diff(t)
        max_gap = float(np.max(gaps)) if len(gaps) else 0.0
        if max_gap >= MAX_GAP_WEEKS_PENALTY:
            score -= 0.15
            reasons.append(f"Very large time gap (max gap {max_gap:.0f}w)")
        else:
            score += 0.10

        # 4) Volatilità: ignora micro-diff sotto 1% mean
        mean_y = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
        mean_y = mean_y if mean_y > 0 else 1.0
        diffs = np.diff(y)

        # maschera: solo movimenti "reali"
        real_moves = np.abs(diffs) >= (SMALL_DIFF_FRAC * mean_y)
        diffs_real = diffs[real_moves]

        if len(diffs_real) >= 3:
            signs = np.sign(diffs_real)
            # rimuovi zeri (già filtrati ma per safety)
            signs = signs[signs != 0]
            if len(signs) >= 3:
                sign_changes = int(np.sum(np.diff(signs) != 0))
                if sign_changes > (len(signs) / 2):
                    score -= 0.15
                    reasons.append("High volatility (zig-zag)")

        # clamp
        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            lvl = ConfidenceLevel.HIGH
        elif score >= 0.45:
            lvl = ConfidenceLevel.MEDIUM
        else:
            lvl = ConfidenceLevel.LOW

        reason = "Dati solidi" if not reasons else "; ".join(reasons)
        return ConfidenceScore(lvl, round(score, 2), reason)


# =============================================================================
# 2) SIGNALS
# =============================================================================

@dataclass
class TemporalSignal:
    metric: str
    current_value: float
    ratio_to_uln: float
    velocity_norm: float         # %/week
    acceleration_norm: float     # %/week^2 (local)
    r_squared: float
    confidence: ConfidenceScore
    forecast_3m: Tuple[float, float]   # prediction interval 95%
    forecast_point: float
    outlier_flag: bool           # ultimo punto outlier vs residui
    notes: str = ""


# =============================================================================
# 3) DATA VECTORIZATION + CLEANING
# =============================================================================

def _safe_get(d: Dict[str, Any], path: List[str], default=np.nan):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _extract_vaf_max(genetics: Dict[str, Any]) -> float:
    """
    Estrae il VAF massimo da un dizionario genetics.
    Cerca pattern: *_vaf, *_cn (copy number), vaf_*, e chiavi specifiche.
    """
    if not isinstance(genetics, dict):
        return np.nan
    vals = []
    
    for k, v in genetics.items():
        k_lower = str(k).lower()
        
        # Pattern 1: chiavi che finiscono con _vaf (es. tp53_vaf, egfr_vaf)
        if k_lower.endswith("_vaf") and isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
        
        # Pattern 2: chiavi che iniziano con vaf_ (es. vaf_tp53)
        elif k_lower.startswith("vaf_") and isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
        
        # Pattern 3: copy number alto indica amplificazione (considera come "surrogate VAF")
        # met_cn > 4 = amplificazione significativa, mappiamo a "pseudo-VAF"
        elif k_lower.endswith("_cn") and isinstance(v, (int, float)) and np.isfinite(v):
            cn = float(v)
            if cn > 4:  # CN > 4 = amplificazione clinicamente rilevante
                # Mappa CN a pseudo-VAF: CN 5 → ~10%, CN 10 → ~30%, CN 15+ → ~50%
                pseudo_vaf = min(50.0, (cn - 2) * 5)  # Approssimazione
                vals.append(pseudo_vaf)
        
        # Pattern 4: chiave esatta "vaf" o "VAF"
        elif k_lower == "vaf" and isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
    
    return max(vals) if vals else np.nan


def _aggregate_by_week(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Se ci sono settimane duplicate, aggrega per week_on_therapy:
    - per LDH/marker: mediana
    - per VAF: massimo (per non perdere segnali)
    - per marker di tossicità/infezione: mediana
    """
    buckets: Dict[float, List[Dict[str, Any]]] = {}
    for r in rows:
        w = r.get("week_on_therapy", np.nan)
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        w = float(w)
        buckets.setdefault(w, []).append(r)

    out: List[Dict[str, Any]] = []
    for w in sorted(buckets.keys()):
        group = buckets[w]

        ldh_vals = []
        vaf_vals = []
        crp_vals = []
        neut_vals = []
        alt_vals = []
        ast_vals = []

        for g in group:
            ldh = _safe_get(g, ["blood_markers", "ldh"])
            if isinstance(ldh, (int, float)) and np.isfinite(ldh):
                ldh_vals.append(float(ldh))

            vaf = _extract_vaf_max(g.get("genetics", {}))
            if np.isfinite(vaf):
                vaf_vals.append(float(vaf))

            crp = _safe_get(g, ["blood_markers", "crp"])
            if isinstance(crp, (int, float)) and np.isfinite(crp):
                crp_vals.append(float(crp))

            neut = _safe_get(g, ["blood_markers", "neutrophils"])
            if isinstance(neut, (int, float)) and np.isfinite(neut):
                neut_vals.append(float(neut))

            alt = _safe_get(g, ["blood_markers", "alt"])
            if isinstance(alt, (int, float)) and np.isfinite(alt):
                alt_vals.append(float(alt))

            ast = _safe_get(g, ["blood_markers", "ast"])
            if isinstance(ast, (int, float)) and np.isfinite(ast):
                ast_vals.append(float(ast))

        agg = {
            "week_on_therapy": w,
            "blood_markers": {},
            "genetics": {},
        }

        if ldh_vals:
            agg["blood_markers"]["ldh"] = float(np.median(ldh_vals))

        # VAF: massimo del gruppo
        if vaf_vals:
            agg["genetics"]["max_vaf"] = float(np.max(vaf_vals))

        if crp_vals:
            agg["blood_markers"]["crp"] = float(np.median(crp_vals))
        if neut_vals:
            agg["blood_markers"]["neutrophils"] = float(np.median(neut_vals))
        if alt_vals:
            agg["blood_markers"]["alt"] = float(np.median(alt_vals))
        if ast_vals:
            agg["blood_markers"]["ast"] = float(np.median(ast_vals))

        out.append(agg)

    return out


# =============================================================================
# 4) PROPHET v4
# =============================================================================

class SentinelProphetV4:
    def __init__(self, patient_history: List[Dict[str, Any]]):
        history = sorted(patient_history, key=lambda x: x.get("week_on_therapy", 0))
        history = _aggregate_by_week(history)
        self.history = history
        self.vectors = self._vectorize()

    def _vectorize(self) -> Dict[str, np.ndarray]:
        t, ldh, vaf, crp, neut, alt, ast = [], [], [], [], [], [], []

        for v in self.history:
            w = v.get("week_on_therapy", np.nan)
            if w is None or (isinstance(w, float) and np.isnan(w)):
                continue

            t.append(float(w))

            ldh.append(float(_safe_get(v, ["blood_markers", "ldh"], np.nan)))
            # VAF: abbiamo salvato in genetics["max_vaf"]
            vaf.append(float(_safe_get(v, ["genetics", "max_vaf"], np.nan)))

            crp.append(float(_safe_get(v, ["blood_markers", "crp"], np.nan)))
            neut.append(float(_safe_get(v, ["blood_markers", "neutrophils"], np.nan)))
            alt.append(float(_safe_get(v, ["blood_markers", "alt"], np.nan)))
            ast.append(float(_safe_get(v, ["blood_markers", "ast"], np.nan)))

        return {
            "t": np.array(t, dtype=float),
            "ldh": np.array(ldh, dtype=float),
            "vaf": np.array(vaf, dtype=float),
            "crp": np.array(crp, dtype=float),
            "neutrophils": np.array(neut, dtype=float),
            "alt": np.array(alt, dtype=float),
            "ast": np.array(ast, dtype=float),
        }

    def analyze_metric(self, key: str, uln_override: Optional[float] = None) -> TemporalSignal:
        t_raw = self.vectors["t"]
        y_raw = self.vectors.get(key, np.array([], dtype=float))

        uln = float(uln_override) if uln_override is not None else float(ULN_DEFAULTS.get(key, 1.0))

        # cleaning
        mask = np.isfinite(t_raw) & np.isfinite(y_raw)
        t = t_raw[mask]
        y = y_raw[mask]

        if len(t) == 0:
            return TemporalSignal(
                metric=key,
                current_value=0.0,
                ratio_to_uln=0.0,
                velocity_norm=0.0,
                acceleration_norm=0.0,
                r_squared=0.0,
                confidence=ConfidenceScore(ConfidenceLevel.LOW, 0.0, "No data"),
                forecast_3m=(0.0, 0.0),
                forecast_point=0.0,
                outlier_flag=False,
                notes="No valid points",
            )

        current = float(y[-1])
        ratio = float(current / uln) if (uln and uln > 0) else 0.0

        if len(t) < 2:
            return TemporalSignal(
                metric=key,
                current_value=round(current, 2),
                ratio_to_uln=round(ratio, 2),
                velocity_norm=0.0,
                acceleration_norm=0.0,
                r_squared=0.0,
                confidence=ConfidenceScore(ConfidenceLevel.LOW, 0.1, "Single point"),
                forecast_3m=(round(current, 2), round(current, 2)),
                forecast_point=round(current, 2),
                outlier_flag=False,
                notes="Insufficient for trend",
            )

        # Regressione globale
        slope, intercept, r_val, p_val, std_err = stats.linregress(t, y)
        r2 = float(r_val**2)

        # Normalizzazione velocità
        mean_y = float(np.mean(y)) if np.mean(y) > 0 else 1.0

        # Fix VAF low-range: floor per evitare %/wk che esplodono
        if key == "vaf":
            mean_y = max(mean_y, VAF_NOISE_FLOOR)

        velocity_norm = (float(slope) / mean_y) * 100.0

        # Accelerazione locale (late vs early)
        accel_norm = 0.0
        n = len(t)
        K = 0
        if n >= 6:
            K = 3
        elif n >= 4:
            K = 2

        if K > 0:
            t_late, y_late = t[-K:], y[-K:]
            t_early, y_early = t[-2 * K : -K], y[-2 * K : -K]

            slope_late, *_ = stats.linregress(t_late, y_late)
            slope_early, *_ = stats.linregress(t_early, y_early)

            accel_raw = float(slope_late - slope_early)
            accel_norm = (accel_raw / mean_y) * 100.0

        # Prediction interval (robusto con Sxx guard)
        future_t = float(t[-1] + 12.0)
        y_pred = float(slope * future_t + intercept)

        residuals = y - (slope * t + intercept)
        resid_std = float(np.std(residuals, ddof=1)) if len(residuals) >= 3 else float(np.std(residuals)) if len(residuals) else 0.0
        resid_std = resid_std if resid_std > 0 else float(std_err) if np.isfinite(std_err) else 0.0

        Sxx = float(np.sum((t - np.mean(t)) ** 2))
        if Sxx < SXX_EPS or n < 3 or not np.isfinite(Sxx):
            # fallback: intervallo basato su residui (più conservativo)
            pi_err = 1.96 * resid_std if resid_std > 0 else 0.0
            ci_lower = max(0.0, y_pred - pi_err)
            ci_upper = y_pred + pi_err
            pi_note = "PI fallback (low Sxx / low n)"
        else:
            Se = math.sqrt(float(np.sum(residuals**2) / (n - 2))) if n > 2 else float(std_err)
            Se = Se if np.isfinite(Se) else resid_std
            pred_error = float(Se * math.sqrt(1.0 + (1.0 / n) + ((future_t - float(np.mean(t))) ** 2 / Sxx)))
            t_crit = float(stats.t.ppf(0.975, df=n - 2))
            ci_lower = max(0.0, y_pred - (t_crit * pred_error))
            ci_upper = y_pred + (t_crit * pred_error)
            pi_note = "PI standard"

        # Outlier flag sull'ultimo punto
        outlier_flag = False
        if resid_std > 0 and len(residuals) >= 3:
            z_last = abs(float(residuals[-1])) / resid_std
            if z_last >= OUTLIER_Z:
                outlier_flag = True

        # Confidence
        conf = ConfidenceEngineV2.evaluate(t, y, r2)
        # se ultimo punto outlier, abbassa conf "di uno step" (soft)
        notes = [pi_note]
        if outlier_flag:
            notes.append(f"Last-point outlier (z>={OUTLIER_Z})")
            if conf.level == ConfidenceLevel.HIGH:
                conf = ConfidenceScore(ConfidenceLevel.MEDIUM, max(0.45, conf.score - 0.15), conf.reason + "; last-point outlier")
            elif conf.level == ConfidenceLevel.MEDIUM:
                conf = ConfidenceScore(ConfidenceLevel.LOW, max(0.20, conf.score - 0.15), conf.reason + "; last-point outlier")

        return TemporalSignal(
            metric=key,
            current_value=round(current, 3 if key == "vaf" else 1),
            ratio_to_uln=round(ratio, 2),
            velocity_norm=round(velocity_norm, 2),
            acceleration_norm=round(accel_norm, 2),
            r_squared=round(r2, 2),
            confidence=conf,
            forecast_3m=(round(ci_lower, 3 if key == "vaf" else 1), round(ci_upper, 3 if key == "vaf" else 1)),
            forecast_point=round(y_pred, 3 if key == "vaf" else 1),
            outlier_flag=outlier_flag,
            notes="; ".join(notes),
        )

    # Convenience per gates
    def latest_marker(self, key: str) -> Optional[float]:
        arr = self.vectors.get(key, None)
        if arr is None:
            return None
        # prendi ultimo finito
        idx = np.where(np.isfinite(arr))[0]
        if len(idx) == 0:
            return None
        return float(arr[idx[-1]])

    def vaf_persistence(self, threshold: float = VAF_CLINICAL_THRESH, k: int = VAF_PERSISTENCE_K) -> bool:
        """
        True se VAF >= threshold in k visite consecutive (considerando solo punti finiti).
        """
        v = self.vectors.get("vaf", np.array([], dtype=float))
        finite_idx = np.where(np.isfinite(v))[0]
        if len(finite_idx) < k:
            return False
        vv = v[finite_idx]
        # consecutive in ordine temporale già garantito
        tail = vv[-k:]
        return bool(np.all(tail >= threshold))

    def vaf_threshold_crossed(self, threshold: float = VAF_CLINICAL_THRESH) -> bool:
        v = self.vectors.get("vaf", np.array([], dtype=float))
        finite_idx = np.where(np.isfinite(v))[0]
        if len(finite_idx) < 2:
            return False
        vv = v[finite_idx]
        return bool((vv[-2] < threshold) and (vv[-1] >= threshold))


# =============================================================================
# 5) FUSION ENGINE v2 (Guardrails + Clinical Gates)
# =============================================================================

class Urgency(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FusionResult:
    archetype: str
    urgency: Urgency
    action: str
    explanation: str
    flags: List[str]


class FusionEngineV2:
    @staticmethod
    def infection_toxicity_gate(prophet: SentinelProphetV4) -> Tuple[bool, List[str]]:
        """
        Ritorna (gate_active, flags).
        Se marker suggeriscono infezione/tossicità che può spiegare LDH, riduce urgenza.
        """
        flags: List[str] = []

        crp = prophet.latest_marker("crp")
        neut = prophet.latest_marker("neutrophils")
        alt = prophet.latest_marker("alt")
        ast = prophet.latest_marker("ast")

        gate = False

        # Infezione/infiammazione probabile
        if crp is not None and crp >= 10.0:  # soglia euristica
            gate = True
            flags.append(f"CRP high ({crp}) -> possible infection/inflammation")

        if neut is not None and neut >= 10000.0:
            gate = True
            flags.append(f"Neutrophils high ({neut}) -> possible infection/stress")

        # Tossicità epatica possibile
        if alt is not None and alt >= (2.0 * ULN_DEFAULTS["alt"]):
            gate = True
            flags.append(f"ALT high ({alt}) -> possible hepatotoxicity")
        if ast is not None and ast >= (2.0 * ULN_DEFAULTS["ast"]):
            gate = True
            flags.append(f"AST high ({ast}) -> possible hepatotoxicity")

        return gate, flags

    @staticmethod
    def diagnose(
        prophet: SentinelProphetV4,
        ldh: TemporalSignal,
        vaf: TemporalSignal,
        vaf_thresh: float = VAF_CLINICAL_THRESH,
    ) -> FusionResult:
        flags: List[str] = []

        reliable_ldh = ldh.confidence.level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)
        reliable_vaf = vaf.confidence.level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)

        ldh_up = ldh.velocity_norm >= LDH_UP_WEEKLY_PCT
        vaf_up = vaf.velocity_norm >= VAF_UP_WEEKLY_PCT
        ldh_accel = ldh.acceleration_norm >= LDH_ACCEL_WEEKLY_PCT

        # Gravità assoluta / forecast
        ldh_grave_now = ldh.ratio_to_uln >= LDH_GRAVITY_RATIO
        ldh_forecast_lower = ldh.forecast_3m[0]
        ldh_forecast_grave = (ldh_forecast_lower / ULN_DEFAULTS["ldh"]) >= LDH_FORECAST_GRAVITY_RATIO if ULN_DEFAULTS["ldh"] > 0 else False

        # VAF robust features
        vaf_persist = prophet.vaf_persistence(threshold=vaf_thresh, k=VAF_PERSISTENCE_K)
        vaf_crossed = prophet.vaf_threshold_crossed(threshold=vaf_thresh)
        vaf_now_high = vaf.current_value >= vaf_thresh

        if vaf_persist:
            flags.append(f"VAF persistence >= {vaf_thresh}% for {VAF_PERSISTENCE_K} visits")
        if vaf_crossed:
            flags.append(f"VAF threshold crossed ({vaf_thresh}%)")

        # Gate confondenti clinici (infezione/tossicità)
        gate_active, gate_flags = FusionEngineV2.infection_toxicity_gate(prophet)
        flags.extend(gate_flags)

        # ---------------------------------------------------------------------
        # LOGICA DECISIONALE (con guardrails)
        # ---------------------------------------------------------------------

        archetype = "STABLE / DORMANT"
        urgency = Urgency.LOW
        action = "Continue monitoring per schedule."
        explanation_parts: List[str] = []

        explanation_parts.append(
            f"LDH: {ldh.current_value} (ratio {ldh.ratio_to_uln}x ULN), vel {ldh.velocity_norm}%/wk, accel {ldh.acceleration_norm}%/wk², conf {ldh.confidence.level.name}, PI3m {ldh.forecast_3m}"
        )
        explanation_parts.append(
            f"VAF: {vaf.current_value}%, vel {vaf.velocity_norm}%/wk, conf {vaf.confidence.level.name}"
        )

        # =================================================================
        # SCENARIO -1: VALORI ASSOLUTI CRITICI (anche senza trend/velocity)
        # Se LDH o VAF sono già a livelli pericolosi, non aspettiamo il trend!
        # =================================================================
        ldh_absolute_critical = ldh.ratio_to_uln >= 2.5  # >2.5x ULN è sempre grave
        vaf_absolute_critical = vaf.current_value >= 10.0  # VAF > 10% è sempre significativo
        
        if ldh_absolute_critical and vaf_absolute_critical:
            # Entrambi critici = situazione grave indipendentemente dal trend
            archetype = "ABSOLUTE CRISIS (Extreme baseline values)"
            urgency = Urgency.CRITICAL
            action = "🔴 CRITICAL: Both LDH and VAF at dangerous levels. Immediate reassessment required."
            flags.append(f"LDH absolute critical ({ldh.current_value}, {ldh.ratio_to_uln:.1f}x ULN)")
            flags.append(f"VAF absolute critical ({vaf.current_value}%)")
            explanation_parts.append("Extreme absolute values detected regardless of trend.")
        
        elif ldh_absolute_critical and ldh.ratio_to_uln >= 3.0:
            # LDH > 3x ULN da solo è già HIGH
            archetype = "METABOLIC CRISIS (Extreme LDH)"
            urgency = Urgency.HIGH
            action = "🟠 HIGH: LDH extremely elevated. Urgent investigation + repeat labs."
            flags.append(f"LDH extreme ({ldh.current_value}, {ldh.ratio_to_uln:.1f}x ULN)")
        
        elif vaf_absolute_critical and vaf.current_value >= 15.0:
            # VAF > 15% da solo suggerisce alto tumor burden
            archetype = "HIGH TUMOR BURDEN (Elevated ctDNA)"
            urgency = Urgency.HIGH
            action = "🟠 HIGH: Significant ctDNA level. Consider imaging + therapy review."
            flags.append(f"VAF elevated ({vaf.current_value}%)")

        # Scenario 0: Dati scarsi (trend presente ma confidence bassa)
        elif (ldh_up or vaf_up) and (not reliable_ldh and not reliable_vaf):
            # NUOVO: Override se LDH è estremamente alto (>3x ULN) - non ignoriamo mai questo!
            if ldh.ratio_to_uln >= 3.0:
                archetype = "METABOLIC CRISIS (Low confidence but extreme value)"
                urgency = Urgency.HIGH
                action = "🟠 HIGH: LDH extremely elevated. Verify with repeat labs + investigate urgently."
                flags.append(f"LDH extreme ({ldh.current_value}, {ldh.ratio_to_uln}x ULN) overrides low confidence")
            else:
                archetype = "NOISY SIGNAL"
                urgency = Urgency.LOW
                action = "Repeat labs/ctDNA in ~2 weeks to confirm trend (data quality gate)."
                explanation_parts.append("Signals present but reliability is LOW for both metrics.")

        # Scenario 1: Progressione molecolare e metabolica (potenziale CRITICAL)
        elif reliable_ldh and reliable_vaf and ldh_up and vaf_up:
            archetype = "MOLECULAR + METABOLIC PROGRESSION"
            # Guardrail CRITICAL:
            # CRITICAL solo se almeno uno: LDH grave OR forecast grave OR VAF persist/high crossing
            critical_ok = (ldh_grave_now or ldh_forecast_grave or vaf_persist or vaf_crossed or vaf_now_high)

            if critical_ok:
                urgency = Urgency.CRITICAL
                action = "🔴 URGENT: Re-staging + therapy reassessment now (switch window)."
                if ldh_accel:
                    archetype += " (ACCELERATING)"
                    action = "🔴 IMMEDIATE: Window closing. Consider switch within 2-4 weeks + imaging/ctDNA."
            else:
                urgency = Urgency.HIGH
                action = "🟠 HIGH: Intensify monitoring; repeat labs/ctDNA sooner; prep next line."
                explanation_parts.append("Guardrail prevented CRITICAL: severity/persistence not yet confirmed.")

        # Scenario 2: Escape molecolare (VAF sale, LDH non sale)
        elif reliable_vaf and vaf_up and (not ldh_up or not reliable_ldh):
            archetype = "MOLECULAR ESCAPE (Silent Progression)"
            # Se VAF persist o crossed -> HIGH, altrimenti MEDIUM
            if vaf_persist or vaf_crossed or vaf_now_high:
                urgency = Urgency.HIGH
                action = "🟠 PRE-EMPTIVE: ctDNA rising. Consider earlier imaging + discuss next line within 8-12 weeks."
            else:
                urgency = Urgency.MEDIUM
                action = "🟡 Monitor closely: repeat ctDNA sooner to confirm persistence."

        # Scenario 3: Dissociazione metabolica (LDH sale, VAF fermo)
        elif reliable_ldh and ldh_up and (not vaf_up or not reliable_vaf):
            archetype = "METABOLIC DISSOCIATION"
            # Se LDH già grave o forecast grave -> HIGH
            # NUOVO: Se LDH è MOLTO grave (>3x ULN) E accelera -> CRITICAL
            ldh_very_grave = ldh.ratio_to_uln >= 2.5  # 2.5x ULN = molto alto
            
            if ldh_very_grave and ldh_accel:
                urgency = Urgency.CRITICAL
                action = "🔴 CRITICAL: Explosive metabolic progression (Warburg). Re-staging + therapy change URGENT."
                archetype = "METABOLIC EXPLOSION (Warburg)"
                explanation_parts.append("LDH very high (>2.5x ULN) + accelerating = critical metabolic failure.")
            elif ldh_grave_now or ldh_forecast_grave:
                urgency = Urgency.HIGH
                action = "🟠 Investigate confounders; if negative, suspect non-shedding progression. Consider imaging sooner."
            else:
                urgency = Urgency.MEDIUM
                action = "🟡 Investigate: infection/toxicity/other LDH sources; repeat soon."

        else:
            archetype = "STABLE / DORMANT"
            urgency = Urgency.LOW
            action = "Continue monitoring per schedule."

        # ---------------------------------------------------------------------
        # Apply Clinical Gate: se infezione/tossicità probabile, abbassa urgenza 1 step
        # ---------------------------------------------------------------------
        if gate_active and urgency in (Urgency.CRITICAL, Urgency.HIGH):
            flags.append("Clinical confounder gate applied (infection/toxicity suspected)")
            if urgency == Urgency.CRITICAL:
                urgency = Urgency.HIGH
                action = "🟠 HIGH (GATED): Check/treat confounders first; repeat LDH/markers in 3-7 days + reassess."
            elif urgency == Urgency.HIGH:
                urgency = Urgency.MEDIUM
                action = "🟡 MEDIUM (GATED): Manage confounders; repeat labs soon; reassess trend."

        explanation = "\n".join(explanation_parts)
        return FusionResult(archetype=archetype, urgency=urgency, action=action, explanation=explanation, flags=flags)


# =============================================================================
# 6) DEMO / TEST DRIVE
# =============================================================================

def _print_signal(sig: TemporalSignal):
    print(f"{sig.metric.upper():>12}: val={sig.current_value} ratio={sig.ratio_to_uln}x "
          f"vel={sig.velocity_norm}%/wk accel={sig.acceleration_norm}%/wk² r2={sig.r_squared} "
          f"conf={sig.confidence.level.name} ({sig.confidence.score}) PI3m={sig.forecast_3m} "
          f"outlier={sig.outlier_flag} notes=[{sig.notes}]")

def run_demo():
    # Caso: progressione reale ma con buco e possibile confondente
    history = [
        {"week_on_therapy": 0,  "blood_markers": {"ldh": 180, "crp": 2.0, "neutrophils": 6000}, "genetics": {"tp53_vaf": 0.4}},
        {"week_on_therapy": 4,  "blood_markers": {"ldh": 190, "crp": 3.0, "neutrophils": 6500}, "genetics": {"tp53_vaf": 0.6}},
        {"week_on_therapy": 12, "blood_markers": {"ldh": 240, "crp": 4.0, "neutrophils": 7000}, "genetics": {"tp53_vaf": 1.2}},
        {"week_on_therapy": 16, "blood_markers": {"ldh": 290, "crp": 18.0, "neutrophils": 12000}, "genetics": {"tp53_vaf": 2.5}},  # confondente: CRP/Neut alti
        {"week_on_therapy": 18, "blood_markers": {"ldh": 350, "crp": 15.0, "neutrophils": 11000}, "genetics": {"tp53_vaf": 4.0}},
    ]

    prophet = SentinelProphetV4(history)

    sig_ldh = prophet.analyze_metric("ldh")
    sig_vaf = prophet.analyze_metric("vaf")
    sig_crp = prophet.analyze_metric("crp")
    sig_neut = prophet.analyze_metric("neutrophils")

    print("=" * 80)
    print("SENTINEL PROPHET v4 - SIGNALS")
    print("=" * 80)
    _print_signal(sig_ldh)
    _print_signal(sig_vaf)
    _print_signal(sig_crp)
    _print_signal(sig_neut)
    print()

    res = FusionEngineV2.diagnose(prophet, sig_ldh, sig_vaf, vaf_thresh=VAF_CLINICAL_THRESH)

    print("=" * 80)
    print("FUSION v2 - FINAL")
    print("=" * 80)
    print(f"ARCHETYPE: {res.archetype}")
    print(f"URGENCY:   {res.urgency.value}")
    print(f"ACTION:    {res.action}")
    print("-" * 80)
    print("EXPLANATION:")
    print(res.explanation)
    print("-" * 80)
    if res.flags:
        print("FLAGS:")
        for f in res.flags:
            print(f" - {f}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
