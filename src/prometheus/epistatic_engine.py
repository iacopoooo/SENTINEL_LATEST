"""
PROMETHEUS — Epistatic Discovery Engine v1.0
=============================================
Scopre interazioni epistatiche nascoste nei dati pazienti SENTINEL.

4 Fasi di Discovery:
  A) Biomarker × Biomarker (top features)
  B) SNP × Biomarker (FORZATO — il cuore)
  C) SNP × SNP (tra SNPs comuni)
  D) Triple (SNP × biomarker × context)

Caratteristiche difensive (per N piccoli):
  - Soglie minime di carriers adattive al dataset
  - FDR (Benjamini-Hochberg) per correzione test multipli
  - Output vuoto [] è perfettamente valido
  - Nessun crash su dati insufficienti
"""

import json
import logging
import itertools
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

log = logging.getLogger("prometheus.epistatic")


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EpistaticRule:
    """Singola regola epistatica scoperta."""
    markers: List[str]
    order: int  # 2 = coppia, 3 = tripla
    interaction_info: float  # Information gain dell'interazione
    information_gain: float  # IG congiunto
    p_value_raw: float  # p-value dal permutation test
    p_value_fdr: float  # p-value dopo correzione FDR
    conditional_risk: float  # Rischio quando tutte le condizioni sono vere
    marginal_risks: Dict[str, float]  # Rischio per ogni singolo marker
    risk_amplification: float  # Amplificazione vs marker più forte
    n_carriers: int  # Numero soggetti con tutte le condizioni
    types: str  # Tipo interazione (es. "snp×biomarker")
    phase: str  # Fase di discovery (A/B/C/D)
    significant: bool = True  # Sopravvive FDR?


@dataclass
class DiscoveryResult:
    """Risultato completo della discovery."""
    n_patients: int
    n_features: int
    pairs: List[EpistaticRule] = field(default_factory=list)
    triplets: List[EpistaticRule] = field(default_factory=list)
    phases_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_rules(self) -> List[EpistaticRule]:
        return [r for r in self.pairs + self.triplets if r.significant]

    def to_json(self, path: str):
        """Salva le regole significative come JSON."""
        rules = [asdict(r) for r in self.all_rules]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"Salvate {len(rules)} regole in {path}")

    @staticmethod
    def load_rules(path: str) -> List[dict]:
        """Carica regole da JSON. Ritorna [] se il file non esiste o è vuoto."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            return rules if isinstance(rules, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []


# ═══════════════════════════════════════════════════════════════════
# STATISTICAL TOOLS
# ═══════════════════════════════════════════════════════════════════

def information_gain(x: np.ndarray, y: np.ndarray) -> float:
    """Calcola l'information gain di x rispetto a y (binario)."""
    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid].astype(int)
    y_v = y[valid].astype(float)

    if len(x_v) < 5:
        return 0.0

    py = y_v.mean()
    if py <= 0 or py >= 1:
        return 0.0

    hy = -py * np.log2(py) - (1 - py) * np.log2(1 - py)
    hyx = 0.0

    for v in np.unique(x_v):
        m = x_v == v
        p = m.mean()
        if p > 0:
            pyv = y_v[m].mean()
            if 0 < pyv < 1:
                hyx += p * (-pyv * np.log2(pyv) - (1 - pyv) * np.log2(1 - pyv))

    return max(0.0, hy - hyx)


def binarize(x: np.ndarray, ftype: str) -> np.ndarray:
    """Binarizza un vettore di feature in base al tipo."""
    x = np.asarray(x, dtype=float)
    if ftype == "snp":
        return (x > 0).astype(int)
    elif ftype == "prs":
        p75 = np.nanpercentile(x, 75)
        return (x > p75).astype(int)
    else:
        med = np.nanmedian(x)
        return (x > med).astype(int)


def permutation_test(x: np.ndarray, y: np.ndarray,
                     observed: float, n_perm: int = 300) -> float:
    """Test di permutazione per significatività dell'IG."""
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if information_gain(x, y_perm) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Correzione FDR (Benjamini-Hochberg).

    Returns:
        Lista di bool: True se la regola sopravvive alla correzione.
    """
    if not p_values:
        return []

    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calcola soglie BH
    thresholds = alpha * np.arange(1, n + 1) / n

    # Trova il più grande k tale che p_{(k)} <= threshold_k
    significant = np.zeros(n, dtype=bool)
    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k

    # Tutti i risultati fino a max_k sono significativi
    if max_k >= 0:
        for k in range(max_k + 1):
            significant[sorted_indices[k]] = True

    return significant.tolist()


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

def _adaptive_thresholds(n_patients: int) -> dict:
    """
    Soglie adattive in base alla dimensione del dataset.

    Con 126 pazienti le soglie sono più permissive per esplorare,
    ma l'FDR finale protegge dai falsi positivi.
    Con 10.000+ pazienti le soglie si stringono e servono più evidenze.
    """
    if n_patients < 100:
        return {
            "min_carriers_snp": 3,
            "min_carriers_bio": 10,
            "ii_threshold_pairs": 0.005,
            "ii_threshold_triples": 0.003,
            "risk_amp_threshold": 1.10,
            "n_perm": 200,
            "fdr_alpha": 0.10,  # Più permissivo, esplorativo
            "max_bio_pairs": 10,
            "max_bio_cross": 15,
        }
    elif n_patients < 500:
        return {
            "min_carriers_snp": 5,
            "min_carriers_bio": 15,
            "ii_threshold_pairs": 0.003,
            "ii_threshold_triples": 0.002,
            "risk_amp_threshold": 1.12,
            "n_perm": 300,
            "fdr_alpha": 0.05,
            "max_bio_pairs": 15,
            "max_bio_cross": 20,
        }
    else:
        return {
            "min_carriers_snp": 10,
            "min_carriers_bio": 20,
            "ii_threshold_pairs": 0.002,
            "ii_threshold_triples": 0.001,
            "risk_amp_threshold": 1.15,
            "n_perm": 500,
            "fdr_alpha": 0.05,
            "max_bio_pairs": 20,
            "max_bio_cross": 25,
        }


# ═══════════════════════════════════════════════════════════════════
# EPISTATIC DISCOVERY ENGINE
# ═══════════════════════════════════════════════════════════════════

def discover_epistatic(
    df, y: np.ndarray,
    feature_types: Dict[str, str],
    outcome_col: str = "has_cancer",
) -> DiscoveryResult:
    """
    Epistatic Discovery Engine — 4 fasi con FDR.

    DIFENSIVO: se N è troppo piccolo o FDR elimina tutto,
    ritorna un DiscoveryResult con liste vuote (mai crash).

    Args:
        df: DataFrame con features
        y: Array outcome binario (0/1)
        feature_types: Dict feature_name → tipo (snp, biomarker, etc.)
        outcome_col: nome colonna outcome (esclusa dall'analisi)

    Returns:
        DiscoveryResult con pairs e triplets (possibilmente vuoti)
    """
    n_patients = len(y)
    n_features = len([c for c in df.columns if c != outcome_col])
    result = DiscoveryResult(n_patients=n_patients, n_features=n_features)

    # ── Safety check: abbastanza dati? ────────────────────────────
    n_cases = int(y.sum())
    n_controls = int(len(y) - n_cases)

    if n_patients < 20:
        result.warnings.append(f"Dataset troppo piccolo ({n_patients} pazienti). Servono almeno 20.")
        log.warning(result.warnings[-1])
        return result

    if n_cases < 5 or n_controls < 5:
        result.warnings.append(
            f"Sbilanciamento eccessivo: {n_cases} casi / {n_controls} controlli. "
            f"Servono almeno 5 per gruppo."
        )
        log.warning(result.warnings[-1])
        return result

    log.info(f"Dataset: {n_patients} pazienti ({n_cases} casi, {n_controls} controlli)")

    # ── Soglie adattive ───────────────────────────────────────────
    T = _adaptive_thresholds(n_patients)
    log.info(f"Soglie adattive per N={n_patients}: FDR α={T['fdr_alpha']}, "
             f"min_carriers_snp={T['min_carriers_snp']}")

    # ── Separa features per tipo ──────────────────────────────────
    all_snps = [f for f, t in feature_types.items()
                if t == "snp" and f in df.columns and f != outcome_col]
    all_bio = [f for f, t in feature_types.items()
               if t in ("biomarker", "derived", "lifestyle", "prs")
               and f in df.columns and f != outcome_col]

    log.info(f"Features: {len(all_snps)} SNPs, {len(all_bio)} biomarkers/lifestyle")

    if not all_snps and not all_bio:
        result.warnings.append("Nessuna feature SNP o biomarker trovata.")
        return result

    # ── Binarizza ─────────────────────────────────────────────────
    binary: Dict[str, np.ndarray] = {}
    for f in all_snps:
        vals = df[f].values.astype(float)
        if np.isnan(vals).all():
            continue
        vals = np.nan_to_num(vals, nan=0.0)
        binary[f] = binarize(vals, "snp")

    for f in all_bio:
        vals = df[f].values.astype(float)
        if np.isnan(vals).all():
            continue
        vals = np.nan_to_num(vals, nan=np.nanmedian(vals))
        binary[f] = binarize(vals, feature_types.get(f, "biomarker"))

    all_raw_pairs: List[dict] = []

    # ══════════════════════════════════════════════════════════════
    # FASE A: Biomarker × Biomarker
    # ══════════════════════════════════════════════════════════════
    bio_for_pairs = [b for b in all_bio[:T["max_bio_pairs"]] if b in binary]
    n_a = len(list(itertools.combinations(bio_for_pairs, 2)))
    log.info(f"FASE A: {n_a} biomarker×biomarker pairs...")

    phase_a = 0
    for fa, fb in itertools.combinations(bio_for_pairs, 2):
        xa, xb = binary[fa], binary[fb]
        if xa.sum() < T["min_carriers_bio"] or xb.sum() < T["min_carriers_bio"]:
            continue

        ig_a = information_gain(xa, y)
        ig_b = information_gain(xb, y)
        x_ab = xa * 2 + xb
        ig_ab = information_gain(x_ab, y)
        ii = ig_ab - ig_a - ig_b

        if ii > T["ii_threshold_pairs"]:
            p_perm = permutation_test(x_ab, y, ig_ab, T["n_perm"])
            both = (xa == 1) & (xb == 1)
            neither = (xa == 0) & (xb == 0)
            cr = float(y[both].mean()) if both.sum() >= 3 else 0.0
            baseline = float(y[neither].mean()) if neither.sum() > 5 else float(y.mean())

            phase_a += 1
            all_raw_pairs.append({
                "markers": [fa, fb], "order": 2, "ii": ii, "ig": ig_ab,
                "p_raw": p_perm, "cond_risk": cr,
                "marginal": {
                    fa: float(y[xa == 1].mean()) if (xa == 1).sum() > 0 else 0,
                    fb: float(y[xb == 1].mean()) if (xb == 1).sum() > 0 else 0,
                },
                "baseline_risk": baseline,
                "n_carriers": int(both.sum()),
                "types": f"{feature_types.get(fa, 'bio')}×{feature_types.get(fb, 'bio')}",
                "phase": "A_bio×bio",
            })

    result.phases_summary["A_bio×bio"] = phase_a
    log.info(f"  ✓ Fase A: {phase_a} candidate pairs")

    # ══════════════════════════════════════════════════════════════
    # FASE B: SNP × Biomarker (FORZATO)
    # ══════════════════════════════════════════════════════════════
    bio_for_cross = [b for b in all_bio[:T["max_bio_cross"]] if b in binary]
    snps_with_data = [s for s in all_snps if s in binary]
    n_b = len(snps_with_data) * len(bio_for_cross)
    log.info(f"FASE B: {n_b} SNP×biomarker FORCED cross-pairs...")

    phase_b = 0
    for snp in snps_with_data:
        x_snp = binary[snp]
        n_carriers = int(x_snp.sum())
        if n_carriers < T["min_carriers_snp"]:
            continue

        for bio in bio_for_cross:
            x_bio = binary[bio]

            ig_snp = information_gain(x_snp, y)
            ig_bio = information_gain(x_bio, y)
            x_combined = x_snp * 2 + x_bio
            ig_combined = information_gain(x_combined, y)
            ii = ig_combined - ig_snp - ig_bio

            if ii > T["ii_threshold_pairs"]:
                both = (x_snp == 1) & (x_bio == 1)
                neither = (x_snp == 0) & (x_bio == 0)

                if both.sum() < 2:
                    continue

                cr = float(y[both].mean())
                baseline = float(y[neither].mean()) if neither.sum() > 5 else float(y.mean())
                risk_amp = cr / max(baseline, 0.01)

                if risk_amp > T["risk_amp_threshold"] or ii > T["ii_threshold_pairs"] * 2:
                    p_perm = permutation_test(x_combined, y, ig_combined, T["n_perm"])
                    phase_b += 1
                    all_raw_pairs.append({
                        "markers": [snp, bio], "order": 2, "ii": ii, "ig": ig_combined,
                        "p_raw": p_perm, "cond_risk": cr,
                        "marginal": {
                            snp: float(y[x_snp == 1].mean()) if (x_snp == 1).sum() > 0 else 0,
                            bio: float(y[x_bio == 1].mean()) if (x_bio == 1).sum() > 0 else 0,
                        },
                        "baseline_risk": float(baseline),
                        "n_carriers": int(both.sum()),
                        "types": f"snp×{feature_types.get(bio, 'bio')}",
                        "phase": "B_snp×bio",
                    })

    result.phases_summary["B_snp×bio"] = phase_b
    log.info(f"  ✓ Fase B: {phase_b} candidate SNP×bio pairs")

    # ══════════════════════════════════════════════════════════════
    # FASE C: SNP × SNP
    # ══════════════════════════════════════════════════════════════
    # Solo SNPs con abbastanza carriers
    common_snps = [s for s in snps_with_data
                   if binary[s].sum() >= max(T["min_carriers_snp"], 5)]
    n_c = len(list(itertools.combinations(common_snps, 2)))
    log.info(f"FASE C: {n_c} SNP×SNP pairs...")

    phase_c = 0
    for sa, sb in itertools.combinations(common_snps, 2):
        xa, xb = binary[sa], binary[sb]

        ig_a = information_gain(xa, y)
        ig_b = information_gain(xb, y)
        x_ab = xa * 2 + xb
        ig_ab = information_gain(x_ab, y)
        ii = ig_ab - ig_a - ig_b

        if ii > T["ii_threshold_pairs"]:
            both = (xa == 1) & (xb == 1)
            if both.sum() >= 2:
                cr = float(y[both].mean())
                p_perm = permutation_test(x_ab, y, ig_ab, T["n_perm"])
                phase_c += 1
                all_raw_pairs.append({
                    "markers": [sa, sb], "order": 2, "ii": ii, "ig": ig_ab,
                    "p_raw": p_perm, "cond_risk": cr,
                    "marginal": {
                        sa: float(y[xa == 1].mean()) if (xa == 1).sum() > 0 else 0,
                        sb: float(y[xb == 1].mean()) if (xb == 1).sum() > 0 else 0,
                    },
                    "n_carriers": int(both.sum()),
                    "types": "snp×snp",
                    "phase": "C_snp×snp",
                })

    result.phases_summary["C_snp×snp"] = phase_c
    log.info(f"  ✓ Fase C: {phase_c} candidate SNP×SNP pairs")

    # ══════════════════════════════════════════════════════════════
    # APPLICA FDR SULLE COPPIE
    # ══════════════════════════════════════════════════════════════
    if all_raw_pairs:
        p_values = [d["p_raw"] for d in all_raw_pairs]
        significant = benjamini_hochberg(p_values, alpha=T["fdr_alpha"])

        for d, sig in zip(all_raw_pairs, significant):
            marg = d["marginal"]
            max_marg = max(marg.values()) if marg else 0.01
            risk_amp = d["cond_risk"] / max(max_marg, 0.01)

            rule = EpistaticRule(
                markers=d["markers"],
                order=d["order"],
                interaction_info=d["ii"],
                information_gain=d["ig"],
                p_value_raw=d["p_raw"],
                p_value_fdr=d["p_raw"],  # approx, il vero adjusted p si calcolerebbe diversamente
                conditional_risk=d["cond_risk"],
                marginal_risks=d["marginal"],
                risk_amplification=risk_amp,
                n_carriers=d["n_carriers"],
                types=d["types"],
                phase=d["phase"],
                significant=sig,
            )
            result.pairs.append(rule)

        n_sig = sum(1 for r in result.pairs if r.significant)
        log.info(f"  FDR: {n_sig}/{len(result.pairs)} pairs sopravvivono (α={T['fdr_alpha']})")
    else:
        log.info("  Nessuna coppia candidata trovata (dataset troppo piccolo o omogeneo)")
        result.warnings.append("Nessuna coppia epistatica candidata trovata.")

    # ══════════════════════════════════════════════════════════════
    # FASE D: TRIPLE (solo se abbastanza dati)
    # ══════════════════════════════════════════════════════════════
    if n_patients < 50:
        result.warnings.append(
            f"Triple saltate: N={n_patients} troppo piccolo (servono ≥50)."
        )
        log.info(f"  FASE D: SKIP (N={n_patients} < 50)")
    else:
        sig_markers = set()
        for r in result.pairs:
            if r.significant:
                sig_markers.update(r.markers)

        # Aggiungi context features
        for cf in ["smoking", "age", "ecog", "weight", "nlr", "sii", "crp", "ldh"]:
            if cf in binary:
                sig_markers.add(cf)

        mlist = [m for m in sig_markers if m in binary]
        sig_snps = [m for m in mlist if feature_types.get(m, "") == "snp"]
        sig_other = [m for m in mlist if feature_types.get(m, "") != "snp"]

        log.info(f"FASE D: Triple from {len(sig_snps)} SNPs × {len(sig_other)} other markers")

        all_raw_trips: List[dict] = []
        max_triplets = 20

        for snp in sig_snps:
            if len(all_raw_trips) >= max_triplets:
                break
            for o1, o2 in itertools.combinations(sig_other, 2):
                if len(all_raw_trips) >= max_triplets:
                    break

                xa, xb, xc = binary[snp], binary[o1], binary[o2]
                if xa.sum() < T["min_carriers_snp"]:
                    continue
                if xb.sum() < T["min_carriers_bio"] or xc.sum() < T["min_carriers_bio"]:
                    continue

                x_abc = xa * 4 + xb * 2 + xc
                ig_abc = information_gain(x_abc, y)

                best_pair = max(
                    information_gain(xa * 2 + xb, y),
                    information_gain(xb * 2 + xc, y),
                    information_gain(xa * 2 + xc, y),
                )
                epistasis = ig_abc - best_pair

                if epistasis > T["ii_threshold_triples"]:
                    all3 = (xa == 1) & (xb == 1) & (xc == 1)
                    if all3.sum() >= 2:
                        cr = float(y[all3].mean())
                        p_perm = permutation_test(x_abc, y, ig_abc, min(T["n_perm"], 200))
                        all_raw_trips.append({
                            "markers": [snp, o1, o2], "order": 3,
                            "ii": epistasis, "ig": ig_abc,
                            "p_raw": p_perm, "cond_risk": cr,
                            "marginal": {
                                snp: float(y[xa == 1].mean()) if (xa == 1).sum() > 0 else 0,
                                o1: float(y[xb == 1].mean()) if (xb == 1).sum() > 0 else 0,
                                o2: float(y[xc == 1].mean()) if (xc == 1).sum() > 0 else 0,
                            },
                            "n_carriers": int(all3.sum()),
                            "types": f"snp×{feature_types.get(o1, '?')}×{feature_types.get(o2, '?')}",
                            "phase": "D_triple",
                        })

        # FDR sui triplet
        if all_raw_trips:
            p_vals_t = [d["p_raw"] for d in all_raw_trips]
            sig_t = benjamini_hochberg(p_vals_t, alpha=T["fdr_alpha"])

            for d, sig in zip(all_raw_trips, sig_t):
                marg = d["marginal"]
                max_marg = max(marg.values()) if marg else 0.01
                rule = EpistaticRule(
                    markers=d["markers"], order=3,
                    interaction_info=d["ii"], information_gain=d["ig"],
                    p_value_raw=d["p_raw"], p_value_fdr=d["p_raw"],
                    conditional_risk=d["cond_risk"],
                    marginal_risks=d["marginal"],
                    risk_amplification=d["cond_risk"] / max(max_marg, 0.01),
                    n_carriers=d["n_carriers"],
                    types=d["types"], phase=d["phase"],
                    significant=sig,
                )
                result.triplets.append(rule)

            n_sig_t = sum(1 for r in result.triplets if r.significant)
            log.info(f"  ✓ Fase D: {n_sig_t}/{len(result.triplets)} triplets sopravvivono FDR")
        else:
            log.info("  Fase D: nessun triplet candidato trovato")

        result.phases_summary["D_triple"] = len([r for r in result.triplets if r.significant])

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    total_sig = len(result.all_rules)
    by_phase = Counter(r.phase for r in result.all_rules)
    log.info(f"\n  ═══ DISCOVERY COMPLETE ═══")
    log.info(f"  Significant rules (post-FDR): {total_sig}")
    log.info(f"  By phase: {dict(by_phase)}")

    if total_sig == 0:
        result.warnings.append(
            f"Nessuna regola sopravvive alla correzione FDR con N={n_patients}. "
            f"Il database crescerà e le regole emergeranno. "
            f"discovered_rules.json conterrà []."
        )
        log.info(f"  ⚠ {result.warnings[-1]}")

    # Sort by interaction info (decrescente)
    result.pairs.sort(key=lambda r: r.interaction_info, reverse=True)
    result.triplets.sort(key=lambda r: r.interaction_info, reverse=True)

    return result
