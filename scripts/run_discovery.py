#!/usr/bin/env python3
"""
PROMETHEUS Discovery Runner
============================
Lancia la pipeline di discovery sul database pazienti SENTINEL.

Uso:
    python scripts/run_discovery.py
    python scripts/run_discovery.py --data data/patients/ --output data/discovered_rules.json
    python scripts/run_discovery.py --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from prometheus.feature_engineering import load_patient_database
from prometheus.epistatic_engine import discover_epistatic


def main():
    parser = argparse.ArgumentParser(description="PROMETHEUS Epistatic Discovery Runner")
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "patients"),
                        help="Directory con i JSON pazienti")
    parser.add_argument("--output", default=str(BASE_DIR / "data" / "discovered_rules.json"),
                        help="File output per le regole scoperte")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Output dettagliato")
    args = parser.parse_args()

    # Logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [PROMETHEUS] %(message)s",
        datefmt="%H:%M:%S"
    )
    log = logging.getLogger("prometheus")
    log.setLevel(logging.INFO)  # Sempre almeno INFO per il runner

    print()
    print("█" * 70)
    print("█  PROMETHEUS v1.0 — Epistatic Discovery Engine for SENTINEL       █")
    print("█" * 70)
    print()

    start = time.time()

    # 1. Carica database pazienti
    print(f"  📂 Loading patients from: {args.data}")
    df, feat_types, patient_ids = load_patient_database(args.data)

    if df.empty:
        print(f"  ⚠ Nessun paziente trovato in {args.data}")
        print(f"  Il file {args.output} conterrà []")
        # Scrivi array vuoto
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("[]")
        return

    print(f"  ✓ {len(df)} pazienti | {len(df.columns)} features")
    print(f"    SNPs: {sum(1 for t in feat_types.values() if t == 'snp')}")
    print(f"    Biomarkers: {sum(1 for t in feat_types.values() if t == 'biomarker')}")
    print(f"    Derived: {sum(1 for t in feat_types.values() if t == 'derived')}")
    print(f"    PRS: {sum(1 for t in feat_types.values() if t == 'prs')}")

    # 2. Prepara outcome
    if "has_cancer" not in df.columns:
        print("  ⚠ Colonna 'has_cancer' non trovata. Nessuna discovery possibile.")
        with open(args.output, "w") as f:
            f.write("[]")
        return

    y = df["has_cancer"].values.astype(float)
    y = np.nan_to_num(y, nan=0.0)
    n_cases = int(y.sum())
    n_controls = int(len(y) - n_cases)
    print(f"  Outcome: {n_cases} casi / {n_controls} controlli")

    # 3. Discovery
    print()
    print("━" * 70)
    print("  EPISTATIC DISCOVERY")
    print("━" * 70)

    result = discover_epistatic(df, y, feat_types)

    # 4. Report
    elapsed = time.time() - start
    total_rules = len(result.all_rules)

    print()
    print("━" * 70)
    print("  RESULTS")
    print("━" * 70)
    print(f"  Regole significative (post-FDR): {total_rules}")
    print(f"    Coppie: {sum(1 for r in result.pairs if r.significant)}")
    print(f"    Triple: {sum(1 for r in result.triplets if r.significant)}")
    print(f"  Per fase: {result.phases_summary}")
    print(f"  Tempo: {elapsed:.1f}s")

    if total_rules > 0:
        print()
        print("  ★ TOP DISCOVERIES:")
        for i, rule in enumerate(result.all_rules[:10]):
            label = " + ".join(rule.markers)
            print(f"    [{i+1}] {label}")
            print(f"        Risk: {rule.conditional_risk:.0%} | "
                  f"Amp: {rule.risk_amplification:.1f}x | "
                  f"p_fdr: {rule.p_value_fdr:.4f} | "
                  f"N: {rule.n_carriers}")
    else:
        print()
        print("  ℹ Nessuna regola scoperta con i dati attuali.")
        print("    Questo è NORMALE con pochi pazienti.")
        print("    Le regole emergeranno man mano che il database cresce.")

    if result.warnings:
        print()
        print("  ⚠ Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    # 5. Salva
    result.to_json(args.output)
    print()
    print(f"  ✓ Regole salvate in: {args.output}")
    print(f"  Tempo totale: {elapsed:.1f}s")
    print()


# Need numpy for nan handling
import numpy as np

if __name__ == "__main__":
    main()
