#!/usr/bin/env python3
"""Compatibility entry point for WN18RR lexical-domain purity.

Historically this filename computed raw synset-name purity. The manuscript now
uses lexical-domain (Supersense) purity, so this script intentionally delegates
to the shared ``synset.lexname()`` evaluator and writes lexical-domain columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wn18rr_lexical_domain_purity import load_id_to_offset, summarize_clusters


def tau_from_name(path: Path) -> float | None:
    match = re.search(r"tau_?([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="directory with idmap and cliques_tau_*.json")
    parser.add_argument("--idmap", default="wn18rr_entity_id_map.json")
    parser.add_argument("--out", default="wn18rr_purity_summary.csv")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    id_to_offset = load_id_to_offset(data_dir / args.idmap)
    rows = []

    for path in sorted(data_dir.glob("cliques_tau*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            clusters = json.load(handle)
        if isinstance(clusters, dict):
            clusters = list(clusters.values())
        _, summary = summarize_clusters(clusters, id_to_offset)
        rows.append(
            {
                "tau": tau_from_name(path),
                "avg_lexical_domain_purity": summary["avg_lexical_domain_purity"],
                "perfect_lexical_domain_ratio": summary["perfect_lexical_domain_ratio"],
                "n_comparable_cliques": summary["n_eval"],
                "n_total_cliques": summary["n_total"],
                "n_perfect": summary["n_perfect"],
            }
        )

    rows.sort(key=lambda row: row["tau"] if row["tau"] is not None else -1)
    out_path = data_dir / args.out
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "tau",
            "avg_lexical_domain_purity",
            "perfect_lexical_domain_ratio",
            "n_comparable_cliques",
            "n_total_cliques",
            "n_perfect",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote lexical-domain purity summary to {out_path}")
    for row in rows:
        print(
            f"tau={row['tau']:.2f} total={row['n_total_cliques']} "
            f"eval={row['n_comparable_cliques']} "
            f"avg={row['avg_lexical_domain_purity']:.4f} "
            f"perfect={row['perfect_lexical_domain_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()