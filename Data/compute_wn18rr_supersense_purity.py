#!/usr/bin/env python3
"""Compute WN18RR lexical-domain (Supersense) purity for TCKGE cliques.

This script resolves 8-digit WN18RR clique members directly as WordNet offsets
and scores their WordNet lexicographer-file labels via ``synset.lexname()``.
It intentionally avoids raw synset-name purity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wn18rr_lexical_domain_purity import load_id_to_offset, summarize_clusters


def tau_from_name(path: Path) -> float:
    return float(path.stem.rsplit("_", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "Data"))
    parser.add_argument("--idmap", default="wn18rr_entity_id_map.json")
    parser.add_argument("--out", default="wn18rr_supersense_purity_summary.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    id_to_offset = load_id_to_offset(data_dir / args.idmap)
    rows = []

    for path in sorted(data_dir.glob("cliques_tau_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            clusters = json.load(handle)
        if isinstance(clusters, dict):
            clusters = list(clusters.values())
        _, summary = summarize_clusters(clusters, id_to_offset)
        rows.append(
            {
                "tau": tau_from_name(path),
                "n_total_cliques": summary["n_total"],
                "n_comparable_cliques": summary["n_eval"],
                "avg_supersense_purity": summary["avg_lexical_domain_purity"],
                "perfect_alignment_ratio": summary["perfect_lexical_domain_ratio"],
                "n_perfect": summary["n_perfect"],
            }
        )

    out_path = data_dir / args.out
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    for row in rows:
        print(
            f"tau={row['tau']:.2f} total={row['n_total_cliques']} "
            f"eval={row['n_comparable_cliques']} "
            f"avg={row['avg_supersense_purity']:.4f} "
            f"perfect={row['perfect_alignment_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()