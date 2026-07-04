#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PYTHON = os.environ.get("TCKGE_PYTHON")
FIGURE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
TABLE_RE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def preferred_python() -> Path:
    return Path(ENV_PYTHON) if ENV_PYTHON else Path(sys.executable)


def ensure_gpu_python() -> None:
    if os.environ.get("TCKGE_NO_REEXEC") == "1" or not ENV_PYTHON:
        return
    target = preferred_python()
    if not target.exists():
        raise SystemExit(f"TCKGE_PYTHON does not exist: {target}")
    if Path(sys.executable).resolve() != target.resolve():
        print(f"Re-running with {target}")
        os.execv(str(target), [str(target), *sys.argv])


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def python_cmd() -> str:
    return str(preferred_python())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_command(cmd: list[str], label: str, dry_run: bool, allow_fail: bool = False) -> dict:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print("Command:", " ".join(cmd))
    if dry_run:
        return {"label": label, "command": cmd, "status": "dry-run"}
    started = time.time()
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return {"label": label, "command": cmd, "status": "completed", "returncode": 0, "elapsed_s": round(time.time() - started, 2)}
    except subprocess.CalledProcessError as exc:
        if not allow_fail:
            raise
        return {"label": label, "command": cmd, "status": "failed-allowed", "returncode": exc.returncode, "elapsed_s": round(time.time() - started, 2)}


def backup_outputs(paths: list[Path], dry_run: bool) -> list[str]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    backup_dir = ROOT / "reproducibility" / "backups" / "phase3" / stamp()
    print(f"Backing up {len(existing)} phase-3 outputs to {rel(backup_dir)}")
    if dry_run:
        return [rel(path) for path in existing]
    copied: list[str] = []
    for path in existing:
        target = backup_dir / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
        copied.append(rel(target))
    return copied


def remove_outputs(paths: list[Path], dry_run: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        print(f"Removing stale output: {rel(path)}")
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def table_sources(data_dir: Path) -> list[dict]:
    sources = [
        ("tab:robust_sparse", data_dir / "summary_robustness.csv"),
        ("tab:robust_dense", data_dir / "wn18rr_analysis_out" / "summary_robustness.csv"),
        ("tab:model_robustness", data_dir / "wn18rr_lexical_domain_purity_recomputed.csv"),
        ("tab:cpm_baseline", data_dir / "cpm_results" / "cpm_summary.csv"),
        ("tab:fuzzy_mmsb", data_dir / "fuzzy_mmsb_baselines.csv"),
        ("tab:cross_benchmark_fb15k237", data_dir / "fb15k237_results" / "coherence_summary_complex.csv"),
        ("tab:cross_benchmark_yago310", data_dir / "yago310_results" / "coherence_summary_complex.csv"),
        ("tab:fb15k237_transe", data_dir / "fb15k237_transe_results" / "coherence_summary_transe.csv"),
        ("tab:nodepiece_fb15k", data_dir / "fb15k237_nodepiece_results" / "coherence_summary_nodepiece.csv"),
        ("tab:cosine_tail", data_dir / "cosine_similarity_comparison.csv"),
    ]
    return [{"label": label, "source": rel(path), "exists": path.exists()} for label, path in sources]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 figure/table generation for the TCKGE reproducibility package")
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-case-studies", action="store_true")
    parser.add_argument("--skip-legacy-tables", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_gpu_python()
    args = parse_args()
    data_dir = (ROOT / args.data_dir).resolve()
    manifests_dir = ROOT / "reproducibility" / "manifests"
    table_dir = ROOT / "reproducibility" / "paper_tables"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    outputs = [ROOT / "figures" / name for name in ["toy_example_tolerance_vs_partition.pdf", "clique_size_distribution.pdf", "comparison_n_cliques.pdf"]]
    outputs.extend([ROOT / "paper_tables.tex", ROOT / "case_studies", table_dir])
    manifest = {"phase": "phase3", "started_at": stamp(), "python": sys.executable,  "items": []}
    if args.force:
        if not args.no_backup:
            manifest["backups"] = backup_outputs(outputs, args.dry_run)
        remove_outputs(outputs, args.dry_run)

    commands = [
        ([python_cmd(), "create_2d_example.py"], "Toy example figure"),
        ([python_cmd(), "plot_transe_clique_size_distribution.py"], "TransE clique-size figure"),
        ([python_cmd(), "plot_model_comparison.py"], "Model comparison figures"),
    ]
    
    if not args.skip_legacy_tables:
        commands.append(([python_cmd(), "generate_tables.py"], "Generated LaTeX table bundle"))
    for cmd, label in commands:
        manifest["items"].append(run_command(cmd, label, args.dry_run, allow_fail=False))

    manifest["table_sources"] = table_sources(data_dir)
    manifest["missing_table_sources"] = [item for item in manifest["table_sources"] if not item["exists"]]
    manifest_path = manifests_dir / f"phase3_outputs_{stamp()}.json"
    if not args.dry_run:
        write_json(manifest_path, manifest)

if __name__ == "__main__":
    main()