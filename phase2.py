
from __future__ import annotations

import argparse

import json

import os

import shutil

import subprocess

import sys

import time

from pathlib import Path

ROOT = Path(__file__).resolve().parent

ENV_PYTHON = os.environ.get("TCKGE_PYTHON")

TAUS = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]

TAUS_BASELINE = [0.45, 0.50, 0.55]

TAUS_EXTERNAL = [0.45, 0.50, 0.55, 0.60, 0.65]

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

    backup_dir = ROOT / "reproducibility" / "backups" / "phase2" / stamp()

    print(f"Backing up {len(existing)} phase-2 outputs to {rel(backup_dir)}")

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

def require(path: Path, allow_missing: bool) -> bool:

    if path.exists():

        return True

    if allow_missing:

        print(f"WARNING: missing required input: {rel(path)}")

        return False

    raise FileNotFoundError(f"Missing required input: {rel(path)}")

def tolerance_command(embed: Path, idmap: Path, outdir: Path, model_name: str) -> list[str]:

    return [python_cmd(), "tolerance_kg_full_pipeline.py", "--embed", str(embed), "--idmap", str(idmap), "--outdir", str(outdir), "--model_name", model_name, "--taus", *[str(tau) for tau in TAUS], "--topk_for_weight", "50"]

def sync_transe_compat(data_dir: Path, dry_run: bool) -> dict:

    source_dir = data_dir / "transe_results"

    copied: list[tuple[str, str]] = []

    for tau in TAUS:

        tag = f"{tau:.3f}"

        pairs = [

            (source_dir / f"cliques_transe_tau_{tag}.json", data_dir / f"cliques_tau_{tag}.json"),

            (source_dir / f"louvain_transe_tau_{tag}.csv", data_dir / f"louvain_tau_{tag}.csv"),

            (source_dir / f"leiden_transe_tau_{tag}.csv", data_dir / f"leiden_tau_{tag}.csv"),

        ]

        for src, dst in pairs:

            if src.exists():

                print(f"Sync {rel(src)} -> {rel(dst)}")

                if not dry_run:

                    shutil.copy2(src, dst)

                copied.append((rel(src), rel(dst)))

    summary = source_dir / "summary_robustness_transe.csv"

    if summary.exists():

        dst = data_dir / "summary_robustness.csv"

        print(f"Sync {rel(summary)} -> {rel(dst)}")

        if not dry_run:

            shutil.copy2(summary, dst)

        copied.append((rel(summary), rel(dst)))

    return {"label": "TransE legacy compatibility sync", "status": "completed", "copied": copied}

def cosine_tail_command() -> list[str]:

    code = (

        "from pathlib import Path; import pandas as pd; "

        "from run_fb15k237_transe import compute_99th_cosine; "

        "d=Path('Data'); specs=["

        "('fb15k237_transe_entity.npy','TransE / FB15k-237'),"

        "('fb15k237_complex_entity.npy','ComplEx / FB15k-237'),"

        "('yago310_complex_entity.npy','ComplEx / YAGO3-10'),"

        "('wn18rr_transe_entity.npy','TransE / WN18RR'),"

        "('wn18rr_complex_entity.npy','ComplEx / WN18RR')]; "

        "rows=[compute_99th_cosine(str(d/p), label) for p,label in specs if (d/p).exists()]; "

        "pd.DataFrame(rows).to_csv(d/'cosine_similarity_comparison.csv', index=False)"

    )

    return [python_cmd(), "-c", code]

def dense_ablation_command(data_dir: Path) -> list[str]:

    return [

        python_cmd(), "Data/wn18rr_full_analysis.py",

        "--embed", str(data_dir / "wn18rr_transe_entity.npy"),

        "--idmap", str(data_dir / "wn18rr_entity_id_map.json"),

        "--out", str(data_dir / "wn18rr_analysis_out"),

        "--taus", *[str(tau) for tau in TAUS],

    ]

def external_coherence_commands(data_dir: Path) -> list[tuple[list[str], str]]:

    commands: list[tuple[list[str], str]] = []

    datasets = [

        ("fb15k237", data_dir / "fb15k237_results", data_dir / "fb15k237_train_triples.npy", data_dir / "fb15k237_relation_id_map.json", data_dir / "fb15k237_complex_entity_id_map.json"),

        ("yago310", data_dir / "yago310_results", data_dir / "yago310_train_triples.npy", data_dir / "yago310_relation_id_map.json", data_dir / "yago310_complex_entity_id_map.json"),

    ]

    for dataset, result_dir, triples, rel_id, ent_id in datasets:

        for tau in TAUS_EXTERNAL:

            commands.append(([

                python_cmd(), "experiments/external_type_coherence_fb15k237.py",

                "--triples", str(triples), "--rel_id", str(rel_id), "--ent_id", str(ent_id),

                "--cliques", str(result_dir / f"cliques_complex_tau_{tau:.3f}.json"),

                "--tau", str(tau), "--dataset", dataset,

                "--out", str(result_dir / f"external_coherence_tau_{tau:.3f}.csv"),

            ], f"External coherence {dataset} tau={tau:.2f}"))

    return commands

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Phase 2 analysis for the TCKGE reproducibility package")

    parser.add_argument("--data-dir", default="Data")

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--no-backup", action="store_true")

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--allow-missing", action="store_true")

    parser.add_argument("--skip-baselines", action="store_true")

    parser.add_argument("--skip-cross-benchmark", action="store_true")

    parser.add_argument("--skip-external-coherence", action="store_true")

    parser.add_argument("--quick-baselines", action="store_true")

    return parser.parse_args()

def main() -> None:

    ensure_gpu_python()

    args = parse_args()

    data_dir = (ROOT / args.data_dir).resolve()

    logs_dir = ROOT / "reproducibility" / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"phase": "phase2", "started_at": stamp(), "python": sys.executable, "data_dir": rel(data_dir), "items": []}

    outputs = [data_dir / name for name in ["transe_results", "rotate_results", "complex_results", "fb15k237_results", "yago310_results", "fb15k237_transe_results", "fb15k237_nodepiece_results", "cpm_results", "wn18rr_analysis_out", "fuzzy_mmsb_baselines.csv", "wn18rr_lexical_domain_purity_recomputed.csv", "cutoff_sensitivity.csv", "variance_louvain_transe.csv", "variance_louvain_transe_summary.csv", "cosine_similarity_comparison.csv"]]

    outputs.extend(data_dir / f"cliques_tau_{tau:.3f}.json" for tau in TAUS)

    outputs.extend(data_dir / f"louvain_tau_{tau:.3f}.csv" for tau in TAUS)

    outputs.extend(data_dir / f"leiden_tau_{tau:.3f}.csv" for tau in TAUS)

    outputs.append(data_dir / "summary_robustness.csv")

    if args.force:

        if not args.no_backup:

            manifest["backups"] = backup_outputs(outputs, args.dry_run)

        remove_outputs(outputs, args.dry_run)

    jobs = [

        ("WN18RR TransE tolerance extraction", data_dir / "wn18rr_transe_entity.npy", data_dir / "wn18rr_entity_id_map.json", data_dir / "transe_results", "transe"),

        ("WN18RR RotatE tolerance extraction", data_dir / "wn18rr_rotate_entity.npy", data_dir / "wn18rr_rotate_entity_id_map.json", data_dir / "rotate_results", "rotate"),

        ("WN18RR ComplEx tolerance extraction", data_dir / "wn18rr_complex_entity.npy", data_dir / "wn18rr_complex_entity_id_map.json", data_dir / "complex_results", "complex"),

    ]

    if not args.skip_cross_benchmark:

        jobs.extend([

            ("FB15k-237 ComplEx tolerance extraction", data_dir / "fb15k237_complex_entity.npy", data_dir / "fb15k237_complex_entity_id_map.json", data_dir / "fb15k237_results", "complex"),

            ("YAGO3-10 ComplEx tolerance extraction", data_dir / "yago310_complex_entity.npy", data_dir / "yago310_complex_entity_id_map.json", data_dir / "yago310_results", "complex"),

            ("FB15k-237 TransE tolerance extraction", data_dir / "fb15k237_transe_entity.npy", data_dir / "fb15k237_transe_entity_id_map.json", data_dir / "fb15k237_transe_results", "transe"),

            ("FB15k-237 NodePiece tolerance extraction", data_dir / "fb15k237_nodepiece" / "entity.npy", data_dir / "fb15k237_nodepiece" / "entity_id_map.json", data_dir / "fb15k237_nodepiece_results", "nodepiece"),

        ])

    for label, embed, idmap, outdir, model_name in jobs:

        if require(embed, args.allow_missing) and require(idmap, args.allow_missing):

            manifest["items"].append(run_command(tolerance_command(embed, idmap, outdir, model_name), label, args.dry_run))

    manifest["items"].append(sync_transe_compat(data_dir, args.dry_run))

    manifest["items"].append(run_command(dense_ablation_command(data_dir), "WN18RR dense-threshold ablation", args.dry_run, allow_fail=True))

    for model_name, result_dir, idmap in [("transe", data_dir / "transe_results", data_dir / "wn18rr_entity_id_map.json"), ("rotate", data_dir / "rotate_results", data_dir / "wn18rr_rotate_entity_id_map.json"), ("complex", data_dir / "complex_results", data_dir / "wn18rr_complex_entity_id_map.json")]:

        manifest["items"].append(run_command([python_cmd(), "compute_purity_generic.py", "--results_dir", str(result_dir), "--model_name", model_name, "--idmap", str(idmap)], f"WN18RR lexical-domain purity ({model_name})", args.dry_run, allow_fail=True))

    manifest["items"].append(run_command([python_cmd(), "evaluate_wn18rr_lexical_domain_purity.py", "--root", str(ROOT)], "Combined WN18RR lexical-domain recomputation", args.dry_run, allow_fail=True))

    if not args.skip_cross_benchmark:

        for model_name, result_dir, idmap, triples in [("complex", data_dir / "fb15k237_results", data_dir / "fb15k237_complex_entity_id_map.json", data_dir / "fb15k237_train_triples.npy"), ("complex", data_dir / "yago310_results", data_dir / "yago310_complex_entity_id_map.json", data_dir / "yago310_train_triples.npy"), ("transe", data_dir / "fb15k237_transe_results", data_dir / "fb15k237_transe_entity_id_map.json", data_dir / "fb15k237_train_triples.npy"), ("nodepiece", data_dir / "fb15k237_nodepiece_results", data_dir / "fb15k237_nodepiece" / "entity_id_map.json", data_dir / "fb15k237_train_triples.npy")]:

            manifest["items"].append(run_command([python_cmd(), "compute_purity_relation.py", "--results_dir", str(result_dir), "--model_name", model_name, "--idmap", str(idmap), "--triples", str(triples)], f"Type-coherence purity ({rel(result_dir)})", args.dry_run, allow_fail=True))

    manifest["items"].append(run_command(cosine_tail_command(), "Cosine similarity tail table", args.dry_run, allow_fail=True))

    if not args.skip_baselines:

        manifest["items"].append(run_command([python_cmd(), "experiments/cpm_baseline_wn18rr.py", "--embed", str(data_dir / "wn18rr_transe_entity.npy"), "--idmap", str(data_dir / "wn18rr_entity_id_map.json"), "--taus", *[str(tau) for tau in TAUS_BASELINE], "--kpercs", "3", "4", "5", "--topk", "50", "--out", str(data_dir / "cpm_results")], "CPM overlapping baseline", args.dry_run, allow_fail=True))

        fuzzy_cmd = [python_cmd(), "run_fuzzy_mmsb_baselines.py", "--data-dir", str(data_dir), "--include-mmsb-k1000"]

        if args.quick_baselines:

            fuzzy_cmd.append("--quick")

        manifest["items"].append(run_command(fuzzy_cmd, "Fuzzy and MMSB baselines", args.dry_run, allow_fail=True))

        manifest["items"].append(run_command([python_cmd(), "experiments/cutoff_sensitivity.py", "--embed", str(data_dir / "wn18rr_transe_entity.npy"), "--taus", "0.45", "0.50", "--cmax", "60", "120", "240", "480", "960", "--topk", "50", "--out", str(data_dir / "cutoff_sensitivity.csv")], "Component cutoff sensitivity", args.dry_run, allow_fail=True))

        manifest["items"].append(run_command([python_cmd(), "experiments/multi_seed_variance.py", "algorithmic", "--embed", str(data_dir / "wn18rr_transe_entity.npy"), "--taus", "0.45", "0.50", "0.55", "--topk", "50", "--cmax", "120", "--n_seeds", "5", "--out", str(data_dir / "variance_louvain_transe.csv")], "Extraction determinism check", args.dry_run, allow_fail=True))

        manifest["items"].append(run_command([python_cmd(), "compute_random_baseline.py"], "Chance-level type-coherence baseline", args.dry_run, allow_fail=True))

    if not args.skip_external_coherence and not args.skip_cross_benchmark:

        for cmd, label in external_coherence_commands(data_dir):

            manifest["items"].append(run_command(cmd, label, args.dry_run, allow_fail=True))

    manifest_path = logs_dir / f"phase2_manifest_{stamp()}.json"

    if not args.dry_run:

        write_json(manifest_path, manifest)

    print("\nPhase 2 complete.")

    print(f"Manifest: {rel(manifest_path)}" if not args.dry_run else "Dry-run: manifest not written")

if __name__ == "__main__":

    main()
