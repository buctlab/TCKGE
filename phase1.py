
from __future__ import annotations

import argparse

import importlib

import importlib.metadata as importlib_metadata

import json

import os

import shutil

import subprocess

import sys

import time

from dataclasses import dataclass

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

ENV_PYTHON = os.environ.get("TCKGE_PYTHON")

TARGETS = [

    "wn18rr-transe",

    "wn18rr-rotate",

    "wn18rr-complex",

    "fb15k237-complex",

    "yago310-complex",

    "fb15k237-transe",

    "fb15k237-nodepiece",

]

@dataclass(frozen=True)

class WN18RRConfig:

    target: str

    model: str

    output_prefix: str

    id_map_name: str

    embedding_dim: int = 100

    epochs: int = 200

    batch_size: int = 512

    lr: float = 1e-3

    negative_samples: int | None = None

WN18RR_CONFIGS = {

    "wn18rr-transe": WN18RRConfig(

        target="wn18rr-transe",

        model="TransE",

        output_prefix="wn18rr_transe",

        id_map_name="wn18rr_entity_id_map.json",

        batch_size=1024,

        lr=1e-3,

        negative_samples=1024,

    ),

    "wn18rr-rotate": WN18RRConfig(

        target="wn18rr-rotate",

        model="RotatE",

        output_prefix="wn18rr_rotate",

        id_map_name="wn18rr_rotate_entity_id_map.json",

        batch_size=512,

        lr=5e-4,

    ),

    "wn18rr-complex": WN18RRConfig(

        target="wn18rr-complex",

        model="ComplEx",

        output_prefix="wn18rr_complex",

        id_map_name="wn18rr_complex_entity_id_map.json",

        batch_size=512,

        lr=1e-3,

    ),

}

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

def write_json(path: Path, payload: object) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:

        json.dump(payload, handle, ensure_ascii=False, indent=2)

def run_command(cmd: list[str], label: str, dry_run: bool) -> dict:

    print("\n" + "=" * 78)

    print(label)

    print("=" * 78)

    print("Command:", " ".join(cmd))

    if dry_run:

        return {"label": label, "command": cmd, "status": "dry-run"}

    started = time.time()

    subprocess.run(cmd, cwd=ROOT, check=True)

    return {"label": label, "command": cmd, "status": "completed", "elapsed_s": round(time.time() - started, 2)}

def output_exists(paths: list[Path]) -> bool:

    return all(path.exists() for path in paths)

def check_training_dependencies(targets: list[str], dry_run: bool) -> dict[str, str]:

    if dry_run or not targets:

        return {}

    required = {

        "torch": "torch",

        "pykeen": "pykeen.pipeline",

    }

    versions: dict[str, str] = {}

    failures: list[str] = []

    for package, module in required.items():

        try:

            importlib.import_module(module)

            versions[package] = importlib_metadata.version(package)

        except Exception as exc:

            failures.append(f"{package} ({exc.__class__.__name__}: {exc})")

    if failures:

        joined = "; ".join(failures)

        raise SystemExit(

            "Missing or incompatible Phase 1 training dependencies: "

            f"{joined}\nInstall them in the GPU environment with:\n"

            "  python -m pip install -r requirements.txt"

        )

    return versions

def backup_outputs(paths: list[Path], dry_run: bool) -> list[str]:

    existing = [path for path in paths if path.exists()]

    if not existing:

        return []

    backup_dir = ROOT / "reproducibility" / "backups" / "phase1" / stamp()

    print(f"Backing up {len(existing)} phase-1 outputs to {rel(backup_dir)}")

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

def read_id_mapping(path: Path) -> dict[str, int]:

    mapping: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as handle:

        _ = handle.readline()

        for line in handle:

            parts = line.strip().split()

            if len(parts) >= 2:

                mapping[parts[0]] = int(parts[1])

    return mapping

def read_id_triples(path: Path) -> np.ndarray:

    triples: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8") as handle:

        _ = handle.readline()

        for line in handle:

            parts = line.strip().split()

            if len(parts) == 3:

                head, tail, relation = parts

                triples.append((int(head), int(relation), int(tail)))

    return np.asarray(triples, dtype=np.int32)

def id_triples_to_strings(triples: np.ndarray, id_to_entity: dict[int, str], id_to_relation: dict[int, str]) -> np.ndarray:

    rows = [(id_to_entity[int(h)], id_to_relation[int(r)], id_to_entity[int(t)]) for h, r, t in triples]

    return np.asarray(rows, dtype=str)

def real_matrix(raw: np.ndarray) -> np.ndarray:

    if np.iscomplexobj(raw):

        return np.concatenate([raw.real, raw.imag], axis=1).astype(np.float32)

    return raw.astype(np.float32)

def train_wn18rr(config: WN18RRConfig, data_dir: Path, force: bool, dry_run: bool) -> dict:

    outputs = [

        data_dir / f"{config.output_prefix}_entity.npy",

        data_dir / f"{config.output_prefix}_relation.npy",

        data_dir / config.id_map_name,

    ]

    if output_exists(outputs) and not force:

        print(f"Skipping {config.target}; outputs already exist.")

        return {"target": config.target, "status": "skipped-existing", "outputs": [rel(path) for path in outputs]}

    if dry_run:

        return {"target": config.target, "status": "dry-run", "outputs": [rel(path) for path in outputs]}

    import torch

    from pykeen.pipeline import pipeline

    from pykeen.triples import TriplesFactory

    entity_to_id = read_id_mapping(data_dir / "entity2id.txt")

    relation_to_id = read_id_mapping(data_dir / "relation2id.txt")

    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}

    id_to_relation = {idx: relation for relation, idx in relation_to_id.items()}

    train = id_triples_to_strings(read_id_triples(data_dir / "train2id.txt"), id_to_entity, id_to_relation)

    valid = id_triples_to_strings(read_id_triples(data_dir / "valid2id.txt"), id_to_entity, id_to_relation)

    test = id_triples_to_strings(read_id_triples(data_dir / "test2id.txt"), id_to_entity, id_to_relation)

    training = TriplesFactory.from_labeled_triples(triples=train)

    validation = TriplesFactory.from_labeled_triples(triples=valid, entity_to_id=training.entity_to_id, relation_to_id=training.relation_to_id)

    testing = TriplesFactory.from_labeled_triples(triples=test, entity_to_id=training.entity_to_id, relation_to_id=training.relation_to_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {

        "training": training,

        "validation": validation,

        "testing": testing,

        "model": config.model,

        "model_kwargs": {"embedding_dim": config.embedding_dim},

        "optimizer": "Adam",

        "optimizer_kwargs": {"lr": config.lr},

        "training_kwargs": {

            "num_epochs": config.epochs,

            "batch_size": config.batch_size,

            "checkpoint_name": f"{config.output_prefix}_checkpoint.pt",

            "checkpoint_frequency": 50,

        },

        "evaluation_kwargs": {"batch_size": config.batch_size},

        "random_seed": 42,

        "device": device,

    }

    if config.negative_samples is not None:

        kwargs["negative_sampler_kwargs"] = {"num_negs_per_pos": config.negative_samples}

    print("\n" + "=" * 78)

    print(f"Training {config.model} on WN18RR with device={device}")

    print("=" * 78)

    started = time.time()

    result = pipeline(**kwargs)

    model = result.model

    entity = real_matrix(model.entity_representations[0](indices=None).detach().cpu().numpy())

    relation = real_matrix(model.relation_representations[0](indices=None).detach().cpu().numpy())

    np.save(outputs[0], entity)

    np.save(outputs[1], relation)

    write_json(outputs[2], training.entity_to_id)

    return {

        "target": config.target,

        "status": "completed",

        "elapsed_s": round(time.time() - started, 2),

        "device": device,

        "outputs": [rel(path) for path in outputs],

        "entity_shape": list(entity.shape),

        "relation_shape": list(relation.shape),

    }

def train_fb15k237_transe(data_dir: Path, epochs: int, force: bool, dry_run: bool) -> dict:

    outputs = [data_dir / "fb15k237_transe_entity.npy", data_dir / "fb15k237_transe_entity_id_map.json", data_dir / "fb15k237_train_triples.npy"]

    if output_exists(outputs) and not force:

        print("Skipping fb15k237-transe; outputs already exist.")

        return {"target": "fb15k237-transe", "status": "skipped-existing", "outputs": [rel(path) for path in outputs]}

    if dry_run:

        return {"target": "fb15k237-transe", "status": "dry-run", "outputs": [rel(path) for path in outputs]}

    from run_fb15k237_transe import train_transe_fb15k237

    started = time.time()

    train_transe_fb15k237(epochs=epochs)

    return {"target": "fb15k237-transe", "status": "completed", "elapsed_s": round(time.time() - started, 2), "outputs": [rel(path) for path in outputs]}

def external_target(target: str, data_dir: Path, epochs: int) -> tuple[list[str], list[Path]]:

    python = str(preferred_python())

    if target == "fb15k237-complex":

        return [python, "train_complex_dataset.py", "--dataset", "fb15k237", "--epochs", str(epochs), "--outdir", str(data_dir)], [data_dir / "fb15k237_complex_entity.npy", data_dir / "fb15k237_complex_entity_id_map.json"]

    if target == "yago310-complex":

        return [python, "train_complex_dataset.py", "--dataset", "yago310", "--epochs", str(epochs), "--outdir", str(data_dir)], [data_dir / "yago310_complex_entity.npy", data_dir / "yago310_complex_entity_id_map.json"]

    if target == "fb15k237-nodepiece":

        return [python, "train_fb15k237_nodepiece.py"], [data_dir / "fb15k237_nodepiece" / "entity.npy", data_dir / "fb15k237_nodepiece" / "entity_id_map.json"]

    raise ValueError(target)

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Phase 1 training for the TCKGE reproducibility package")

    parser.add_argument("--data-dir", default="Data")

    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)

    parser.add_argument("--force", action="store_true", help="Re-train even if target outputs exist")

    parser.add_argument("--no-backup", action="store_true", help="Do not back up existing phase-1 outputs before --force")

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--wn18rr-epochs", type=int, default=200)

    parser.add_argument("--cross-epochs", type=int, default=100)

    parser.add_argument("--fb15k237-transe-epochs", type=int, default=200)

    parser.add_argument("--quick", action="store_true", help="Use one epoch for orchestration smoke tests")

    return parser.parse_args()

def main() -> None:

    ensure_gpu_python()

    args = parse_args()

    if args.quick:

        args.wn18rr_epochs = 1

        args.cross_epochs = 1

        args.fb15k237_transe_epochs = 1

    dependency_versions = check_training_dependencies(args.targets, args.dry_run)

    data_dir = (ROOT / args.data_dir).resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = ROOT / "reproducibility" / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for config in WN18RR_CONFIGS.values():

        all_outputs.extend([data_dir / f"{config.output_prefix}_entity.npy", data_dir / f"{config.output_prefix}_relation.npy", data_dir / config.id_map_name])

    all_outputs.extend([

        data_dir / "fb15k237_complex_entity.npy",

        data_dir / "fb15k237_complex_entity_id_map.json",

        data_dir / "yago310_complex_entity.npy",

        data_dir / "yago310_complex_entity_id_map.json",

        data_dir / "fb15k237_transe_entity.npy",

        data_dir / "fb15k237_transe_entity_id_map.json",

        data_dir / "fb15k237_nodepiece",

    ])

    manifest = {"phase": "phase1", "started_at": stamp(), "python": sys.executable, "dependencies": dependency_versions, "data_dir": rel(data_dir), "targets": args.targets, "items": []}

    if args.force and not args.no_backup:

        manifest["backups"] = backup_outputs(all_outputs, args.dry_run)

    for target in args.targets:

        if target in WN18RR_CONFIGS:

            base = WN18RR_CONFIGS[target]

            config = WN18RRConfig(**{**base.__dict__, "epochs": args.wn18rr_epochs})

            manifest["items"].append(train_wn18rr(config, data_dir, args.force, args.dry_run))

        elif target == "fb15k237-transe":

            manifest["items"].append(train_fb15k237_transe(data_dir, args.fb15k237_transe_epochs, args.force, args.dry_run))

        else:

            cmd, outputs = external_target(target, data_dir, args.cross_epochs)

            if output_exists(outputs) and not args.force:

                print(f"Skipping {target}; outputs already exist.")

                manifest["items"].append({"target": target, "status": "skipped-existing", "outputs": [rel(path) for path in outputs]})

            else:

                item = run_command(cmd, f"Training {target}", args.dry_run)

                item["target"] = target

                item["outputs"] = [rel(path) for path in outputs]

                manifest["items"].append(item)

    manifest_path = logs_dir / f"phase1_manifest_{stamp()}.json"

    if not args.dry_run:

        write_json(manifest_path, manifest)

    print("\nPhase 1 complete.")

    print(f"Manifest: {rel(manifest_path)}" if not args.dry_run else "Dry-run: manifest not written")

if __name__ == "__main__":

    main()
