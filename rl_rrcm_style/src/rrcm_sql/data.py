import hashlib
import json
from pathlib import Path
import random


def read_json(path):
    return json.loads(Path(path).read_text())


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def serialize_schema(schema):
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]
    pk = set(schema["primary_keys"])
    lines = []
    for tid, table in enumerate(tables):
        definitions = []
        for cid, (owner, name) in enumerate(columns):
            if owner == tid:
                definitions.append(f"  {quote(name)} {schema['column_types'][cid]}")
        table_pk = [quote(name) for cid, (owner, name) in enumerate(columns) if owner == tid and cid in pk]
        if table_pk:
            definitions.append("  PRIMARY KEY (" + ", ".join(table_pk) + ")")
        for source, target in schema["foreign_keys"]:
            if columns[source][0] == tid:
                target_table, target_column = columns[target]
                definitions.append(f"  FOREIGN KEY ({quote(columns[source][1])}) REFERENCES "
                                   f"{quote(tables[target_table])} ({quote(target_column)})")
        lines.append(f"CREATE TABLE {quote(table)} (\n" + ",\n".join(definitions) + "\n);")
    return "\n\n".join(lines)


def database_path(root, db_id):
    root = Path(root).resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Database missing or invalid db_id: {db_id}")
    return path


def prepare(source, output, validation_fraction=0.1, seed=42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    examples = read_json(source)
    dbs = sorted({row["db_id"] for row in examples})
    if len(dbs) < 2:
        raise ValueError("Need at least two databases for a disjoint split")
    random.Random(seed).shuffle(dbs)
    n = max(1, min(len(dbs) - 1, round(len(dbs) * validation_fraction)))
    validation = set(dbs[:n])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, selected in (("train", set(dbs[n:])), ("validation", validation)):
        rows = [dict(row, example_id=str(i)) for i, row in enumerate(examples)
                if row["db_id"] in selected]
        (output / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    manifest = {"source": str(Path(source).resolve()),
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "seed": seed, "train_databases": sorted(dbs[n:]),
                "validation_databases": sorted(validation)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(config, split):
    if config.split_mode == "official":
        split = "dev" if split == "validation" else split
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        rows = read_json(getattr(config, f"{split}_json"))
        if split == "train" and config.train_others_json:
            rows += read_json(config.train_others_json)
        if not rows:
            raise ValueError(f"Empty {split} data")
        return [dict(row, example_id=f"{split}:{i}") for i, row in enumerate(rows)]
    split = "validation" if split == "dev" else split
    if split not in {"train", "validation"}:
        raise ValueError("Legacy internal data supports train/dev only; select official mode for test")
    directory = Path(config.prepared_dir)
    manifest = read_json(directory / "manifest.json")
    train, valid = set(manifest["train_databases"]), set(manifest["validation_databases"])
    if train & valid or not train or not valid:
        raise ValueError("Invalid database-disjoint split manifest")
    rows = read_json(directory / f"{split}.json")
    expected = train if split == "train" else valid
    if not rows or {row["db_id"] for row in rows} != expected:
        raise ValueError("Dataset does not match split manifest")
    return rows


def schema_map(tables):
    return {row["db_id"]: serialize_schema(row) for row in read_json(tables)}


def split_config(cfg, split):
    from copy import deepcopy
    result = deepcopy(cfg)
    if split == "test":
        result.data.database_dir = cfg.data.test_database_dir
        result.data.tables = cfg.data.test_tables
    if split in {"dev", "test"}:
        # Evaluation correctness is reported independently of training reward shaping.
        result.sql.reward_metric = "execution"
        result.sql.suite_database_dir = getattr(cfg.evaluation, f"{split}_suite_database_dir")
    return result


def official_manifest(cfg):
    """Verify official split resources without executing test questions or gold SQL."""
    manifest = {"version": 2, "split_mode": "official", "splits": {}}
    seen = set()
    for split in ("train", "dev", "test"):
        rows = load_split(cfg.data, split)
        dbs = {r["db_id"] for r in rows}
        if seen & dbs:
            raise ValueError(f"Database overlap in {split}: {sorted(seen & dbs)}")
        seen.update(dbs)
        resolved = split_config(cfg, split)
        schemas = schema_map(resolved.data.tables)
        if not dbs <= schemas.keys():
            raise ValueError(f"Missing {split} schemas: {sorted(dbs - schemas.keys())}")
        for db in dbs:
            database_path(resolved.data.database_dir, db)
        sources = [getattr(cfg.data, f"{split}_json")]
        if split == "train" and cfg.data.train_others_json:
            sources.append(cfg.data.train_others_json)
        manifest["splits"][split] = {
            "count": len(rows), "databases": sorted(dbs),
            "sources": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            "tables_sha256": hashlib.sha256(Path(resolved.data.tables).read_bytes()).hexdigest(),
            "database_dir": str(Path(resolved.data.database_dir).resolve()),
        }
    return manifest


def prepare_official(cfg, output):
    manifest = official_manifest(cfg)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
