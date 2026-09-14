import json
from pathlib import Path


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


def schema_map(tables):
    return {row["db_id"]: serialize_schema(row) for row in read_json(tables)}
