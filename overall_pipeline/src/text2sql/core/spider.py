from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

from text2sql.config import SpiderConfig
from text2sql.core.models import SchemaMetadata, SpiderExample, ValidationReport


class SpiderDataError(RuntimeError):
    """Raised when the local Spider data violates the expected contract."""


_SAFE_DB_ID = re.compile(r"^[A-Za-z0-9_]+$")


def _load_json_list(path: Path) -> List[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise SpiderDataError("Required Spider file does not exist: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise SpiderDataError("Invalid JSON in Spider file %s: %s" % (path, exc)) from exc
    if not isinstance(payload, list):
        raise SpiderDataError("Spider file must contain a JSON list: %s" % path)
    return payload


def _sqlite_read_only_uri(path: Path) -> str:
    return "%s?mode=ro&immutable=1" % path.resolve().as_uri()


def _quote_identifier(identifier: str) -> str:
    return '"%s"' % identifier.replace('"', '""')


class SpiderDataset:
    def __init__(self, config: SpiderConfig, *, include_gold: bool = True):
        self.config = config
        self.include_gold = include_gold
        self.root = config.root
        self.examples_path = self.root / config.examples_file
        self.tables_path = self.root / config.tables_file
        self.database_root = self.root / config.database_dir
        self._examples = self._read_examples()
        self._schemas = self._read_schemas()

    def _read_examples(self) -> Sequence[SpiderExample]:
        records = _load_json_list(self.examples_path)
        examples: List[SpiderExample] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise SpiderDataError("Example %d is not an object" % index)
            db_id = record.get("db_id")
            question = record.get("question")
            query = record.get("query") if self.include_gold else ""
            parsed_sql = record.get("sql", {}) if self.include_gold else {}
            if not isinstance(db_id, str) or not _SAFE_DB_ID.fullmatch(db_id):
                raise SpiderDataError("Example %d has an unsafe or invalid db_id" % index)
            if not isinstance(question, str) or not question.strip():
                raise SpiderDataError("Example %d has no question" % index)
            if self.include_gold and (
                not isinstance(query, str) or not query.strip()
            ):
                raise SpiderDataError("Example %d has no gold SQL" % index)
            if not isinstance(parsed_sql, dict):
                raise SpiderDataError("Example %d has invalid parsed SQL metadata" % index)
            examples.append(
                SpiderExample(
                    index=index,
                    split=self.config.split,
                    db_id=db_id,
                    question=question,
                    gold_sql=query,
                    parsed_sql=parsed_sql,
                )
            )
        return tuple(examples)

    def _read_schemas(self) -> Mapping[str, SchemaMetadata]:
        records = _load_json_list(self.tables_path)
        schemas: Dict[str, SchemaMetadata] = {}
        required = (
            "db_id",
            "table_names_original",
            "column_names_original",
            "column_types",
            "primary_keys",
            "foreign_keys",
        )
        for position, record in enumerate(records):
            if not isinstance(record, dict):
                raise SpiderDataError("Schema record %d is not an object" % position)
            missing = [key for key in required if key not in record]
            if missing:
                raise SpiderDataError(
                    "Schema record %d is missing: %s" % (position, ", ".join(missing))
                )
            db_id = record["db_id"]
            if not isinstance(db_id, str) or not _SAFE_DB_ID.fullmatch(db_id):
                raise SpiderDataError("Schema record %d has invalid db_id" % position)
            if db_id in schemas:
                raise SpiderDataError("Duplicate schema metadata for db_id=%s" % db_id)
            schemas[db_id] = SchemaMetadata(
                db_id=db_id,
                table_names=tuple(record["table_names_original"]),
                column_names=tuple(tuple(value) for value in record["column_names_original"]),
                column_types=tuple(record["column_types"]),
                primary_keys=tuple(record["primary_keys"]),
                foreign_keys=tuple(tuple(value) for value in record["foreign_keys"]),
            )
        return schemas

    @property
    def examples(self) -> Sequence[SpiderExample]:
        return self._examples

    @property
    def schemas(self) -> Mapping[str, SchemaMetadata]:
        return self._schemas

    def get_example(self, index: int) -> SpiderExample:
        try:
            return self._examples[index]
        except IndexError as exc:
            raise SpiderDataError("Example index is out of range: %d" % index) from exc

    def get_schema(self, db_id: str) -> SchemaMetadata:
        try:
            return self._schemas[db_id]
        except KeyError as exc:
            raise SpiderDataError("No schema metadata for db_id=%s" % db_id) from exc

    def database_path(self, db_id: str) -> Path:
        if not _SAFE_DB_ID.fullmatch(db_id):
            raise SpiderDataError("Unsafe db_id: %s" % db_id)
        root = self.database_root.resolve()
        path = (root / db_id / (db_id + ".sqlite")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SpiderDataError("Database path escapes the configured root") from exc
        return path

    def _validate_schema_against_sqlite(
        self, db_id: str, metadata: SchemaMetadata, path: Path
    ) -> List[str]:
        errors: List[str] = []
        try:
            connection = sqlite3.connect(_sqlite_read_only_uri(path), uri=True)
        except sqlite3.Error as exc:
            return ["Could not open %s read-only: %s" % (db_id, exc)]
        try:
            actual_tables = {
                row[0].casefold(): row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            columns_by_table: Dict[str, Set[str]] = {}
            for table_name in metadata.table_names:
                actual_name = actual_tables.get(str(table_name).casefold())
                if actual_name is None:
                    errors.append("%s: missing table %s" % (db_id, table_name))
                    continue
                pragma = "PRAGMA table_info(%s)" % _quote_identifier(actual_name)
                columns_by_table[str(table_name).casefold()] = {
                    str(row[1]).casefold() for row in connection.execute(pragma)
                }
            for column_index, column in enumerate(metadata.column_names):
                if len(column) != 2:
                    errors.append("%s: malformed column metadata at %d" % (db_id, column_index))
                    continue
                table_index, column_name = column
                if table_index == -1:
                    continue
                if not isinstance(table_index, int) or not (0 <= table_index < len(metadata.table_names)):
                    errors.append("%s: invalid table index for column %s" % (db_id, column_name))
                    continue
                table_name = str(metadata.table_names[table_index])
                actual_columns = columns_by_table.get(table_name.casefold(), set())
                if str(column_name).casefold() not in actual_columns:
                    errors.append(
                        "%s: missing column %s.%s" % (db_id, table_name, column_name)
                    )
        except sqlite3.Error as exc:
            errors.append("%s: schema inspection failed: %s" % (db_id, exc))
        finally:
            connection.close()
        return errors

    def validate(self) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        db_ids = sorted({example.db_id for example in self._examples})
        validated = 0
        if not self.root.is_dir():
            errors.append("Spider root is not a directory: %s" % self.root)
        if not self.database_root.is_dir():
            errors.append("Database root is not a directory: %s" % self.database_root)
        for db_id in db_ids:
            metadata = self._schemas.get(db_id)
            if metadata is None:
                errors.append("No tables.json entry for db_id=%s" % db_id)
                continue
            path = self.database_path(db_id)
            if not path.is_file():
                errors.append("Missing SQLite database: %s" % path)
                continue
            schema_errors = self._validate_schema_against_sqlite(db_id, metadata, path)
            if schema_errors:
                errors.extend(schema_errors)
            else:
                validated += 1
        unused_schema_count = len(set(self._schemas) - set(db_ids))
        if unused_schema_count:
            warnings.append(
                "%d schema entries are not used by split=%s"
                % (unused_schema_count, self.config.split)
            )
        return ValidationReport(
            split=self.config.split,
            example_count=len(self._examples),
            database_count=len(db_ids),
            schema_count=len(self._schemas),
            validated_database_count=validated,
            errors=errors,
            warnings=warnings,
        )
