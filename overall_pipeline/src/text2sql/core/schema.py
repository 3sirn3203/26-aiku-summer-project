from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from text2sql.core.models import SchemaMetadata


def _flatten_primary_keys(values: Iterable[Any]) -> Set[int]:
    flattened: Set[int] = set()
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            flattened.add(value)
        elif isinstance(value, (list, tuple)):
            flattened.update(_flatten_primary_keys(value))
    return flattened


def _quote(identifier: str) -> str:
    return '"%s"' % identifier.replace('"', '""')


def serialize_schema(metadata: SchemaMetadata) -> str:
    """Serialize Spider tables.json metadata without reading schema.sql or DB rows."""
    primary_keys = _flatten_primary_keys(metadata.primary_keys)
    columns_by_table: Dict[int, list] = {
        index: [] for index in range(len(metadata.table_names))
    }
    for column_index, column in enumerate(metadata.column_names):
        if len(column) != 2:
            raise ValueError("Malformed column metadata at index %d" % column_index)
        table_index, column_name = column
        if table_index == -1:
            continue
        if not isinstance(table_index, int) or table_index not in columns_by_table:
            raise ValueError("Invalid table index at column %d" % column_index)
        column_type = (
            str(metadata.column_types[column_index]).upper()
            if column_index < len(metadata.column_types)
            else "UNKNOWN"
        )
        suffix = " PRIMARY KEY" if column_index in primary_keys else ""
        columns_by_table[table_index].append(
            "  - %s %s%s" % (_quote(str(column_name)), column_type, suffix)
        )

    lines = ["Database: %s" % metadata.db_id, "Tables:"]
    for table_index, table_name in enumerate(metadata.table_names):
        lines.append("Table %s" % _quote(str(table_name)))
        lines.extend(columns_by_table[table_index])

    lines.append("Foreign keys:")
    if not metadata.foreign_keys:
        lines.append("  - none")
    for pair in metadata.foreign_keys:
        if len(pair) != 2:
            raise ValueError("Malformed foreign key metadata")
        source_index, target_index = pair
        source_table_index, source_column = metadata.column_names[source_index]
        target_table_index, target_column = metadata.column_names[target_index]
        source_table = metadata.table_names[source_table_index]
        target_table = metadata.table_names[target_table_index]
        lines.append(
            "  - %s.%s -> %s.%s"
            % (
                _quote(str(source_table)),
                _quote(str(source_column)),
                _quote(str(target_table)),
                _quote(str(target_column)),
            )
        )
    return "\n".join(lines)

