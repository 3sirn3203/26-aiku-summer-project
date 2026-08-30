from __future__ import annotations

import re
from typing import List, Set


def _without_comments(sql: str) -> str:
    output: List[str] = []
    index = 0
    state = "normal"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "normal":
            if char == "-" and next_char == "-":
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                state = "block_comment"
                index += 2
                continue
            if char == "'":
                state = "single_quote"
            elif char == '"':
                state = "double_quote"
            elif char == "`":
                state = "backtick"
            elif char == "[":
                state = "bracket"
            output.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char in "\r\n":
                output.append("\n")
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
            else:
                index += 1
            continue
        output.append(char)
        if state == "single_quote" and char == "'":
            if next_char == "'":
                output.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "double_quote" and char == '"':
            if next_char == '"':
                output.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "backtick" and char == "`":
            if next_char == "`":
                output.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "bracket" and char == "]":
            state = "normal"
        index += 1
    return "".join(output)


def split_sql_statements(sql: str) -> List[str]:
    """Split on semicolons outside quotes and comments.

    This is a preflight guard only. SQLite's parser and authorizer remain the
    authoritative safety boundary.
    """
    statements: List[str] = []
    current: List[str] = []
    index = 0
    state = "normal"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "normal":
            if char == "-" and next_char == "-":
                current.extend((char, next_char))
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                current.extend((char, next_char))
                state = "block_comment"
                index += 2
                continue
            if char == "'":
                state = "single_quote"
            elif char == '"':
                state = "double_quote"
            elif char == "`":
                state = "backtick"
            elif char == "[":
                state = "bracket"
            elif char == ";":
                candidate = "".join(current).strip()
                if _without_comments(candidate).strip():
                    statements.append(candidate)
                current = []
                index += 1
                continue
            current.append(char)
            index += 1
            continue
        current.append(char)
        if state == "line_comment" and char in "\r\n":
            state = "normal"
        elif state == "block_comment" and char == "*" and next_char == "/":
            current.append(next_char)
            state = "normal"
            index += 2
            continue
        elif state == "single_quote" and char == "'":
            if next_char == "'":
                current.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "double_quote" and char == '"':
            if next_char == '"':
                current.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "backtick" and char == "`":
            if next_char == "`":
                current.append(next_char)
                index += 2
                continue
            state = "normal"
        elif state == "bracket" and char == "]":
            state = "normal"
        index += 1
    candidate = "".join(current).strip()
    if _without_comments(candidate).strip():
        statements.append(candidate)
    return statements


def first_keyword(sql: str) -> str:
    match = re.search(r"[A-Za-z_]+", _without_comments(sql))
    return match.group(0).upper() if match else ""


def top_level_statement_keyword(sql: str) -> str:
    """Return the main statement keyword, including after top-level CTEs.

    Quoted text, comments, and tokens inside parentheses are ignored.  For
    example, ``WITH x AS (SELECT 1) DELETE ...`` returns ``DELETE`` while a
    read-only CTE query returns ``SELECT``.
    """

    statement_keywords = {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "REPLACE",
        "CREATE",
        "DROP",
        "ALTER",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "VACUUM",
        "REINDEX",
        "ANALYZE",
    }
    top_level_words: List[str] = []
    index = 0
    depth = 0
    state = "normal"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "normal":
            if char == "-" and next_char == "-":
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                state = "block_comment"
                index += 2
                continue
            if char == "'":
                state = "single_quote"
                index += 1
                continue
            if char == '"':
                state = "double_quote"
                index += 1
                continue
            if char == "`":
                state = "backtick"
                index += 1
                continue
            if char == "[":
                state = "bracket"
                index += 1
                continue
            if char == "(":
                depth += 1
                index += 1
                continue
            if char == ")":
                depth = max(0, depth - 1)
                index += 1
                continue
            if depth == 0 and (char.isalpha() or char == "_"):
                end = index + 1
                while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                    end += 1
                top_level_words.append(sql[index:end].upper())
                index = end
                continue
            index += 1
            continue
        if state == "line_comment":
            if char in "\r\n":
                state = "normal"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "normal"
                index += 2
            else:
                index += 1
            continue
        if state == "single_quote" and char == "'":
            if next_char == "'":
                index += 2
                continue
            state = "normal"
        elif state == "double_quote" and char == '"':
            if next_char == '"':
                index += 2
                continue
            state = "normal"
        elif state == "backtick" and char == "`":
            if next_char == "`":
                index += 2
                continue
            state = "normal"
        elif state == "bracket" and char == "]":
            state = "normal"
        index += 1

    if not top_level_words:
        return ""
    first = top_level_words[0]
    if first != "WITH":
        return first
    for word in top_level_words[1:]:
        if word in statement_keywords:
            return word
    return "WITH"


def function_calls(sql: str) -> Set[str]:
    """Return SQLite function-like identifiers outside string literals.

    SQLite permits function names to be quoted as identifiers, and comments
    are legal between the name and opening parenthesis.  This lexer therefore
    handles bare, double-quoted, backtick-quoted, and bracket-quoted names and
    skips both whitespace and comments before checking for ``(``.
    """

    def skip_trivia(position: int) -> int:
        while position < len(sql):
            while position < len(sql) and sql[position].isspace():
                position += 1
            if sql.startswith("--", position):
                position += 2
                while position < len(sql) and sql[position] not in "\r\n":
                    position += 1
                continue
            if sql.startswith("/*", position):
                closing = sql.find("*/", position + 2)
                if closing < 0:
                    return len(sql)
                position = closing + 2
                continue
            return position
        return position

    def read_quoted_identifier(position: int) -> tuple[str, int]:
        opener = sql[position]
        if opener == "[":
            closing = sql.find("]", position + 1)
            if closing < 0:
                return "", len(sql)
            return sql[position + 1 : closing], closing + 1
        output: List[str] = []
        closing = opener
        position += 1
        while position < len(sql):
            char = sql[position]
            if char == closing:
                if position + 1 < len(sql) and sql[position + 1] == closing:
                    output.append(closing)
                    position += 2
                    continue
                return "".join(output), position + 1
            output.append(char)
            position += 1
        return "", len(sql)

    def skip_single_quoted_literal(position: int) -> int:
        position += 1
        while position < len(sql):
            if sql[position] == "'":
                if position + 1 < len(sql) and sql[position + 1] == "'":
                    position += 2
                    continue
                return position + 1
            position += 1
        return position

    calls: Set[str] = set()
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if char == "-" and next_char == "-":
            index = skip_trivia(index)
            continue
        if char == "/" and next_char == "*":
            index = skip_trivia(index)
            continue
        if char == "'":
            index = skip_single_quoted_literal(index)
            continue
        if char in ('"', "`", "["):
            name, end = read_quoted_identifier(index)
            cursor = skip_trivia(end)
            if name and cursor < len(sql) and sql[cursor] == "(":
                calls.add(name.casefold())
            index = end
            continue
        if char.isalpha() or char == "_":
            end = index + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            name = sql[index:end].casefold()
            cursor = skip_trivia(end)
            if cursor < len(sql) and sql[cursor] == "(":
                calls.add(name)
            index = end
            continue
        index += 1
    return calls
