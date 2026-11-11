"""Source rewrites that let dspy code run in Monty.

Monty has no class inheritance, no ``super()`` and no ``dspy`` package. The
sandbox gets a ``dspy`` namespace from the bridge's prelude instead, so code
that imports dspy, subclasses ``dspy.Module`` or declares a
``dspy.Signature`` class is rewritten before it runs:

- ``import dspy`` binds nothing new; ``import dspy as d`` and
  ``from dspy import X`` become assignments from the namespace.
- A ``dspy.Module`` subclass loses that base and its ``super().__init__()``
  calls, and gains a ``__call__`` that runs ``forward``.
- A ``dspy.Signature`` subclass becomes a ``dspy.Signature(...)`` call that
  the host builds from the field declarations. Nothing is executed on the
  host: annotations travel as text and field options as sandbox values.

Every edit keeps the line count, so line numbers in tracebacks still match
the code the model wrote.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any

_DSPY = "dspy"
_MODULE_BASES = {"dspy.Module", "Module"}
_SIGNATURE_BASES = {"dspy.Signature", "Signature"}
_FIELD_KINDS = {"InputField": "input", "OutputField": "output"}

# Flex drives a module with this exact snippet (dspy.predict.flex.bridge).
_FLEX_DRIVER_RE = re.compile(
    r"\Afrom dspy\.primitives\.sandbox_guest import run_forward as __dspy_run_forward\n"
    r"__dspy_run_forward\(globals\(\), '(?P<cls>[A-Za-z_]\w*)', (?P<inputs>[A-Za-z_]\w*)\)\Z"
)
FLEX_RESULT_VAR = "_dspy_flex_result"


@dataclass
class FunctionInfo:
    """What a REPL-defined function looks like to a tool-calling LM."""

    name: str
    params: list[dict[str, Any]] = field(default_factory=list)
    doc: str = ""


def flex_driver(code: str) -> tuple[str, str] | None:
    """``(class_name, inputs_var)`` when ``code`` is Flex's forward driver."""
    m = _FLEX_DRIVER_RE.match(code)
    return (m["cls"], m["inputs"]) if m else None


def flex_driver_code(class_name: str, inputs_var: str) -> str:
    """Monty code that runs one Flex forward and evaluates to ``{"fields": ...}``."""
    r = FLEX_RESULT_VAR
    return (
        f"{r} = {class_name}().forward(**{inputs_var})\n"
        f'{{"fields": dict({r}._store) if isinstance({r}, dspy.Prediction) else {r}}}'
    )


def functions(code: str) -> dict[str, FunctionInfo]:
    """The functions ``code`` defines, by name: module-level ones and those
    nested in other functions, such as helpers inside a module's
    ``forward``. Methods are skipped, and a module-level function wins over
    a nested one of the same name."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {}
    top: dict[str, FunctionInfo] = {}
    nested: dict[str, FunctionInfo] = {}

    def visit(node: ast.AST, depth: int, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not in_class:
                    (top if depth == 0 else nested)[child.name] = _function_info(child, code)
                visit(child, depth + 1, False)
            elif isinstance(child, ast.ClassDef):
                visit(child, depth + 1, True)
            else:
                visit(child, depth, in_class)

    visit(tree, 0, False)
    return {**nested, **top}


def _function_info(node: ast.FunctionDef | ast.AsyncFunctionDef, code: str) -> FunctionInfo:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[Any] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = []
    for arg, default in [*zip(positional, defaults, strict=True), *zip(args.kwonlyargs, args.kw_defaults, strict=True)]:
        if arg.arg in ("self", "cls"):
            continue
        param: dict[str, Any] = {"name": arg.arg}
        if arg.annotation is not None:
            param["annotation"] = ast.get_source_segment(code, arg.annotation) or ""
        if default is not None:
            try:
                param["default"] = ast.literal_eval(default)
            except Exception:  # not a literal, or one that fails to build, like {[1]: 2}
                param["default"] = None
        params.append(param)
    return FunctionInfo(node.name, params, ast.get_docstring(node) or "")


def rewrite(code: str) -> str:
    """``code`` with its dspy imports, Module subclasses and Signature classes
    made runnable in Monty. Code that does not mention dspy is returned as is."""
    if _DSPY not in code and "Module" not in code and "Signature" not in code:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code  # Monty reports the error itself
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def pos(lineno: int, col: int) -> int:
        # ast columns are UTF-8 byte offsets.
        return offsets[lineno - 1] + len(lines[lineno - 1].encode()[:col].decode(errors="ignore"))

    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (text := _import_text(node)) is not None:
                edits.append((pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset),
                              _keep_lines(text, node)))
        elif isinstance(node, ast.ClassDef):
            bases = {ast.unparse(b) for b in node.bases}
            if bases & _SIGNATURE_BASES and (text := _signature_text(node, code)) is not None:
                start = node.decorator_list[0] if node.decorator_list else node
                edits.append((pos(start.lineno, start.col_offset), pos(node.end_lineno, node.end_col_offset),
                              _keep_lines(text, node, start.lineno)))
            elif bases & _MODULE_BASES:
                edits.extend(_module_edits(node, code, pos))
    if not edits:
        return code
    out = code
    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + text + out[end:]
    return out


def _keep_lines(text: str, node: ast.AST, first_line: int | None = None) -> str:
    """``text`` padded with newlines to span as many lines as ``node`` did."""
    span = node.end_lineno - (first_line or node.lineno)
    return text + "\n" * max(0, span - text.count("\n"))


def _import_text(node: ast.Import | ast.ImportFrom) -> str | None:
    """Replacement for an import that names dspy, or None to leave it."""
    if isinstance(node, ast.ImportFrom):
        if node.module != _DSPY or node.level:
            return None
        parts = [f"{a.asname or a.name} = dspy.{a.name}" for a in node.names if a.name != "*"]
        return "; ".join(parts) or "pass"
    if not any(a.name == _DSPY for a in node.names):
        return None
    parts = []
    other = [a for a in node.names if a.name != _DSPY]
    if other:
        parts.append("import " + ", ".join(f"{a.name} as {a.asname}" if a.asname else a.name for a in other))
    parts += [f"{a.asname} = dspy" for a in node.names if a.name == _DSPY and a.asname]
    return "; ".join(parts) or "pass"


def _module_edits(node: ast.ClassDef, code: str, pos: Any) -> list[tuple[int, int, str]]:
    """Drop the Module base, its ``super()`` calls, and add ``__call__``."""
    edits = []
    kept = [b for b in node.bases if ast.unparse(b) not in _MODULE_BASES]
    paren = _base_parens(node, code)
    if paren is not None:
        start, end = pos(*paren[0]), pos(*paren[1])
        span = paren[1][0] - paren[0][0]
        kept_text = ", ".join(ast.get_source_segment(code, b) or ast.unparse(b) for b in kept)
        # Newlines may only stay inside the parentheses.
        text = f"({kept_text}{chr(10) * span})" if kept or span else ""
        edits.append((start, end, text))
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Expr) and _is_super_init(stmt.value):
            edits.append((pos(stmt.lineno, stmt.col_offset), pos(stmt.end_lineno, stmt.end_col_offset),
                          _keep_lines("pass", stmt)))
    has_call = any(isinstance(s, ast.FunctionDef) and s.name == "__call__" for s in node.body)
    if not has_call:
        edits.append(_call_edit(node, code, pos))
    return edits


def _call_edit(node: ast.ClassDef, code: str, pos: Any) -> tuple[int, int, str]:
    """Add ``__call__`` after the class body, in the blank lines that follow
    it when there are two, so later line numbers stay put."""
    indent = " " * node.body[0].col_offset
    last = node.body[-1]
    if last.lineno == node.lineno:  # a one-line class body
        at = pos(last.end_lineno, last.end_col_offset)
        return at, at, "; __call__ = lambda self, *args, **kwargs: self.forward(*args, **kwargs)"
    lines = code.splitlines(keepends=True)
    end_line = node.end_lineno  # 1-based; the class's last line
    blank = 0
    while end_line + blank < len(lines) and blank < 2 and not lines[end_line + blank].strip():
        blank += 1
    method = f"{indent}def __call__(self, *args, **kwargs):\n{indent}    return self.forward(*args, **kwargs)"
    at = pos(end_line, len(lines[end_line - 1].rstrip("\r\n").encode()))
    if blank == 2:
        stop = pos(end_line + 2, len(lines[end_line + 1].rstrip("\r\n").encode()))
        return at, stop, "\n" + method
    return at, at, "\n" + method


def _base_parens(node: ast.ClassDef, code: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Start of ``(`` and end of ``)`` around a class's bases, as (line, col)."""
    header_end = node.body[0].lineno
    segment = "".join(code.splitlines(keepends=True)[node.lineno - 1:header_end])
    depth = 0
    start = None
    seen_class = False
    try:
        for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
            row = tok.start[0] + node.lineno - 1
            if tok.type == tokenize.NAME and tok.string == "class" and row >= node.lineno:
                seen_class = True
            elif not seen_class:
                continue
            elif tok.type == tokenize.OP and tok.string == "(":
                if depth == 0:
                    start = (row, _byte_col(segment, tok.start))
                depth += 1
            elif tok.type == tokenize.OP and tok.string == ")":
                depth -= 1
                if depth == 0 and start is not None:
                    return start, (tok.end[0] + node.lineno - 1, _byte_col(segment, tok.end))
            elif tok.type == tokenize.OP and tok.string == ":" and depth == 0:
                return None
    except (tokenize.TokenError, SyntaxError):
        return None
    return None


def _byte_col(segment: str, rowcol: tuple[int, int]) -> int:
    line = segment.splitlines(keepends=True)[rowcol[0] - 1]
    return len(line[: rowcol[1]].encode())


def _is_super_init(call: ast.expr) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "__init__"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
    )


def _signature_text(node: ast.ClassDef, code: str) -> str | None:
    """``Name = dspy.Signature(...)`` for a declarative Signature class, or
    None when the class does more than declare fields."""
    instructions = ast.get_docstring(node) or ""
    inputs: list[str] = []
    outputs: list[str] = []
    options: list[str] = []
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name, annotation, value = stmt.target.id, stmt.annotation, stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, annotation, value = stmt.targets[0].id, None, stmt.value
        else:
            return None
        kind = _field_kind(value)
        if kind is None or value.args:
            return None
        text = name
        if annotation is not None:
            text += ": " + " ".join((ast.get_source_segment(code, annotation) or ast.unparse(annotation)).split())
        (inputs if kind == "input" else outputs).append(text)
        if value.keywords:
            kwargs = ", ".join(ast.get_source_segment(code, kw) or ast.unparse(kw) for kw in value.keywords)
            options.append(f"{name!r}: dict({kwargs})")
    signature = f"{', '.join(inputs)} -> {', '.join(outputs)}"
    return f"{node.name} = dspy.Signature({signature!r}, {instructions!r}, {{{', '.join(options)}}})"


def _field_kind(value: ast.expr | None) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    return _FIELD_KINDS.get(name or "")
