"""Stubs that describe the sandbox's host names to Monty's type checker.

With ``type_check=True`` Monty checks each snippet before running it. The
checker only knows names it has seen declared, so the adapter declares the
host functions sandbox code may call (tools, ``llm_query``, ``SUBMIT``) as
stubs at checkout, and each input variable as a bare annotation
(``context: str``) before the first snippet that receives it.

The checker also fixes a variable's type at the value a snippet first gave
it, so ``n = 1`` in one snippet makes ``n = 2`` in the next an error. After
each snippet runs, the adapter redeclares the names it assigned as ``Any``.

Types the checker cannot resolve in the sandbox, such as a tool's pydantic
model, are declared as ``Any`` so they never cause false errors.
"""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

# Names an annotation may use and still be written into a stub as is.
_SAFE_NAMES = frozenset({
    "str", "int", "float", "bool", "bytes", "complex", "list", "dict", "tuple", "set", "frozenset",
    "None", "NoneType", "Any", "Literal", "Optional", "Union", "typing",
})
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLAIN_VALUE_TYPES = {str: "str", int: "int", float: "float", bool: "bool", bytes: "bytes"}

# Every typing name annotation_text may write.
_TYPING_IMPORT = "from typing import Any, Literal, Optional, Union"
SUBMIT_STUB = "def SUBMIT(*args: Any, **kwargs: Any) -> None: ..."


class _Text:
    """Renders as its text inside ``str(inspect.Signature)``."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def annotation_text(annotation: Any) -> str:
    """``annotation`` as stub source, or ``Any`` when it names other types."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "Any"
    if annotation is None or annotation is type(None):
        return "None"
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    text = text.replace("typing.", "").replace("NoneType", "None")
    names = set(_IDENTIFIER_RE.findall(re.sub(r"(['\"]).*?\1", "", text)))
    return text if names <= _SAFE_NAMES else "Any"


def function_stub(name: str, fn: Callable[..., Any]) -> str:
    """A ``def`` stub with ``fn``'s parameters, or one that accepts anything."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"def {name}(*args: Any, **kwargs: Any) -> Any: ..."
    params = [
        p.replace(
            annotation=_Text(annotation_text(p.annotation)),
            default=p.default if p.default is inspect.Parameter.empty else _Text("..."),
        )
        for p in sig.parameters.values()
    ]
    returns = _Text(annotation_text(sig.return_annotation))
    return f"def {name}{inspect.Signature(params, return_annotation=returns)}: ..."


def value_annotation(value: Any) -> str:
    """The annotation an input variable is declared with."""
    if value is None:
        return "None"
    if type(value) in _PLAIN_VALUE_TYPES:
        return _PLAIN_VALUE_TYPES[type(value)]
    if isinstance(value, list):
        return "list[Any]"
    if isinstance(value, dict):
        return "dict[Any, Any]"
    return "Any"


def stubs(functions: Mapping[str, Callable[..., Any]], names: Iterable[str] = (), extra: str = "") -> str:
    """Stub source declaring ``functions`` by signature, ``names`` as ``Any``,
    and the declarations in ``extra``."""
    lines = [_TYPING_IMPORT, SUBMIT_STUB, extra.rstrip()]
    lines += [function_stub(name, fn) for name, fn in functions.items() if name.isidentifier()]
    lines += [f"{name}: Any" for name in names if name.isidentifier()]
    return "\n".join(lines) + "\n"


def declarations(annotations: Mapping[str, str]) -> str:
    """A snippet that declares each name with its annotation and runs nothing."""
    return _TYPING_IMPORT + "\n" + "\n".join(f"{name}: {text}" for name, text in annotations.items())


def assigned_names(code: str) -> set[str]:
    """Names ``code`` assigns at module level without declaring a type."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        if isinstance(node, ast.AnnAssign):
            return  # its annotation already declares the type
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return names
