"""Strip code a generated script can never run for the settings it was made with.

The script generators paste in shared helpers and setup code covering every
model and option, then fix the chosen ones as literals in the script. Three
passes clean that up, in this order:

1. Branches whose outcome is already fixed: ``if False:`` / ``if True:``, and
   tests on the model flags (``is_mace_polar``, ``is_orbmol``) and ``device``
   when the script assigns them a literal exactly once. The branch that runs is
   kept, the other goes. Other settings (``fix_symmetry``, ``optimizer_type``
   and friends) are left alone so they can still be switched by hand.
2. Top-level ``def``/``class`` blocks nothing else in the script refers to.
3. ``name = <literal>`` lines for the model-only variables above once nothing
   reads them.

Everything errs towards keeping code: a helper counts as used if its name
appears anywhere outside its own definition, even as a word inside a string,
and a script that does not parse is returned untouched.
"""
import ast
import functools
import io
import re
import tokenize

_KEEP = {"main"}

# Names fixed by the model/device choice; branches on them can be resolved.
_SETUP_NAMES = ("is_mace_polar", "is_orbmol", "device")
# Variables that only feed such branches; dropped once nothing reads them.
_SETUP_ONLY_VARS = ("is_mace_polar", "is_orbmol", "polar_settings", "polar_results_all",
                    "polar_logger", "device")

# Expression nodes a resolvable test may be built from.
_STATIC_NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Compare, ast.Eq,
                 ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.BoolOp, ast.And,
                 ast.Or, ast.UnaryOp, ast.Not, ast.Tuple, ast.List)


def prune_unused_definitions(code):
    try:
        code = _resolve_static_branches(code)
        code = _prune_helpers(code)
        return _drop_unread_setup_vars(code)
    except SyntaxError:
        return code


def _drop_lines(code, ranges):
    lines = code.splitlines(keepends=True)
    drop = set()
    for start, end in ranges:
        drop.update(range(start - 1, end))
    return "".join(line for i, line in enumerate(lines) if i not in drop)


def _walk_bodies(body, visit):
    """Call ``visit(stmt, body)`` for every statement, depth first."""
    for stmt in body:
        if visit(stmt, body):
            return True
        for field in ("body", "orelse", "finalbody"):
            if _walk_bodies(getattr(stmt, field, None) or [], visit):
                return True
        for handler in getattr(stmt, "handlers", None) or []:
            if _walk_bodies(handler.body, visit):
                return True
    return False


# ---------------------------------------------------------------------------
# 1. Branches with a fixed outcome
# ---------------------------------------------------------------------------
def _setup_constants(tree):
    """Setup names bound exactly once in the whole script, to a literal."""
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bindings.setdefault(node.id, []).append(node)
        elif isinstance(node, ast.arg):
            bindings.setdefault(node.arg, []).append(node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bindings.setdefault(name, []).extend([None, None])

    consts = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in _SETUP_NAMES
                and len(bindings.get(node.targets[0].id, [])) == 1):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return consts


def _static_value(test, consts):
    """(True, value) when ``test`` only uses literals and known constants."""
    expr = ast.Expression(test)
    for node in ast.walk(expr):
        if not isinstance(node, _STATIC_NODES):
            return False, None
        if isinstance(node, ast.Name) and node.id not in consts:
            return False, None
    try:
        return True, bool(eval(compile(expr, "<test>", "eval"), {"__builtins__": {}}, consts))
    except Exception:
        return False, None


def _lines_inside_strings(code):
    """Line numbers that continue a string token begun on an earlier line.

    Only those lines are unsafe to re-indent. Adjacent literals split over
    several lines are separate tokens and do not count.
    """
    inside = set()
    starts = []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        kind = tokenize.tok_name[tok.type]
        if kind == "STRING":
            inside.update(range(tok.start[0] + 1, tok.end[0] + 1))
        elif kind == "FSTRING_START":
            starts.append(tok.start[0])
        elif kind == "FSTRING_END" and starts:
            inside.update(range(starts.pop() + 1, tok.end[0] + 1))
    return inside


def _dedent(lines, amount):
    out = []
    for line in lines:
        strip = min(amount, len(line) - len(line.lstrip(" ")))
        out.append(line[strip:])
    return out


def _resolve_one(code):
    """Resolve the first fixed-outcome ``if``; None when there is none left."""
    tree = ast.parse(code)
    consts = _setup_constants(tree)
    lines = code.splitlines(keepends=True)
    in_strings = None
    result = []

    def visit(stmt, body):
        if not isinstance(stmt, ast.If):
            return False
        known, value = _static_value(stmt.test, consts)
        if not known:
            return False
        before, after = lines[:stmt.lineno - 1], lines[stmt.end_lineno:]
        kept = stmt.body if value else stmt.orelse

        if not kept:
            if len(body) == 1:
                return False  # dropping it would leave an empty block
            result.append("".join(before + after))
            return True

        if kept[0].lineno == stmt.lineno:
            return False  # one-line ``if x: y``
        if not value and len(kept) == 1 and isinstance(kept[0], ast.If) and \
                lines[kept[0].lineno - 1].lstrip().startswith("elif"):
            # ``elif`` becomes the new ``if``; it already sits at this indent.
            first = lines[kept[0].lineno - 1]
            first = first.replace("elif", "if", 1)
            block = [first] + lines[kept[0].lineno:kept[0].end_lineno]
            result.append("".join(before + block + after))
            return True
        nonlocal in_strings
        if in_strings is None:
            in_strings = _lines_inside_strings(code)
        if any(n in in_strings for n in range(kept[0].lineno, kept[-1].end_lineno + 1)):
            return False  # dedenting would change a string's text

        block = lines[kept[0].lineno - 1:kept[-1].end_lineno]
        block = _dedent(block, kept[0].col_offset - stmt.col_offset)
        result.append("".join(before + block + after))
        return True

    _walk_bodies(tree.body, visit)
    return result[0] if result else None


def _resolve_static_branches(code):
    while True:
        new = _resolve_one(code)
        if new is None:
            return code
        code = new


# ---------------------------------------------------------------------------
# 2. Unreferenced helpers
# ---------------------------------------------------------------------------
def _names_used(node, skip_name=None):
    used = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id != skip_name:
            used.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            used.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            used.update(w for w in re.findall(r"\w+", sub.value) if w != skip_name)
    return used


def _prune_helpers(code):
    tree = ast.parse(code)
    defs = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    removed = set()

    # Repeat until stable: dropping one helper can leave another unreferenced.
    while True:
        used = set()
        for node in tree.body:
            if getattr(node, "name", None) in removed:
                continue
            used |= _names_used(node, skip_name=getattr(node, "name", None))
        newly = {d.name for d in defs
                 if d.name not in removed and d.name not in _KEEP and d.name not in used}
        if not newly:
            break
        removed |= newly

    if not removed:
        return code

    lines = code.splitlines(keepends=True)
    ranges = []
    for d in defs:
        if d.name in removed:
            start = min([d.lineno] + [dec.lineno for dec in d.decorator_list])
            end = d.end_lineno
            # Take the blank separator lines after it too; they sit outside any
            # string literal, so this cannot touch the script's own text.
            while end < len(lines) and not lines[end].strip():
                end += 1
            ranges.append((start, end))
    return _drop_lines(code, ranges)


# ---------------------------------------------------------------------------
# 3. Setup variables nothing reads any more
# ---------------------------------------------------------------------------
def _drop_unread_setup_vars(code):
    tree = ast.parse(code)
    loads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    ranges = []
    dropped_in = {}

    def visit(stmt, body):
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in _SETUP_ONLY_VARS
                and stmt.targets[0].id not in loads
                and dropped_in.get(id(body), 0) < len(body) - 1):
            try:
                ast.literal_eval(stmt.value)
            except ValueError:
                return False
            ranges.append((stmt.lineno, stmt.end_lineno))
            dropped_in[id(body)] = dropped_in.get(id(body), 0) + 1
        return False

    _walk_bodies(tree.body, visit)
    return _drop_lines(code, ranges) if ranges else code


def pruned_script(generator):
    """Decorator: run ``prune_unused_definitions`` on a generator's output."""
    @functools.wraps(generator)
    def wrapper(*args, **kwargs):
        return prune_unused_definitions(generator(*args, **kwargs))
    return wrapper
