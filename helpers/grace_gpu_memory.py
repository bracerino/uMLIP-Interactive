"""Stop GRACE (TensorFlow) from reserving the whole GPU in generated scripts.

TensorFlow grabs nearly all GPU memory the moment it initialises the card, even
when the model needs a fraction of it. ``TF_FORCE_GPU_ALLOW_GROWTH=true`` makes
it allocate only what it actually uses, which leaves room for other jobs on the
same GPU. TF reads the variable once, when it sets the GPU up, so the line has
to run before anything imports tensorflow. It goes at the very top of the
script, ahead of the package check that already imports tensorpotential.

``setdefault`` keeps a value the user exported themselves.
"""
import ast
import functools

_PREAMBLE = (
    "# Allow growth so it takes only the GPU memory it actually uses\n"
    "import os\n"
    "os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')\n"
    "\n"
)

# Every GRACE script, foundation or custom model, imports from this package.
_GRACE_MARKER = "tensorpotential"


def add_grace_gpu_memory_growth(code):
    """Insert the allow-growth preamble into a GRACE script; others pass through."""
    if _GRACE_MARKER not in code or "TF_FORCE_GPU_ALLOW_GROWTH" in code:
        return code

    lines = code.splitlines(keepends=True)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Still get ahead of the imports: just below a shebang, if there is one.
        at = 1 if lines and lines[0].startswith("#!") else 0
        return "".join(lines[:at] + [_PREAMBLE] + lines[at:])

    # After the module docstring and any ``from __future__`` imports, which
    # Python requires to come first.
    at = 0
    for stmt in tree.body:
        is_docstring = (stmt is tree.body[0] and isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str))
        is_future = isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__"
        if not (is_docstring or is_future):
            break
        at = stmt.end_lineno
    if at == 0 and lines and lines[0].startswith("#!"):
        at = 1
    return "".join(lines[:at] + [_PREAMBLE] + lines[at:])


def grace_gpu_memory_growth(generator):
    """Decorator: apply ``add_grace_gpu_memory_growth`` to a generator's output."""
    @functools.wraps(generator)
    def wrapper(*args, **kwargs):
        return add_grace_gpu_memory_growth(generator(*args, **kwargs))
    return wrapper
