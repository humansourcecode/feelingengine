"""Execute the Colab demo notebook's code cells locally. Catches broken
notebook code without needing an actual Colab runtime. The first cell
(git clone + pip install) is skipped — it only makes sense in Colab."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOK = REPO_ROOT / "examples" / "colab" / "feeling_engine_demo.ipynb"

# Which cells (by zero-based index among CODE cells only) are safe to
# execute locally. Cell 0 of the code cells is the clone/install step —
# skip that because we're running against the repo we're already inside.
LOCAL_SAFE_CODE_CELL_INDICES = [1, 2, 3, 4, 5, 6, 7]


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text())


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid(notebook):
    assert notebook.get("nbformat") == 4
    assert "cells" in notebook and notebook["cells"], "empty notebook"


def test_notebook_has_expected_cell_count(code_cells):
    # Cell 0 = install; cells 1-7 = the demo. Update if notebook is
    # intentionally restructured.
    assert len(code_cells) >= 8, f"expected ≥8 code cells, got {len(code_cells)}"


def test_all_code_cells_parse_as_python(code_cells):
    """Every code cell should compile as valid Python (after stripping
    Colab shell-escape lines like `!pip install ...`)."""
    import ast
    for i, cell in enumerate(code_cells):
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        # Strip Colab shell (!) and magic (%) lines — not valid Python
        cleaned = "\n".join(
            ln for ln in source.splitlines()
            if not ln.lstrip().startswith(("!", "%"))
        )
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            pytest.fail(f"code cell {i} is not valid Python:\n{cleaned}\n{e}")


def test_local_safe_cells_execute(code_cells, tmp_path, monkeypatch):
    """Execute each safe code cell in sequence against the local repo.
    Uses a shared namespace so later cells see variables from earlier ones."""
    ns = {"__name__": "__colab_test__"}

    # Run from the repo root so relative paths like examples/arcs/... resolve
    monkeypatch.chdir(REPO_ROOT)

    for idx in LOCAL_SAFE_CODE_CELL_INDICES:
        cell = code_cells[idx]
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        # Strip shell/magic lines
        cleaned = "\n".join(
            ln for ln in source.splitlines()
            if not ln.lstrip().startswith(("!", "%"))
        )
        # Force matplotlib to non-interactive backend so test doesn't open windows
        if "matplotlib" in cleaned and "use(" not in cleaned:
            cleaned = "import matplotlib; matplotlib.use('Agg')\n" + cleaned
        try:
            exec(compile(cleaned, f"<colab_cell_{idx}>", "exec"), ns)
        except Exception as e:
            pytest.fail(f"code cell {idx} raised {type(e).__name__}: {e}\n\n{cleaned}")
