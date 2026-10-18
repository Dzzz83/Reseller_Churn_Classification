"""Fail packaging checks if a wheel contains anything beyond production code."""

import ast
import sys
import zipfile
from pathlib import Path


def check_package(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert "reseller_churn/config/retained_model.json" in names
        for name in names:
            assert name.startswith("reseller_churn/") or ".dist-info/" in name, name
            assert not name.endswith((".csv", ".joblib", ".pkl", ".ipynb")), name
            if name.endswith(".py"):
                tree = ast.parse(archive.read(name), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    assert all(
                        m.split(".")[0] not in {"comparisons", "pipeline", "tests"}
                        for m in modules
                    ), (name, modules)
    print(f"Production wheel contents verified: {Path(path).name}")


if __name__ == "__main__":
    check_package(sys.argv[1])
