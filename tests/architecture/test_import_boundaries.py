import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _imports_for(package: str) -> set[str]:
    package_dir = PROJECT_ROOT / package
    imports: set[str] = set()
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    return imports


def _has_forbidden(imports: set[str], forbidden: tuple[str, ...]) -> list[str]:
    matches = []
    for module in sorted(imports):
        if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
            matches.append(module)
    return matches


class ImportBoundaryTests(unittest.TestCase):
    def test_lfm3_has_no_serving_dependencies(self):
        imports = _imports_for("lfm3")
        forbidden = ("app", "adapters", "runtime", "fastapi")
        self.assertEqual(_has_forbidden(imports, forbidden), [])

    def test_inference_has_no_http_or_runtime_dependencies(self):
        imports = _imports_for("inference")
        forbidden = ("fastapi", "starlette", "adapters.openai", "app", "runtime")
        self.assertEqual(_has_forbidden(imports, forbidden), [])

    def test_runtime_has_no_http_dependencies(self):
        imports = _imports_for("runtime")
        forbidden = ("fastapi", "starlette", "adapters.openai", "app")
        self.assertEqual(_has_forbidden(imports, forbidden), [])

    def test_app_has_no_http_or_openai_dependencies(self):
        imports = _imports_for("app")
        forbidden = ("fastapi", "starlette", "adapters.openai")
        self.assertEqual(_has_forbidden(imports, forbidden), [])

    def test_openai_adapter_does_not_import_torch(self):
        imports = _imports_for("adapters/openai")
        self.assertEqual(_has_forbidden(imports, ("torch",)), [])


if __name__ == "__main__":
    unittest.main()
