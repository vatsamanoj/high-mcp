import ast
import argparse
import json
import os
from typing import Dict, List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".pytest_cache",
    "trust_store",
    "logs",
    "testing_scripts",
    "verification_scripts",
}
API_DECORATORS = {"get", "post", "put", "patch", "delete", "options", "head"}
CORE_ALLOWLIST = {"ui_server.py", "quota_server.py", "server.py"}


def iter_python_files() -> List[str]:
    out: List[str] = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            if fn.startswith("test_") or fn.endswith("_test.py"):
                continue
            out.append(os.path.join(base, fn))
    return out


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def has_setup_fn(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            return True
    return False


def router_declared(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "router":
                    return True
    return False


def setup_references_router(tree: ast.AST) -> bool:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.FunctionDef) or node.name != "setup":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "include_router":
                return True
    return False


def find_direct_app_routes(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in API_DECORATORS:
            base = fn.value
            if isinstance(base, ast.Name) and base.id == "app":
                count += 1
    return count


def _run_audit() -> Dict[str, object]:
    report: Dict[str, List[Dict[str, object]]] = {
        "plugin_issues": [],
        "direct_app_routes_outside_core": [],
    }

    for path in iter_python_files():
        r = rel(path)
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                src = f.read()
            tree = ast.parse(src)
        except Exception as e:
            report["plugin_issues"].append({"file": r, "issue": f"parse_error: {e}"})
            continue

        in_plugins_or_components = r.startswith("plugins/") or r.startswith("components/")
        if in_plugins_or_components:
            if not has_setup_fn(tree):
                report["plugin_issues"].append({"file": r, "issue": "missing_setup_function"})
            if router_declared(tree) and not setup_references_router(tree):
                report["plugin_issues"].append({"file": r, "issue": "router_declared_but_not_included_in_setup"})

        direct_routes = find_direct_app_routes(tree)
        if direct_routes > 0 and r not in CORE_ALLOWLIST and not r.startswith("components/"):
            report["direct_app_routes_outside_core"].append(
                {"file": r, "route_decorators": direct_routes}
            )

    total_issues = len(report["plugin_issues"]) + len(report["direct_app_routes_outside_core"])
    return {"total_issues": total_issues, **report}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit plugin architecture safety.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when issues are found. Default is warning-only exit 0.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress JSON output (exit code still reflects mode).",
    )
    args = parser.parse_args()

    result = _run_audit()
    if not args.quiet:
        print(json.dumps(result, indent=2))
    if args.strict and int(result.get("total_issues", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
