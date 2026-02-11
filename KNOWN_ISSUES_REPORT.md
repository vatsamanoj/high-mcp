# Known Issues Scan Report

## Scope and Method
Scanned the repository for common known-issue patterns and ran baseline health checks.

Checks run:
- `python -m compileall -q .`
- `pytest -q`
- `rg -n "except\\s*:" --glob '*.py'`
- `rg -n "TODO|FIXME|HACK|XXX|BUG" --glob '*.py' --glob '*.js' --glob '*.md' --glob '*.bat' --glob '*.json' -g '!key_harvester/chrome_profile/**'`
- `rg -n "shell=True" --glob '*.py'`
- `rg -n "\\beval\\(|\\bexec\\(" --glob '*.py'`

## Findings

### 1) Test suite cannot be collected in current environment
`pytest -q` fails during test collection with missing runtime dependencies:
- `ModuleNotFoundError: No module named 'aiosqlite'`
- `ModuleNotFoundError: No module named 'requests'`

These modules are imported by tests and app modules, but are currently absent from the local environment.

### 2) Dependency declaration gap in `requirements.txt`
`requirements.txt` does **not** include `aiosqlite` or `requests`, even though project files import them.

Impact:
- Fresh environment setup from `requirements.txt` is incomplete.
- CI/local test execution fails before running tests.

### 3) Bare `except:` blocks present in multiple modules
Pattern search found multiple `except:` usages (no exception type), including in:
- `notification_system.py`
- `start_cluster.py`
- `testing_system.py`
- `error_manager.py`
- `ai_engine.py`
- and others.

Impact:
- Can hide programming/runtime errors and make debugging difficult.
- May mask operational faults that should fail fast or be logged with context.

### 4) Extensive DEBUG logging in runtime paths
Several modules include unconditional `DEBUG:` prints/writes (not gated by log level), especially `ui_server.py` and `ai_engine.py`.

Impact:
- Noisy logs in production.
- Potential leakage of operational metadata into logs.

## Non-findings from this scan
- No direct `shell=True` subprocess usage found.
- No direct `eval(...)`/`exec(...)` usage found.
- Python syntax compilation check passed (`compileall`).

## Recommended Next Steps
1. Add missing dependencies (`aiosqlite`, `requests`) to `requirements.txt` and re-run test collection.
2. Replace bare `except:` with explicit exception types and structured logging.
3. Gate debug output behind logger levels/config flags.
4. Add a lightweight CI check: `python -m compileall -q .` + `pytest --collect-only`.
