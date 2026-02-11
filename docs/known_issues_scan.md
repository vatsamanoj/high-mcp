# Known Issues Scan Report

Date: 2026-02-11
Repository: `/workspace/high-mcp`

## Scan commands used

- `python -m compileall -q .`
- `pytest -q`
- `rg -n "TODO|FIXME|BUG|HACK|XXX|pass #|NotImplemented"`
- `rg -n "subprocess|create_subprocess|requests\.|httpx\.|verify=False|api_key|SECRET|password" *.py verification_scripts/*.py testing_scripts/*.py components/*.py`

## Findings

### 1) Test suite cannot be collected in a clean environment due to missing runtime dependencies

`pytest -q` fails at collection time with `ModuleNotFoundError` for:

- `aiosqlite`
- `requests`

This indicates required imports used by tests and runtime modules are not fully declared in `requirements.txt`.

### 2) Hardcoded API key discovered in verification script

A literal NVIDIA API key string was present in `verification_scripts/debug_nvidia.py`.

Risk:

- credential leakage if committed/shared
- accidental unauthorized usage

Mitigation implemented:

- script now reads `NVIDIA_API_KEY` from environment
- exits with a clear message if variable is not set

### 3) Extensive DEBUG logging remains in production server path

Multiple `DEBUG:` log writes/prints were found in `ui_server.py` and `ai_engine.py`.

Risk:

- possible sensitive operational metadata leakage
- noisy logs in production

Recommendation:

- gate debug logs behind a dedicated debug flag or logger level checks
- avoid printing request internals unless explicitly enabled

## Changes made from this scan

- Added missing dependencies in `requirements.txt`: `aiosqlite`, `requests`
- Removed hardcoded API key from `verification_scripts/debug_nvidia.py` in favor of env var usage

## Follow-up recommendations

1. Run `pip install -r requirements.txt` and then rerun `pytest -q`.
2. Add a CI job that performs dependency validation and test collection.
3. Optionally add secret scanning in CI (e.g., gitleaks/trufflehog) to prevent future key leaks.
