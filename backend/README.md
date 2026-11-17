# jarvis-core

Python package for the JARVIS agent core. Phase 1 contains the configuration
loader and the unified, hot-swappable model router (LLM + STT + TTS).

See the repository root `README.md` and `docs/model-router.md`.

```bash
cd backend
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
PYTHONPATH=src .venv/bin/python -m pytest -q
```
