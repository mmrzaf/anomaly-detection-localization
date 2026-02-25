set dotenv-load := true
set shell := ["bash", "-cu"]

APP      := "app.main:app"
HOST     := "0.0.0.0"
PORT     := "8008"
LOGLEVEL := "info"
RELOAD   := "true"
WORKERS  := "1"
UVICORN  := "uvicorn"

VENV := ".venv"
PY   := "python3"
#PY   := "{{VENV}}/bin/python3"
PIP  := "{{VENV}}/bin/pip"
BIN  := "{{VENV}}/bin"

default: dev

help:
	@echo "Available tasks:"
	@just --list

ensure-venv:
	@test -d {{VENV}} || (python -m venv {{VENV}} && {{PIP}} install -U pip wheel)

# ---- Environment / deps ----
# Fresh install from requirements (dev takes precedence if present)
install:
	@just ensure-venv
	@if [ -f requirements-dev.txt ]; then \
		{{PIP}} install -r requirements-dev.txt; \
	elif [ -f requirements.txt ]; then \
		{{PIP}} install -r requirements.txt; \
	else \
		echo "No requirements*.txt found; skipping."; \
	fi

# Create/refresh a pinned requirements.txt without touching the env if pip-tools is available.
# Fallback: freeze current env.
lock:
	@if [ -f requirements.in ]; then \
		if {{PY}} -c "import piptools" >/dev/null 2>&1; then \
			{{PY}} -m piptools compile --upgrade -o requirements.txt requirements.in; \
		else \
			echo "pip-tools not installed; run '$(PIP) install pip-tools' or use 'just requirements' to freeze."; \
			exit 1; \
		fi \
	else \
		echo "No requirements.in found; freezing current env to requirements.txt."; \
		{{PIP}} freeze > requirements.txt; \
	fi

# Update all deps to latest allowed by requirements.* (simple pip upgrade)
update:
	@just ensure-venv
	@if [ -f requirements.txt ]; then \
		{{PIP}} install -U -r requirements.txt; \
	else \
		echo "requirements.txt not found."; \
		exit 1; \
	fi

# Export requirements by freezing the current environment
requirements:
	@just ensure-venv
	@{{PIP}} freeze > requirements.txt

# ---- Code quality ----
# Lint (no changes)
lint:
	@{{BIN}}/ruff check .

# Auto-fix + format
fmt:
	@{{BIN}}/ruff check . --fix
	@{{BIN}}/ruff format .

# Type check
typecheck:
	@{{BIN}}/mypy .

# Test (quiet)
test:
	@{{BIN}}/pytest -q

# Full style gate
check: fmt lint typecheck test

# ---- Running the app ----
# Dev server
dev:
	@{{PY}} -m {{UVICORN}} {{APP}} \
		--host {{HOST}} \
		--port {{PORT}} \
		--env-file .env \
		--log-level {{LOGLEVEL}} \
		$( [ "{{RELOAD}}" = "true" ] && echo --reload )

# Prod-ish serve
serve:
	@{{PY}} -m {{UVICORN}} {{APP}} \
		--host {{HOST}} \
		--port {{PORT}} \
		--log-level {{LOGLEVEL}} \
		--workers {{WORKERS}} \
		--env-file .env

# ---- Utilities ----
# Drop into the venv shell
shell:
	@. {{VENV}}/bin/activate && exec "$$SHELL"

# Print current config
print-config:
	@echo "APP={{APP}}"
	@echo "HOST={{HOST}}"
	@echo "PORT={{PORT}}"
	@echo "LOGLEVEL={{LOGLEVEL}}"
	@echo "RELOAD={{RELOAD}}"
	@echo "WORKERS={{WORKERS}}"

# Clean typical Python cruft
clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@rm -rf .mypy_cache .pytest_cache .ruff_cache dist build

# Build wheel + sdist using the 'build' module
build:
	@just ensure-venv
	@{{PIP}} install -U build
	@{{PY}} -m build

