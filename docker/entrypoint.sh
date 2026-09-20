#!/usr/bin/env bash
# Entrypoint for the FRGfp development container.
#
# Installs the package in editable mode on first start, so that the
# bind-mounted sources are importable without rebuilding the image. Python
# changes are then picked up immediately; re-run `pip install -e .` after
# changing the C++ sources.
set -e

if ! python -c "import frgfp" >/dev/null 2>&1; then
    echo "[dev-container] Installing FRGfp in editable mode (first start)..."
    pip install -e . || echo "[dev-container] WARNING: 'pip install -e .' failed; run it manually."
fi

exec "$@"
