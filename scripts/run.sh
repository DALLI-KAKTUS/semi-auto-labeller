#!/bin/sh
# TUI'yi açar. Komut satırı sürümü: .venv/bin/python -m app.cli --help
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -W ignore -m app "$@"
