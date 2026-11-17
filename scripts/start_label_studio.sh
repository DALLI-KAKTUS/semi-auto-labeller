#!/bin/sh
# Label Studio'yu yerelde başlatır (http://localhost:8080). Veritabanı ve yüklenen görüntüler .ls-data/ altında.
cd "$(dirname "$0")/.." || exit 1
export LABEL_STUDIO_BASE_DATA_DIR="$PWD/.ls-data"
exec .venv-ls/bin/label-studio start --port 8080 "$@"
