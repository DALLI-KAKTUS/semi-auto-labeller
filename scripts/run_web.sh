#!/bin/sh
# Web uygulamasını yerelde başlatır (http://localhost:8000). Ayarlar ortam değişkenlerinden (web/settings.py):
#   LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY, ENG_ADMIN_USERNAME, ENG_ADMIN_PASSWORD, ENG_DATA_DIR
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -m uvicorn web.server:create_app --factory --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
