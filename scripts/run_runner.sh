#!/bin/sh
# Yerel çalıştırıcı: modeli bu bilgisayarda çalıştırır, web uygulamasından iş alır.
#   scripts/run_runner.sh --server https://eng.ornek.com --token engr_... --model models/lite_v8.json=/yol/lite_net.pt
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -W ignore -m runner "$@"
