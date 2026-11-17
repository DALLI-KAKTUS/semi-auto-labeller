#!/bin/sh
# ENG yerel çalıştırıcı (macOS: ENG-Calistirici.command'a çift tıklayın; Linux: sh eng-calistirici.sh).
# İlk açılışta uv (Python ortam yöneticisi) ve gerekli paketleri kurar, sonra çalıştırıcıyı başlatır. Ek argümanlar
# çalıştırıcıya geçer (ör. --device cpu).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

pause() {
  [ -t 0 ] && { printf '\nKapatmak için Enter tuşuna basın.'; read -r _; }
}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv kuruluyor (bir kez; https://docs.astral.sh/uv/)..."
  if command -v curl >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh
  else wget -qO- https://astral.sh/uv/install.sh | sh; fi
  command -v uv >/dev/null 2>&1 || { echo "uv kurulamadı; https://docs.astral.sh/uv/ adresinden kurun."; pause; exit 1; }
fi

if [ ! -x .venv/bin/python ] || [ ! -f .venv/.kurulum-tamam ] || [ requirements-runner.txt -nt .venv/.kurulum-tamam ]; then
  echo "Python ortamı hazırlanıyor (ilk açılışta birkaç dakika sürebilir)..."
  uv venv --python 3.11 --allow-existing .venv \
    && uv pip install --python .venv --torch-backend auto -r requirements-runner.txt \
    && touch .venv/.kurulum-tamam \
    || { echo "Kurulum başarısız (internet bağlantısını kontrol edip yeniden deneyin)."; pause; exit 1; }
fi

.venv/bin/python -W ignore -m runner "$@"
status=$?
pause
exit $status
