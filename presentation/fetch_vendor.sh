#!/usr/bin/env bash
# Vendor reveal.js + Chart.js locally so the deck records with NO network dependency.
# If a download fails (offline), the deck still works: index.html falls back to CDN
# <script> tags automatically when vendor/ files are missing.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor"
mkdir -p "$VENDOR"

REVEAL_VER="5.1.0"
CHARTJS_VER="4.4.3"

fetch() {  # url  dest
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest" && echo "  ok  $(basename "$dest")" && return 0
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dest" && echo "  ok  $(basename "$dest")" && return 0
  fi
  echo "  FAIL  $(basename "$dest")  (deck will fall back to CDN)"
  rm -f "$dest"
  return 1
}

echo "Vendoring reveal.js ${REVEAL_VER} + Chart.js ${CHARTJS_VER} into vendor/ ..."
fetch "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/${REVEAL_VER}/reveal.min.js"          "$VENDOR/reveal.min.js"
fetch "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/${REVEAL_VER}/reveal.min.css"         "$VENDOR/reveal.min.css"
fetch "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/${REVEAL_VER}/theme/black.min.css"    "$VENDOR/reveal-theme-black.min.css"
fetch "https://cdnjs.cloudflare.com/ajax/libs/reveal.js/${REVEAL_VER}/plugin/notes/notes.min.js" "$VENDOR/reveal-notes.min.js"
fetch "https://cdn.jsdelivr.net/npm/chart.js@${CHARTJS_VER}/dist/chart.umd.min.js"             "$VENDOR/chart.umd.min.js"

echo "Done. Open index.html in a browser (or: python -m http.server, then visit the page)."
