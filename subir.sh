#!/usr/bin/env bash
#
# Sobe a aplicação inteira: treina o modelo e publica o painel.
#
#   ./subir.sh              treina e sobe o site
#   ./subir.sh --abrir      idem, e abre o navegador no painel
#   ./subir.sh --rebuild    refaz a imagem antes (use ao mexer no requirements.txt)
#   ./subir.sh --parar      derruba o site
#
set -euo pipefail

# Roda a partir da pasta do script, então funciona chamado de qualquer lugar
cd "$(dirname "$0")"

PORTA=8080
URL="http://localhost:${PORTA}"

abrir=0
rebuild=0
parar=0

for arg in "$@"; do
  case "$arg" in
    --abrir)   abrir=1 ;;
    --rebuild) rebuild=1 ;;
    --parar)   parar=1 ;;
    -h|--help) sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Opção desconhecida: $arg (use --help)" >&2; exit 2 ;;
  esac
done

# --- Docker está de pé? ---------------------------------------------------
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose não encontrado. Instale o Docker Desktop." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "O Docker não está rodando. Abra o Docker Desktop, espere ele terminar de subir" >&2
  echo "e rode este script de novo." >&2
  exit 1
fi

# --- Parar ----------------------------------------------------------------
if [ "$parar" -eq 1 ]; then
  docker compose down
  echo "Site derrubado."
  exit 0
fi

# --- Imagem ---------------------------------------------------------------
if [ "$rebuild" -eq 1 ]; then
  echo "==> Refazendo a imagem"
  docker compose build
fi

# --- Treino ---------------------------------------------------------------
# Em primeiro plano de propósito: as métricas do modelo aparecem na tela.
# Se o treino falhar, o set -e derruba o script aqui e o site não sobe com
# uma página velha.
echo "==> Treinando o modelo"
docker compose run --rm trainer

# --- Site -----------------------------------------------------------------
echo
echo "==> Subindo o site"
docker compose up -d --no-deps site

# Espera o nginx responder antes de dizer que está pronto
if command -v curl >/dev/null 2>&1; then
  for _ in $(seq 1 20); do
    if curl -fsS -o /dev/null "$URL"; then break; fi
    sleep 0.5
  done
  if ! curl -fsS -o /dev/null "$URL"; then
    echo "O site subiu mas não respondeu em ${URL}." >&2
    echo "Veja o que houve com: docker compose logs site" >&2
    exit 1
  fi
fi

echo
echo "Aplicação no ar:"
echo "  ${URL}               painel de retenção"
echo "  ${URL}/contas.html   só a lista de contas em risco"
echo "  ${URL}/resumo.html   resumo para apresentação"
echo
echo "Para derrubar: ./subir.sh --parar"

# --- Navegador ------------------------------------------------------------
if [ "$abrir" -eq 1 ]; then
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 &
  elif command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$URL" >/dev/null 2>&1 &
  else echo "Não consegui abrir o navegador sozinho; acesse ${URL}."
  fi
fi
