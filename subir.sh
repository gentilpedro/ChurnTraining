#!/usr/bin/env bash
#
# Sobe a aplicação inteira: treina o modelo, publica o painel e a API.
#
#   ./subir.sh              treina, sobe o site e abre o painel no navegador
#   ./subir.sh --sem-abrir  idem, sem abrir o navegador
#   ./subir.sh --rebuild    refaz a imagem antes (use ao mexer no requirements.txt)
#   ./subir.sh --parar      derruba o site e a API
#
set -euo pipefail

# Roda a partir da pasta do script, então funciona chamado de qualquer lugar
cd "$(dirname "$0")"

PORTA=8080
URL="http://localhost:${PORTA}"
API_URL="http://localhost:8000"

# Abrir o painel é o padrão: quem roda isso quer ver a aplicação, não um prompt
abrir=1
rebuild=0
parar=0

for arg in "$@"; do
  case "$arg" in
    --sem-abrir) abrir=0 ;;
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
  echo "Site e API derrubados."
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

# --- Site e API -----------------------------------------------------------
# --force-recreate nos dois: a API lê o modelo só na subida, e o nginx lê o
# nginx.conf só na subida; reiniciar garante que os dois usam o que está no disco
echo
echo "==> Subindo o site e a API"
docker compose up -d --no-deps --force-recreate site
docker compose up -d --no-deps --force-recreate api

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
  for _ in $(seq 1 30); do
    if curl -fsS -o /dev/null "${API_URL}/saude" 2>/dev/null; then break; fi
    sleep 0.5
  done
  if ! curl -fsS -o /dev/null "${API_URL}/saude"; then
    echo "A API subiu mas não respondeu em ${API_URL}." >&2
    echo "Veja o que houve com: docker compose logs api" >&2
    exit 1
  fi
fi

# --- Navegador ------------------------------------------------------------
# No Windows é explorer.exe, não 'cmd.exe /c start': o Git Bash converte o /c
# em caminho (C:\...) antes do cmd ver, e o que abre é um prompt interativo.
# O explorer devolve código 1 mesmo quando abre certo, daí o '|| true' — sem ele
# o set -e derrubaria o script depois da aplicação já estar no ar.
if [ "$abrir" -eq 1 ]; then
  echo
  echo "==> Abrindo o painel no navegador"
  if command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$URL" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  else
    echo "Não achei como abrir o navegador daqui; abra ${URL} na mão."
  fi
fi

echo
echo "Aplicação no ar:"
echo "  ${URL}               painel de retenção"
echo "  ${URL}/contas.html   só a lista de contas em risco"
echo "  ${URL}/resumo.html   resumo para apresentação"
echo "  ${URL}/injetar.html  injetar contas na API pelo navegador"
echo "  ${API_URL}/docs               Churn API (consulta interativa)"
echo
echo "Injetar as contas guardadas fora do treino:"
echo "  ${URL}/injetar.html  e envie data/contas_10_porcento.csv"
echo "  python cliente/injetar.py csv data/contas_novas.csv"
echo "  python cliente/injetar.py            (menu)"
echo
echo "Para derrubar: ./subir.sh --parar"
