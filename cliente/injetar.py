"""Injeta contas bancárias na Churn API e mostra quais têm mais chance de sair.

Só usa a biblioteca padrão: roda com qualquer Python 3.10+, sem pip install.

    python cliente/injetar.py                           menu interativo
    python cliente/injetar.py csv data/contas_novas.csv injeta um CSV inteiro
    python cliente/injetar.py nova                      digita uma conta na mão
    python cliente/injetar.py ranking [--limite 20] [--todas]
    python cliente/injetar.py conta 15634602            nota de uma conta injetada
    python cliente/injetar.py conferir data/contas_novas_gabarito.csv
    python cliente/injetar.py limpar                    apaga as contas injetadas

A API padrão é http://localhost:8000; troque com --api ou a variável CHURN_API.
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

LOTE = 1000  # contas por requisição ao injetar um CSV

# Campo -> (pergunta, conversor). A ordem é a do dataset.
CAMPOS = {
    "customer_id":      ("Número da conta (customer_id)", int),
    "credit_score":     ("Score de crédito (300-900)", int),
    "country":          ("País [France/Germany/Spain]", str),
    "gender":           ("Gênero [Female/Male]", str),
    "age":              ("Idade", int),
    "tenure":           ("Anos como cliente", int),
    "balance":          ("Saldo em conta", float),
    "products_number":  ("Quantidade de produtos", int),
    "credit_card":      ("Tem cartão de crédito? [0/1]", int),
    "active_member":    ("Cliente ativo? [0/1]", int),
    "estimated_salary": ("Salário estimado", float),
}

# Campos de resposta fechada: conferidos já na digitação, antes de ir para a API
OPCOES = {"country": ("France", "Germany", "Spain"), "gender": ("Female", "Male"),
          "credit_card": (0, 1), "active_member": (0, 1)}

PAIS = {"France": "França", "Germany": "Alemanha", "Spain": "Espanha"}


def moeda(v: float) -> str:
    # 164769.02 -> 164.769,02, no formato brasileiro do painel
    return f"{v:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


class ErroApi(Exception):
    pass


def chamar(api: str, metodo: str, caminho: str, corpo=None):
    dados = json.dumps(corpo).encode() if corpo is not None else None
    req = urllib.request.Request(
        api.rstrip("/") + caminho, data=dados, method=metodo,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detalhe = e.read().decode(errors="replace")
        try:
            detalhe = json.loads(detalhe).get("detail", detalhe)
        except ValueError:
            pass
        if isinstance(detalhe, list):  # erro de validação do FastAPI
            detalhe = "; ".join(
                f"{'.'.join(str(p) for p in d['loc'][1:])}: {d['msg']}" for d in detalhe)
        raise ErroApi(f"A API recusou ({e.code}): {detalhe}") from None
    except urllib.error.URLError as e:
        raise ErroApi(f"Não consegui falar com a API em {api} ({e.reason}). "
                      "Ela está no ar? Rode ./subir.sh") from None


def converter(linha: dict, n: int) -> dict:
    try:
        return {k: conv(linha[k].strip()) if conv is not str else linha[k].strip()
                for k, (_, conv) in CAMPOS.items()}
    except KeyError as e:
        raise ErroApi(f"O CSV não tem a coluna {e}.") from None
    except ValueError as e:
        raise ErroApi(f"Linha {n} do CSV com valor inválido: {e}") from None


def tabela(contas: list[dict]):
    if not contas:
        print("  (nenhuma conta)")
        return
    print(f"  {'Conta':>10}  {'Risco':>6}  {'':8}  {'Idade':>5}  {'País':9} "
          f"{'Prod.':>5}  {'Ativo':5}  {'Saldo':>12}")
    for c in contas:
        marca = "EM RISCO" if c["em_risco"] else ""
        print(f"  {c['customer_id']:>10}  {100 * c['risco_churn']:5.1f}%  {marca:8}  "
              f"{c['age']:>5}  {PAIS.get(c['country'], c['country']):9} "
              f"{c['products_number']:>5}  {'sim' if c['active_member'] else 'não':5}  "
              f"{moeda(c['balance']):>12}")


def cmd_csv(api: str, caminho: str, mostrar: int = 20):
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        contas = [converter(l, i) for i, l in enumerate(csv.DictReader(f), start=2)]
    if not contas:
        raise ErroApi(f"'{caminho}' não tem nenhuma conta.")

    pontuadas = []
    for i in range(0, len(contas), LOTE):
        pontuadas += chamar(api, "POST", "/contas", contas[i:i + LOTE])
        print(f"  enviadas {min(i + LOTE, len(contas))}/{len(contas)}", end="\r")
    print()

    pontuadas.sort(key=lambda c: c["risco_churn"], reverse=True)
    em_risco = sum(c["em_risco"] for c in pontuadas)
    print(f"{len(pontuadas)} contas injetadas; {em_risco} acima do corte de risco "
          f"({100 * em_risco / len(pontuadas):.1f}%).\n")
    print(f"As {min(mostrar, len(pontuadas))} com mais chance de sair:")
    tabela(pontuadas[:mostrar])


def perguntar(pergunta: str, conv, opcoes=None):
    while True:
        valor = input(f"  {pergunta}: ").strip()
        try:
            valor = conv(valor) if conv is not str else valor
        except ValueError:
            print("    valor inválido, tente de novo")
            continue
        if opcoes is None or valor in opcoes:
            return valor
        print(f"    use um destes: {', '.join(map(str, opcoes))}")


def cmd_nova(api: str):
    print("Nova conta (a API valida os valores):")
    conta = {k: perguntar(p, conv, OPCOES.get(k)) for k, (p, conv) in CAMPOS.items()}
    [c] = chamar(api, "POST", "/contas", [conta])
    print(f"\nConta {c['customer_id']}: {100 * c['risco_churn']:.1f}% de chance de sair"
          f" — {'EM RISCO' if c['em_risco'] else 'estável'}.")


def cmd_ranking(api: str, limite: int, todas: bool):
    contas = chamar(api, "GET", f"/contas?limite={limite}&so_em_risco={str(not todas).lower()}")
    titulo = "Contas injetadas" if todas else "Contas injetadas em risco"
    print(f"{titulo}, da maior chance de sair para a menor:")
    tabela(contas)


def cmd_conta(api: str, customer_id: int):
    tabela([chamar(api, "GET", f"/contas/{customer_id}")])


def cmd_conferir(api: str, gabarito: str):
    """Compara as notas da API com o que de fato aconteceu com as contas guardadas."""
    with open(gabarito, newline="", encoding="utf-8-sig") as f:
        real = {int(l["customer_id"]): int(l["churn"]) for l in csv.DictReader(f)}
    notas = chamar(api, "GET", "/contas?so_em_risco=false&limite=10000")
    pares = [(c["em_risco"], real[c["customer_id"]]) for c in notas
             if c["customer_id"] in real]
    if not pares:
        raise ErroApi("Nenhuma conta do gabarito foi injetada ainda. "
                      "Injete antes: python cliente/injetar.py csv data/contas_novas.csv")

    tp = sum(p and r for p, r in pares)
    fp = sum(p and not r for p, r in pares)
    fn = sum(not p and r for p, r in pares)
    tn = len(pares) - tp - fp - fn
    print(f"Conferência de {len(pares)} contas que o modelo nunca viu:")
    print(f"  Acurácia:  {100 * (tp + tn) / len(pares):.1f}%")
    print(f"  Recall:    {100 * tp / max(tp + fn, 1):.1f}%  (pegou {tp} dos {tp + fn} que cancelaram)")
    print(f"  Precisão:  {100 * tp / max(tp + fp, 1):.1f}%  (dos {tp + fp} sinalizados, {tp} cancelaram mesmo)")
    print(f"  Falsos alarmes: {fp}   |   Escaparam: {fn}")


def cmd_limpar(api: str):
    print(f"{chamar(api, 'DELETE', '/contas')['apagadas']} contas apagadas.")


def menu(api: str):
    opcoes = {
        "1": ("Injetar um CSV", lambda: cmd_csv(api, input("  Caminho do CSV [data/contas_novas.csv]: ").strip() or "data/contas_novas.csv")),
        "2": ("Digitar uma conta nova", lambda: cmd_nova(api)),
        "3": ("Ver contas em risco", lambda: cmd_ranking(api, 20, False)),
        "4": ("Ver todas as contas injetadas", lambda: cmd_ranking(api, 50, True)),
        "5": ("Conferir com o gabarito", lambda: cmd_conferir(api, "data/contas_novas_gabarito.csv")),
        "6": ("Apagar contas injetadas", lambda: cmd_limpar(api)),
    }
    while True:
        print(f"\n== Churn API ({api}) ==")
        for k, (nome, _) in opcoes.items():
            print(f"  {k}. {nome}")
        print("  0. Sair")
        escolha = input("> ").strip()
        if escolha in ("0", "", "q"):
            return
        if escolha not in opcoes:
            continue
        print()
        try:
            opcoes[escolha][1]()
        except (ErroApi, OSError) as e:
            print(f"Erro: {e}")


def main():
    # No Git Bash a saída não é um console do Windows e o Python cairia no cp1252,
    # trocando "País" por "Pa�s"
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Injeta contas na Churn API.")
    ap.add_argument("--api", default=os.getenv("CHURN_API", "http://localhost:8000"))
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("csv", help="injeta um CSV com as colunas do dataset")
    p.add_argument("arquivo")
    p.add_argument("--mostrar", type=int, default=20, help="quantas mostrar no fim")
    sub.add_parser("nova", help="digita uma conta na mão")
    p = sub.add_parser("ranking", help="contas injetadas por risco")
    p.add_argument("--limite", type=int, default=20)
    p.add_argument("--todas", action="store_true", help="inclui as fora de risco")
    p = sub.add_parser("conta", help="nota de uma conta injetada")
    p.add_argument("customer_id", type=int)
    p = sub.add_parser("conferir", help="compara as notas com o desfecho real")
    p.add_argument("gabarito", nargs="?", default="data/contas_novas_gabarito.csv")
    sub.add_parser("limpar", help="apaga as contas injetadas")
    a = ap.parse_args()

    try:
        match a.cmd:
            case None: menu(a.api)
            case "csv": cmd_csv(a.api, a.arquivo, a.mostrar)
            case "nova": cmd_nova(a.api)
            case "ranking": cmd_ranking(a.api, a.limite, a.todas)
            case "conta": cmd_conta(a.api, a.customer_id)
            case "conferir": cmd_conferir(a.api, a.gabarito)
            case "limpar": cmd_limpar(a.api)
    except (ErroApi, OSError) as e:
        sys.exit(f"Erro: {e}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
