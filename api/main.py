"""Minimal API de consulta de churn.

Carrega o modelo gerado pelo train_model.py e pontua contas novas. As contas
injetadas por POST /contas ficam guardadas num SQLite, para depois consultar
quais delas têm mais chance de largar o banco.

    uvicorn api.main:app --port 8000      (docs interativas em /docs)
"""
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "churn_model.pkl"))
DB_PATH = os.getenv("DB_PATH", os.path.join("data", "contas_injetadas.db"))

# Mesmas colunas do dataset, sem o churn: é o que o modelo usa para dar a nota
CAMPOS = ["customer_id", "credit_score", "country", "gender", "age", "tenure",
          "balance", "products_number", "credit_card", "active_member",
          "estimated_salary"]


class Conta(BaseModel):
    customer_id: int = Field(ge=1)
    credit_score: int = Field(ge=300, le=900)
    country: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]
    age: int = Field(ge=18, le=110)
    tenure: int = Field(ge=0, le=80, description="Anos como cliente")
    balance: float = Field(ge=0)
    products_number: int = Field(ge=1, le=10)
    credit_card: Literal[0, 1]
    active_member: Literal[0, 1]
    estimated_salary: float = Field(ge=0)


class ContaPontuada(Conta):
    risco_churn: float = Field(description="Probabilidade de sair, de 0 a 1")
    em_risco: bool = Field(description="risco_churn >= threshold do modelo")


# Até 5 mil contas por requisição; o cliente manda CSVs maiores em lotes
Lote = Annotated[list[Conta], Body(min_length=1, max_length=5000)]

modelo: dict = {}


def conectar() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@asynccontextmanager
async def lifespan(_: FastAPI):
    # O modelo é lido uma vez na subida: depois de treinar de novo, reinicie a API
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em '{MODEL_PATH}'. Rode o treino antes.")
    modelo.update(joblib.load(MODEL_PATH))

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with closing(conectar()) as con, con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS contas (
                customer_id      INTEGER PRIMARY KEY,
                credit_score     INTEGER NOT NULL,
                country          TEXT    NOT NULL,
                gender           TEXT    NOT NULL,
                age              INTEGER NOT NULL,
                tenure           INTEGER NOT NULL,
                balance          REAL    NOT NULL,
                products_number  INTEGER NOT NULL,
                credit_card      INTEGER NOT NULL,
                active_member    INTEGER NOT NULL,
                estimated_salary REAL    NOT NULL,
                risco_churn      REAL    NOT NULL,
                injetada_em      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
    yield


app = FastAPI(
    title="Churn API",
    description="Consulta a chance de cada conta bancária largar o banco.",
    lifespan=lifespan,
)


def pontuar(contas: list[Conta]) -> list[ContaPontuada]:
    df = pd.DataFrame([c.model_dump() for c in contas])
    # Mesmo one-hot do treino. Uma conta sozinha não tem todos os países, então o
    # reindex completa as colunas que faltam com 0 e descarta as de referência
    # (o treino usou drop_first) — a ordem fica idêntica à que a árvore aprendeu.
    X = pd.get_dummies(df.drop(columns=["customer_id"]),
                       columns=["country", "gender"])
    X = X.reindex(columns=modelo["features"], fill_value=0)
    probas = modelo["model"].predict_proba(X)[:, 1]
    t = modelo["threshold"]
    return [ContaPontuada(**c.model_dump(), risco_churn=round(float(p), 4),
                          em_risco=bool(p >= t))
            for c, p in zip(contas, probas)]


def da_linha(linha: sqlite3.Row) -> ContaPontuada:
    d = {k: linha[k] for k in CAMPOS}
    return ContaPontuada(**d, risco_churn=linha["risco_churn"],
                         em_risco=linha["risco_churn"] >= modelo["threshold"])


@app.get("/saude", tags=["api"])
def saude():
    with closing(conectar()) as con:
        n = con.execute("SELECT COUNT(*) FROM contas").fetchone()[0]
    return {"status": "ok", "threshold": modelo["threshold"],
            "features": modelo["features"], "contas_injetadas": n}


@app.post("/prever", response_model=list[ContaPontuada], tags=["consulta"])
def prever(contas: Lote):
    """Só consulta: devolve a nota das contas sem guardar nada."""
    return sorted(pontuar(contas), key=lambda c: c.risco_churn, reverse=True)


@app.post("/contas", response_model=list[ContaPontuada], tags=["contas"])
def injetar(contas: Lote):
    """Pontua e guarda as contas. Um customer_id repetido substitui o anterior."""
    pontuadas = pontuar(contas)
    with closing(conectar()) as con, con:
        con.executemany(
            f"INSERT OR REPLACE INTO contas ({', '.join(CAMPOS)}, risco_churn) "
            f"VALUES ({', '.join('?' * (len(CAMPOS) + 1))})",
            [[getattr(c, k) for k in CAMPOS] + [c.risco_churn] for c in pontuadas],
        )
    return sorted(pontuadas, key=lambda c: c.risco_churn, reverse=True)


@app.get("/contas", response_model=list[ContaPontuada], tags=["contas"])
def ranking(
    so_em_risco: bool = Query(True, description="Só contas acima do threshold"),
    limite: int = Query(50, ge=1, le=10000),
):
    """As contas injetadas, da maior chance de sair para a menor."""
    sql = "SELECT * FROM contas"
    args: list = []
    if so_em_risco:
        sql += " WHERE risco_churn >= ?"
        args.append(modelo["threshold"])
    sql += " ORDER BY risco_churn DESC, balance DESC LIMIT ?"
    args.append(limite)
    with closing(conectar()) as con:
        return [da_linha(r) for r in con.execute(sql, args)]


@app.get("/contas/{customer_id}", response_model=ContaPontuada, tags=["contas"])
def conta(customer_id: int):
    with closing(conectar()) as con:
        linha = con.execute("SELECT * FROM contas WHERE customer_id = ?",
                            (customer_id,)).fetchone()
    if linha is None:
        raise HTTPException(404, f"Conta {customer_id} não foi injetada.")
    return da_linha(linha)


@app.delete("/contas", tags=["contas"])
def limpar():
    """Apaga todas as contas injetadas."""
    with closing(conectar()) as con, con:
        apagadas = con.execute("DELETE FROM contas").rowcount
    return {"apagadas": apagadas}
