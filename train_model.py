import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import joblib
import json
import glob
import os

# Diretório de saída dos artefatos (sobrescrevível via variável de ambiente)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
RISK_PATH = os.path.join(MODEL_DIR, "clientes_em_risco.csv")
# Base inteira pontuada, consumida pelo painel do Banco Avenida
DASH_PATH = os.path.join(MODEL_DIR, "dashboard_data.json")
# Pasta dos templates de página; cada .html daqui vira uma página em MODEL_DIR
PANEL_DIR = os.getenv("PANEL_DIR", "painel")
os.makedirs(MODEL_DIR, exist_ok=True)

# Onde o CSV do dataset fica guardado (baixado uma vez e reaproveitado)
DATA_DIR = os.getenv("DATA_DIR", "data")
DATA_PATH = os.getenv("DATA_PATH", os.path.join(DATA_DIR, "bank_customer_churn.csv"))
# Contas separadas antes do treino: o modelo nunca as vê. Ficam guardadas para
# serem injetadas depois na API, como se fossem contas novas chegando no banco.
NOVAS_PATH = os.path.join(DATA_DIR, "contas_novas.csv")
# O desfecho real dessas contas, à parte, para conferir as notas da API depois
GABARITO_PATH = os.path.join(DATA_DIR, "contas_novas_gabarito.csv")
# Os mesmos 10% inteiros, no formato da planilha original (com a coluna churn)
SEPARADAS_PATH = os.path.join(DATA_DIR, "contas_10_porcento.csv")
# Fração da base guardada fora do treino (0.10 -> treina com 90%)
HOLDOUT = float(os.getenv("HOLDOUT", "0.10"))
# Espelho público do Bank Customer Churn Dataset (Kaggle: gauravtopre)
DATA_URL = os.getenv(
    "DATA_URL",
    "https://media.githubusercontent.com/media/akshara-a/open-data-intelligence-hub/main/"
    "Mentee%20Contribution/Task%203%20-%20Custom%20Churn%20-%20Exploratory%20Data%20Analysis/"
    "Prashant_Kasar_G40_AIML/Bank%20Customer%20Churn%20Prediction.csv",
)

# A partir de qual probabilidade o cliente entra na lista de risco
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
# Quantos clientes de maior risco exportar no CSV
RISK_TOP = int(os.getenv("RISK_TOP", "200"))

# 1. Carrega o Bank Customer Churn Dataset (cache local, baixa só na primeira vez)
if os.path.exists(DATA_PATH):
    print(f"[1/7] Lendo dataset local '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
else:
    print("[1/7] Baixando Bank Customer Churn Dataset...")
    df = pd.read_csv(DATA_URL)
    os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"      Cache salvo em '{DATA_PATH}'.")

# 2. Tratamento mínimo de dados
df_all = df.dropna().copy()

# Separa as contas guardadas ANTES de qualquer coisa. random_state fixo: toda
# execução guarda as mesmas contas, então elas nunca vazam para um treino futuro.
df_clean, df_novas = train_test_split(
    df_all, test_size=HOLDOUT, random_state=42, stratify=df_all["churn"])

os.makedirs(DATA_DIR, exist_ok=True)
df_novas.drop(columns=["churn"]).to_csv(NOVAS_PATH, index=False)
df_novas[["customer_id", "churn"]].to_csv(GABARITO_PATH, index=False)
df_novas.to_csv(SEPARADAS_PATH, index=False)

# Target já vem binário na coluna 'churn': 1 = Cancelou, 0 = Ativo
y = df_clean["churn"].astype(int)

# customer_id é só identificador, não ajuda a prever nada
# country (France/Germany/Spain) e gender viram colunas numéricas (one-hot)
X = pd.get_dummies(
    df_clean.drop(columns=["customer_id", "churn"]),
    columns=["country", "gender"],
    drop_first=True,
)  # features/caracteristicas/atributos são as variáveis independentes

print(f"[2/7] {len(df_clean)} clientes para treino ({100 - HOLDOUT * 100:.0f}%), "
      f"{X.shape[1]} features, {y.mean() * 100:.1f}% de churn.")
print(f"      {len(df_novas)} contas guardadas fora do treino em '{NOVAS_PATH}' "
      f"(desfecho em '{GABARITO_PATH}', completas em '{SEPARADAS_PATH}').")

# 3. Treinamento do Modelo
# class_weight='balanced' -> só 20% da base cancelou; sem isso a árvore aprende
# a chutar "fica" pra todo mundo e nunca sinaliza risco nenhum
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)

# 4. Avaliação por validação cruzada (5 partes)
# O treino usa os 90% inteiros, então não sobra conjunto de teste separado. Em vez
# disso, cada cliente recebe a nota de uma árvore treinada nas outras 4 partes:
# toda nota abaixo vem de um modelo que nunca viu aquele cliente.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
print("[3/7] Validação cruzada em 5 partes concluída.")

# Modelo final: a mesma árvore treinada com os 90% inteiros — é ela que a API usa
model.fit(X, y)
print(f"[4/7] Modelo final treinado com {len(X)} clientes.")

# 5. Avaliação — o que importa aqui é o RECALL da classe 1 (quantos dos que
# realmente cancelaram o modelo conseguiu sinalizar antes)
pred_oof = (proba_oof >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y, pred_oof).ravel()
print(f"[5/7] Avaliação fora do treino (threshold {THRESHOLD:.2f}):")
print(f"      Acurácia:  {accuracy_score(y, pred_oof) * 100:.2f}%")
print(f"      Recall:    {recall_score(y, pred_oof) * 100:.2f}%  "
      f"(pegou {tp} dos {tp + fn} que cancelaram)")
print(f"      Precisão:  {precision_score(y, pred_oof, zero_division=0) * 100:.2f}%  "
      f"(dos {tp + fp} sinalizados, {tp} cancelaram mesmo)")
print(f"      F1:        {f1_score(y, pred_oof) * 100:.2f}%")
print(f"      Falsos alarmes: {fp}   |   Escaparam: {fn}")

# 6. Exportação dos artefatos
# Salva o modelo junto das colunas e do threshold: a ordem das features do
# one-hot precisa ser a mesma na hora de pontuar clientes novos
joblib.dump(
    {"model": model, "features": list(X.columns), "threshold": THRESHOLD},
    MODEL_PATH,
)

# Ranqueia quem tem mais chance de sair, usando a nota da validação cruzada:
# a do modelo final seria otimista, porque ele treinou com esses mesmos clientes.
risco = df_clean.copy()
risco["risco_churn"] = proba_oof
risco = risco[risco["risco_churn"] >= THRESHOLD].sort_values(
    "risco_churn", ascending=False)

colunas_saida = ["customer_id", "risco_churn", "age", "country", "gender",
                 "tenure", "balance", "products_number", "active_member",
                 "credit_score", "estimated_salary", "churn"]
risco.head(RISK_TOP)[colunas_saida].to_csv(RISK_PATH, index=False)

print(f"[6/7] Modelo salvo em '{MODEL_PATH}'.")
print(f"      {len(risco)} clientes acima do threshold; "
      f"top {min(RISK_TOP, len(risco))} exportados em '{RISK_PATH}'.")

# 7. Base inteira pontuada em JSON, para o painel do Banco Avenida
# O painel refaz o corte de risco no navegador, então ele precisa da nota de
# todo mundo — não só de quem passou do threshold — mais as métricas do passo 5.
dash = df_clean.copy()
dash["risco_churn"] = proba_oof
# Toda nota veio de uma árvore que não viu o cliente, então todos contam como teste
dash["teste"] = True

colunas_dash = ["customer_id", "risco_churn", "credit_score", "country",
                "gender", "age", "tenure", "balance", "products_number",
                "credit_card", "active_member", "estimated_salary", "churn",
                "teste"]

# Formato colunar (lista de listas) em vez de um objeto por cliente: mesmo
# conteúdo, arquivo bem menor com 10 mil linhas
linhas = [
    [int(r.customer_id), round(float(r.risco_churn), 4), int(r.credit_score),
     r.country, r.gender, int(r.age), int(r.tenure), round(float(r.balance), 2),
     int(r.products_number), int(r.credit_card), int(r.active_member),
     round(float(r.estimated_salary), 2), int(r.churn), bool(r.teste)]
    for r in dash[colunas_dash].itertuples(index=False)
]

payload = json.dumps(
    {
        "threshold": THRESHOLD,
        "n_clientes": len(dash),
        "taxa_churn_base": round(float(y.mean()), 4),
        "metricas_teste": {
            "n": int(len(y)),
            "acuracia": round(float(accuracy_score(y, pred_oof)), 4),
            "recall": round(float(recall_score(y, pred_oof)), 4),
            "precisao": round(float(precision_score(y, pred_oof, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred_oof)), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        # O quanto cada feature pesou nas decisões da árvore
        "importancias": sorted(
            ((c, round(float(i), 4)) for c, i in zip(X.columns, model.feature_importances_)),
            key=lambda kv: kv[1], reverse=True,
        ),
        "colunas": colunas_dash,
        "clientes": linhas,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

with open(DASH_PATH, "w", encoding="utf-8") as f:
    f.write(payload)

print(f"[7/7] Base pontuada exportada em '{DASH_PATH}' "
      f"({len(linhas)} clientes).")

# Cada página é um HTML só: o template vem de painel/ e os dados entram no lugar
# do marcador, então o arquivo em models/ abre no navegador sem servidor nem rede.
# Hoje são dois: painel.html (console completo) e resumo.html (apresentação).
templates = sorted(glob.glob(os.path.join(PANEL_DIR, "*.html")))
if templates:
    # '</' escapado para nenhum dado conseguir fechar a tag <script> antes da hora
    dados_js = payload.replace("</", "<\\/")
    for template_path in templates:
        with open(template_path, encoding="utf-8") as f:
            pagina = f.read().replace("/*DADOS*/null", dados_js)
        destino = os.path.join(MODEL_DIR, os.path.basename(template_path))
        with open(destino, "w", encoding="utf-8") as f:
            f.write(pagina)
        print(f"      Página gerada em '{destino}'.")
else:
    print(f"      Nenhum template .html em '{PANEL_DIR}'; páginas não geradas.")
