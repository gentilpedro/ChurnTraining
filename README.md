

# Sistema de Previsão de Churn

## 1. Qual dataset foi escolhido e qual problema ele representa?

Foi utilizado o **Bank Customer Churn Dataset**, disponível no Kaggle.

O dataset possui informações de aproximadamente **10.000 clientes bancários** e representa um problema de **Customer Churn Prediction**, ou seja, identificar clientes que possuem maior probabilidade de abandonar o banco.

Na base utilizada, aproximadamente **20,4% dos clientes cancelaram seus serviços**.

O objetivo do modelo é utilizar as características dos clientes para identificar possíveis clientes em risco de cancelamento.

---

## 2. Qual é a variável-alvo (Target)?

A variável-alvo é:

```text
churn
```

Ela representa se o cliente cancelou ou não o serviço.

| Valor | Significado              |
| ----: | ------------------------ |
|   `0` | Cliente permaneceu ativo |
|   `1` | Cliente cancelou         |

No código:

```python
y = df_clean["churn"].astype(int)
```

A variável `y` representa aquilo que o modelo deverá aprender a prever.

---

## 3. Quais são as classes possíveis?

O problema é uma **classificação binária**.

Existem duas classes:

```text
0 → Cliente ativo
1 → Cliente que cancelou
```

Portanto, o modelo precisa decidir entre essas duas possibilidades.

Além disso, o modelo fornece uma **probabilidade de churn**, por exemplo:

```text
Cliente 1001 → 0.82 → 82% de probabilidade de sair
Cliente 1002 → 0.17 → 17% de probabilidade de sair
```

---

## 4. Quais informações serão utilizadas como entrada do modelo?

As informações utilizadas como entrada são:

* `credit_score`
* `country`
* `gender`
* `age`
* `tenure`
* `balance`
* `products_number`
* `credit_card`
* `active_member`
* `estimated_salary`

O `customer_id` não é utilizado para treinar o modelo, pois ele serve apenas para identificar o cliente.

No código:

```python
X = pd.get_dummies(
    df_clean.drop(columns=["customer_id", "churn"]),
    columns=["country", "gender"],
    drop_first=True,
)
```

As informações `country` e `gender` são categóricas. Por isso, são transformadas em valores numéricos através de **One-Hot Encoding**.

---

## 5. Qual modelo de Machine Learning foi utilizado?

Foi utilizado um:

```text
DecisionTreeClassifier
```

Ou seja, uma **Árvore de Decisão**.

A configuração utilizada é:

```python
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)
```

O `class_weight="balanced"` é importante porque existe uma diferença entre a quantidade de clientes que permaneceram e a quantidade que cancelaram.

A base é dividida assim:

```text
90% → treinamento  (9.000 clientes)
10% → guardados    (1.000 contas que o modelo nunca vê)
```

Os 10% são separados **antes de tudo** e gravados em `data/contas_novas.csv` (sem a coluna
`churn`) e `data/contas_novas_gabarito.csv` (só `customer_id` e `churn`). As mesmas 1.000 contas
também saem inteiras, no formato da planilha original, em `data/contas_10_porcento.csv`. Elas
servem para serem injetadas depois na API, como contas novas chegando no banco:

```python
df_clean, df_novas = train_test_split(
    df_all, test_size=HOLDOUT, random_state=42, stratify=df_all["churn"])
```

Como o treino usa os 90% inteiros, a avaliação é feita por **validação cruzada em 5 partes**:
cada cliente recebe a nota de uma árvore treinada nas outras 4 partes. Depois, o modelo final
é treinado com os 90% e salvo para a API:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
model.fit(X, y)
```

A fração guardada pode ser mudada com a variável `HOLDOUT` (padrão `0.10`).

---

# 6. Quem utilizaria essa aplicação?

A aplicação poderia ser utilizada principalmente por:

* Gerentes bancários;
* Equipes de retenção;
* Analistas de dados;
* Equipes de relacionamento com clientes;
* Gestores comerciais.

O objetivo seria identificar antecipadamente os clientes que apresentam maior probabilidade de abandonar o banco.

Por exemplo:

```text
Cliente 8542
Probabilidade de churn: 87%

→ Entrar em contato
→ Oferecer benefício
→ Verificar satisfação
→ Tentar evitar o cancelamento
```

---

# 7. O que a aplicação faz com a classificação?

O modelo calcula uma probabilidade de churn para cada cliente.

Por exemplo:

```text
Cliente A → 91%
Cliente B → 78%
Cliente C → 63%
Cliente D → 21%
Cliente E → 8%
```

Existe um **threshold**, ou corte de risco.

O padrão utilizado no projeto é:

```text
Threshold = 0.50
```

Assim:

```text
probabilidade >= 50% → Em risco
probabilidade < 50%  → Estável
```

O threshold pode ser alterado no painel.

Isso permite controlar o equilíbrio entre:

* Recall;
* Precisão;
* Falsos positivos;
* Clientes que escaparam da identificação.

---

# 8. Como os dados chegam até o HTML?

Essa é uma das partes mais interessantes do projeto.

O painel **não usa a API** para buscar os clientes (a API é para contas novas, veja a seção 16).

Em vez disso, o Python treina o modelo e gera arquivos que serão utilizados pelo HTML.

O fluxo é:

```text
Dataset
   │
   ▼
Python / Pandas
   │
   ▼
Treinamento do modelo
   │
   ▼
Decision Tree
   │
   ▼
Probabilidade de churn
   │
   ├──────────────► clientes_em_risco.csv
   │
   ├──────────────► churn_model.pkl
   │
   └──────────────► dashboard_data.json
                              │
                              ▼
                         painel.html
                              │
                              ▼
                        JavaScript
                              │
                              ▼
                            Tela
```

---

# 9. Como o Python gera os dados do Dashboard?

Depois de treinar o modelo, o código calcula a probabilidade de churn de todos os clientes:

```python
dash = df_clean.copy()

dash["risco_churn"] = proba_oof
```

Aqui acontece algo importante.

O método:

```python
model.predict_proba(X)
```

retorna a probabilidade de cada classe.

O código:

```python
[:, 1]
```

pega a probabilidade da classe `1`, que representa:

```text
Cliente cancelou
```

Por exemplo:

```text
Cliente      Probabilidade
10001        0.87
10002        0.12
10003        0.64
```

---

# 10. O Python transforma os dados em JSON

Depois disso, o projeto monta um objeto contendo:

* threshold;
* quantidade de clientes;
* taxa de churn;
* métricas;
* importância das features;
* informações dos clientes.

Por exemplo, conceitualmente:

```json
{
  "threshold": 0.5,
  "n_clientes": 10000,
  "taxa_churn_base": 0.204,
  "metricas_teste": {
    "acuracia": 0.81,
    "recall": 0.72,
    "precisao": 0.64
  },
  "clientes": [
    [10001, 0.87, 650, "Germany", "Male", 42, 3, 12500.50],
    [10002, 0.12, 720, "France", "Female", 35, 7, 3200.00]
  ]
}
```

No projeto isso é feito através de:

```python
payload = json.dumps(
    {
        "threshold": THRESHOLD,
        "n_clientes": len(dash),
        "taxa_churn_base": round(float(y.mean()), 4),
        "metricas_teste": {
            ...
        },
        "importancias": ...,
        "colunas": colunas_dash,
        "clientes": linhas,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)
```

Depois o JSON é salvo em:

```text
models/dashboard_data.json
```

---

# 11. Mas como o HTML recebe esse JSON?

Aqui está a parte principal.

O projeto possui um template HTML:

```text
painel/painel.html
```

Dentro dele existe:

```javascript
const RAW = /*DADOS*/null;
```

Esse:

```text
/*DADOS*/
```

é um **marcador**.

O Python abre o HTML e substitui esse marcador pelo JSON gerado.

No código Python:

```python
with open(template_path, encoding="utf-8") as f:
    pagina = f.read().replace("/*DADOS*/null", dados_js)
```

Ou seja, antes:

```javascript
const RAW = /*DADOS*/null;
```

Depois da execução do Python, o HTML fica conceitualmente assim:

```javascript
const RAW = {
    "threshold": 0.5,
    "n_clientes": 10000,
    "taxa_churn_base": 0.204,
    "clientes": [
        [10001, 0.87, 650, "Germany", "Male", 42],
        [10002, 0.12, 720, "France", "Female", 35]
    ]
};
```

Então o navegador já recebe os dados **dentro do próprio HTML**.

---

# 12. Como o JavaScript utiliza esses dados?

Depois que o JSON foi colocado dentro do HTML, o JavaScript consegue acessar os dados normalmente.

O código do projeto faz:

```javascript
const RAW = /*DADOS*/null;

const C = Object.fromEntries(
    RAW.colunas.map((c, i) => [c, i])
);

const B = RAW.clientes;
```

`RAW` contém todas as informações geradas pelo Python.

E:

```javascript
RAW.clientes
```

contém os clientes.

Então:

```javascript
const B = RAW.clientes;
```

coloca todos os clientes na variável `B`.

---

# 13. Como ele sabe qual posição representa cada coluna?

O projeto utiliza uma estrutura chamada:

```javascript
C
```

Ela transforma o nome da coluna em seu índice.

Por exemplo:

```text
customer_id       → 0
risco_churn       → 1
credit_score      → 2
country           → 3
gender            → 4
age               → 5
```

Isso é necessário porque o JSON utiliza uma estrutura mais compacta:

```json
[
    10001,
    0.87,
    650,
    "Germany",
    "Male",
    42
]
```

Em vez de:

```json
{
    "customer_id": 10001,
    "risco_churn": 0.87,
    "credit_score": 650,
    "country": "Germany",
    "gender": "Male",
    "age": 42
}
```

Isso reduz o tamanho do arquivo.

---

# 14. Como o cliente aparece na tabela?

O JavaScript pega os dados:

```javascript
const B = RAW.clientes;
```

Depois filtra os clientes de acordo com o threshold.

Conceitualmente:

```javascript
cliente[C.risco_churn] >= estado.t
```

Se:

```text
risco_churn = 0.82
threshold = 0.50
```

então:

```text
0.82 >= 0.50
```

O cliente é considerado:

```text
EM RISCO
```

Se:

```text
risco_churn = 0.23
```

então:

```text
0.23 < 0.50
```

e ele fica:

```text
ESTÁVEL
```

---

# 15. O usuário consegue alterar o threshold sem treinar novamente?

Sim.

Essa é uma característica interessante do projeto.

O Python fornece a probabilidade de cada cliente:

```text
Cliente A → 82%
Cliente B → 61%
Cliente C → 43%
Cliente D → 17%
```

O navegador recebe essas probabilidades.

Quando o usuário altera o threshold:

```text
50%
```

para:

```text
70%
```

o JavaScript simplesmente refaz o filtro.

Por exemplo:

### Threshold = 50%

```text
82% → Risco
61% → Risco
43% → Estável
17% → Estável
```

### Threshold = 70%

```text
82% → Risco
61% → Estável
43% → Estável
17% → Estável
```

Ou seja:

> **O modelo não é treinado novamente. O navegador apenas muda o critério utilizado para classificar o risco.**

---

# 16. A API de consulta

Além do painel, o projeto tem uma **Minimal API em FastAPI** (`api/main.py`) que carrega o
`churn_model.pkl` e pontua contas novas. Ela sobe junto com o painel pelo `./subir.sh`, em
**http://localhost:8000**, com documentação interativa em **http://localhost:8000/docs**.

| Método   | Rota                  | O que faz                                                               |
| -------- | --------------------- | ----------------------------------------------------------------------- |
| `GET`    | `/saude`              | Status, threshold, features do modelo e quantas contas foram injetadas. |
| `POST`   | `/prever`             | Só consulta: devolve a nota das contas enviadas, sem guardar.           |
| `POST`   | `/contas`             | Pontua e **guarda** as contas (até 5.000 por requisição).               |
| `GET`    | `/contas`             | Ranking das contas injetadas, da maior chance de sair para a menor.     |
| `GET`    | `/contas/{id}`        | A nota de uma conta injetada.                                           |
| `DELETE` | `/contas`             | Apaga as contas injetadas.                                              |

`GET /contas` aceita `so_em_risco` (padrão `true`) e `limite` (padrão `50`).

Cada conta segue as colunas do dataset, sem o `churn`:

```json
[{
  "customer_id": 15634602, "credit_score": 619, "country": "France",
  "gender": "Female", "age": 42, "tenure": 2, "balance": 0.0,
  "products_number": 1, "credit_card": 1, "active_member": 1,
  "estimated_salary": 101348.88
}]
```

E volta com a nota:

```json
[{ "customer_id": 15634602, "...": "...", "risco_churn": 0.8123, "em_risco": true }]
```

A API aplica o mesmo one-hot do treino e reordena as colunas pela lista `features` salva no
`.pkl`, então uma conta sozinha é pontuada igual a um lote. Valores fora do esperado
(país desconhecido, idade negativa, `credit_card` diferente de 0/1) são recusados com 422.

As contas injetadas ficam em `data/contas_injetadas.db` (SQLite) e sobrevivem a reinícios. Um
`customer_id` repetido substitui o anterior.

> O modelo é lido na subida da API. O `./subir.sh` recria a API depois de cada treino; se
> treinar na mão, rode `docker compose restart api`.

---

# 17. Injetar contas: página web e programa

**Pelo navegador:** `http://localhost:8080/injetar.html`. A página fala com a API pelo próprio
nginx (`/api/...` é repassado para o serviço `api`), então não precisa de CORS. Nela dá para:

- enviar um CSV com as colunas da planilha — se ele tiver a coluna `churn`, como
  `data/contas_10_porcento.csv`, a página já confere as notas e mostra recall e precisão;
- digitar uma conta e injetar, ou só consultar a nota sem guardar;
- ver o ranking das contas injetadas e apagar todas.

**Pelo terminal:** `cliente/injetar.py` conversa com a API. Usa só a biblioteca padrão do Python, então roda na
sua máquina sem `pip install`:

```bash
python cliente/injetar.py                                  # menu interativo
python cliente/injetar.py csv data/contas_novas.csv        # injeta um CSV inteiro
python cliente/injetar.py nova                             # digita uma conta na mão
python cliente/injetar.py ranking --limite 20              # contas em risco
python cliente/injetar.py ranking --todas                  # todas as injetadas
python cliente/injetar.py conta 15634602                   # nota de uma conta
python cliente/injetar.py conferir                         # compara com o desfecho real
python cliente/injetar.py limpar                           # apaga as injetadas
```

O fluxo para usar os 10% guardados:

1. `./subir.sh` — treina com os 90% e sobe painel e API.
2. `python cliente/injetar.py csv data/contas_novas.csv` — injeta as 1.000 contas e mostra as de
   maior risco.
3. `python cliente/injetar.py conferir` — cruza as notas com `contas_novas_gabarito.csv` e mostra
   recall e precisão em contas que o modelo realmente nunca viu.

Qualquer CSV com as colunas do dataset serve (a coluna `churn`, se existir, é ignorada). Para
apontar para outra API: `--api http://host:porta` ou a variável `CHURN_API`.

---

# 18. Por que o projeto atual utiliza essa abordagem?

Para um projeto de treinamento/apresentação, essa abordagem é bastante interessante porque simplifica a execução.

O HTML pode ser gerado com os dados já incorporados.

O próprio código explica:

```python
# Cada página é um HTML só:
# o template vem de painel/ e os dados entram no lugar
# do marcador, então o arquivo em models/ abre no navegador
# sem servidor nem rede.
```

Assim, depois do treinamento, temos algo parecido com:

```text
models/
├── churn_model.pkl
├── clientes_em_risco.csv
├── dashboard_data.json
├── painel.html
└── resumo.html
```

O usuário pode simplesmente abrir:

```text
painel.html
```

e visualizar o resultado.

---

# 19. Resumo do funcionamento completo

```text
                    DATASET
                       │
                       ▼
                 ┌───────────┐
                 │   Pandas  │
                 └─────┬─────┘
                       │
                       ▼
              Tratamento dos dados
                       │
                       ▼
                Train / Test
                       │
                       ▼
             Decision Tree Classifier
                       │
                       ▼
             Probabilidade de Churn
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Modelo         CSV          JSON
       .pkl        clientes       Dashboard
          │                         │
          │                         ▼
          │                    Template HTML
          │                         │
          │                         ▼
          │                    JavaScript
          │                         │
          └─────────────────────────┤
                                    ▼
                              DASHBOARD
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
                  Clientes       Métricas       Risco
                  em risco       do modelo      por cliente
```

## Conclusão

A solução utiliza Machine Learning para transformar dados históricos de clientes em uma **probabilidade de churn**.

O diferencial da aplicação é que ela não apenas apresenta o resultado do modelo, mas permite ao usuário explorar os dados através de um dashboard.

A comunicação entre o modelo e a interface, no projeto atual, acontece através da **geração de JSON e incorporação desse JSON diretamente no HTML**.

O trecho mais importante para entender essa comunicação é:

### Python

```python
dash["risco_churn"] = proba_oof
```

↓

```python
payload = json.dumps({
    "clientes": linhas,
    "metricas_teste": {...},
    "importancias": {...}
})
```

↓

```python
pagina = f.read().replace(
    "/*DADOS*/null",
    dados_js
)
```

↓

### HTML/JavaScript

```javascript
const RAW = /*DADOS*/null;

const B = RAW.clientes;
```

Assim, o caminho dos dados é:

```text
Modelo ML
   ↓
Python
   ↓
JSON
   ↓
HTML
   ↓
JavaScript
   ↓
Tabela / Gráficos / Indicadores
```