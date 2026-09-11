# Churn Training

Projeto de **treinamento de Machine Learning** para previsão de churn (Bank Customer Churn Dataset).
O modelo aprende com os clientes que **já saíram** do banco e usa esse padrão para apontar,
entre os clientes atuais, quais têm mais chance de sair.
Sem API e sem servidor: o treino termina gerando um painel HTML autocontido com os dados já
embutidos. Roda inteiramente em Docker, nada de Python instalado na máquina.

## O dataset

[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
(Kaggle, `gauravtopre`): 10.000 clientes de banco, 20,4% deles cancelaram.

| Coluna               | Descrição                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `customer_id`      | Identificador (não entra no treino, só identifica quem está em risco). |
| `credit_score`     | Score de crédito.                                                        |
| `country`          | França, Alemanha ou Espanha.                                             |
| `gender`           | Female / Male.                                                            |
| `age`              | Idade.                                                                    |
| `tenure`           | Anos como cliente.                                                        |
| `balance`          | Saldo em conta.                                                           |
| `products_number`  | Quantidade de produtos contratados.                                       |
| `credit_card`      | Tem cartão de crédito (0/1).                                            |
| `active_member`    | Cliente ativo (0/1).                                                      |
| `estimated_salary` | Salário estimado.                                                        |
| `churn`            | **Target**: 1 = cancelou, 0 = ativo.                                |

O download do Kaggle exige token de API, então o script busca o CSV de um espelho público e
**guarda em `./data/bank_customer_churn.csv`** — baixa uma vez só, as execuções seguintes leem do disco.

Se preferir o arquivo oficial, baixe do Kaggle e salve como `./data/bank_customer_churn.csv`
(mesmo nome); o script detecta o arquivo e pula o download.

## O que ele faz

1. Carrega o dataset (cache local em `data/`, ou baixa na primeira vez).
2. Descarta linhas nulas e separa `customer_id` do conjunto de features.
3. Transforma `country` e `gender` em colunas numéricas (one-hot) — 11 features no total.
4. Separa treino/teste (80/20, `random_state=42`, estratificado pelo churn).
5. Treina uma Árvore de Decisão com `class_weight="balanced"` e avalia no conjunto de teste.
6. Exporta o modelo em `models/churn_model.pkl` e a lista de clientes em risco em
   `models/clientes_em_risco.csv`.
7. Pontua a base inteira em `models/dashboard_data.json` e injeta esses dados em cada template
   de `painel/`, gerando `models/painel.html` e `models/resumo.html`.

### Por que o foco não é acurácia

Só 20,4% da base cancelou. Um modelo que chuta "ninguém sai" para todo mundo já acerta 79,6%
e é **inútil** — nunca sinaliza ninguém. Por isso o treino usa `class_weight="balanced"`
(dá mais peso a quem cancelou) e a avaliação olha:

- **Recall** — dos clientes que realmente cancelaram, quantos o modelo conseguiu sinalizar. É a métrica principal.
- **Precisão** — dos que ele sinalizou, quantos cancelaram mesmo. Mede o desperdício de esforço em falso alarme.
- **Falsos alarmes / escaparam** — os dois erros em número absoluto.

## Saída: `models/clientes_em_risco.csv`

Depois de treinar, o script pontua a base inteira e exporta os clientes com risco acima do
threshold, do maior para o menor:

| Coluna          | Descrição                                                  |
| --------------- | ------------------------------------------------------------ |
| `customer_id` | Quem é o cliente.                                           |
| `risco_churn` | Probabilidade estimada de sair (0 a 1).                      |
| demais colunas  | Os dados do cliente, para entender o motivo do alerta.       |
| `churn`       | O que de fato aconteceu no dataset — serve de conferência. |

> As linhas que entraram no treino recebem nota otimista. As métricas confiáveis são as do
> passo 5, calculadas só no conjunto de teste.

O `.pkl` guarda um dicionário com `model`, `features` e `threshold` — a ordem das colunas do
one-hot precisa ser a mesma na hora de pontuar clientes novos:

```python
bundle = joblib.load("models/churn_model.pkl")
bundle["model"].predict_proba(novos[bundle["features"]])[:, 1]
```

## As páginas geradas

O treino termina montando duas páginas para o **Banco Avenida** (nome fictício sobre os dados
do Kaggle), a partir dos mesmos dados: uma para operar, outra para apresentar.

### `models/painel.html` — o console completo

Um arquivo HTML só, com as notas de todos os 10.000 clientes embutidas: dá para servir em
<http://localhost:8080> com o serviço `site`, ou simplesmente abrir com duplo clique —
sem servidor e sem internet, funciona igual.

O que dá para fazer nele:

- **Mexer no corte de risco** e ver recall, precisão, falsos alarmes e as duas listas de
  clientes recalcularem ao vivo. É a forma mais direta de sentir o custo do threshold.
- **Separar a carteira** entre quem está em risco e quem está estável, com busca por
  `customer_id`, ordenação por qualquer coluna e filtros de país e atividade.
- **Ver onde o churn se concentra** por faixa etária, número de produtos, país, uso da conta,
  saldo e tempo de casa — taxa real do histórico, não previsão.
- **Ligar “só quem ficou fora do treino”**, que restringe a carteira aos 2.000 clientes do
  conjunto de teste. É a visão sem a nota otimista das linhas de treino.

As métricas do modelo no painel são sempre calculadas no conjunto de teste e não respondem aos
filtros de país, atividade ou busca — só ao corte.

### `models/resumo.html` — a versão de apresentação

Para mostrar o trabalho, o painel completo atrapalha: slider, filtros e 10.000 linhas de tabela
são coisas para operar, não para projetar. O `resumo.html` conta a mesma história sem controle
nenhum, com número grande e uma ideia por faixa:

1. **A base** — 10.000 clientes, 2.037 cancelamentos, 20,4% de churn.
2. **Por que acurácia engana** — o espantalho (“ninguém sai”, 79,6% de acurácia, 0 clientes
   avisados) lado a lado com a árvore (76,1% de acurácia, 319 dos 407 avisados).
3. **O resultado** — recall, alcançados, escapados e falsos alarmes no conjunto de teste.
4. **Onde o churn se concentra** — dez grupos ranqueados contra a média da base.
5. **No que a árvore se apoia** — as seis características de maior peso.
6. **Os dez mais em risco** — só clientes de fora do treino, para a coluna de desfecho ser
   conferência de verdade em vez de memorização.
7. **Como foi construído** — os quatro passos do método.

A página também tem estilo de impressão, então dá para gerar PDF pelo Ctrl+P sem sair tudo preto.

### Mexendo nas páginas

Os templates ficam em `painel/*.html`, cada um com o marcador `/*DADOS*/null` no lugar dos
dados. O passo 7 troca o marcador pelo JSON e escreve o resultado em `models/`, mantendo o
nome do arquivo — jogar um `.html` novo nessa pasta basta para ele virar mais uma página.
Para mexer no visual, edite o template e rode o treino de novo: o volume é montado, não
precisa rebuild.

## Como rodar

São dois serviços: `trainer` treina e gera os artefatos, `site` serve o painel.

**Treinar e subir o site de uma vez:**

```bash
docker compose up -d site
```

O `site` espera o `trainer` terminar bem antes de subir — treino que falha não vira página no ar.
Quando voltar o prompt:

- **<http://localhost:8080>** — o painel completo, para operar.
- **<http://localhost:8080/resumo.html>** — o resumo, para apresentar.

**Só treinar,** sem subir servidor nenhum:

```bash
docker compose run --rm trainer
```

Os artefatos aparecem em `./models` na sua máquina (volume montado) e o
`models/painel.html` abre com duplo clique, sem precisar do servidor.

**Só servir,** reaproveitando o último treino:

```bash
docker compose up -d --no-deps site
```

**Derrubar:**

```bash
docker compose down
```

A porta é publicada em `127.0.0.1:8080`, então o painel responde só na sua máquina e não fica
exposto na rede local. Como a raiz do site é a pasta `models/`, o `clientes_em_risco.csv` também
fica baixável em <http://localhost:8080/clientes_em_risco.csv>.

Primeira execução baixa as imagens, instala as dependências e busca o dataset;
as seguintes reaproveitam o cache da imagem e o CSV em `./data`.

## Experimentando

`train_model.py` está montado como volume — edite os hiperparâmetros e rode de novo,
sem rebuild:

```bash
docker compose run --rm trainer
```

Só refaça a imagem se mudar o `requirements.txt`:

```bash
docker compose build
```

Variáveis de ambiente disponíveis:

| Variável     | Padrão                          | Papel                                                                                      |
| ------------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `THRESHOLD` | `0.5`                          | Corte de probabilidade. Mais baixo = sinaliza mais gente (mais recall, mais falso alarme). |
| `RISK_TOP`  | `200`                          | Quantos clientes de maior risco exportar no CSV.                                           |
| `MODEL_DIR` | `models`                       | Diretório de saída dos artefatos.                                                        |
| `DATA_DIR`  | `data`                         | Pasta do cache do CSV (usada só para montar o `DATA_PATH` padrão).                       |
| `DATA_PATH` | `data/bank_customer_churn.csv` | Caminho do CSV (lido se existir, senão baixado e salvo aí).                              |
| `DATA_URL`  | espelho público                 | De onde baixar quando não há CSV local.                                                  |
| `PANEL_DIR` | `painel`                       | Pasta dos templates. Cada `.html` dela vira uma página em `models/`.                 |

```bash
# rede mais larga: pega mais clientes em risco, aceitando mais falso alarme
docker compose run --rm -e THRESHOLD=0.35 -e RISK_TOP=500 trainer
```

No código, os hiperparâmetros da árvore ficam em `DecisionTreeClassifier`
(`max_depth=6`, `min_samples_leaf=20`) — mexer neles muda o equilíbrio entre
decorar a base e generalizar.

## Estrutura

| Arquivo                | Papel                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `train_model.py`     | Pipeline completo: dataset → treino → avaliação → clientes em risco → painel. |
| `painel/painel.html` | Template do painel completo, com `/*DADOS*/null` no lugar dos dados.              |
| `painel/resumo.html` | Template da versão de apresentação, mesmo marcador e mesmos dados.               |
| `painel/nginx.conf`  | Config do serviço `site`: raiz em `models/`, índice `painel.html`, sem cache.  |
| `requirements.txt`   | pandas, scikit-learn, joblib, numpy.                                               |
| `Dockerfile`         | Imagem`python:3.12-slim` com as dependências.                                   |
| `docker-compose.yml` | Serviços`trainer` (treino) e `site` (nginx na 8080) + os volumes.           |
| `data/`              | Cache do CSV do dataset (ignorado no git).                                         |
| `models/`            | Saída:`churn_model.pkl`, `clientes_em_risco.csv`, `dashboard_data.json` e `painel.html` (ignorados no git). |
