# Churn Training

Projeto de **treinamento de Machine Learning** para previsão de churn (Telco Customer Churn Dataset).
Só a parte de treino — sem API e sem frontend. Roda inteiramente em Docker, nada de Python instalado na máquina.

## O que ele faz

1. Baixa o Telco Customer Churn Dataset de um repositório público.
2. Limpa os dados (`TotalCharges` para numérico, descarta nulos) e cria o target binário `ChurnBinary`.
3. Separa treino/teste (80/20, `random_state=42`).
4. Treina uma Árvore de Decisão (`max_depth=4`, `min_samples_leaf=10`) e imprime a acurácia.
5. Exporta o artefato em `models/churn_model.pkl`.

Features usadas: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`.

## Como rodar

```bash
docker compose run --rm trainer
```

O `.pkl` gerado aparece em `./models/churn_model.pkl` na sua máquina (volume montado).

Primeira execução baixa a imagem e instala as dependências; as seguintes reaproveitam o cache.
Precisa de internet: o dataset é baixado a cada treino.

## Experimentando

`train_model.py` está montado como volume — edite os hiperparâmetros (`max_depth`, `min_samples_leaf`,
`test_size`, trocar o `DecisionTreeClassifier` por outro modelo) e rode de novo:

```bash
docker compose run --rm trainer
```

Sem rebuild. Só refaça a imagem se mudar o `requirements.txt`:

```bash
docker compose build
```

Para gravar o modelo em outro diretório, use a variável `MODEL_DIR`:

```bash
docker compose run --rm -e MODEL_DIR=/app/models/v2 trainer
```

## Estrutura

| Arquivo | Papel |
| --- | --- |
| `train_model.py` | Pipeline de treino, do download do dataset ao export do `.pkl`. |
| `requirements.txt` | pandas, scikit-learn, joblib, numpy. |
| `Dockerfile` | Imagem `python:3.12-slim` com as dependências. |
| `docker-compose.yml` | Serviço `trainer` + volumes de `models/` e do script. |
| `models/` | Saída dos artefatos treinados (ignorado no git). |
