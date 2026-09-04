import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Diretório de saída do artefato (sobrescrevível via variável de ambiente)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Carrega o Telco Churn Dataset diretamente de repositório público
url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
print("[1/5] Baixando Telco Customer Churn Dataset...")
df = pd.read_csv(url)

# 2. Tratamento mínimo de dados
# Converte TotalCharges para float e descarta valores nulos/espaços em branco
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_clean = df.dropna(subset=['TotalCharges']).copy()

# Target binário: 'Yes' vira 1 (Cancelou), 'No' vira 0 (Ativo)
df_clean['ChurnBinary'] = (df_clean['Churn'] == 'Yes').astype(int)

features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
X = df_clean[features] #features/caracteristicas/atributos são as variáveis independentes
y = df_clean['ChurnBinary'] # target/label/rótulo/classe é a variável dependente

# 3. Divisão Treino e Teste
# evita não decorar respostas
# random_state -> determinístico
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Treinamento do Modelo
# max_depth 1..10 (9) * min_sample_leaf 1..10 (9) * ..
model = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"[4/5] Árvore treinada! Acurácia de teste: {acc * 100:.2f}%")

# 5. Exportação do Artefato
joblib.dump(model, MODEL_PATH)
print(f"[5/5] Modelo salvo com sucesso em '{MODEL_PATH}'!")