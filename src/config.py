"""Constantes do projeto: caminhos, colunas e parâmetros fixos."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ / "data" / "raw" / "diabetes.csv"
CAMINHO_MODELO = RAIZ / "models" / "modelo_diabetes.joblib"
CAMINHO_METRICAS = RAIZ / "reports" / "metricas.json"

RANDOM_STATE = 42

ALVO = "Outcome"

# Nestas colunas o valor 0 não existe na prática: é um valor ausente registrado como 0.
# Em Pregnancies o 0 é legítimo (nenhuma gestação), por isso ela não entra aqui.
COLUNAS_ZERO_INVALIDO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Dois grupos de colunas no pré-processamento. O grupo A é o que recebe o indicador de
# ausência quando ele é ligado (experimento da seção 9 do notebook).
COLUNAS_GRUPO_A = ["SkinThickness", "Insulin"]
COLUNAS_GRUPO_B = ["Pregnancies", "Glucose", "BloodPressure", "BMI",
                   "DiabetesPedigreeFunction", "Age"]

# Proporções da divisão: 70% treino, 15% validação, 15% teste.
FRACAO_TESTE_E_VALIDACAO = 0.30

# Meta de recall da classe positiva usada para escolher o limiar de decisão.
RECALL_MINIMO = 0.85
