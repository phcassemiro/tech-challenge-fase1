"""Carga, divisão e pré-processamento dos dados do Pima Indians Diabetes."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (ALVO, CAMINHO_DADOS, COLUNAS_GRUPO_A, COLUNAS_GRUPO_B,
                        COLUNAS_ZERO_INVALIDO, FRACAO_TESTE_E_VALIDACAO,
                        RANDOM_STATE)


def carregar_dados(caminho=CAMINHO_DADOS):
    """Lê o CSV e troca por NaN os zeros que na verdade são valores ausentes.

    Devolve X (variáveis) e y (Outcome, 0 ou 1).
    """
    df = pd.read_csv(caminho)
    df[COLUNAS_ZERO_INVALIDO] = df[COLUNAS_ZERO_INVALIDO].replace(0, float("nan"))
    return df.drop(columns=ALVO), df[ALVO]


def dividir_dados(X, y):
    """Divide em treino (70%), validação (15%) e teste (15%), de forma estratificada.

    São duas divisões em sequência: primeiro separa 30% para validação + teste e depois
    divide esses 30% ao meio. A estratificação mantém a proporção de positivos (~35%)
    nas três partes.

    Devolve (X_treino, X_val, X_teste, y_treino, y_val, y_teste).
    """
    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=FRACAO_TESTE_E_VALIDACAO, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste


def montar_preprocessador(com_flag):
    """Monta o pré-processamento: mediana para ausentes e padronização (StandardScaler).

    Com `com_flag=True`, o grupo A (SkinThickness e Insulin) ganha uma coluna 0/1 que
    marca se o valor estava ausente. O modelo final usa `com_flag=False`.

    O preprocessador só usa classes do scikit-learn, de propósito: assim o modelo salvo
    em .joblib pode ser carregado pela API sem precisar do código de `src/`. Os nomes
    "grupo_a" e "grupo_b" aparecem como prefixo (antes do "__") nos nomes das features
    e a API depende deles.
    """
    return ColumnTransformer([
        ("grupo_a", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=com_flag)),
            ("escala", StandardScaler()),
        ]), COLUNAS_GRUPO_A),
        ("grupo_b", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]), COLUNAS_GRUPO_B),
    ])
