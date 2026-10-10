"""Treino do modelo final, escolha do limiar de decisão e avaliação.

Uso, a partir da raiz do projeto:

    python -m src.treino

O script é o único responsável por gerar `models/modelo_diabetes.joblib`. O conjunto de
teste é usado uma única vez, no final, só para reportar o resultado: nada é ajustado
com base nele.
"""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import (CAMINHO_METRICAS, CAMINHO_MODELO, RANDOM_STATE,
                        RECALL_MINIMO)
from src.preprocessamento import (carregar_dados, dividir_dados,
                                  montar_preprocessador)


def criar_pipeline():
    """Pipeline final: pré-processamento sem indicador de ausência + regressão logística balanceada.

    Os passos se chamam "pre" e "clf" porque a API acessa o pipeline por esses nomes.
    """
    return Pipeline([
        ("pre", montar_preprocessador(com_flag=False)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


def escolher_limiar(pipeline, X, y, recall_minimo=RECALL_MINIMO):
    """Escolhe o maior limiar cujo recall out-of-fold (validação cruzada) é >= recall_minimo.

    As probabilidades vêm de uma validação cruzada estratificada de 5 dobras, então cada
    previsão é feita por um modelo que não viu aquela paciente. Entre os limiares de 0,10
    a 0,79 (passo 0,01) que atingem a meta, fica o maior, que é o de maior precisão.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    p_oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    return max(t for t in np.arange(0.10, 0.80, 0.01)
               if recall_score(y, (p_oof >= t).astype(int)) >= recall_minimo)


def avaliar(pipeline, X, y, limiar):
    """Calcula as métricas de um conjunto de dados usando o limiar de decisão dado."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= limiar).astype(int)
    return {
        "recall": round(float(recall_score(y, pred)), 3),
        "precisao": round(float(precision_score(y, pred)), 3),
        "f1": round(float(f1_score(y, pred)), 3),
        "acuracia": round(float(accuracy_score(y, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "matriz_confusao": confusion_matrix(y, pred).tolist(),
    }


def main(caminho_modelo=CAMINHO_MODELO, caminho_metricas=CAMINHO_METRICAS):
    """Treina, escolhe o limiar, avalia na validação e no teste e salva modelo e métricas."""
    X, y = carregar_dados()
    X_treino, X_val, X_teste, y_treino, y_val, y_teste = dividir_dados(X, y)

    pipeline = criar_pipeline()
    limiar = float(escolher_limiar(pipeline, X_treino, y_treino))
    pipeline.fit(X_treino, y_treino)

    metricas = {
        "limiar": round(limiar, 2),
        "validacao": avaliar(pipeline, X_val, y_val, limiar),
        "teste": avaliar(pipeline, X_teste, y_teste, limiar),
    }

    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "limiar": limiar}, caminho_modelo)
    caminho_metricas.parent.mkdir(parents=True, exist_ok=True)
    caminho_metricas.write_text(json.dumps(metricas, indent=2, ensure_ascii=False))

    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    return metricas


if __name__ == "__main__":
    main()
