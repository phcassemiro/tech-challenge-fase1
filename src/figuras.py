"""Gera a figura da matriz de confusão do teste a partir de reports/metricas.json.

Uso, a partir da raiz do projeto:

    python -m src.figuras
"""
import json

import matplotlib.pyplot as plt

from src.config import CAMINHO_METRICAS, RAIZ

DESTINO = RAIZ / "reports" / "figures" / "matriz_confusao_teste.png"


def main():
    """Desenha a matriz de confusão do conjunto de teste, com contagens por célula."""
    metricas = json.loads(CAMINHO_METRICAS.read_text())
    matriz = metricas["teste"]["matriz_confusao"]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matriz, cmap="Blues")
    rotulos = ["Sem diabetes", "Diabetes"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão no teste (limiar {metricas['limiar']})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i][j], ha="center", va="center", fontsize=16,
                    color="white" if matriz[i][j] > 30 else "black")
    plt.tight_layout()
    plt.savefig(DESTINO, dpi=150)
    print("figura salva em", DESTINO)


if __name__ == "__main__":
    main()
