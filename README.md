# Apoio à triagem de diabetes (Tech Challenge – Fase 1)

Projeto da Fase 1 da pós-graduação em IA para Devs (FIAP Postech). O objetivo é classificar, com aprendizado de máquina sobre dados tabulares, se uma paciente tem diabetes a partir de medidas clínicas.

> **Aviso:** este projeto é uma ferramenta de **apoio à triagem**, não um diagnóstico. O resultado indica quem deve fazer um exame confirmatório, e a decisão final é sempre do médico.

**Vídeo de apresentação:** https://www.youtube.com/watch?v=IFiFOobmkm8  
**Relatório técnico (PDF):** [`reports/relatorio-tecnico.pdf`](reports/relatorio-tecnico.pdf)

## Visão geral

- **Dados:** Pima Indians Diabetes, com 768 pacientes (mulheres de origem Pima, 21 anos ou mais), 8 variáveis clínicas e o rótulo `Outcome` (1 = diabetes, 0 = não). Cerca de 35% dos casos são positivos.
- **Pré-processamento:** zeros que na verdade são valores ausentes (glicose, pressão, dobra cutânea, insulina e IMC) viram `NaN`; depois, mediana para preencher e padronização, tudo dentro de um `Pipeline` do scikit-learn.
- **Modelos comparados:** regressão logística (com e sem balanceamento de classes) e Random Forest balanceado, por validação cruzada de 5 dobras no treino.
- **Modelo final:** regressão logística balanceada, escolhida pela simplicidade e pela interpretabilidade, já que o Random Forest ficou praticamente empatado.
- **Métrica principal:** recall da classe positiva, porque numa triagem o falso negativo é o erro mais caro. O limiar de decisão é o maior limiar cujo recall out-of-fold no treino é pelo menos 0,85, o que dá **0,34**.
- **Interpretação:** coeficientes e odds ratios da regressão logística, e SHAP (beeswarm e waterfall) de um Random Forest como análise complementar.
- **API:** serviço FastAPI que devolve o escore de risco, se a paciente é sinalizada e a contribuição de cada variável.

### Resultados

Divisão estratificada 70/15/15 (537 treino, 115 validação, 116 teste), com `random_state=42`. O conjunto de teste foi usado uma única vez, com o modelo e o limiar já definidos.

| Conjunto | Recall | Precisão | F1 | ROC-AUC |
|---|---|---|---|---|
| Validação | 0,925 | 0,536 | 0,679 | 0,810 |
| Teste | 0,854 (35 de 41) | 0,636 | 0,729 | 0,861 |

As métricas completas (incluindo as matrizes de confusão) ficam em [`reports/metricas.json`](reports/metricas.json). Com apenas 41 pacientes positivas no teste, o recall tem uma margem de incerteza grande, e vale lê-lo como uma estimativa e não como um valor exato.

## Estrutura do repositório

```
.
├── api/
│   └── main.py                 # API FastAPI (/saude e /predict)
├── data/
│   └── raw/diabetes.csv        # dataset Pima Indians Diabetes
├── models/
│   └── modelo_diabetes.joblib  # pipeline treinado + limiar de decisão
├── notebooks/
│   └── 01-eda.ipynb            # EDA, comparação de modelos, interpretação e avaliação
├── reports/
│   ├── figures/                # gráficos gerados pelo notebook
│   └── metricas.json           # métricas geradas pelo script de treino
├── src/
│   ├── config.py               # caminhos, colunas e parâmetros fixos
│   ├── download_data.py        # baixa o dataset do OpenML
│   ├── preprocessamento.py     # carga, divisão e pré-processador
│   └── treino.py               # treino, escolha do limiar, avaliação e salvamento
├── Dockerfile
├── requirements.txt            # ambiente completo (notebook, SHAP, API)
└── requirements-api.txt        # só o necessário para a API (usado no Docker)
```

## Como rodar localmente

Requer Python 3.12. As versões das bibliotecas estão fixadas porque o modelo salvo (`.joblib`) depende da versão do scikit-learn.

```bash
git clone https://github.com/phcassemiro/tech-challenge-fase1.git
cd tech-challenge-fase1

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dados

O dataset já está em `data/raw/diabetes.csv`. Para baixá-lo de novo do OpenML (opcional):

```bash
python src/download_data.py
```

### Treinar o modelo

O script divide os dados, escolhe o limiar, avalia na validação e no teste e grava `models/modelo_diabetes.joblib` e `reports/metricas.json`:

```bash
python -m src.treino
```

Em alguns Macs com numpy 2.x o scikit-learn emite avisos `RuntimeWarning ... encountered in matmul`. Os dados e os coeficientes foram verificados como finitos, e os avisos podem ser ocultados com `python -W ignore -m src.treino`.

### Notebook

```bash
jupyter notebook notebooks/01-eda.ipynb
```

O notebook cobre a análise exploratória, a comparação dos modelos, a escolha do limiar, a interpretação (coeficientes e SHAP) e a avaliação no teste. Ele não sobrescreve o modelo: a última célula só carrega `models/modelo_diabetes.joblib` e confere que as previsões batem.

### API

```bash
uvicorn api.main:app --reload
```

A documentação interativa (Swagger) fica em http://127.0.0.1:8000/docs.

## Usando a API

`GET /saude` confere se o serviço está no ar e mostra o limiar em uso:

```bash
curl http://127.0.0.1:8000/saude
```

`POST /predict` recebe as medidas de uma paciente. `BloodPressure`, `SkinThickness` e `Insulin` são opcionais (use `null` quando não foram medidas) e são preenchidas com a mediana do treino:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 70,
    "SkinThickness": null,
    "Insulin": null,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 45
  }'
```

A resposta traz:

- `escore_risco`: saída do modelo, entre 0 e 1. É um escore de risco e **não uma probabilidade calibrada**.
- `limiar`: ponto de corte usado (0,34).
- `sinalizada_para_exame_confirmatorio`: `true` quando o escore passa do limiar.
- `fatores`: contribuição de cada variável para o escore (coeficiente × valor padronizado), ordenada pela magnitude, indicando também quais valores estavam ausentes e foram preenchidos.
- `aviso`: lembrete de que é apoio à triagem.

## Docker

O Dockerfile empacota só a API e o modelo, e usa `requirements-api.txt`.

```bash
docker build -t triagem-diabetes .
docker run --rm -p 8000:8000 triagem-diabetes
```

Com o contêiner no ar, a API responde em http://127.0.0.1:8000 (`/saude`, `/predict` e `/docs`), como no exemplo acima.

## Limitações

- O dataset tem apenas mulheres de origem Pima com 21 anos ou mais, e o modelo não foi validado em outras populações nem com dados externos.
- O conjunto de teste é pequeno (116 pacientes, 41 positivas), então as métricas têm incerteza relevante.
- O escore não é uma probabilidade calibrada, porque o balanceamento de classes desloca os valores.
- A glicose domina a previsão, e valores ausentes ou mal medidos nessa variável afetam muito o resultado.
- O modelo erra: no teste, 6 das 41 pacientes com diabetes não foram sinalizadas. Por isso o resultado serve para priorizar exames, nunca para descartar a doença.
