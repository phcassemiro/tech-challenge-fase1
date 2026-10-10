from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo_diabetes.joblib"

artefato = joblib.load(CAMINHO_MODELO)
pipeline = artefato["pipeline"]
limiar = round(artefato["limiar"], 2)

app = FastAPI(
    title="Apoio à triagem de diabetes",
    description="Ferramenta de apoio à triagem. Não é diagnóstico: a decisão é sempre do médico.",
)


class Paciente(BaseModel):
    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(gt=0, le=300)
    BloodPressure: float | None = Field(default=None, gt=0, le=200)
    SkinThickness: float | None = Field(default=None, gt=0, le=100)
    Insulin: float | None = Field(default=None, gt=0, le=900)
    BMI: float = Field(gt=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=21, le=120)


@app.get("/saude")
def saude():
    return {"status": "ok", "limiar": limiar}


@app.post("/predict")
def prever(paciente: Paciente):
    dados = pd.DataFrame([paciente.model_dump()]).astype(float)  # None vira NaN

    escore = float(pipeline.predict_proba(dados)[0, 1])

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    x_escalado = pre.transform(dados)[0]
    nomes = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    contribuicoes = x_escalado * clf.coef_[0]

    fatores = [
        {
            "variavel": nome,
            "contribuicao": round(float(c), 3),
            "valor_informado": getattr(paciente, nome),
            "valor_ausente_preenchido": getattr(paciente, nome) is None,
        }
        for nome, c in zip(nomes, contribuicoes)
    ]
    fatores.sort(key=lambda f: abs(f["contribuicao"]), reverse=True)

    return {
        "escore_risco": round(escore, 3),
        "limiar": limiar,
        "sinalizada_para_exame_confirmatorio": escore >= limiar,
        "fatores": fatores,
        "aviso": "Escore de risco, não probabilidade calibrada. Apoio à triagem; a decisão final é do médico.",
    }