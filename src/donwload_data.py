from pathlib import Path
from sklearn.datasets import fetch_openml

DESTINO = Path("data/raw/diabetes.csv")

df = fetch_openml("diabetes", version=1, as_frame=True).frame

df = df.rename(columns={
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
    "class": "Outcome",
})
df["Outcome"] = (df["Outcome"] == "tested_positive").astype(int)

DESTINO.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DESTINO, index=False)
print(df.shape)
print(df["Outcome"].value_counts())