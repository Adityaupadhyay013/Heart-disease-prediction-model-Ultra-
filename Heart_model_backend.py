from fastapi import FastAPI 
import joblib
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all websites (development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class InputData(BaseModel):
    age: int                
    sex: int                
    cp: int                 # chest pain type (0–3)
    trestbps: int           # resting blood pressure
    chol: int               # cholesterol mg/dl
    fbs: int                # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg: int            # ECG results (0,1,2)
    thalach: int            # max heart rate achieved
    exang: int              # exercise induced angina (1 = yes, 0 = no)
    oldpeak: float          # ST depression
    slope: int              # slope (0–2)
    ca: int                 # number of vessels (0–3)
    thal: int  
model = joblib.load("Heart disease prediction ml model.joblib")
@app.post("/predict")
def predict(data : InputData):
    df = {
        "age":data.age , "sex":data.sex , "cp":data.cp , "trestbps":data.trestbps , "chol":data.chol , "fbs":data.fbs , "restecg":data.restecg , 
        "thalach":data.thalach , "exang":data.exang , "oldpeak":data.oldpeak , "slope": data.slope , 
            "ca":data.ca , "thal":data.thal}
    df = pd.DataFrame([df])
    reply = model.predict(df)
    prediction = model.predict_proba(df)[: , 1][0]
    return {"Chances of Heart disease": f"{round(prediction*100 , 4)}%" , "Risk Level":["Low" if reply == 0 else "High"]}

