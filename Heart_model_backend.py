from fastapi import FastAPI 
import joblib
import shap
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
def Shap_explainations(df):
    X_processed = model.named_steps['CT'].transform(df)
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values = explainer(X_processed).values[0 , : , 1]
    impt = pd.Series(shap_values , index = model.named_steps['CT'].get_feature_names_out())
    top_ind = impt.abs().sort_values(ascending = False).head(3)
    return top_ind
@app.post("/predict")
def predict(data : InputData):
    df = {
        "age":data.age , "sex":data.sex , "cp":data.cp , "trestbps":data.trestbps , "chol":data.chol , "fbs":data.fbs , "restecg":data.restecg , 
        "thalach":data.thalach , "exang":data.exang , "oldpeak":data.oldpeak , "slope": data.slope , 
            "ca":data.ca , "thal":data.thal}
    df = pd.DataFrame([df])
    Explain = Shap_explainations(df)
    reply = model.predict(df)
    prediction = model.predict_proba(df)[: , 1][0]
    return {"Explain":Explain , "Chances of Heart disease": f"{round(prediction*100 , 4)}%" , "Risk Level":["Low" if reply == 0 else "High"]}



