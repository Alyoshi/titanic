from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBClassifier
import traceback
import pandas as pd

app = FastAPI()

model = XGBClassifier()
model.load_model('titanic.json')
print("Model loaded")

# Allow all origins for CORS (insecure, should be restricted in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request_data: dict):
    try:
        query = pd.DataFrame([request_data])
        predict_array = model.predict_proba(query)
        predict = round(predict_array[0][1], 2)
        print(f"Predicted survival chance is {predict * 100}%")
        return {"prediction": predict}
    except Exception as e:
        return {"trace": traceback.format_exc()}

@app.get("/health")
def health():
    return 'Ok'
