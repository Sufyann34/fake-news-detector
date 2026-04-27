import re
import pickle
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    t0 = time.perf_counter()
    cleaned = clean(req.text)
    transformed = vec.transform([cleaned])
    label_id = int(model.predict(transformed)[0])
    proba = model.predict_proba(transformed)[0]
    confidence = float(proba[label_id])
    return {
        "label": "fake" if label_id == 1 else "real",
        "confidence": round(confidence, 4),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
    }