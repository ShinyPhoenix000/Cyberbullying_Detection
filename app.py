from fastapi import FastAPI, Request
from scripts.preprocess import clean_text
# from scripts.train import load_model_and_tokenizer  # To be implemented
import torch

app = FastAPI()

@app.post('/predict')
async def predict(request: Request):
    data = await request.json()
    text = clean_text(data['text'])
    # Tokenize, run model, return prediction (to be implemented)
    return {"prediction": "neutral"}
