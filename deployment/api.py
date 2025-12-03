from fastapi import FastAPI
import torch
from models.pytorch_model import SimpleMLP

app = FastAPI()
model = SimpleMLP(10, 20, 1)

@app.get("/predict")
def predict():
    x = torch.randn(1, 10)
    y = model(x)
    return {"prediction": y.tolist()}
