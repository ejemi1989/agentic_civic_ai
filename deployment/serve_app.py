# deployment/serve_app.py
import ray
from ray import serve
from fastapi import FastAPI
import yaml
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from agents import Agent, Environment

# read config
cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model"]["hf_model"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# simple model endpoint (loads HF model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

@app.get("/")
def index():
    return {"status": "agentic-civic-ai serve running"}

@app.post("/predict")
def predict(text: str):
    inputs = tokenizer([text], truncation=True, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
    return {"prediction": pred, "probs": probs}

# Agent orchestration endpoint
AGENTS = [Agent("agent-A", hf_model=MODEL_NAME, device=DEVICE),
          Agent("agent-B", hf_model=MODEL_NAME, device=DEVICE)]

env = Environment(AGENTS)

@app.post("/agent_orchestrate")
def orchestrate(tasks: list):
    """
    tasks: list of strings
    """
    res = env.distribute_tasks(tasks)
    return {"results": res}

# If you want to run with Ray Serve programmatically:
def start_ray_serve():
    ray.init(ignore_reinit_error=True)
    serve.start(detached=True)
    serve.run(app)  # serve the FastAPI app
