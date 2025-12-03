# src/agents.py
import logging
import random
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)

class Agent:
    def __init__(self, name: str, model_path: str = None, hf_model: str = "distilbert-base-uncased", device="cpu"):
        self.name = name
        self.device = device
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(hf_model).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def evaluate_text(self, text: str):
        inputs = self.tokenizer([text], truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            score = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
        logging.info(f"[{self.name}] -> pred={pred}, score={score}")
        return {"agent": self.name, "pred": pred, "score": score}

    def assign_and_act(self, task_text: str):
        # simple pipeline: evaluate and return result
        return self.evaluate_text(task_text)

class Environment:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def distribute_tasks(self, tasks: List[str]):
        results = []
        for t in tasks:
            # random agent selection to mimic allocation
            agent = random.choice(self.agents)
            res = agent.assign_and_act(t)
            results.append(res)
        return results

if __name__ == "__main__":
    # demo
    a1 = Agent("agent-1")
    a2 = Agent("agent-2")
    env = Environment([a1, a2])
    tasks = ["Is this news item misinformation? 'Fake rumor about hospital.'", "Public meeting scheduled downtown."]
    print(env.distribute_tasks(tasks))
