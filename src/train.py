# src/train.py
import os
import yaml
import mlflow
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from preprocess import get_tokenizer, tokenize_dataframe
from data_utils import load_sample_csv, ensure_dirs
from dataset import HFEncodedDataset

def run_training(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ensure_dirs()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    train_df, test_df = load_sample_csv(cfg["dataset"]["sample_csv"],
                                        text_col=cfg["dataset"]["text_col"],
                                        label_col=cfg["dataset"]["label_col"])
    tokenizer = get_tokenizer(cfg["model"]["hf_model"], max_length=cfg["model"]["max_length"])

    train_enc, train_labels = tokenize_dataframe(train_df, tokenizer, text_col=cfg["dataset"]["text_col"], label_col=cfg["dataset"]["label_col"], max_length=cfg["model"]["max_length"])
    val_enc, val_labels = tokenize_dataframe(test_df, tokenizer, text_col=cfg["dataset"]["text_col"], label_col=cfg["dataset"]["label_col"], max_length=cfg["model"]["max_length"])

    train_dataset = HFEncodedDataset(train_enc, train_labels)
    val_dataset = HFEncodedDataset(val_enc, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["hf_model"], num_labels=cfg["model"]["num_labels"])

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        evaluation_strategy="epoch",
        logging_steps=cfg["training"]["logging_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # simple compute_metrics
    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="binary")}

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    with mlflow.start_run():
        mlflow.log_params({
            "model": cfg["model"]["hf_model"],
            "max_length": cfg["model"]["max_length"],
            "epochs": cfg["training"]["num_train_epochs"],
            "batch_size": cfg["training"]["per_device_train_batch_size"],
            "learning_rate": cfg["training"]["learning_rate"]
        })
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        trainer.save_model(os.path.join(cfg["training"]["output_dir"], "best_model"))
        mlflow.pytorch.log_model(trainer.model, "hf_model")
        print("Training complete. Metrics:", metrics)

if __name__ == "__main__":
    run_training()
