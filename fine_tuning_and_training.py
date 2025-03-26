import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import login
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# Log in to Hugging Face
huggingface_token = "TOPSECRET"
login(token=huggingface_token)

# Load JSON files with the data.
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_json("train_data.json")
test_data = load_json("test_data.json")

# Initialize tokenizer of the base model
model_name = "viktoroo/sberbank-rubert-base-collection3" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# This function creates token-level labels.
# For each example, if the marker candidate is found in the sentence,
# tokens overlapping its character span get label 1; otherwise, all tokens get 0.
def create_token_classification_examples(data, tokenizer, max_length=128):
    examples = []
    for item in data:
        text = item["input"]
        # Only consider the candidate if is_discourse_marker is True.
        if item.get("is_discourse_marker", False):
            candidate = item.get("output", "").strip()
        else:
            candidate = ""
        
        # If a candidate exists and is found in the text, determine its character span.
        if candidate and candidate in text:
            start_idx = text.find(candidate)
            end_idx = start_idx + len(candidate)
        else:
            start_idx, end_idx = -1, -1

        # Tokenize the text with offsets so we can map tokens back to character positions.
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True
        )
        labels = [0] * len(tokenized["input_ids"])
        # Assign label 1 to tokens overlapping with the candidate span (if any).
        for idx, (offset_start, offset_end) in enumerate(tokenized["offset_mapping"]):
            # Skip special tokens (which typically have (0, 0) offsets).
            if offset_start is None or offset_end is None or (offset_start == 0 and offset_end == 0):
                continue
            if start_idx != -1 and not (offset_end <= start_idx or offset_start >= end_idx):
                labels[idx] = 1
        tokenized["labels"] = labels
        # Remove offsets from the example (not needed for training).
        tokenized.pop("offset_mapping")
        examples.append(tokenized)
    return examples


# Create Hugging Face Datasets for training and evaluation.
train_examples = create_token_classification_examples(train_data, tokenizer)
test_examples = create_token_classification_examples(test_data, tokenizer)
train_dataset = Dataset.from_list(train_examples)
test_dataset = Dataset.from_list(test_examples)

# Initialize a token-classification model with 2 labels (0: non-marker, 1: marker)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a metrics function.
# Here we compute token-level accuracy and f1-score (using seqeval for sequence labeling metrics).
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    # Remove ignored index (if any) - in our case, all tokens are valid.
    true_labels = [[str(l) for l in label] for label in labels]
    true_predictions = [[str(p) for (p, l) in zip(prediction, label)] for prediction, label in zip(predictions, labels)]
    
    acc = accuracy_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Set up training arguments.
training_args = TrainingArguments(
    output_dir="./RussianDMrecognizer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_model_id="MariaOls/RussianDMrecognizer",
    weight_decay=0.01,
    learning_rate=2e-5,
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model!
trainer.train()

# After training, push your model to the Hugging Face Hub.
trainer.push_to_hub()

print("Token classification model training and upload completed successfully!")
