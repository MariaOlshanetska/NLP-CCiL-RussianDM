# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 20:19:12 2025

@author: Usuario
"""

import torch
import json
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from fine_tuning_and_training import preprocess_data, DMModel, DMOutput
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load original marker2id
with open("marker2id.json", "r", encoding="utf-8") as f:
    marker2id = json.load(f)
id2marker = {int(v): k for k, v in marker2id.items()}

# Load test data
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Preprocess data
test_dataset = preprocess_data(test_data, marker2id)

# Tokenizer
model_name = "viktoroo/sberbank-rubert-base-collection3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_dataset.map(tokenize, batched=True)

# Load model
config_path = "./RussianDMrecognizer"
model = DMModel.from_pretrained(config_path, config=model_name, num_marker_labels=len(marker2id))
model.eval()
if torch.cuda.is_available():
    model.to(device)

# Evaluation loop
true_binary, pred_binary = [], []
true_marker, pred_marker = [], []
loss_fn = torch.nn.CrossEntropyLoss()
total_loss = 0

for ex in tqdm(test_dataset):
    input_ids = torch.tensor([ex["input_ids"]]).to(device)
    attention_mask = torch.tensor([ex["attention_mask"]]).to(device)
    marker_label = torch.tensor([ex["marker_label"]]).to(device)
    binary_label = torch.tensor([ex["binary_label"]]).to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       marker_label=marker_label, binary_label=binary_label)
    
    total_loss += output.loss.item()
    pred_bin = torch.argmax(output.binary_logits, dim=-1).item()
    pred_mark = torch.argmax(output.marker_logits, dim=-1).item()

    true_binary.append(binary_label.item())
    pred_binary.append(pred_bin)
    true_marker.append(marker_label.item())
    pred_marker.append(pred_mark)

# Report
avg_loss = total_loss / len(test_dataset)
print("\nEvaluation Results")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Binary Accuracy: {accuracy_score(true_binary, pred_binary):.4f}")
print(f"Binary F1 Score: {f1_score(true_binary, pred_binary):.4f}")
print(f"Marker Accuracy: {accuracy_score(true_marker, pred_marker):.4f}")
print(f"Marker F1 Score (macro): {f1_score(true_marker, pred_marker, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(true_binary, pred_binary, target_names=["Not DM", "DM"]))