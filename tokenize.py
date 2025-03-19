# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:56:25 2025

@author: Usuario
"""

import json
from transformers import AutoTokenizer

# Load tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3")

# Load training and testing datasets from JSON files
with open('train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Define a function to tokenize data
def tokenize_data(data):
    tokenized_data = []
    for entry in data:
        tokenized_input = tokenizer(entry["input"], truncation=True, padding='max_length', max_length=128)
        tokenized_entry = {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "labels": tokenizer.encode(entry["output"], truncation=True, max_length=10)  # Tokenizing output (marker)
        }
        tokenized_data.append(tokenized_entry)

    return tokenized_data

# Tokenize datasets
tokenized_train_data = tokenize_data(train_data)
tokenized_test_data = tokenize_data(test_data)

# Save the tokenized datasets into JSON files
with open('tokenized_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(tokenized_train_data, f, ensure_ascii=False, indent=4)

with open('tokenized_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(tokenized_test_data, f, ensure_ascii=False, indent=4)

print("Data tokenization is complete. Files 'tokenized_train_data.json' and 'tokenized_test_data.json' created successfully.")

