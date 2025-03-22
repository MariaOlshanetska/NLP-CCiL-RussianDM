# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 18:37:37 2025

@author: Usuario
"""
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, 
    AutoTokenizer,
    pipeline
)
import torch

# STEP A: Token Classification (Identifying potential discourse markers)
token_model_name = "MariaOls/RussianDMrecognizer"
token_classifier = pipeline("token-classification", model=token_model_name, aggregation_strategy="simple")

# STEP B: Sequence Classification (Confirming if the detected marker functions as a discourse marker)
seq_model_name = "sberbank-ai/ruBert-base"
seq_model = AutoModelForSequenceClassification.from_pretrained(sberbank-ai/ruBert-base)
seq_tokenizer = AutoTokenizer.from_pretrained(sberbank-ai/ruBert-base)
seq_model.eval()

def confirm_discourse_marker(sentence, marker):
    """Confirm if a detected marker functions as a discourse marker."""
    inputs = seq_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)  # ✅ seq_tokenizer is now properly defined
    with torch.no_grad():
        outputs = seq_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return "TRUE (Discourse Marker)" if predicted_label == 1 else "FALSE (Not a Discourse Marker)"

def analyze_sentence(sentence):
    """Analyze sentence to first detect, then confirm discourse markers."""
    detected_tokens = token_classifier(sentence)

    if not detected_tokens:
        print("🔍 No potential discourse markers detected automatically.")
        return

    print(f"📖 Sentence: '{sentence}'\n")
    print("🔎 Detected potential discourse markers and their classifications:\n")

    for entity in detected_tokens:
        marker = entity['word']
        classification = confirm_discourse_marker(sentence, marker)
        print(f"• Marker: '{marker}' → {classification}")

# Example usage:
if __name__ == "__main__":
    test_sentence = "Это, конечно, была отличная идея, однако я не уверен, что все согласны."
    analyze_sentence(test_sentence)