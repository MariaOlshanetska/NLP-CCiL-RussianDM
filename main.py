# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:33:57 2025

@author: Usuario
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def extract_marker(sentence, tokenizer, model, device):
    # Ensure the sentence is a non-empty string.
    sentence = str(sentence)
    
    # Tokenize the sentence with offset mapping.
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = encoding["offset_mapping"][0].tolist()

    # Run inference.
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    # Gather contiguous token spans predicted as discourse markers (label == 1).
    candidate_spans = []
    current_span = None
    for idx, label in enumerate(predictions):
        # Skip special tokens (which have offset [0, 0])
        if offsets[idx] in ([0, 0], (0, 0)):
            continue
        if label == 1:
            token_offset = offsets[idx]
            if current_span is None:
                current_span = list(token_offset)
            else:
                # Extend span if tokens are contiguous.
                if token_offset[0] == current_span[1]:
                    current_span[1] = token_offset[1]
                else:
                    candidate_spans.append(current_span)
                    current_span = list(token_offset)
        else:
            if current_span is not None:
                candidate_spans.append(current_span)
                current_span = None
    if current_span is not None:
        candidate_spans.append(current_span)

    candidate_texts = [sentence[start:end] for start, end in candidate_spans if end > start]
    return candidate_texts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the tokenizer from the base model that was used during training.
    base_model_id = "viktoroo/sberbank-rubert-base-collection3"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    
    # Load your fine-tuned model (from your Hub) for token classification.
    model_id = "MariaOls/RussianDMrecognizer"
    model = AutoModelForTokenClassification.from_pretrained(model_id, ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()

    # List the sentences you want to analyze.
    sentences = [
        "Кажется, он уже ушел.",
	"Объяснение проблемы, конечно, выглядело очевидным.",
	"Правда причина этого была известна всем.",
	"Напротив, некоторые участники высказались резко против.",
	"Документ содержал, скажем, указания для исполнителей.",
	"Конечно, влияние на процесс оказалось значительным.",
	"Во-первых, это важно.",
	"Это кажется маловероятным.",
	"Пожалуй, стоит начать подготовку заранее.",
	"Наверное только после вмешательства властей.",
	"Скажем прямо, это провал.",
	"Итак продолжим наше собеседование.",
	"Она работает в Итак — новой технологической компании.",
	"Он конечно был прав.",
	"Их реакция была, может, оправданной.",
	"по-моему, он не прав",
	"Он выбрал вариант по-моему вкусу."
    ]

    # Process each sentence.
    for sentence in sentences:
        candidate_texts = extract_marker(sentence, tokenizer, model, device)
        print("\nInput sentence:", sentence)
        if candidate_texts:
            print("Extracted candidate marker(s):")
            for candidate in candidate_texts:
                print(" -", candidate)
        else:
            print("No discourse markers found.")

if __name__ == "__main__":
    main()






