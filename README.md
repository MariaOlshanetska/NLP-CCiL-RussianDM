# NLP-CCiL-RussianDM
Exercise 3: Russian Discourse Markers
# Russian Discourse Marker Classifier

Welcome! This repository contains a fine-tuned large language model designed to identify **discourse markers** in **Russian sentences**.

Discourse markers are words or phrases that manage the flow of conversation, establish connections between parts of a discourse, and signal the speaker's attitudes or intentions.

In Russian, discourse markers like "ну", "вот", "короче", and "значит" can be ambiguous when detecting Discourse Markers as they only work as such in some contexts!

Our model has a dual-head design: one head classifies among multiple marker labels, and the other performs a binary classification to determine if the candidate is actually acting as a discourse marker in that sentence.

---

## 🚀 Project Overview

This project is about training a model to recognize when certain Russian words are being used as **discourse markers** (DMs), based on sentence context.

### 🧾 What we did:

1. **Collected and curated data**:
   - We started from corpora and dictionaries containing lists and examples of possible discourse markers.
   - We built a structured dataset with annotated usage examples.

2. **Preprocessed the data**:
   - Turned spreadsheet and text corpus data into a structured format.
   - Combined each sentence with:
     - the candidate discourse marker,
     - a label (`true` or `false`) indicating whether the word functions as a discourse marker.

3. **Split into train/test**:
   - Data was split 80/20 into `train_data.json` and `test_data.json`.

4. **Fine-tuned a Russian LLM**:
   - Based on `viktoroo/sberbank-rubert-base-collection3`.
   - We built a multi-task model that:
     - predicts the candidate marker,
     - classifies it as a discourse marker or not.

5. **Trained and evaluated**:
   - Training was performed with Hugging Face's `Trainer` and `transformers` stack.
   - Final model saved and pushed to the 🤗 Hub.

---

## How It Works

1. **Model & Tokenizer Loading**  
   - **Tokenizer:**  
     The script loads a tokenizer from the base model `viktoroo/sberbank-rubert-base-collection3` to ensure proper tokenization with offset mappings.
   - **Model:**  
     It loads your fine-tuned token classification model from the Hugging Face Hub (with model ID `MariaOls/RussianDMtokenClassifier`). The model's classification head is reinitialized to work with 2 labels (0: non-marker, 1: marker).

2. **Input Sentences**  
   A list of sentences is hard-coded into the script. You can update the list in the script to analyze any Russian sentences you desire.

3. **Tokenization & Offset Mapping**  
   For each sentence:
   - The sentence is tokenized with offset mappings returned. These mappings provide the start and end character indices for each token.
   - The model performs inference and predicts a label for each token.

4. **Candidate Marker Extraction**  
   Contiguous tokens with the predicted label `1` are grouped together, and their offset mappings are used to extract the exact text of the candidate discourse marker.

5. **Output**  
   The script prints for each sentence:
   - The original input sentence.
   - The extracted discourse marker(s).  
   If no markers are detected, it prints "No discourse markers found."


Install the required packages with:

```bash
pip install torch transformers datasets seqeval huggingface_hub
```

## 🤗 Hugging Face
Find the fine-tuned model in https://huggingface.co/MariaOls/RussianDMtokenClassifier

## 📍 Coming Soon

This model will soon be updated with an expanded and improved dataset, enhanced training parameters, and new features—including the ability to process full paragraphs by splitting them into sentences and analyzing each one individually. We also plan to integrate a Russian language detector and, potentially, extend the tool to support additional languages.

## Prerequisites

- **Python 3.x**
- **PyTorch** (GPU support recommended for faster inference)
- **Transformers**
- **Datasets**
- **Seqeval** (for evaluation, if needed)
- **HuggingFace Hub**



