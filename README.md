# NLP-CCiL-RussianDM
Exercise 3: Russian Discourse Markers
# Russian Discourse Marker Classifier

Welcome! This repository contains a fine-tuned large language model designed to identify **discourse markers** in **Russian sentences** — that is, words like _например_ or _впрочем_, and whether they're really functioning as discourse markers in context.

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

## 🧠 How it works

The project is missing a `main.py` file (coming soon!), but here's how the **ideal user flow** will look:

```bash
$ python main.py
```

The script will:

1. Ask the user to input a text in Russian.

2. Split the input into individual sentences.

3. Search each sentence for discourse marker candidates through the fine-tuned model.

4. Run each candidate through the fine-tuned model.

5. Output whether each candidate is used as a discourse marker or not.

## 🤗 Hugging Face
Find the fine-tuned model in https://huggingface.co/MariaOls/RussianDMrecognizer

## 📍 Coming Soon

This model will be soon updated with extra (and better) sentences in the dataset, improved training parameters and new functionalities.

## 📦 Dependencies
You'll need:
transformers
datasets
evaluate
sklearn
torch
pandas
huggingface_hub



