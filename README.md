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

## How it works

You can run `main.py` in your command line or edit it (and run) from a python editor (we used Spyder from Anaconda)

```bash
$ python main.py
```

The script will look for Discourse Markers in the given sentences and confirm if they are actually acting as a discourse marker. The sentences can be edited in the script itself.

## 🤗 Hugging Face
Find the fine-tuned model in https://huggingface.co/MariaOls/RussianDMrecognizer

## 📍 Coming Soon

This model will be soon updated with extra (and better) sentences in the dataset, improved training parameters and new functionalities like asking an input paragraph, splitting it in sentences and analyzing each sentence. Also we want to have a russian language detector and (why not) expand the tool to other languages. 

## 📦 Dependencies
You'll need:
transformers
datasets
evaluate
sklearn
torch
pandas
huggingface_hub



