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

## How It Works

This is a provisional script designed to test the model on custom input sentences. While it functions as intended, it is not optimized for efficiency and is meant primarily for quick experimentation or demonstration.

You can run `main.py` directly from the command line or modify and execute it using a Python editor (we used Spyder via Anaconda):

```bash
$ python main.py
```
The script will identify potential discourse markers in the provided sentences and evaluate whether they are functioning as discourse markers in context. Sentences can be edited directly within the script.

For a more efficient and flexible experience, we encourage users to try the hosted model on Hugging Face, where you can test it interactively or integrate it into your own workflows via the API.

## 🤗 Hugging Face
Find the fine-tuned model in https://huggingface.co/MariaOls/RussianDMrecognizer

## 📍 Coming Soon

This model will soon be updated with an expanded and improved dataset, enhanced training parameters, and new features—including the ability to process full paragraphs by splitting them into sentences and analyzing each one individually. We also plan to integrate a Russian language detector and, potentially, extend the tool to support additional languages.

## 📦 Dependencies
You'll need:
transformers - 
datasets - 
evaluate - 
sklearn - 
torch - 
pandas - 
huggingface_hub



