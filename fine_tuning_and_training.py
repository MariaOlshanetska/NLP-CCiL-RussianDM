import json
import os
import torch
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoConfig,
    PreTrainedModel
)
from huggingface_hub import login, create_repo, upload_folder
import numpy as np
from sklearn.metrics import accuracy_score

# Additional imports for ModelOutput and dataclass
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

# Here we had to specify our huggingface_token that has been erased for security resons
"""huggingface_token = "TOP_SECRET"
login(token=huggingface_token)"""

# =============================================================================
# STEP 1. Prepare the data and marker vocabulary
# =============================================================================

def load_json_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_json_data("train_data.json")
test_data = load_json_data("test_data.json")

# Build a marker vocabulary from the training data
def build_marker_vocab(data):
    markers = set(item["output"] for item in data)
    marker2id = {marker: idx for idx, marker in enumerate(sorted(markers))}
    marker2id["UNKNOWN"] = len(marker2id)  # Add an "UNKNOWN" marker class to avoid previous problems with extra labels
    id2marker = {idx: marker for marker, idx in marker2id.items()}
    return marker2id, id2marker

marker2id, id2marker = build_marker_vocab(train_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save marker2id for consistent evaluation
with open("marker2id.json", "w", encoding="utf-8") as f:
    json.dump(marker2id, f, ensure_ascii=False, indent=4)

def preprocess_data(data, marker2id):
    processed = []
    for item in data:
        marker_text = item["output"]
        marker_id = marker2id.get(marker_text, marker2id["UNKNOWN"])  # Use "UNKNOWN" if not found
        processed.append({
            "text": item["input"],
            "marker": marker_text,
            "binary_label": 1 if item["is_discourse_marker"] else 0,
            "marker_label": marker_id
        })
    return Dataset.from_list(processed)

train_dataset = preprocess_data(train_data, marker2id)
test_dataset = preprocess_data(test_data, marker2id)

# =============================================================================
# STEP 2. Tokenize the input
# =============================================================================

model_name = "viktoroo/sberbank-rubert-base-collection3"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_data(example):
    # We tokenize the full sentence.
    # The labels (binary_label, marker_label) are kept as-is.
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

# We do not remove the label columns now since we need them for multi-task training.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =============================================================================
# STEP 3. Define a custom multi-task model using PreTrainedModel
# =============================================================================

# Define a custom ModelOutput for our multi-task output.
@dataclass
class DMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None
    marker_logits: Optional[torch.FloatTensor] = None
    binary_logits: Optional[torch.FloatTensor] = None

    def to_tuple(self):
        # Return exactly the tuple (marker_logits, binary_logits)
        return (self.marker_logits, self.binary_logits)

    def __iter__(self):
        return iter(self.to_tuple())

# In this model we use the shared encoder from the pretrained model.
# One head (binary_classifier) is used to decide if the candidate marker in the sentence is a discourse marker.
# The other head (marker_classifier) is used to “identify” the candidate marker from a fixed vocabulary.
class DMModel(PreTrainedModel):
    config_class = AutoConfig  # specify the config class

    def __init__(self, config, num_marker_labels, dropout_rate=0.1):
        super().__init__(config)
        self.num_marker_labels = num_marker_labels
        # Load the encoder using the model name stored in the config.
        self.encoder = AutoModel.from_pretrained(config.name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.marker_classifier = torch.nn.Linear(hidden_size, num_marker_labels)  # Marker head
        self.binary_classifier = torch.nn.Linear(hidden_size, 2)  # Binary head
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()  # for initializing weights

    def forward(self, input_ids, attention_mask, marker_label=None, binary_label=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Assume that the second element is the pooled output (works for BERT-like models)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        marker_logits = self.marker_classifier(pooled_output)
        binary_logits = self.binary_classifier(pooled_output)

        loss = None
        if marker_label is not None and binary_label is not None:
            loss_marker = self.loss_fct(marker_logits, marker_label)
            loss_binary = self.loss_fct(binary_logits, binary_label)
            loss = loss_marker + loss_binary

        # Return a proper ModelOutput instead of a plain dict.
        return DMOutput(
            loss=loss,
            logits=(marker_logits, binary_logits),
            marker_logits=marker_logits,
            binary_logits=binary_logits
        )

# Instantiate the custom model using AutoConfig.
config = AutoConfig.from_pretrained(model_name)
config.name_or_path = model_name  # ensure the model name is stored in the config
config.num_marker_labels = len(marker2id)
num_marker_labels = len(marker2id)
model_path = "./RussianDMrecognizer"
model = DMModel.from_pretrained(model_path, config=config, num_marker_labels=config.num_marker_labels)


# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

# =============================================================================
# STEP 4. Define training arguments and metrics
# =============================================================================

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
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="RussianDMrecognizer",
    weight_decay=0.01,
    learning_rate=2e-5,
)


# =============================================================================
# STEP 5. Instantiate Trainer and start training
# =============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer locally
model.save_pretrained("./RussianDMrecognizer")
tokenizer.save_pretrained("./RussianDMrecognizer")

# Local save path
save_path = "./RussianDMrecognizer"
os.makedirs(save_path, exist_ok=True)

# Save the model weights (as a PyTorch file)
torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

# Save the tokenizer
tokenizer.save_pretrained(save_path)
# This should be uncommented when actually updating to HuggingFace
repo_id = "MariaOls/RussianDMrecognizer"
"""create_repo(repo_id, token=huggingface_token, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path=save_path,
    path_in_repo=".",  # upload all contents to the root of the repo
    token=huggingface_token
)"""

print(f"Model has been successfully uploaded to: https://huggingface.co/{repo_id}")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Push the model to the Hugging Face Hub
trainer.push_to_hub()

# =============================================================================
# STEP 6. Inference: Define a function for custom input
# =============================================================================

def classify_marker(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs)
    binary_pred = torch.argmax(outputs.binary_logits, dim=-1).item()
    marker_pred = torch.argmax(outputs.marker_logits, dim=-1).item()

    binary_result = "TRUE (Discourse Marker)" if binary_pred == 1 else "FALSE (Not a Discourse Marker)"
    marker_text = id2marker.get(marker_pred, "Unknown")

    return {"sentence": sentence, "predicted_marker": marker_text, "classification": binary_result}

# Example classification
custom_sentence = "Это, конечно, хорошая идея."
prediction = classify_marker(custom_sentence)

print("\nCustom Input Analysis:")
print(f"Sentence: {prediction['sentence']}")
print(f"Predicted Marker: {prediction['predicted_marker']}")
print(f"Classification: {prediction['classification']}")

