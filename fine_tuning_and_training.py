from datasets import load_dataset
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
import torch

from transformers.utils import logging
logging.set_verbosity_info()  # Enable logging
tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3", use_fast=True, cache_dir="/path/to/cache")

# Load the tokenized datasets
dataset = load_dataset("json", data_files={"train": "tokenized_train_data.json", "test": "tokenized_test_data.json"})


subset_train_size = 20  # Number of examples to use for training
subset_test_size = 10

train_dataset = dataset["train"].select(range(subset_train_size))
test_dataset = dataset["test"].select(range(subset_test_size))

# Adjust the function to align labels with tokens without tokenization
def align_labels_with_tokens(examples):
    input_ids = examples["input_ids"]
    attention_mask = examples["attention_mask"]
    labels = examples["labels"]

    aligned_labels = []
    for i in range(len(input_ids)):
        # Pad or truncate labels to match input_ids length
        if len(labels[i]) < len(input_ids[i]):
            labels[i] += [-100] * (len(input_ids[i]) - len(labels[i]))
        elif len(labels[i]) > len(input_ids[i]):
            labels[i] = labels[i][:len(input_ids[i])]

        # Ignore padding tokens
        aligned_labels.append([label if mask == 1 else -100 for label, mask in zip(labels[i], attention_mask[i])])

    examples["labels"] = aligned_labels
    return examples


# Apply the alignment function to the datasets
train_dataset = dataset['train'].map(align_labels_with_tokens, batched=True)
test_dataset = dataset['test'].map(align_labels_with_tokens, batched=True)



# Load the pre-trained model for token classification
model = AutoModelForTokenClassification.from_pretrained("viktoroo/sberbank-rubert-base-collection3", num_labels=2, ignore_mismatched_sizes=True)

# Now you can change the classifier layer to match the number of labels you want
model.num_labels = 2
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for saved models and logs
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2, 
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Start the training process
#trainer.train()


##-----------------------------EVALUATE----------------------------------

# Evaluate the model on the test dataset
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Make predictions on the test dataset
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)

# Function to extract and visualize discourse markers
def extract_discourse_markers(input_ids, labels, predicted_labels, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    discourse_markers = []

    for token, true_label, pred_label in zip(tokens, labels, predicted_labels):
        if true_label != -100:  # Ignore padding tokens
            if true_label == 1:  # Assuming 1 is the label for discourse markers
                discourse_markers.append({
                    "token": token,
                    "true_label": true_label,
                    "predicted_label": pred_label
                })

    return discourse_markers

# Analyze discourse markers in the test dataset
for i in range(len(test_dataset)):
    input_ids = test_dataset[i]["input_ids"]
    labels = test_dataset[i]["labels"]
    preds = predicted_labels[i]

    # Extract discourse markers for this example
    discourse_markers = extract_discourse_markers(input_ids, labels, preds, tokenizer)

    # Print results for discourse markers
    print(f"\nExample {i + 1}:")
    for marker in discourse_markers:
        print(f"Token: {marker['token']}, True Label: {marker['true_label']}, Predicted Label: {marker['predicted_label']}")

# Test with custom input (optional)
custom_input = "Это, конечно, хорошая идея."
custom_input_ids = tokenizer(custom_input, return_tensors="pt", truncation=True, padding=True)["input_ids"]

# Make predictions for custom input
with torch.no_grad():
    logits = model(custom_input_ids).logits
    custom_predicted_labels = logits.argmax(axis=-1).squeeze().tolist()

# Extract discourse markers from custom input
custom_tokens = tokenizer.convert_ids_to_tokens(custom_input_ids.squeeze().tolist())
custom_discourse_markers = []

for token, pred_label in zip(custom_tokens, custom_predicted_labels):
    if pred_label == 1:  # Assuming 1 is the label for discourse markers
        custom_discourse_markers.append({
            "token": token,
            "predicted_label": pred_label
        })

# Print results for custom input
print("\nCustom Input Analysis:")
for marker in custom_discourse_markers:
    print(f"Token: {marker['token']}, Predicted Label: {marker['predicted_label']}")