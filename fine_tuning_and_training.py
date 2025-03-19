from datasets import load_dataset
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
import torch

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3")

# Load the tokenized datasets
dataset = load_dataset("json", data_files={"train": "tokenized_train_data.json", "test": "tokenized_test_data.json"})

# Adjust the function to align labels with tokens without tokenization
def align_labels_with_tokens(examples):
    # Get the input_ids, attention_mask, and labels
    input_ids = examples["input_ids"]
    attention_mask = examples["attention_mask"]
    labels = examples["labels"]
    
    # Initialize the final labels list
    aligned_labels = []
    
    # Iterate through each example in the dataset
    for i in range(len(input_ids)):
        # Get the tokens from input_ids
        word_ids = tokenizer.convert_ids_to_tokens(input_ids[i])  # Get the tokenized words from input_ids
        label_ids = labels[i]  # Labels are already aligned with the tokens in this case
        
        # Ensure labels are the same length as input_ids
        # If labels are shorter, pad with -100 (ignored by the loss function)
        # If labels are longer, truncate them
        if len(label_ids) < len(input_ids[i]):
            label_ids += [-100] * (len(input_ids[i]) - len(label_ids))
        elif len(label_ids) > len(input_ids[i]):
            label_ids = label_ids[:len(input_ids[i])]
        
        # Ignore padding tokens (assign -100 to padding tokens)
        aligned_labels.append([label if token != tokenizer.pad_token else -100 for token, label in zip(word_ids, label_ids)])
    
    # Add aligned labels to the dataset
    examples["labels"] = aligned_labels
    return examples

# Load and preprocess the dataset (replace 'your_dataset' with your actual dataset)
dataset = load_dataset('json', data_files={'train': 'tokenized_train_data.json', 'test': 'tokenized_test_data.json'})

# Tokenized datasets should already contain 'input_ids', 'attention_mask', and 'labels'
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
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
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
trainer.train()




