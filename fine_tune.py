from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = load_dataset("sms_spam")

# Print the dataset structure and inspect the columns
print(dataset)
print(dataset['train'][0])  # Print the first row of the 'train' split

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize the dataset using the correct column
def tokenize_function(examples):
    return tokenizer(examples["sms"], padding="max_length", truncation=True)

# Apply the tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Check if 'test' split exists, else use 'validation' or create your own split
train_dataset = tokenized_datasets["train"]

# If there is no 'test' split, you can use 'validation' or manually split the dataset
eval_dataset = tokenized_datasets.get("test", tokenized_datasets.get("validation"))

# If neither 'test' nor 'validation' exists, manually split the dataset
if eval_dataset is None:
    eval_dataset = train_dataset.shuffle(seed=42).select([i for i in range(len(train_dataset)//10)])  # Take 10% as eval dataset
    train_dataset = train_dataset.select([i for i in range(len(train_dataset)//10, len(train_dataset))])  # Take the remaining 90% as train dataset

# Set up training arguments
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",  # Evaluate every 'eval_steps'
    save_strategy="steps",  # Save every 'save_steps'
    eval_steps=500,  # Evaluate every 500 steps
    save_steps=500,  # Save every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


# Define compute_metrics function (optional, if you want to track metrics)
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  # Optional: to compute accuracy
)

# Train the model
trainer.train()

# Save the model after training
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")


# Optionally, push the model to Hugging Face Hub
# from huggingface_hub import HfApi, HfFolder

# model.push_to_hub("Anuj02003/Spam-classification-using-LLM")
# tokenizer.push_to_hub("Anuj02003/Spam-classification-using-LLM")
