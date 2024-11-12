# Spam Detection Using LLM

This project implements a spam detection system using a fine-tuned DistilBERT model to classify SMS messages as "Spam" or "Not Spam." It includes:

- **`fine_tune.py`**: Python script for training and fine-tuning the DistilBERT model on a spam dataset.
- **`app.py`**: Streamlit web app for making predictions on new messages.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement and Objectives](#problem-statement-and-objectives)
3. [Proposed System](#proposed-system)
4. [Modules](#modules)
5. [Key Files](#key-files)
6. [Requirements](#requirements)
7. [Dataset](#dataset)
8. [Training the Model](#training-the-model)
9. [Running the Application](#running-the-application)
10. [App Usage](#app-usage)
11. [Screenshots](#screenshots)
12. [Results](#results)

## Overview

This system leverages **DistilBERT**, a lightweight transformer model, fine-tuned on an SMS message dataset to classify messages as spam or not. It offers an efficient solution for filtering unwanted or harmful content in real-time.

## Problem Statement and Objectives

### Problem Statement

The increasing volume of SMS communications necessitates effective filtering of unwanted or malicious messages. Traditional spam filters often struggle against modern spam tactics. By utilizing transformer-based models like DistilBERT, this system provides a more accurate and robust solution.

### Objectives

1. Develop a high-performing spam classification model using a fine-tuned DistilBERT.
2. Create a user-friendly web app for real-time spam detection.
3. Demonstrate the effectiveness of large language models (LLMs) in text classification tasks.

## Proposed System

The proposed system uses a fine-tuned DistilBERT transformer model to classify SMS messages as "Spam" or "Not Spam." The system includes the following components:

1. **Training Pipeline**: Fine-tunes the DistilBERT model for spam classification.
2. **Prediction Interface**: A Streamlit web app that classifies SMS messages in real-time.

## Modules

### 1. Data Loading and Preprocessing
- Loads the SMS spam dataset.
- Tokenizes the data using DistilBERT's tokenizer.

### 2. Model Fine-Tuning
- Fine-tunes the DistilBERT model for binary classification (Spam vs Not Spam).

### 3. Evaluation
- Evaluates the model’s performance using a validation set.

### 4. Web Application
- Streamlit-based app for making real-time predictions.

## Key Files

- **`fine_tune.py`**: Fine-tunes the DistilBERT model on the SMS spam dataset.
- **`app.py`**: Streamlit web app that loads the fine-tuned model for real-time predictions.

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Additional Recommendations

- **GPU**: For faster training, it is recommended to use a GPU-enabled environment.
- **CUDA**: Ensure that CUDA is properly installed and configured for optimal performance on compatible GPUs.
- **Data Augmentation**: To further improve model accuracy, consider augmenting the dataset with additional samples if needed.
- **Hyperparameter Tuning**: Experiment with hyperparameter tuning (e.g., learning rate, batch size) for potentially better model performance.

## Dataset

The dataset used in this project is the **`sms_spam`** dataset from the Hugging Face Datasets library. It contains SMS messages that are labeled as either "spam" or "ham" (not spam). The dataset is balanced, with approximately equal distributions of spam and ham messages. 

You can access the dataset and further details here: [Hugging Face Datasets - SMS Spam](https://huggingface.co/datasets/sms_spam).

## Training the Model

To train the model, run the script **`fine_tune.py`**. This script performs the following steps:

1. Loads and preprocesses the SMS spam dataset.
2. Tokenizes the data using DistilBERT's tokenizer.
3. Fine-tunes the DistilBERT model on the dataset using the Hugging Face `Trainer` class.
4. Saves the fine-tuned model and tokenizer to the `./fine_tuned_model` directory for future use.

Run the following command to start the training:

```bash
python fine_tune.py



