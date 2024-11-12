# Spam-Detection-Using-LLM

This project is a spam detection application that uses a fine-tuned DistilBERT model to classify SMS messages as "Spam" or "Not Spam." It includes:
- **`fine_tune.py`**: Python script for training and fine-tuning the DistilBERT model on a spam dataset.
- **`app.py`**: Streamlit web app for making predictions on new messages.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Additional Recommendations](#additional-recommendations)
4. [Dataset](#dataset)
5. [Training the Model](#training-the-model)
6. [Running the Application](#running-the-application)
7. [App Usage](#app-usage)
8. [Screenshots](#screenshots)

## Overview

This spam detection system leverages **DistilBERT**, a lightweight transformer model, fine-tuned on SMS message data to identify whether a message is spam or not. This tool is useful for filtering unwanted or potentially harmful content.

## Problem Statement and Objectives

### Problem Statement

With the increasing volume of text-based communication, there is a rising need to filter out unwanted spam messages that can be disruptive or harmful. Traditional rule-based spam filters often struggle with modern spam techniques. Leveraging transformer-based models like DistilBERT allows us to enhance the accuracy and robustness of spam detection systems.

### Objectives

1. Develop a robust spam classification model using a fine-tuned DistilBERT.
2. Create a user-friendly web app to classify messages as "Spam" or "Not Spam."
3. Demonstrate the use of large language models (LLMs) in real-world text classification tasks.

## Proposed System

The proposed system uses a DistilBERT transformer model fine-tuned on SMS spam datasets. The model classifies incoming messages in real-time with high accuracy. The solution includes:

1. **Training Pipeline**: A Python script (`fine_tune.py`) for fine-tuning the model.
2. **Prediction Interface**: A Streamlit-based web application (`app.py`) that loads the fine-tuned model to classify messages in real-time.

### Module Description

1. **Data Loading and Preprocessing**: Loads the SMS spam dataset, tokenizes it using DistilBERT's tokenizer.
2. **Model Fine-Tuning**: Fine-tunes DistilBERT for binary classification on spam detection.
3. **Evaluation**: Evaluates the model's performance on the validation set.
4. **Web Application**: Streamlit app for real-time prediction.


### Key Files
- **`fine_tune.py`**: Fine-tunes DistilBERT on an SMS spam dataset.
- **`app.py`**: Streamlit web application that loads the fine-tuned model and allows users to classify messages in real time.

## Requirements

To install required packages, run:

```bash
pip install -r requirements.txt
```


## Additional Recommendations
- **GPU**: For faster training, a GPU-enabled environment is recommended.
- **CUDA**: Make sure CUDA is installed and configured for optimal performance on compatible GPUs.

## Dataset

The dataset used is the `sms_spam` dataset from the Hugging Face Datasets library, which contains SMS messages labeled as "spam" or "ham" (not spam).

## Training the Model

To train the model, run `fine_tune.py`. This script:

1. Loads and preprocesses the SMS spam dataset.
2. Tokenizes the data using DistilBERT's tokenizer.
3. Fine-tunes DistilBERT using the `Trainer` class.
4. Saves the fine-tuned model and tokenizer to the `./fine_tuned_model` directory.

### Block Diagram

                   +----------------+
                   |                |
                   | SMS Spam Data  |
                   |                |
                   +----------------+
                           |
                           v
                   +----------------+
                   |                |
                   |   Tokenizer    |
                   |                |
                   +----------------+
                           |
                           v
                   +----------------+
                   |                |
                   |   DistilBERT   |
                   |   Fine-Tuning  |
                   |                |
                   +----------------+
                           |
                           v
                   +----------------+
                   |                |
                   |    Web App     |
                   |                |
                   +----------------+
                           |
                           v
                  User enters message and receives prediction

## Results
![Screenshot 2024-11-12 172040](https://github.com/user-attachments/assets/69c9ede9-0e79-4d6b-a8a9-307fedb6501a)
![Screenshot 2024-11-12 172018](https://github.com/user-attachments/assets/9515e0a8-c1d1-40d8-9d07-736c1f536d7f)


