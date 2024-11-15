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

This system leverages **DistilBERT**, a lightweight transformer model, fine-tuned on an Email/SMS etc message dataset to classify messages as spam or not. It offers an efficient solution for filtering unwanted or harmful content in real-time.

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
- **Virtual Enviroment**: Create a venv files using ``` python -m venv myenv ``` and activate it using ``` source myenv/Scripts/activate ```



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
```

## Running the Application

Once the model has been fine-tuned, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```
## App Usage

1. **Launch the Streamlit app**:
    - Open a terminal or command prompt.
    - Run the following command to start the app:
      ```bash
      streamlit run app.py
      ```
    - This will start a local server and open the web application in your default browser.

2. **Enter an SMS message**:
    - In the web app, you'll see a text box where you can type or paste an SMS message.

3. **Classify the message**:
    - After entering the message, click the **"Classify"** button.

4. **View the result**:
    - The app will classify the message as either **"Spam"** or **"Not Spam"** based on the model's prediction.

## Screenshots

Below are some screenshots of the app interface in action:

![Screenshot 1](https://github.com/user-attachments/assets/69c9ede9-0e79-4d6b-a8a9-307fedb6501a)
![Screenshot 2](https://github.com/user-attachments/assets/9515e0a8-c1d1-40d8-9d07-736c1f536d7f)


## Results
The fine-tuned DistilBERT model provides high accuracy in classifying SMS messages as either "Spam" or "Not Spam." The results from the real-time web app demonstrate the effectiveness of using a transformer model for spam detection, with minimal latency and fast predictions.

This system can help users filter out unwanted messages and protect against potential spam or phishing attempts.

This project represents not only a powerful tool for detecting SMS spam but also a stepping stone toward a larger, more comprehensive solution for managing text-based communications in the digital age. By building on this foundation, it is possible to create systems that not only protect users but also evolve to stay ahead of increasingly sophisticated spam tactics, making digital spaces safer and more enjoyable for all users.

