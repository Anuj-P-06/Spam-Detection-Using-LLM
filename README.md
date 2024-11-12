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

### Key Files
- **`fine_tune.py`**: Fine-tunes DistilBERT on an SMS spam dataset.
- **`app.py`**: Streamlit web application that loads the fine-tuned model and allows users to classify messages in real time.

## Requirements

To install required packages, run:
pip install -r requirements.txt


To install required packages, run:
pip install -r requirements.txt

