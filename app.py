import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="Spam Detection", page_icon="📧")

# Load fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./fine_tuned_model")

# Function to predict whether a message is spam or not
def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Not Spam"

def main():
    st.title("Spam Detection")
    st.write("This is a Spam Detection App using a fine-tuned DistilBERT model.")

    # Input text box for the user
    message = st.text_area("Enter message to classify as spam or not:")

    if st.button("Predict"):
        if message:
            prediction = predict_spam(message)
            st.write(f"The message is: {prediction}")
        else:
            st.write("Please enter a message to classify.")

if __name__ == "__main__":
    main()
