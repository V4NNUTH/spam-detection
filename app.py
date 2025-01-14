import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Initialize required objects
lstem = LancasterStemmer()
tfvec = TfidfVectorizer(stop_words='english')

# Load the trained classifier and data
@st.cache_resource
def load_data():
    datafile = pickle.load(open("training_data.pkl", "rb"))
    return datafile["message_x"], datafile["classifier"]

message_x, classifier = load_data()

# Function to preprocess the input message
def preprocess_message(messages):
    processed_messages = []
    for message in messages:
        # Filter only alphabets
        message = ''.join(filter(lambda char: (char.isalpha() or char == " "), message))
        # Tokenize and stem words
        words = word_tokenize(message)
        processed_messages.append(' '.join([lstem.stem(word) for word in words]))
    return processed_messages

# Function to create Bag of Words representation
def bow_transform(message):
    tfidf_transformer = tfvec.fit(message_x)
    return tfidf_transformer.transform(message).toarray()

# Streamlit app UI
st.title("Spam Detector")
st.subheader("Spam or Not Spam? Message Detector")
st.write("Enter a message below to classify it as Spam or Not Spam:")

# Input text box
user_message = st.text_area("Your message:", height=150)

# if st.button("Check Message"):
#     if user_message.strip():
#         # Preprocess and classify the message
#         preprocessed_message = preprocess_message([user_message])
#         prediction = classifier.predict(bow_transform(preprocessed_message)).reshape(1, -1)
#         result = "spam" if prediction else "Not Spam"
#         st.write(f"Your message is classified as: **{result.upper()}**")
#     else:
#         st.warning("Please enter a valid message to classify.")

if st.button("Check Message"):
    if user_message.strip():
        # Preprocess and classify the message
        preprocessed_message = preprocess_message([user_message])
        prediction = classifier.predict(bow_transform(preprocessed_message)).reshape(1, -1)
        result = "spam" if prediction else "not Spam"
        if result == "spam":
            st.write(f"Your message is classified as: **{result.upper()}**")
            st.error("Spam detected", icon="❌")
        else:
            st.write(f"Your message is classified as: **{result.upper()}**")
            st.success("Message is Not Spam", icon="✔️")
    else:
        st.warning("Please enter a valid message to classify.")
