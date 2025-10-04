import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# -----------------------------
# Load model & tokenizer
# -----------------------------
@st.cache_resource
def load_assets():
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    # Load Keras model
    model = load_model('next_word_predictor_model.keras')  # <-- make sure this matches your GitHub filename
    return model, tokenizer

model, tokenizer = load_assets()
MAX_SEQUENCE_LEN = 98  # adjust if your model uses a different sequence length

# -----------------------------
# Prediction function
# -----------------------------
def predict_next_words(seed_text, top_k=3):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list) == 0:
        return ["No prediction available"]

    padded = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN, padding='pre')
    probs = model.predict(padded, verbose=0)[0]

    top_indices = probs.argsort()[-top_k:][::-1]
    index_word = {v: k for k, v in tokenizer.word_index.items()}

    suggestions = []
    for idx in top_indices:
        word = index_word.get(int(idx))
        if word:
            suggestions.append(word)
    return suggestions

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Next Word Predictor", layout="wide")

# Top-right corner name
st.markdown(
    """
    <div style="
        position: absolute;
        top: 10px;
        right: 20px;
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: underline;
        z-index: 1000;
    ">
        Ayush Dwivedi
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Next Word Predictor (LSTM)")
st.markdown("Type a few words, and the model will predict the next words.")

# User input
user_input = st.text_area("Enter your sentence:", height=100)
top_k = st.slider("Number of suggestions:", min_value=1, max_value=10, value=3)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        suggestions = predict_next_words(user_input, top_k=top_k)
        st.success("Predicted Next Word(s):")
        for i, word in enumerate(suggestions, 1):
            st.write(f"{i}. {word}")
