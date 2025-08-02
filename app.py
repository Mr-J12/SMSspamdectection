import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

data = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_test = X_test.fillna('')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors, y_train) 


st.set_page_config(page_title="TextGuardAI", page_icon="📩", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stTextArea textarea {
        background-color: #2c2c2c;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🚀 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Quickly check if a message is <b>Spam</b> or <b>Ham</b> using Machine Learning!</p>", unsafe_allow_html=True)

user_input = st.text_area("💬 Enter your message here:")
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://plus.unsplash.com/premium_photo-1674811564431-4be5bd37fb6a?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fHdlYnNpdGUlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fHww");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 20px;
        border-radius: 10px;
    }}
    .stTextInput > div > div > input {{
        background-color: #ffffffdd;
    }}

    </style>
""", unsafe_allow_html=True)

if st.button("🔍 Predict"):
    if user_input.strip() != "":
        with st.spinner('Analyzing the message... 🔄'):
            time.sleep(1)
            user_vector = vectorizer.transform([user_input])
            prediction = model.predict(user_vector)
            if prediction[0] == 1:
                st.error("🚫 It's a SPAM message!")
            else:
                st.success("✅ It's a HAM (not spam) message!")
    else:
        st.warning("⚠️ Please enter a message first.")

 