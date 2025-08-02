import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

data = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

data = data.dropna(subset=['label', 'message'])

data = data.dropna(subset=['label', 'message'])

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_test = X_test.fillna('')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(y_test, y_pred)

st.set_page_config(page_title="TextGuard", page_icon="📩", layout="centered", initial_sidebar_state="collapsed")

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

st.subheader("Classification Report:")
st.text(report)
st.markdown("<h1 style='text-align: center;'>🚀 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Quickly check if a message is <b>Spam</b> or <b>Ham</b> using Machine Learning!</p>", unsafe_allow_html=True)
user_input = st.text_area("💬 Enter your message here:")

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
        

st.subheader("Model Evaluation Metrics:")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
st.pyplot(fig)

st.subheader("📊 Dataset Class Distribution")
class_counts = data['label'].value_counts()
fig_bar, ax_bar = plt.subplots()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis", ax=ax_bar)
ax_bar.set_title("Distribution of Ham and Spam Messages")
ax_bar.set_ylabel("Count")
ax_bar.set_xlabel("Message Type")
st.pyplot(fig_bar)

st.subheader("📈 Spam vs Ham Proportion")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
ax_pie.axis('equal')
st.pyplot(fig_pie)

st.subheader("📝 Most Common Words in Spam Messages")

spam_messages = data[data['label'] == 'spam']['message']
all_words = " ".join(spam_messages).lower().split()
common_words = Counter(all_words).most_common(10)

words, counts = zip(*common_words)
fig_words, ax_words = plt.subplots()
sns.barplot(x=list(counts), y=list(words), palette="magma", ax=ax_words)
ax_words.set_title("Top 10 Common Words in Spam Messages")
ax_words.set_xlabel("Frequency")
st.pyplot(fig_words)