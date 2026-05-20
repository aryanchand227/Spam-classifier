import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# CUSTOM CSS (PREMIUM UI)
# =========================
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Glassmorphism container for the main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #b2bec3;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #00C9FF !important;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%) !important;
        color: #1e272e !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        font-size: 1.1rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 201, 255, 0.5) !important;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CACHED RESOURCES
# =========================
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')
    return set(stopwords.words('english')), WordNetLemmatizer()

@st.cache_resource
def load_models():
    m = joblib.load("saved_models/best_spam_classifier.pkl")
    v = joblib.load("saved_models/tfidf_vectorizer.pkl")
    return m, v

stop_words, lemmatizer = setup_nltk()
model, vectorizer = load_models()

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# =========================
# STREAMLIT UI
# =========================
st.markdown("<h1>🛡️ NLP Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect whether a message is Spam or Ham with AI</div>", unsafe_allow_html=True)

st.write("") # Spacer

message = st.text_area(
    "Message Content",
    placeholder="Type or paste your email/SMS message here...",
    height=200,
    label_visibility="collapsed"
)

st.write("") # Spacer

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("Analyze Message 🚀")

if predict_btn:
    if not message.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing with AI models..."):
            clean_message = preprocess_text(message)
            vectorized_message = vectorizer.transform([clean_message])
            prediction = model.predict(vectorized_message)
            probabilities = model.predict_proba(vectorized_message)
            
            spam_probability = probabilities[0][1]

        st.write("") # Spacer
        
        if prediction[0] == 1:
            st.error(f"🚨 **SPAM DETECTED**\n\nThe AI is **{spam_probability:.1%}** confident that this message is spam.")
        else:
            st.success(f"✅ **SAFE MESSAGE (HAM)**\n\nThe AI is **{(1-spam_probability):.1%}** confident that this message is safe.")