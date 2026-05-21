import streamlit as st
import joblib
import re
import nltk
import os
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Spam Classifier Dashboard",
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
    
    h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
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
    
    /* Secondary/feedback buttons */
    div.stButton > button[key^="btn_correct"],
    div.stButton > button[key^="btn_incorrect"],
    div.stButton > button[key^="btn_submit_correction"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
        border-radius: 12px !important;
    }
    div.stButton > button[key^="btn_correct"]:hover {
        background: rgba(46, 204, 113, 0.2) !important;
        border-color: #2ecc71 !important;
    }
    div.stButton > button[key^="btn_incorrect"]:hover {
        background: rgba(231, 76, 60, 0.2) !important;
        border-color: #e74c3c !important;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-radius: 8px 8px 0px 0px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        color: #b2bec3 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: none !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00C9FF !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #00C9FF !important;
        border-bottom: 2px solid #00C9FF !important;
    }
    
    /* Metric container styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* DataFrame table styling */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
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
# FEEDBACK LOGGING FUNCTION
# =========================
def save_feedback(text, clean_text, label):
    feedback_file = "data/processed/feedback_dataset.csv"
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    new_data = pd.DataFrame([{
        "text": text,
        "clean_text": clean_text,
        "label": int(label),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
        try:
            new_data.to_csv(feedback_file, mode='a', header=False, index=False)
        except Exception as e:
            st.error(f"Error appending feedback: {e}")
    else:
        try:
            new_data.to_csv(feedback_file, mode='w', header=True, index=False)
        except Exception as e:
            st.error(f"Error writing feedback file: {e}")

# =========================
# STATE INITIALIZATION
# =========================
if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "spam_probability" not in st.session_state:
    st.session_state.spam_probability = 0.0
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
if "show_correction_input" not in st.session_state:
    st.session_state.show_correction_input = False

# =========================
# MAIN APP HEADERS
# =========================
st.markdown("<h1>🛡️ Interactive Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Self-Improving NLP Classifier with Active Learning Feedback Loop</div>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2 = st.tabs(["🔍 Message Analysis", "⚙️ Active Learning & Retraining"])

# =========================
# TAB 1: MESSAGE ANALYSIS
# =========================
with tab1:
    message = st.text_area(
        "Message Content",
        placeholder="Type or paste your email/SMS message here...",
        height=180,
        label_visibility="collapsed"
    )
    
    # Auto-reset state if message is modified
    if message != st.session_state.analyzed_text:
        st.session_state.prediction = None
        st.session_state.spam_probability = 0.0
        st.session_state.feedback_submitted = False
        st.session_state.show_correction_input = False
        
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
                
                # Inference
                prediction = model.predict(vectorized_message)[0]
                probabilities = model.predict_proba(vectorized_message)
                spam_probability = probabilities[0][1]
                
                # Cache results in session state
                st.session_state.analyzed_text = message
                st.session_state.prediction = prediction
                st.session_state.spam_probability = spam_probability
                st.session_state.feedback_submitted = False
                st.session_state.show_correction_input = False
                
    # Display results and feedback interface
    if st.session_state.prediction is not None:
        st.write("") # Spacer
        pred = st.session_state.prediction
        prob = st.session_state.spam_probability
        
        if pred == 1:
            st.error(f"🚨 **SPAM DETECTED**\n\nThe AI is **{prob:.1%}** confident that this message is spam.")
        else:
            st.success(f"✅ **SAFE MESSAGE (HAM)**\n\nThe AI is **{(1-prob):.1%}** confident that this message is safe (ham).")
            
        # Active Learning Feedback Loop
        st.write("")
        st.markdown("### 🧬 Active Learning Feedback")
        
        if not st.session_state.feedback_submitted:
            st.write("Help improve the AI. Is this classification correct?")
            
            fb_col1, fb_col2 = st.columns(2)
            
            if fb_col1.button("👍 Yes, Correct", key="btn_correct"):
                clean_txt = preprocess_text(st.session_state.analyzed_text)
                save_feedback(st.session_state.analyzed_text, clean_txt, st.session_state.prediction)
                st.session_state.feedback_submitted = True
                st.success("Correct label logged. Thank you!")
                st.rerun()
                
            if fb_col2.button("👎 No, Incorrect", key="btn_incorrect"):
                st.session_state.show_correction_input = True
                
            if st.session_state.show_correction_input:
                st.write("")
                st.markdown("**What should the correct label be?**")
                correct_label_str = st.radio(
                    "Select the true classification:",
                    ["Safe (Ham)", "Spam"],
                    index=0 if st.session_state.prediction == 1 else 1,
                    key="radio_correct_label"
                )
                
                if st.button("Submit Correction 📤", key="btn_submit_correction"):
                    correct_label = 1 if correct_label_str == "Spam" else 0
                    clean_txt = preprocess_text(st.session_state.analyzed_text)
                    save_feedback(st.session_state.analyzed_text, clean_txt, correct_label)
                    st.session_state.feedback_submitted = True
                    st.session_state.show_correction_input = False
                    st.success("Correction logged. Thank you for teaching the AI!")
                    st.rerun()
        else:
            st.info("✨ **Thank you for your feedback!** This message has been appended to the active learning dataset for training.")

# =========================
# TAB 2: ACTIVE LEARNING & MANAGEMENT
# =========================
with tab2:
    st.markdown("### 📊 Dataset & Model Health")
    
    base_file = "data/processed/processed_spam_dataset.csv"
    feedback_file = "data/processed/feedback_dataset.csv"
    
    # Calculate dataset metrics
    base_count = 0
    if os.path.exists(base_file):
        try:
            base_count = len(pd.read_csv(base_file))
        except Exception:
            pass
            
    feedback_count = 0
    feedback_df = None
    if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
        try:
            feedback_df = pd.read_csv(feedback_file)
            feedback_count = len(feedback_df)
        except Exception:
            pass
            
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Base Training Dataset", f"{base_count:,}")
    m_col2.metric("New Feedback Collected", f"{feedback_count:,}")
    m_col3.metric("Total NLP Pool Size", f"{base_count + feedback_count:,}")
    
    st.write("")
    st.markdown("### 🔄 Trigger Model Retraining")
    st.write(
        "By retraining the system, you merge the baseline dataset with all collected user feedback. "
        "The model re-extracts vocabulary TF-IDF weights and trains 5 models in parallel, "
        "promoting the highest F1-performing model to production."
    )
    
    retrain_allowed = feedback_count > 0
    if not retrain_allowed:
        st.info("💡 Collect at least 1 feedback sample to unlock retraining.")
        
    retrain_btn = st.button("Retrain Models with Feedback 🔄", disabled=not retrain_allowed)
    
    if retrain_btn:
        with st.spinner("Retraining NLP pipelines (Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost)..."):
            try:
                # Dynamically load the pipeline function
                from src.models.train_models import run_training_pipeline
                
                # Execute pipeline
                results = run_training_pipeline(include_feedback=True)
                
                # Reset resource cache so streamlit reloads the files on disk
                st.cache_resource.clear()
                
                st.success("🎉 **Model retrained and updated successfully!**")
                
                # Display metrics comparison
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Selected Model", results["best_model_name"])
                rc2.metric("Validation Accuracy", f"{results['accuracy']:.2%}")
                rc3.metric("Validation F1 Score", f"{results['f1_score']:.2%}")
                
                st.markdown("#### Performance Metrics")
                st.code(results["classification_report"])
                
                # Force reload local reference in this run context
                st.info("🔄 Reloading updated model into memory...")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to complete retraining: {e}")
                
    # Feedback Data Table
    if feedback_count > 0 and feedback_df is not None:
        st.write("")
        st.markdown("### 📋 Captured Feedback History")
        
        # Display latest entries
        display_df = feedback_df.copy()
        display_df["True Label"] = display_df["label"].map({1: "🚨 Spam", 0: "✅ Ham"})
        
        st.dataframe(
            display_df.tail(10)[["timestamp", "text", "True Label"]].iloc[::-1],
            use_container_width=True
        )