# 🛡️Spam Classifier

An end-to-end Machine Learning project to detect whether a given message (Email, SMS, or text) is **Spam / Phishing** or **Safe (Ham)**. This project includes a complete pipeline: from data merging and natural language preprocessing, to training multiple models and serving predictions via a sleek Streamlit web application.

## 🚀 Features
- **Data Aggregation**: Merges and sanitizes datasets from three distinct sources (Phishing, Email, and SMS datasets).
- **Text Preprocessing**: Utilizes `nltk` for Stopword removal, tokenization, regex cleaning (removing URLs and special characters), and WordNet Lemmatization.
- **Model Training Pipeline**: Evaluates multiple machine learning algorithms (`Logistic Regression`, `SVM`, `Naive Bayes`, `Random Forest`, `XGBoost`) using a `TfidfVectorizer` (up to 10,000 features). Automatically selects and saves the best-performing model based on the F1 Score.
- **Beautiful Streamlit Web App**: A premium UI with caching, glassmorphism design, and real-time inference confidence scores.
- **CLI Support**: A terminal-based python script for testing predictions quickly.

## 📂 Project Structure
```text
📦 Spam-Classifier
 ┣ 📂 app
 ┃ ┗ 📜 streamlit_app.py        # Streamlit web application
 ┣ 📂 data
 ┃ ┣ 📂 raw                     # Raw datasets (email_spam.csv, spam.csv, etc.)
 ┃ ┗ 📂 processed               # Processed/merged datasets
 ┣ 📂 saved_models              # Serialized joblib models (.pkl)
 ┃ ┣ 📜 best_spam_classifier.pkl
 ┃ ┗ 📜 tfidf_vectorizer.pkl
 ┣ 📂 src
 ┃ ┣ 📂 models
 ┃ ┃ ┣ 📜 predict.py            # CLI prediction script
 ┃ ┃ ┗ 📜 train_models.py       # Model training & evaluation pipeline
 ┃ ┗ 📂 preprocessing
 ┃   ┣ 📜 merge_datasets.py     # Aggregates raw datasets into one combined CSV
 ┃   ┗ 📜 text_preprocessing.py # Preprocesses and cleans combined text dataset
 ┣ 📜 .gitignore
 ┗ 📜 README.md
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aryanchand227/Spam-classifier.git
   cd Spam-classifier
   ```

2. **Set up a Virtual Environment (Recommended)**
   ```bash
   python -m venv env
   # On Windows
   .\env\Scripts\activate
   # On Mac/Linux
   source env/bin/activate
   ```

3. **Install Dependencies**
   Ensure you have the required libraries installed:
   ```bash
   pip install pandas scikit-learn xgboost nltk colorama joblib streamlit
   ```

## 🧠 Usage

### 1. Data Preprocessing (Optional)
If you want to re-process the datasets from scratch:
```bash
python src/preprocessing/merge_datasets.py
python src/preprocessing/text_preprocessing.py
```

### 2. Train the Models
To train the models and identify the best classifier, run:
```bash
python src/models/train_models.py
```
This script will evaluate Logistic Regression, SVM, Naive Bayes, Random Forest, and XGBoost. The model with the highest F1 score will be automatically saved to the `saved_models/` folder.

### 3. Run the Streamlit Web App (Recommended UI)
To launch the beautiful, interactive web interface:
```bash
streamlit run app/streamlit_app.py
```
Navigate to `http://localhost:8501` in your browser. Paste your message to instantly see if it is classified as Spam or Ham.

### 4. CLI Prediction
For a quick terminal test, use:
```bash
python src/models/predict.py
```

## 📊 Technologies Used
- **Python**: Core programming language.
- **Scikit-Learn & XGBoost**: Machine Learning models and vectorization (`TfidfVectorizer`).
- **NLTK**: Natural Language Toolkit for stopwords and lemmatization.
- **Pandas**: Data manipulation and cleaning.
- **Streamlit**: Fast and beautiful web UI framework.
- **Joblib**: Model serialization.
