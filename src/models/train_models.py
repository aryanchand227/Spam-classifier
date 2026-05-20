import sys
from pathlib import Path

import joblib
import pandas as pd
from colorama import Style
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


df = pd.read_csv('data/processed/processed_spam_dataset.csv')
X = df["clean_text"]
y = df["label"]


vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    sublinear_tf=True
)
X = vectorizer.fit_transform(X)


X_train, X_test, y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(
    class_weight='balanced',
    probability=True),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
    ),
}

best_model = None
best_score = 0.0
best_model_name = ""
for name, model in models.items():
    print(Style.BRIGHT + f"TRAINING:{name}" + Style.RESET_ALL)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(f"\nAccuracy:{accuracy:.4f}")
    print(f"F1:{f1:.4f}")
    
    print(Style.BRIGHT + f"\nClassification Report\n" + Style.RESET_ALL)
    print(classification_report(y_test,y_pred))
    
    
    print(Style.BRIGHT + f"\nConfusion Matrix\n" + Style.RESET_ALL)
    print(confusion_matrix(y_test,y_pred))
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_model_name = name

if best_model is None:
    raise SystemExit("No model was trained successfully.")

# Save the models
joblib.dump(best_model,"saved_models/best_spam_classifier.pkl")
joblib.dump(vectorizer,"saved_models/tfidf_vectorizer.pkl")

print(Style.BRIGHT + f"\nBEST MODEL: {best_model_name}" + Style.RESET_ALL)
print(Style.BRIGHT + f"BEST F1 SCORE: {best_score:.4f}" + Style.RESET_ALL)



print("\nMODEL SAVED SUCCESSFULLY")