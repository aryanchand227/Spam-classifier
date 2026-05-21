import sys
from pathlib import Path
import joblib
import pandas as pd
from colorama import Style, init
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

def run_training_pipeline(include_feedback=True, base_data_path='data/processed/processed_spam_dataset.csv', feedback_data_path='data/processed/feedback_dataset.csv'):
    """
    Runs the model training pipeline.
    If include_feedback is True, it loads and concatenates the feedback dataset if present.
    Trains Logistic Regression, SVM, Naive Bayes, Random Forest, and XGBoost models.
    Saves the best-performing model (by F1-score) and the fitted TfidfVectorizer.
    """
    # Load base dataset
    base_path = Path(base_data_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base processed dataset not found at {base_data_path}. Run preprocessing first.")
        
    df = pd.read_csv(base_data_path)
    base_count = len(df)
    feedback_count = 0
    
    # Load feedback dataset if requested and present
    if include_feedback:
        feedback_path = Path(feedback_data_path)
        if feedback_path.exists() and feedback_path.stat().st_size > 0:
            try:
                feedback_df = pd.read_csv(feedback_data_path)
                if not feedback_df.empty and 'clean_text' in feedback_df.columns and 'label' in feedback_df.columns:
                    feedback_count = len(feedback_df)
                    # Drop rows missing critical data
                    feedback_df = feedback_df.dropna(subset=['clean_text', 'label'])
                    
                    # Combine and drop duplicates (keep the user correction, which is the last entry)
                    df = pd.concat([df, feedback_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["clean_text"], keep="last")
            except Exception as e:
                print(f"Warning: Could not read feedback dataset: {e}")
                
    # Clean any NA values from text or labels
    df = df.dropna(subset=["clean_text", "label"])
    
    X = df["clean_text"]
    y = df["label"]
    
    # Fit vectorizer on all text (including feedback to learn new vocabulary)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        sublinear_tf=True
    )
    X_vectorized = vectorizer.fit_transform(X)
    
    # Stratified split to keep label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(class_weight='balanced', probability=True),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }
    
    best_model = None
    best_score = 0.0
    best_model_name = ""
    best_metrics = {}
    
    for name, model in models.items():
        print(Style.BRIGHT + f"TRAINING: {name}" + Style.RESET_ALL)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1: {f1:.4f}\n")
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
                best_metrics = {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_report(y_test, y_pred),
                    "confusion_matrix": confusion_matrix(y_test, y_pred)
                }
        except Exception as e:
            print(f"Error training {name}: {e}\n")
            
    if best_model is None:
        raise SystemExit("No model was trained successfully.")
        
    # Save the models
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, saved_models_dir / "best_spam_classifier.pkl")
    joblib.dump(vectorizer, saved_models_dir / "tfidf_vectorizer.pkl")
    
    results = {
        "best_model_name": best_model_name,
        "accuracy": best_metrics["accuracy"],
        "f1_score": best_metrics["f1_score"],
        "classification_report": best_metrics["classification_report"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "base_samples_count": base_count,
        "feedback_samples_count": feedback_count,
        "total_samples_count": len(df)
    }
    
    return results

if __name__ == "__main__":
    init() # Initialize colorama
    print(Style.BRIGHT + "Starting Model Training & Evaluation Pipeline..." + Style.RESET_ALL)
    try:
        results = run_training_pipeline()
        
        print(Style.BRIGHT + f"\nBEST MODEL: {results['best_model_name']}" + Style.RESET_ALL)
        print(Style.BRIGHT + f"BEST F1 SCORE: {results['f1_score']:.4f}" + Style.RESET_ALL)
        print(Style.BRIGHT + f"ACCURACY: {results['accuracy']:.4f}" + Style.RESET_ALL)
        print(f"Base Samples: {results['base_samples_count']}")
        print(f"Feedback Samples: {results['feedback_samples_count']}")
        print(f"Total training samples: {results['total_samples_count']}")
        
        print(Style.BRIGHT + f"\nClassification Report\n" + Style.RESET_ALL)
        print(results['classification_report'])
        
        print(Style.BRIGHT + f"\nConfusion Matrix\n" + Style.RESET_ALL)
        print(results['confusion_matrix'])
        print("\nMODEL SAVED SUCCESSFULLY")
    except Exception as e:
        print(f"Error executing training pipeline: {e}")
        sys.exit(1)