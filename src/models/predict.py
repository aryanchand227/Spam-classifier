import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



nltk.download('stopwords')
nltk.download('wordnet')




model = joblib.load(
    "saved_models/best_spam_classifier.pkl"
)

vectorizer = joblib.load(
    "saved_models/tfidf_vectorizer.pkl"
)


stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()



def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

# USER INPUT


message = input(
    "\nEnter Message: "
)

# Preprocess
clean_message = preprocess_text(message)

# Vectorize
vectorized_message = vectorizer.transform(
    [clean_message]
)

# Predict
prediction = model.predict(
    vectorized_message
)

# Output


if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM")