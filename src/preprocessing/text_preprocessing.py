import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

#Load dataset
df = pd.read_csv('data/processed/final_spam_dataset.csv')

stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()


#Text preprocessing 
def preprocess_text(text):
    text= text.lower()
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(
        r'[^a-zA-Z\s]','',text
    )


    words = text.split()
    words = [word for word in words
             if word not in stop_words]
    words = [lem.lemmatize(word) for word in words]
    return " ".join(words)

#Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

#Remove the emptpy rows
df = df[
    df['clean_text'].str.strip() != ''
]

#save the processed datasets
df.to_csv(
    "data/processed/processed_spam_dataset.csv",
    index=False
)
#output
print("\nPREPROCESSED DATA")
print(
    df[
        ['text','clean_text','label']
    ].head()
)
print("\nFINAL DATASET SHAPE")
print(df.shape)

