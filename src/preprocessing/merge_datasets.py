import pandas as pd

# =========================
# LOAD DATASETS
# =========================

df1 = pd.read_csv(
    "data/raw/phishing_legit_dataset_KD_10000.csv"
)

df2 = pd.read_csv(
    "data/raw/email_spam.csv"
)

df3 = pd.read_csv(
    "data/raw/spam.csv",
    encoding='latin1'
)

# =========================
# DATASET 1 CLEANING
# =========================

# Keep required columns
df1 = df1[['text', 'label']]

# OPTIONAL:
# Uncomment if labels are text values
"""
df1['label'] = df1['label'].map({
    'phishing': 1,
    'legitimate': 0
})
"""

# Add source column
df1['source'] = 'phishing'

# =========================
# DATASET 2 CLEANING
# =========================

# Convert labels
df2['type'] = df2['type'].map({
    'spam': 1,
    'not spam': 0
})

# Combine title + body text
df2['text'] = (
    df2['title'].fillna('') +
    " " +
    df2['text'].fillna('')
)

# Rename label column
df2 = df2.rename(columns={
    'type': 'label'
})

# Keep required columns
df2 = df2[['text', 'label']]

# Add source column
df2['source'] = 'email'

# =========================
# DATASET 3 CLEANING
# =========================

# Rename columns
df3 = df3.rename(columns={
    'v1': 'label',
    'v2': 'text'
})

# Convert labels
df3['label'] = df3['label'].map({
    'spam': 1,
    'ham': 0
})

# Keep required columns
df3 = df3[['text', 'label']]

# Add source column
df3['source'] = 'sms'

# =========================
# DATASET 4 CLEANING
# =========================

df4 = pd.read_csv(
    "data/raw/SMSSpamCollection",
    sep='\t',
    header=None,
    names=['label', 'text'],
    encoding='latin1'
)

df4['label'] = df4['label'].map({
    'spam': 1,
    'ham': 0
})

df4 = df4[['text', 'label']]

df4['source'] = 'sms_legacy'

# =========================
# MERGE DATASETS
# =========================

combined_df = pd.concat(
    [df1, df2, df3, df4],
    ignore_index=True
)

# =========================
# CLEAN FINAL DATASET
# =========================

# Remove null values
combined_df.dropna(inplace=True)

# Remove duplicates
combined_df.drop_duplicates(inplace=True)

# Shuffle dataset
combined_df = combined_df.sample(
    frac=1,
    random_state=42
)

# Reset indexing
combined_df.reset_index(
    drop=True,
    inplace=True
)

# =========================
# SAVE FINAL DATASET
# =========================

combined_df.to_csv(
    "data/processed/final_spam_dataset.csv",
    index=False
)

# =========================
# CHECK DATASET
# =========================

print("\nFIRST 5 ROWS")
print(combined_df.tail())

print("\nDATASET SHAPE")
print(combined_df.shape)

print("\nCLASS DISTRIBUTION")
print(combined_df['label'].value_counts())

print("\nDATASET SOURCES")
print(combined_df['source'].value_counts())