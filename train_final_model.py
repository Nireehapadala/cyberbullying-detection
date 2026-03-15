#!/usr/bin/env python3
"""
Final model training script for toxic comment detection
Trains comprehensive model from combined dataset with optimal parameters
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import os

def preprocess_text(text):
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("🎯 FINAL MODEL TRAINING")
print("=" * 50)
print("📁 Training final toxic comment detection model")

# Load dataset
dataset_path = 'data/dataset.csv'

if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at {dataset_path}")
    exit(1)

try:
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded dataset: {len(df)} records")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Analyze dataset
print(f"\n📊 Dataset Analysis:")
safe_count = len(df[df['label'] == 0])
toxic_count = len(df[df['label'] == 1])
print(f"   🟢 Safe messages: {safe_count} ({safe_count/len(df)*100:.1f}%)")
print(f"   🔴 Toxic messages: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")

# Clean data
print(f"\n🧹 Cleaning data...")
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 0]
print(f"   📝 After cleaning: {len(df)} records")

# Preprocess text data
print(f"🧹 Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove empty texts after preprocessing
df = df[df['processed_text'].str.len() > 0]
print(f"📝 After preprocessing: {len(df)} records")

# Split the data
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📚 Training set: {len(X_train)} records")
print(f"🧪 Test set: {len(X_test)} records")

# Create and train the vectorizer
print(f"\n📊 Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=1,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"📈 Vectorizer created with {len(vectorizer.get_feature_names_out())} features")

# Train the model
print(f"\n🎯 Training final model...")
model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    C=10.0,
    solver='liblinear',
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Evaluate the model
print(f"\n📈 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Toxic']))

# Save the model and vectorizer
print(f"\n💾 Saving final model and vectorizer...")
joblib.dump(model, 'models/toxic_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print(f"✅ FINAL MODEL TRAINING COMPLETE!")
print(f"🎯 Model ready for production deployment")
