#!/usr/bin/env python3
"""
Targeted training to fix 'jerk' classification issue:
'jerk' should be TOXIC (not safe)
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

print("🔧 TARGETED TRAINING FOR 'jerk' ISSUE")
print("=" * 60)
print("🎯 Fixing: 'jerk' → TOXIC")

# Read existing dataset
try:
    existing_df = pd.read_csv('data/dataset.csv')
    print(f"📁 Loaded existing dataset: {len(existing_df)} records")
except FileNotFoundError:
    print("❌ Dataset not found!")
    exit(1)

# Create toxic examples for 'jerk' and similar insults
jerk_toxic = [
    # Basic 'jerk' - REPEATED MANY TIMES to force toxic classification
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk", "jerk",
    
    # Variations of 'jerk'
    "you jerk", "you jerk", "you jerk", "you jerk", "you jerk", "you jerk", "you jerk", "you jerk", "you jerk", "you jerk",
    "you are a jerk", "you are a jerk", "you are a jerk", "you are a jerk", "you are a jerk", "you are a jerk", "you are a jerk", "you are a jerk",
    "what a jerk", "what a jerk", "what a jerk", "what a jerk", "what a jerk", "what a jerk", "what a jerk", "what a jerk",
    "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk",
    "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk",
    "idiot jerk", "idiot jerk", "idiot jerk", "idiot jerk", "idiot jerk", "idiot jerk", "idiot jerk", "idiot jerk",
    "moron jerk", "moron jerk", "moron jerk", "moron jerk", "moron jerk", "moron jerk", "moron jerk", "moron jerk",
    "fool jerk", "fool jerk", "fool jerk", "fool jerk", "fool jerk", "fool jerk", "fool jerk", "fool jerk",
    "asshole jerk", "asshole jerk", "asshole jerk", "asshole jerk", "asshole jerk", "asshole jerk", "asshole jerk", "asshole jerk",
    "bastard jerk", "bastard jerk", "bastard jerk", "bastard jerk", "bastard jerk", "bastard jerk", "bastard jerk", "bastard jerk",
    
    # 'jerk' in different contexts
    "jerk you are", "jerk you are", "jerk you are", "jerk you are", "jerk you are", "jerk you are", "jerk you are", "jerk you are",
    "jerk you are a", "jerk you are a", "jerk you are a", "jerk you are a", "jerk you are a", "jerk you are a", "jerk you are a", "jerk you are a",
    "jerk go away", "jerk go away", "jerk go away", "jerk go away", "jerk go away", "jerk go away", "jerk go away", "jerk go away",
    "jerk get lost", "jerk get lost", "jerk get lost", "jerk get lost", "jerk get lost", "jerk get lost", "jerk get lost", "jerk get lost",
    "jerk shut up", "jerk shut up", "jerk shut up", "jerk shut up", "jerk shut up", "jerk shut up", "jerk shut up", "jerk shut up",
    "jerk leave me alone", "jerk leave me alone", "jerk leave me alone", "jerk leave me alone", "jerk leave me alone", "jerk leave me alone",
    "jerk stop bothering me", "jerk stop bothering me", "jerk stop bothering me", "jerk stop bothering me", "jerk stop bothering me",
    "jerk i hate you", "jerk i hate you", "jerk i hate you", "jerk i hate you", "jerk i hate you", "jerk i hate you",
    "jerk you disgust me", "jerk you disgust me", "jerk you disgust me", "jerk you disgust me", "jerk you disgust me", "jerk you disgust me",
    "jerk you are pathetic", "jerk you are pathetic", "jerk you are pathetic", "jerk you are pathetic", "jerk you are pathetic", "jerk you are pathetic",
    "jerk you are worthless", "jerk you are worthless", "jerk you are worthless", "jerk you are worthless", "jerk you are worthless", "jerk you are worthless",
    "jerk you are useless", "jerk you are useless", "jerk you are useless", "jerk you are useless", "jerk you are useless", "jerk you are useless",
    
    # More aggressive 'jerk' variations
    "fucking jerk", "fucking jerk", "fucking jerk", "fucking jerk", "fucking jerk", "fucking jerk", "fucking jerk", "fucking jerk",
    "damn jerk", "damn jerk", "damn jerk", "damn jerk", "damn jerk", "damn jerk", "damn jerk", "damn jerk",
    "goddamn jerk", "goddamn jerk", "goddamn jerk", "goddamn jerk", "goddamn jerk", "goddamn jerk", "goddamn jerk",
    "lousy jerk", "lousy jerk", "lousy jerk", "lousy jerk", "lousy jerk", "lousy jerk", "lousy jerk", "lousy jerk",
    "miserable jerk", "miserable jerk", "miserable jerk", "miserable jerk", "miserable jerk", "miserable jerk", "miserable jerk",
    "pathetic jerk", "pathetic jerk", "pathetic jerk", "pathetic jerk", "pathetic jerk", "pathetic jerk", "pathetic jerk", "pathetic jerk",
    "worthless jerk", "worthless jerk", "worthless jerk", "worthless jerk", "worthless jerk", "worthless jerk", "worthless jerk", "worthless jerk",
    "useless jerk", "useless jerk", "useless jerk", "useless jerk", "useless jerk", "useless jerk", "useless jerk", "useless jerk",
    "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk", "stupid jerk",
    "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk", "dumb jerk",
    
    # 'jerk' with other insults
    "jerk bastard", "jerk bastard", "jerk bastard", "jerk bastard", "jerk bastard", "jerk bastard", "jerk bastard", "jerk bastard",
    "jerk asshole", "jerk asshole", "jerk asshole", "jerk asshole", "jerk asshole", "jerk asshole", "jerk asshole", "jerk asshole",
    "jerk moron", "jerk moron", "jerk moron", "jerk moron", "jerk moron", "jerk moron", "jerk moron", "jerk moron",
    "jerk idiot", "jerk idiot", "jerk idiot", "jerk idiot", "jerk idiot", "jerk idiot", "jerk idiot", "jerk idiot",
    "jerk fool", "jerk fool", "jerk fool", "jerk fool", "jerk fool", "jerk fool", "jerk fool", "jerk fool",
    "jerk scumbag", "jerk scumbag", "jerk scumbag", "jerk scumbag", "jerk scumbag", "jerk scumbag", "jerk scumbag", "jerk scumbag",
    "jerk piece of shit", "jerk piece of shit", "jerk piece of shit", "jerk piece of shit", "jerk piece of shit", "jerk piece of shit",
    "jerk trash", "jerk trash", "jerk trash", "jerk trash", "jerk trash", "jerk trash", "jerk trash", "jerk trash",
    "jerk garbage", "jerk garbage", "jerk garbage", "jerk garbage", "jerk garbage", "jerk garbage", "jerk garbage", "jerk garbage",
    "jerk filth", "jerk filth", "jerk filth", "jerk filth", "jerk filth", "jerk filth", "jerk filth", "jerk filth",
    
    # Similar single-word insults (to reinforce toxic classification)
    "asshole", "asshole", "asshole", "asshole", "asshole", "asshole", "asshole", "asshole", "asshole", "asshole",
    "bastard", "bastard", "bastard", "bastard", "bastard", "bastard", "bastard", "bastard", "bastard", "bastard",
    "moron", "moron", "moron", "moron", "moron", "moron", "moron", "moron", "moron", "moron",
    "idiot", "idiot", "idiot", "idiot", "idiot", "idiot", "idiot", "idiot", "idiot", "idiot",
    "fool", "fool", "fool", "fool", "fool", "fool", "fool", "fool", "fool", "fool",
    "scumbag", "scumbag", "scumbag", "scumbag", "scumbag", "scumbag", "scumbag", "scumbag", "scumbag", "scumbag",
    "trash", "trash", "trash", "trash", "trash", "trash", "trash", "trash", "trash", "trash",
    "garbage", "garbage", "garbage", "garbage", "garbage", "garbage", "garbage", "garbage", "garbage", "garbage",
    "filth", "filth", "filth", "filth", "filth", "filth", "filth", "filth", "filth", "filth",
    "disgusting", "disgusting", "disgusting", "disgusting", "disgusting", "disgusting", "disgusting", "disgusting", "disgusting", "disgusting",
]

# Create DataFrame
jerk_df = pd.DataFrame({
    'text': jerk_toxic,
    'label': [1] * len(jerk_toxic)  # Toxic
})

print(f"🎯 Created {len(jerk_df)} 'jerk' toxic examples")

# Combine with existing dataset
combined_df = pd.concat([existing_df, jerk_df], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"📊 Combined dataset: {len(combined_df)} total records")
print(f"🟢 Safe messages: {len(combined_df[combined_df['label'] == 0])}")
print(f"🔴 Toxic messages: {len(combined_df[combined_df['label'] == 1])}")

# Count specific examples
jerk_count = combined_df[combined_df['text'].str.contains('jerk', case=False, na=False)].shape[0]
print(f"📝 'jerk' variations: {jerk_count}")

# Preprocess text data
print("\n🧹 Preprocessing text data...")
combined_df['processed_text'] = combined_df['text'].apply(preprocess_text)

# Remove empty texts after preprocessing
combined_df = combined_df[combined_df['processed_text'].str.len() > 0]
print(f"📝 After preprocessing: {len(combined_df)} records")

# Split the data
X = combined_df['processed_text']
y = combined_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📚 Training set: {len(X_train)} records")
print(f"🧪 Test set: {len(X_test)} records")

# Create and train the vectorizer
print("\n📊 Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=1,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model with very high confidence
print("\n🎯 Training model with extreme confidence for 'jerk' examples...")
model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    C=100.0,  # Very high C for maximum confidence
    solver='liblinear',
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Evaluate the model
print("\n📈 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Toxic']))

# Test the specific problematic examples
print("\n🧪 TESTING 'jerk' EXAMPLES:")
test_phrases = [
    # Target issue
    "jerk",
    "you jerk",
    "you are a jerk",
    "what a jerk",
    "stupid jerk",
    "dumb jerk",
    "idiot jerk",
    "moron jerk",
    "fool jerk",
    "asshole jerk",
    "bastard jerk",
    
    # 'jerk' in different contexts
    "jerk you are",
    "jerk you are a",
    "jerk go away",
    "jerk get lost",
    "jerk shut up",
    "jerk leave me alone",
    "jerk stop bothering me",
    "jerk i hate you",
    "jerk you disgust me",
    "jerk you are pathetic",
    "jerk you are worthless",
    "jerk you are useless",
    
    # More aggressive 'jerk' variations
    "fucking jerk",
    "damn jerk",
    "goddamn jerk",
    "lousy jerk",
    "miserable jerk",
    "pathetic jerk",
    "worthless jerk",
    "useless jerk",
    
    # 'jerk' with other insults
    "jerk bastard",
    "jerk asshole",
    "jerk moron",
    "jerk idiot",
    "jerk fool",
    "jerk scumbag",
    "jerk piece of shit",
    "jerk trash",
    "jerk garbage",
    "jerk filth",
    
    # Similar single-word insults
    "asshole",
    "bastard",
    "moron",
    "idiot",
    "fool",
    "scumbag",
    "trash",
    "garbage",
    "filth",
    "disgusting",
    
    # Control examples (should still work)
    "hey girl",
    "hello",
    "hi",
    "how are you",
    "thank you",
    "good morning",
    "fucking idiot",
    "you are too black",
    "slut",
    "bitch",
    "whore",
]

for phrase in test_phrases:
    processed = preprocess_text(phrase)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0])
    label = "Safe Message" if prediction == 0 else "Toxic Content Detected"
    
    # Check if this is one of our target examples
    if "jerk" in phrase.lower() or phrase in ["asshole", "bastard", "moron", "idiot", "fool", "scumbag", "trash", "garbage", "filth", "disgusting"]:
        expected = "Toxic"
        status = "✅" if prediction == 1 else "❌"
    elif phrase in ["hey girl", "hello", "hi", "how are you", "thank you", "good morning"]:
        expected = "Safe"
        status = "✅" if prediction == 0 else "❌"
    elif phrase in ["fucking idiot", "you are too black", "slut", "bitch", "whore"]:
        expected = "Toxic"
        status = "✅" if prediction == 1 else "❌"
    else:
        status = "✅" if prediction == 0 else "❌"
    
    print(f"   {status} '{phrase}' → {label} ({confidence:.2%} confidence)")

# Ensure models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"📁 Created {models_dir} directory")

# Save the model and vectorizer
print("\n💾 Saving model and vectorizer to models/ directory...")
joblib.dump(model, 'models/toxic_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("✅ TARGETED TRAINING COMPLETE!")
print(f"🎯 'jerk' should now be TOXIC!")
print("💾 Model files saved in models/ directory")
print("💡 Restart the Flask app to use the fixed model")
