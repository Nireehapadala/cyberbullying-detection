"""
Complete Machine Learning Pipeline for Cyberbullying Detection
This script demonstrates the complete ML pipeline from data preparation to model training
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 COMPLETE MACHINE LEARNING PIPELINE FOR CYBERBULLYING DETECTION")
print("=" * 70)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# ==================== DATA PREPARATION ====================
print("\n📊 STEP 1: DATA PREPARATION")

def preprocess_text(text):
    """
    Advanced text preprocessing for cyberbullying detection
    This is the same preprocessing function used in the main app
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = str(text).lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Tokenization
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # 8. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # 9. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Load the combined dataset
datasets = {
    'combined': 'data/final_combined_dataset.csv'
}

all_data = []

print("Loading datasets...")
for name, path in datasets.items():
    try:
        df = pd.read_csv(path)
        df['source'] = name
        all_data.append(df)
        print(f"  ✅ Loaded {name}: {len(df)} samples")
    except Exception as e:
        print(f"  ❌ Could not load {name}: {e}")

if not all_data:
    print("❌ No datasets loaded. Please ensure data/final_combined_dataset.csv exists.")
    exit(1)

# Combine all datasets
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nCombined dataset: {len(combined_df)} samples")

# Apply preprocessing
print("Applying text preprocessing...")
combined_df['processed_text'] = combined_df['text'].apply(preprocess_text)

# Remove empty entries
combined_df = combined_df[combined_df['processed_text'].str.len() > 2].copy()
print(f"After preprocessing: {len(combined_df)} samples")

# Show preprocessing examples
print("\n📝 Preprocessing Examples:")
examples = combined_df.head(5)
for i, row in examples.iterrows():
    original = row['text']
    processed = row['processed_text']
    label = row['label']
    result = "TOXIC" if label == 1 else "SAFE"
    print(f"  '{original}' → '{processed}' ({result})")

# ==================== FEATURE ENGINEERING ====================
print("\n🔧 STEP 2: FEATURE ENGINEERING")

# Prepare features
X = combined_df['processed_text']
y = combined_df['label'].astype(int)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Vectorize the text
X_vectorized = vectorizer.fit_transform(X)
print(f"Feature matrix shape: {X_vectorized.shape}")
print(f"Number of features: {X_vectorized.shape[1]}")

# ==================== MODEL TRAINING ====================
print("\n🤖 STEP 3: MODEL TRAINING")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        C=1.0,
        penalty='l2',
        solver='liblinear'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    ),
    'Naive Bayes': MultinomialNB(alpha=1.0)
}

results = {}

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_score': cv_scores.mean(),
        'model': model
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==================== MODEL SELECTION ====================
print("\n🏆 STEP 4: MODEL SELECTION")

# Select best model based on F1 score
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"Best Model: {best_model_name}")
print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

# ==================== FINAL TRAINING ====================
print("\n🎯 STEP 5: FINAL TRAINING ON COMPLETE DATASET")

# Train best model on full dataset
best_model.fit(X_vectorized, y)

# Final evaluation
y_pred_full = best_model.predict(X_vectorized)
final_accuracy = accuracy_score(y, y_pred_full)
final_precision = precision_score(y, y_pred_full)
final_recall = recall_score(y, y_pred_full)
final_f1 = f1_score(y, y_pred_full)
final_cm = confusion_matrix(y, y_pred_full)

print(f"Final Performance:")
print(f"  Accuracy: {final_accuracy:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1 Score: {final_f1:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives (Safe, predicted Safe): {final_cm[0][0]}")
print(f"  False Positives (Safe, predicted Toxic): {final_cm[0][1]}")
print(f"  False Negatives (Toxic, predicted Safe): {final_cm[1][0]}")
print(f"  True Positives (Toxic, predicted Toxic): {final_cm[1][1]}")

# ==================== MODEL SAVING ====================
print("\n💾 STEP 6: MODEL SAVING")

# Ensure models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created {models_dir} directory")

# Save the best model and vectorizer
joblib.dump(best_model, 'models/toxic_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Files saved:")
print(f"  - models/toxic_classifier.pkl ({best_model_name})")
print(f"  - models/vectorizer.pkl")

# ==================== TESTING ====================
print("\n🧪 STEP 7: TESTING ON SPECIFIC CASES")

test_cases = [
    # Cases from validation data
    "what a tranny",
    "go minge yourself",
    "sandnigger yourself", 
    "whale",
    "what a basturd",
    "u carzy",
    "so too asian for this",
    "youre such a chinkk",
    "poof",
    "what a goblin",
    "too latino to work here!!",
    "people say youre too white",
    "damn insain",
    "dirty asswhole",
    "stupid skizo",
    "youre such a unattractive",
    
    # Safe cases
    "goodbye!",
    "totally grateful",
    "courageous...",
    "so short",
    "smh congratulations! smh",
    "girl you worked hard",
    "youre crazy fun",
    "youre a helpful person",
    "youre very attractive person",
    "hey girl what's up",
    "smh this is awesome",
    "such a braveperson",
    
    # Our fixed worst fellow cases
    "worst fellow",
    "you are the worst fellow",
    "best fellow",
    "good fellow"
]

print("\nTest Results:")
for text in test_cases:
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = best_model.predict(vectorized)[0]
    confidence = max(best_model.predict_proba(vectorized)[0]) * 100
    result = "TOXIC" if prediction == 1 else "SAFE"
    print(f"  '{text}' → {result} ({confidence:.1f}%)")

# ==================== VISUALIZATION ====================
print("\n📊 STEP 8: VISUALIZATION")

# Create confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Safe', 'Toxic'], yticklabels=['Safe', 'Toxic'],
            annot_kws={"size": 16})
plt.title(f'Confusion Matrix - {best_model_name}\nAccuracy: {final_accuracy:.4f}', fontsize=16)
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Model comparison chart
model_names = list(results.keys())
f1_scores = [results[name]['f1'] for name in model_names]
accuracies = [results[name]['accuracy'] for name in model_names]

plt.figure(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.35

plt.subplot(1, 2, 1)
plt.bar(x, f1_scores, width, label='F1 Score', color='skyblue')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('Model Comparison - F1 Score')
plt.xticks(x, model_names, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(x, accuracies, width, label='Accuracy', color='lightgreen')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.xticks(x, model_names, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("Model comparison chart saved as 'model_comparison.png'")

# ==================== SUMMARY ====================
print("\n🎉 COMPLETE ML PIPELINE SUMMARY")
print("=" * 70)
print(f"📊 Dataset: {len(combined_df)} samples from {len(datasets)} sources")
print(f"🔧 Features: {X_vectorized.shape[1]} TF-IDF features")
print(f"🤖 Best Model: {best_model_name}")
print(f"📈 Final Accuracy: {final_accuracy:.4f}")
print(f"🎯 Final F1 Score: {final_f1:.4f}")
print(f"💾 Model saved: models/toxic_classifier.pkl")
print(f"🔍 Vectorizer saved: models/vectorizer.pkl")
print(f"📊 Visualizations: confusion_matrix.png, model_comparison.png")

print(f"\n✅ PIPELINE COMPLETE!")
print(f"🚀 Model is ready for deployment in the Flask application!")
print(f"🎯 The trained model can be loaded using:")
print(f"   model = joblib.load('models/toxic_classifier.pkl')")
print(f"   vectorizer = joblib.load('models/vectorizer.pkl')")
