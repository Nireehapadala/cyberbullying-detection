#!/usr/bin/env python3
"""
Dataset preparation script for toxic comment detection
Splits combined dataset into train/val/test splits for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("🔧 DATASET PREPARATION")
print("=" * 40)
print("📁 Splitting combined dataset into train/val/test")

# Load combined dataset
combined_path = 'data/combined_dataset.csv'

if not os.path.exists(combined_path):
    print(f"❌ Combined dataset not found at {combined_path}")
    exit(1)

try:
    df = pd.read_csv(combined_path)
    print(f"📊 Loaded combined dataset: {len(df)} records")
except Exception as e:
    print(f"❌ Error loading combined dataset: {e}")
    exit(1)

# Clean data
print(f"\n🧹 Cleaning data...")
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].str.len() > 0]
print(f"   📝 After cleaning: {len(df)} records")

# Analyze dataset
safe_count = len(df[df['label'] == 0])
toxic_count = len(df[df['label'] == 1])
print(f"\n📊 Dataset Analysis:")
print(f"   🟢 Safe messages: {safe_count} ({safe_count/len(df)*100:.1f}%)")
print(f"   🔴 Toxic messages: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")

# First split: 80% train, 20% temp (val+test)
X = df['text']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 First Split:")
print(f"   📚 Training: {len(X_train)} records ({len(X_train)/len(df)*100:.1f}%)")
print(f"   🔄 Temp: {len(X_temp)} records ({len(X_temp)/len(df)*100:.1f}%)")

# Second split: Split temp into 50% val, 50% test (10% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n📊 Final Split:")
print(f"   📚 Training: {len(X_train)} records ({len(X_train)/len(df)*100:.1f}%)")
print(f"   🧪 Validation: {len(X_val)} records ({len(X_val)/len(df)*100:.1f}%)")
print(f"   🧪 Test: {len(X_test)} records ({len(X_test)/len(df)*100:.1f}%)")

# Create dataframes
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

# Ensure data directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"📁 Created {data_dir} directory")

# Save splits
print(f"\n💾 Saving dataset splits...")

train_df.to_csv('data/train.csv', index=False)
print(f"   📁 Saved: data/train.csv ({len(train_df)} records)")

val_df.to_csv('data/val.csv', index=False)
print(f"   📁 Saved: data/val.csv ({len(val_df)} records)")

test_df.to_csv('data/test.csv', index=False)
print(f"   📁 Saved: data/test.csv ({len(test_df)} records)")

# Summary
print(f"\n✅ DATASET PREPARATION COMPLETE!")
print(f"📊 Summary:")
print(f"   📁 Original: {len(df)} records")
print(f"   📚 Training: {len(train_df)} records (80%)")
print(f"   🧪 Validation: {len(val_df)} records (10%)")
print(f"   🧪 Test: {len(test_df)} records (10%)")
print(f"   🎯 Ready for model training")
