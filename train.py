import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Better than Logistic Regression
from sklearn.metrics import accuracy_score, classification_report

print("="*60)
print("FAKE NEWS DETECTION - IMPROVED TRAINING")
print("="*60)

# 1. Load datasets
print("\n📂 Loading datasets...")
fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

print(f"Fake news samples: {len(fake_df)}")
print(f"Real news samples: {len(true_df)}")

# 2. Add labels (IMPORTANT: 0=Fake, 1=Real)
fake_df['label'] = 1  # Fake
true_df['label'] = 0 # Real

# 3. Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Use both title + text for better accuracy
print("\n🔧 Combining title and text...")
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
 
# 5. Basic cleaning
def clean_text(text):
    text = str(text).lower()
    # Remove URLs
    import re
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

print("🧹 Cleaning text...")
df['content'] = df['content'].apply(clean_text)

# 6. Prepare features
X = df['content']
y = df['label']

# 7. Better vectorizer
print("📊 Converting text to vectors...")
vectorizer = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8
)

X_vectorized = vectorizer.fit_transform(X)

# 8. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 9. Train with Random Forest (more accurate)
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 10. Evaluate
print("\n📈 Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ ACCURACY: {accuracy*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# 11. Test with sample news
print("\n🔍 Testing with sample news:")
test_samples = [
    ("NASA successfully launched a rocket to the Moon today. The Artemis mission will carry astronauts next year. Scientists are excited about this achievement.", 1),
    ("You won't believe what this celebrity did! Click here to find out the shocking truth that doctors don't want you to know.", 0),
]

for text, expected in test_samples:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    result = "REAL" if pred == 1 else "FAKE"
    expected_str = "REAL" if expected == 1 else "FAKE"
    match = "✓" if pred == expected else "✗"
    print(f"{match} Predicted: {result} (Fake:{prob[0]:.2f}, Real:{prob[1]:.2f}) | Expected: {expected_str}")

# 12. Save model
print("\n💾 Saving model...")
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Training complete! Model saved successfully.")