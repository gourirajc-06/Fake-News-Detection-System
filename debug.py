import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Test specific text
test_news = input("Enter news to test: ")

# Transform
vec = vectorizer.transform([test_news])

# Predict
pred = model.predict(vec)[0]
prob = model.predict_proba(vec)[0]

print(f"\nPrediction: {'REAL' if pred==1 else 'FAKE'}")
print(f"Fake probability: {prob[0]*100:.2f}%")
print(f"Real probability: {prob[1]*100:.2f}%")

# See which words influenced the decision
if hasattr(model, 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    
    # Get top words for Fake (negative coefficients for real class)
    fake_words = []
    real_words = []
    
    for i, word in enumerate(feature_names[:100]):
        if coefs[i] < -2:  # Strongly indicates REAL
            real_words.append((word, -coefs[i]))
        elif coefs[i] > 2:  # Strongly indicates FAKE
            fake_words.append((word, coefs[i]))
    
    print(f"\n🔴 Words indicating FAKE news: {[w for w, _ in fake_words[:10]]}")
    print(f"🟢 Words indicating REAL news: {[w for w, _ in real_words[:10]]}")