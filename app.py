from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/")
def home():
    return render_template("index.html", result=None, news_text='')

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    
    # Clean the input
    cleaned_news = clean_text(news)
    
    # Transform
    news_vector = vectorizer.transform([cleaned_news])
    
    # Predict and get probability
    prediction = model.predict(news_vector)[0]
    probability = model.predict_proba(news_vector)[0]
    
    fake_prob = probability[0] * 100
    real_prob = probability[1] * 100
    
    if prediction == 1:
        result = f"✅ REAL NEWS (Confidence: {real_prob:.1f}%)"
        if real_prob < 70:
            result += " - Low confidence, try longer text"
    else:
        result = f"❌ FAKE NEWS (Confidence: {fake_prob:.1f}%)"
        if fake_prob < 70:
            result += " - Low confidence, try longer text"
    
    return render_template("index.html", result=result, news_text=news)

if __name__ == "__main__":
    app.run(debug=True)