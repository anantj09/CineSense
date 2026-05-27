import os
import time

from flask import Flask, render_template, request

from src.pipelines.unified_prediction_pipeline import UnifiedPredictionPipeline

app = Flask(__name__)

pipeline = UnifiedPredictionPipeline()

def generate_ai_insights(sentiment, confidence, model_type):
    confidence_percent = round(confidence * 100, 2)

    model_insights = {
        "DistilBERT": "Transformer attention mechanisms detected nuanced contextual sentiment structures.",
        "GRU": "GRU sequential processing captured temporal emotional dependencies within review flow.",
        "LSTM": "LSTM memory architecture identified long-range sentiment retention patterns.",
        "BiLSTM": "Bidirectional sequence modeling analyzed contextual sentiment progression in both directions.",
    }

    if confidence_percent >= 92:
        confidence_comment = "Near-certain emotional polarity identified with exceptionally strong inference confidence."
    elif confidence_percent >= 82:
        confidence_comment = "High-confidence sentiment prediction generated from stable linguistic indicators."
    elif confidence_percent >= 70:
        confidence_comment = "Moderately strong emotional patterns detected across review structure."
    else:
        confidence_comment = "Mixed or ambiguous emotional signals observed within review semantics."

    if sentiment == "Positive":
        sentiment_comment = "Audience satisfaction, approval-oriented phrasing, and positive engagement markers were detected."
    else:
        sentiment_comment = "Negative emotional polarity and dissatisfaction-oriented language patterns were identified."

    final_insight = (
        f"{model_insights[model_type]} "
        f"{confidence_comment} "
        f"{sentiment_comment}"
    )

    return final_insight


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        review = request.form.get("review", "").strip()
        model_type = request.form.get("model_type", "GRU")

        if not review:
            return render_template(
                "dashboard.html",
                error=("Review text cannot be empty."),
                review=review,
                selected_model=model_type,
            )

        start_time = time.time()
        prediction = pipeline.predict(text=review, model_type=model_type)
        end_time = time.time()
        inference_time = round(end_time - start_time, 2)

        ai_insights = generate_ai_insights(
            sentiment=prediction["sentiment"],
            confidence=prediction["confidence"],
            model_type=prediction["model"],
        )

        return render_template(
            "dashboard.html",
            prediction=prediction,
            review=review,
            selected_model=model_type,
            ai_insights=ai_insights,
            inference_time=inference_time,
        )

    except Exception as e:

        return render_template(
            "dashboard.html",
            error=("Inference failed. " "Please try again."),
            review=request.form.get("review", ""),
            selected_model=request.form.get("model_type", "GRU"),
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)