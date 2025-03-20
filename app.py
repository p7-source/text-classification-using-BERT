from flask import Flask, request, jsonify, render_template_string
from textClassifier.pipeline.prediction import PredictionPipeline
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the PredictionPipeline
try:
    sentiment_inference = PredictionPipeline()
    logger.info("PredictionPipeline initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PredictionPipeline: {e}")
    raise e

# Home page route
@app.route("/", methods=["GET"])
def home():
    return """
    <h1>Welcome to Sentiment Analysis / Text Classification</h1>
    <p>Click <a href="/predict">here</a> to go to the prediction page.</p>
    """

# Predict page route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Render a basic HTML form for text input
        return render_template_string('''
        <h1>Sentiment Prediction</h1>
        <form method="POST">
            <label for="text">Enter your text:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
        ''')
    elif request.method == "POST":
        # Get the input text from the form
        text = request.form.get("text")
        
        # Validate the input
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Predict the sentiment
        try:
            sentiment = sentiment_inference.predict(text)
            logger.info(f"Prediction successful for text: {text}")
            return jsonify({"sentiment": sentiment})
        except Exception as e:
            logger.error(f"Prediction failed for text: {text}. Error: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)