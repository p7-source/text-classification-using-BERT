import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        """
        Initializes the PredictionPipeline class.
        """
        self.model_path = Path("artifacts/training/trained_model")  # Path to the trained model
        self.sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Loads the trained model and tokenizer.
        """
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise e

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment of the input text.

        Args:
            text (str): Input text for sentiment prediction.

        Returns:
            str: Predicted sentiment ('negative', 'neutral', 'positive').
        """
        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            # Get model predictions
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Map the predicted class to sentiment
            return self.sentiment_map[predicted_class]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e